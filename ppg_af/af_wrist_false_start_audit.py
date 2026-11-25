"""Compare native ECG timing at false AF starts with matched negative decisions."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .af_events import _event_at_decision, _union
from .af_wrist_direct_fusion import SPECTRAL_ROUTE
from .af_wrist_exercise import PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_monitor_extension import _archive_features, _person_runs
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import SPECTRAL_SHA, freeze_from_receipts, verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import load_subject
from .features import ECG_FEATURE_NAMES


RMSSD = ECG_FEATURE_NAMES.index("successive_interval_rmssd_over_median")
PNN50 = ECG_FEATURE_NAMES.index("successive_interval_pnn50")
Q = len(ECG_FEATURE_NAMES)
WIDTH = WINDOW_SECONDS // STRIDE_SECONDS


def stage_reference(ecg_source, wrist_source, person, blocks, digest, output):
    with load_subject(ecg_source, wrist_source, subject_id=person, include_acceleration=False) as subject:
        native = subject._reference[0]
        starts = [item["evaluation_s"][0] for item in blocks]
        intervals = [native[np.searchsorted(native, first):np.searchsorted(native, last, side="right")]
                     for first, last in (item["evaluation_s"] for item in blocks)]
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{person}.npz"
    partial = target.with_suffix(".partial.npz")
    np.savez_compressed(partial, start_s=np.asarray(starts), native_peaks=np.concatenate(intervals),
                        offsets=np.r_[0, np.cumsum([len(row) for row in intervals])], plan_sha256=digest)
    partial.replace(target)


def _reference_peaks(path, starts, digest):
    with np.load(path, allow_pickle=False) as source:
        offsets, peaks = source["offsets"], source["native_peaks"]
        if (str(source["plan_sha256"].item()) != digest or not np.array_equal(source["start_s"], starts)
                or offsets.shape != (len(starts) + 1,) or offsets[0] != 0 or offsets[-1] != len(peaks)
                or np.any(np.diff(offsets) < 0) or not np.isfinite(peaks).all()):
            raise ValueError("Reference beats must match the original frozen monitoring plan")
    return [peaks[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]


def non_af_context(run):
    ends = run.start_s + STRIDE_SECONDS * np.arange(1, len(run.decisions) + 1)
    overlaps = (run.af_intervals_s[:, 0] < ends[:, None]) & (run.af_intervals_s[:, 1] > (ends - WINDOW_SECONDS)[:, None])
    complete = np.r_[np.zeros(WIDTH - 1, dtype=bool),
                     np.lib.stride_tricks.sliding_window_view(run.reference_valid, WIDTH).all(axis=1)]
    return ends, overlaps.any(axis=1), complete


def native_irregularity(beats, end):
    start, stop = np.searchsorted(beats, [end - WINDOW_SECONDS, end])
    rr = np.diff(beats[start:stop])
    if len(rr) < 2 or np.any(rr <= 0):
        return np.nan
    changes = np.diff(rr)
    return float(np.sqrt(np.mean(np.square(changes))) / np.median(rr) + np.mean(np.abs(changes) > 0.05))


def _streak_length(decisions, first):
    trailing = np.flatnonzero(decisions[first:] != 1)
    return int(trailing[0]) if len(trailing) else None


def audit_run(run, peaks, template_features):
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("Verified source QRS marks must be sorted by native ECG time")
    ends, recent, complete = non_af_context(run)
    false = (run.alert_start_status == 1) & run.reference_valid & (_event_at_decision(ends, _union(run.af_intervals_s)) < 0)
    stable = complete & ~recent
    cases = np.flatnonzero(false & stable)
    available = np.flatnonzero((run.decisions == 0) & stable)
    output = Counter({"false_starts": int(false.sum()), "with_af_in_model_input": int(np.sum(false & recent)),
                      "fully_observed_30_second_non_af_input": len(cases),
                      "one_positive_decision_without_block_censoring": sum(_streak_length(run.decisions, row) == 1 for row in np.flatnonzero(false))})
    pairs = []
    for case in cases:
        candidates = available[np.abs(available - case) >= WIDTH]
        if not len(candidates):
            continue
        control = min(candidates, key=lambda row: (abs(row - case), row))
        native = [native_irregularity(peaks, ends[row]) for row in (case, control)]
        features = template_features[[case, control]]
        if not np.isfinite(native).all() or not np.isfinite(features).all():
            continue
        optical = features[:, RMSSD] + features[:, PNN50]
        pairs.append(np.column_stack((native, optical, optical - native, features[:, Q])))
        available = available[available != control]
    return output, pairs


def _summarize(pairs):
    if not pairs:
        return {"matched_pairs": 0}
    array = np.stack(pairs)
    flags = ("native_ecg_irregularity", "template_irregularity", "template_minus_native", "ppg_quality")
    return {"matched_pairs": len(array), "within_block_comparisons": {name: {
        "false_start_median": float(np.median(array[:, 0, index])), "negative_decision_median": float(np.median(array[:, 1, index])),
        "paired_median_change": float(np.median(array[:, 0, index] - array[:, 1, index]))}
        for index, name in enumerate(flags)}}


def evaluate(args, protocol, schedule, digest):
    models, _, _, _ = freeze_from_receipts(protocol, args.historical_model_lock, args.variant_lock)
    spectral = _read_pinned(args.spectral_model_lock, SPECTRAL_SHA)
    models[SPECTRAL_ROUTE] = spectral["frozen_model"]
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    counts = {name: Counter() for name in models}
    matched = {name: [] for name in models}
    for person, blocks in schedule.items():
        archive = _archive_features(args.monitoring_cache / f"{person}.npz", person, blocks, digest, template,
                                    families=tuple(models))
        peaks = _reference_peaks(args.reference_cache / f"{person}.npz", archive[0], digest)
        features = archive[3]["quality_interactions"][:, -archive[1].shape[1]:]
        for name, option in models.items():
            runs = _person_runs(person, archive, option["model"], option["cutoff"], name)
            for run, reference, optical in zip(runs, peaks, features, strict=True):
                exposure, pairs = audit_run(run, reference, optical)
                counts[name].update(exposure)
                matched[name].extend(pairs)
    historical = _read_pinned(args.monitoring_results, args.monitoring_results_sha256, public=True)
    expected = {"baseline": historical["baseline"]["wrist_false_alerts"],
                **{name: item["summary"]["wrist_false_alerts"] for name, item in historical["variants"].items()}}
    if historical["plan_sha256"] != digest or {name: value["false_starts"] for name, value in counts.items()} != expected:
        raise ValueError("The audit must reproduce all 25-person supplemental false-start counts")
    return {"scope": "Descriptive false-start audit on the previously scored 25-person wrist roster",
            "plan_sha256": digest, "routes": {name: {**counts[name], **_summarize(matched[name])} for name in models}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "audit"))
    for name in ("plan", "lock", "prior-inputs", "reference-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-dir", "monitoring-cache", "historical-model-lock", "variant-lock", "spectral-model-lock", "ptt-dir",
                 "monitoring-results", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--monitoring-results-sha256")
    args = parser.parse_args()
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    schedule, _, digest = verify_schedule(args.plan, args.lock, args.prior_inputs)
    if args.phase == "stage":
        if args.source_dir is None:
            parser.error("Staging needs the original local source files")
        for person, blocks in schedule.items():
            stage_reference(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat", person, blocks,
                            digest, args.reference_cache)
        return
    if not all((args.monitoring_cache, args.historical_model_lock, args.variant_lock, args.spectral_model_lock, args.ptt_dir,
                args.monitoring_results, args.monitoring_results_sha256, args.output)):
        parser.error("Auditing requires both source caches and all previous model and result receipts")
    _write_json(args.output, evaluate(args, protocol, schedule, digest))


if __name__ == "__main__":
    main()
