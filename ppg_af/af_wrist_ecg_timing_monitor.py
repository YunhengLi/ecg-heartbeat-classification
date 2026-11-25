"""Compare wrist PPG and native ECG timing on the same half-hour wear blocks."""

import argparse
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np

from .af_state import state_durations, state_summary
from .af_wrist_clock_study import _summarize
from .af_wrist_ecg_timing_replay import freeze
from .af_wrist_ecg_timing_stage import DURATIONS, _read_qrs, qrs_features
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import EARLIER_S, _cluster_uncertainty, _person_runs
from .af_wrist_operating import score_decisions
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_short_monitor import archive_for_both
from .af_wrist_short_retrospective import freeze_short
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_compare import compare_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import ECG_FS
from .features import ECG_FEATURE_NAMES


SHORT_RESULTS_SHA = "7ecb074b42ae0207631d37d732f5b8ee6dc5b46bfc9f7ea85c889e175a84b71c"


def monitoring_clock(optical_file, person, blocks, digest):
    with np.load(optical_file, allow_pickle=False) as optical:
        pulse, starts = optical["raw_ppg"], optical["start_s"]
        width = (BLOCK_SECONDS + EARLIER_S - WINDOW_SECONDS) // STRIDE_SECONDS + 1
        if (optical["subject_id"].item() != person or optical["plan_sha256"].item() != digest
                or starts.tolist() != [item["evaluation_s"][0] for item in blocks]
                or pulse.shape != (len(starts), width, WINDOW_SECONDS * PPG_RATE)):
            raise ValueError("The ECG timing sidecar must use the sealed native PPG archive and plan")
        end = starts[:, None] - EARLIER_S + WINDOW_SECONDS + STRIDE_SECONDS * np.arange(width)
        return (end * ECG_FS).astype(np.int64), sha256(pulse.tobytes()).hexdigest()


def stage_monitor_source(ecg_source, person, blocks, digest, optical, output):
    ends, optical_hash = monitoring_clock(optical, person, blocks, digest)
    with h5py.File(ecg_source, "r") as source:
        peaks, gaps, samples = _read_qrs(source)
    matrices = {f"features_{length}s": qrs_features(ends.ravel(), peaks, gaps, samples, length).reshape(
        *ends.shape, len(ECG_FEATURE_NAMES)) for length in DURATIONS}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.npz")
    np.savez_compressed(partial, subject_id=person, decision_end_samples=ends, plan_sha256=digest,
                        frozen_optical_sha256=optical_hash, **matrices)
    partial.replace(output)


def common_archives(path, sidecar, person, blocks, digest, template):
    optical, _ = archive_for_both(path, person, blocks, digest, template)
    ends, expected_hash = monitoring_clock(path, person, blocks, digest)
    with np.load(sidecar, allow_pickle=False) as source:
        if (source["subject_id"].item() != person or source["plan_sha256"].item() != digest
                or source["frozen_optical_sha256"].item() != expected_hash
                or not np.array_equal(source["decision_end_samples"], ends)):
            raise ValueError("Use native ECG timing only on its verified companion clock")
        electrical = {f"ecg_{length}": source[f"features_{length}s"] for length in DURATIONS}
    native = np.logical_and.reduce([np.isfinite(matrix).all(axis=-1) for matrix in electrical.values()])
    starts, reference, episodes, features, available = optical
    common = native & np.isfinite(features["short"]).all(axis=-1)
    joint = {name: np.where(common[..., None], values, np.nan) for name, values in
             {"original": features["baseline"], "short": features["short"], **electrical}.items()}
    for name, values in electrical.items():
        electrical[name] = np.where(native[..., None], values, np.nan)
    return optical, (starts, reference, episodes, joint, available), (starts, reference, episodes, electrical, native)


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    native_models, _ = freeze(args, protocol, model_lock)
    short, short_cutoffs, short_receipt = freeze_short(args, protocol, model_lock)
    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    prior = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    short_prior = _read_pinned(args.short_results, SHORT_RESULTS_SHA, public=True)
    original, roster, _, _ = _locked_model(model_lock, protocol)
    if (digest != prior["plan_sha256"] or digest != short_prior["plan_sha256"] or set(schedule) != roster
            or short_receipt != short_prior["fitting_receipt"]):
        raise ValueError("The monitoring schedule and short model must match the previously frozen inputs")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    routes = {"original_full": [], "short_full": [], "original_common": [], "short_common": []}
    for length in DURATIONS:
        routes[f"native_{length}s_common"] = []
        routes[f"native_{length}s_full"] = []
    for person, blocks in schedule.items():
        if not blocks:
            continue
        name = f"{person}.npz"
        before, common, native = common_archives(args.monitoring_cache / name, args.ecg_monitor / name,
                                                 person, blocks, digest, template)
        for cohort, archive, original_route in (("full", before, "baseline"), ("common", common, "original")):
            routes[f"original_{cohort}"].extend(_person_runs(person, archive, original["model"], original["cutoff"], original_route))
            routes[f"short_{cohort}"].extend(_person_runs(person, archive, short, short_cutoffs["short_matched"], "short"))
        for length, parameters in native_models.items():
            cutoff = parameters["thresholds"]["matched"]
            for cohort, archive in (("full", native), ("common", common)):
                routes[f"native_{length}s_{cohort}"].extend(_person_runs(person, archive, parameters["model"], cutoff, f"ecg_{length}"))
    summaries = {name: _summarize({**score_decisions(runs), "monitor_blocks_with_unknown_cutoff_state": 0,
                                   "episode_blocks_with_unknown_cutoff_state": 0}) for name, runs in routes.items()}
    if summaries["original_full"] != prior["original"] or summaries["short_full"] != short_prior["results"]["short_matched"]["summary"]:
        raise ValueError("Both published optical monitoring summaries must reproduce")
    output = {name: {"summary": summaries[name], "state": state_summary(state_durations(runs)),
                     "scoreable_rows": sum(int((run.decisions >= 0).sum()) for run in runs)} for name, runs in routes.items()}
    baseline = routes["short_common"]
    candidate = routes["native_10s_common"]
    return {"scope": "Short actual ECG timing versus short optical timing, with frozen development calibration",
            "plan_sha256": digest, "results": output,
            "short_ecg_vs_short_ppg_changes": compare_alert_runs(baseline, candidate, baseline, candidate),
            "short_ecg_vs_short_ppg_uncertainty": _cluster_uncertainty((summaries["short_common"], baseline, baseline),
                                                                      (summaries["native_10s_common"], candidate, candidate),
                                                                      sorted(roster))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "historical-model-lock", "prior-monitor-plan", "plan", "monitoring-cache",
                 "half-hour-results", "short-results", "ptt-dir", "ecg-timing", "ecg-monitor", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
