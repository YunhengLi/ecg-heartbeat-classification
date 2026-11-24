"""Compare waveform reconstruction on exact paired native wrist development windows."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_clock_study import load_grid
from .af_wrist_cohort import CONTEXT, WINDOW, core_label, in_wrist_segment
from .af_wrist_cv import _subject_folds
from .af_wrist_data_preflight import _inventory
from .af_wrist_expansion_stage import _open_source
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_train import load_features
from .af_wrist_training_expansion import _check_alignment
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import ECG_FS, _read_vector, _sample_bounds, load_subject
from .features import ecg_peaks
from .tcn_metrics import _peak_score, summarize_beats
from .template import fit_template_from_stats, template_stats
from .waveform import fit_waveform_from_stats, normalize_ecg, waveform_stats


SEED = 2026
FOLDS = 5
RIDGE = 0.01
EDGE = PPG_RATE
NATIVE_SAMPLES = int(ECG_FS) * WINDOW
ARCHIVE_ASSUMED_DELAY_MS = 300


def _stable_rows(raw_dir, feat_dir, subject):
    fields = ("subject_id", "start_s", "stable_label", "reference_valid", "run_kind", "sync_method")
    name = f"{subject}.npz"
    with (np.load(raw_dir / name, allow_pickle=False) as parent, np.load(feat_dir / name, allow_pickle=False) as feat):
        if str(parent["subject_id"].item()) != subject or any(not np.array_equal(parent[k], feat[k]) for k in fields):
            raise ValueError("Original paired row, ECG labels, and training feature clocks must match")
        rows = np.flatnonzero(_shared_features(feat) & (feat["stable_label"] >= 0) & (feat["run_kind"] != "enriched_episode"))
        if not parent["reference_valid"][rows].all():
            raise ValueError("Every original training row needs ECG reference")
        return list(zip(parent["start_s"][rows], parent["stable_label"][rows], parent["ppg"][rows], parent["ecg"][rows],
                        parent["sync_method"][rows], ["original"] * len(rows), strict=True))


def _supplemental_rows(path, receipt, lock_hash, development_raw, stride_raw, roster):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if receipt["extra_bundle_sha256"] != digest or receipt["source_protocol_sha256"] != lock_hash:
        raise ValueError("Supplemental pair archive must match its staging receipt")
    with np.load(path, allow_pickle=False) as expanded:
        if (expanded["source_protocol_sha256"].item() != lock_hash or expanded["ppg_fs"].item() != PPG_RATE
                or expanded["ecg_reference_fs"].item() != ECG_FS or expanded["feature_window_seconds"].item() != WINDOW
                or expanded["ppg_flank_seconds"].item() != CONTEXT):
            raise ValueError("Expansion archive must share the native clock protocol")
        people, starts, ends, labels, pulses, methods = (expanded[key] for key in (
            "subjects", "start_s", "decision_end_s", "stable_label", "ppg_with_context", "sync_method"))
    if (len(people) != 109 or not set(people) <= roster or not np.isin(labels, (0, 1)).all()
            or pulses.shape != (109, (WINDOW + 2 * CONTEXT) * PPG_RATE)):
        raise ValueError("Use the locked expansion of 109 ECG-verified stable windows")
    _check_alignment(people, labels, starts, ends, development_raw, stride_raw)
    declared = {person: [] for person in roster}
    for person, start, label, wave, method in zip(people, starts, labels, pulses, methods, strict=True):
        declared[str(person)].append((start, label, wave, None, method, "supplemental"))
    return declared


def _read_native(subject, raw):
    reference = subject.reference_rhythm_intervals()
    native_r = subject._reference[0]
    all_rows = []
    for start, label, stored_ppg, stored_ecg, method, origin in raw:
        end = start + WINDOW
        segment = in_wrist_segment(subject, start - CONTEXT, end + CONTEXT)
        if segment is None or segment.sync_method != method:
            raise ValueError("The stored PPG method disagrees with the native segment clock")
        ppg_data = subject._ppg_data[segment.index]
        low, high = _sample_bounds(start - CONTEXT, end + CONTEXT, segment.start_time, PPG_RATE, ppg_data.size)
        support = _read_vector(ppg_data, slice(low, high))
        if not np.array_equal(support, stored_ppg, equal_nan=True):
            raise ValueError("A staged padded PPG row did not reproduce from the native source")
        pulse = support[PPG_RATE:-PPG_RATE]
        lo = int(start * ECG_FS)
        measured = _read_vector(subject._voltage, slice(lo, lo + NATIVE_SAMPLES))
        if stored_ecg is not None and not np.array_equal(measured, stored_ecg, equal_nan=True):
            raise ValueError("An original 500 Hz staged ECG row did not reproduce")
        if (pulse.shape != (WINDOW * PPG_RATE,) or not np.isfinite(pulse).all() or np.ptp(pulse) <= 0
                or measured.shape != (NATIVE_SAMPLES,) or not np.isfinite(measured).all() or np.ptp(measured) <= 0
                or core_label(reference, start, end, measured) != (label, True)):
            raise ValueError("Every window must preserve its native paired stable reference")

        phase = (segment.start_time - start) + (low + PPG_RATE) / PPG_RATE
        # Strip subtraction roundoff at the several-day source time offset.
        phase = round(float(phase), 8) or 0.0
        if not 0.0 <= phase < 1.0 / PPG_RATE:
            raise ValueError("The native PPG core starts outside its nominal sample cell")
        decimated = resample_poly(measured, 1, int(ECG_FS) // PPG_RATE)
        grid = np.arange(len(pulse))
        target = np.interp(grid + phase * PPG_RATE, grid, decimated, right=np.nan)
        peaks = native_r[np.searchsorted(native_r, start):np.searchsorted(native_r, end)]
        peaks = (peaks - start - phase) * PPG_RATE
        all_rows.append((subject.subject_id, start, label, origin, method, phase, pulse, measured, target, peaks))
    return all_rows


def _metrics(targets, ppg, annotations, waveforms):
    observed = np.isfinite(targets) & np.isfinite(ppg)
    observed[:, :EDGE] = False
    observed[:, -EDGE:] = False
    common = observed & np.logical_and.reduce([np.isfinite(values) for values in waveforms.values()])
    available = int(observed.sum())
    support = int(common.sum())
    if not support:
        raise ValueError("Every reported rhythm and synchronization stratum needs shared waveform support")
    zero_errors = [np.mean(np.abs(target[mask])) for target, mask in zip(targets, common, strict=True) if mask.any()]
    groups = {}
    for name, prediction in waveforms.items():
        errors, correlations, beats = [], [], []
        for target, forecast, peaks, mask in zip(targets, prediction, annotations, common, strict=True):
            if not mask.any():
                continue
            actual, estimate = target[mask], forecast[mask]
            errors.append(float(np.mean(np.abs(actual - estimate))))
            if len(actual) > 1 and np.std(actual) > 0 and np.std(estimate) > 0:
                correlations.append(float(np.corrcoef(actual, estimate)[0, 1]))
            beats.append(_peak_score(peaks, ecg_peaks(forecast, PPG_RATE), mask, PPG_RATE))
        groups[name] = {"normalized_mae": float(np.mean(errors)),
                        "median_waveform_correlation": float(np.median(correlations)) if correlations else None,
                        "correlation_windows": len(correlations), "own_support_fraction": float(
                            (observed & np.isfinite(prediction)).sum() / available), "native_qrs_scores": summarize_beats(beats)}
    return {"windows": len(targets), "trimmed_measured_samples": available, "common_samples": support,
            "common_samples_fraction": support / available, "windows_with_common_support": int(common.any(axis=1).sum()),
            "zero_waveform_normalized_mae": float(np.mean(zero_errors)),
            "model_metrics": groups}


def participant_template_stats(subjects, ppg, ecg):
    """Build one 100 Hz template-statistics record per paired participant."""
    subjects, ppg, ecg = np.asarray(subjects), np.asarray(ppg), np.asarray(ecg)
    if (subjects.ndim != 1 or ppg.shape != (len(subjects), WINDOW * PPG_RATE) or ppg.shape != ecg.shape
            or not len(subjects)):
        raise ValueError("Provide participant IDs and aligned 30-second ECG/PPG cores at 100 Hz")
    result = {}
    for person in np.unique(subjects):
        rows = subjects == person
        result[person] = template_stats(zip(ppg[rows], ecg[rows], strict=True), fs=PPG_RATE)
    return result


def _evaluate(rows, outer, ptt_dir):
    people = np.asarray([row[0] for row in rows])
    labels = np.asarray([row[2] for row in rows])
    methods = np.asarray([row[4] for row in rows])
    pulse = np.stack([row[6] for row in rows])
    measured = np.stack([row[8] for row in rows])
    target = np.stack([normalize_ecg(signal) for signal in measured])
    annotations = [row[9] for row in rows]
    known = np.asarray([outer[person] for person in people])
    _, fixed = fit_ptt_calibration(ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    predictions = {"external_ptt_template": np.stack([fixed.predict(row) for row in pulse]),
                   "local_held_out_template": np.full(target.shape, np.nan),
                   "local_held_out_ridge": np.full(target.shape, np.nan)}
    templates = participant_template_stats(people, pulse, measured)
    ridges = {}
    for person in templates:
        select = people == person
        ridges[person] = waveform_stats(zip(pulse[select], measured[select], strict=True), fs=PPG_RATE)
    fold_receipts = []
    for fold in range(FOLDS):
        train = [person for person in templates if outer[person] != fold]
        held_out = known == fold
        if not held_out.any():
            raise ValueError("Each development fold needs at least one paired test window")
        pulse_template = fit_template_from_stats(templates[person] for person in train)
        ridge = fit_waveform_from_stats((ridges[person] for person in train), ridge=RIDGE)
        for name, model in (("local_held_out_template", pulse_template), ("local_held_out_ridge", ridge)):
            predictions[name][held_out] = np.stack([model.predict(row) for row in pulse[held_out]])
        fold_receipts.append({"fold": fold, "locked_train_participants": sum(value != fold for value in outer.values()),
                              "locked_test_participants": sum(value == fold for value in outer.values()),
                              "paired_train_participants": len(train), "paired_test_participants": len(set(people[held_out])),
                              "paired_test_windows": int(held_out.sum()), "paired_test_af_windows": int(labels[held_out].sum()),
                              "local_template_delay_ms": pulse_template.delay_samples * 1000 / PPG_RATE})

    def group(mask, waveforms):
        return {"participants": len(set(people[mask])), **_metrics(target[mask], pulse[mask],
                [annotations[i] for i in np.flatnonzero(mask)], {k: values[mask] for k, values in waveforms.items()})}

    all_rows = np.ones(len(rows), dtype=bool)
    sync_methods = np.unique(methods)
    beat = methods == "beat"
    archived_timing = replace(fixed, delay_samples=round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000))
    timing_forecast = predictions["external_ptt_template"].copy()
    timing_forecast[beat] = np.stack([archived_timing.predict(row) for row in pulse[beat]])
    timing_models = {**predictions, "external_ptt_shape_with_archive_beat_delay": timing_forecast}
    return {"overall": group(all_rows, predictions),
            "by_verified_rhythm": {"AF": group(labels == 1, predictions), "non_AF": group(labels == 0, predictions)},
            "by_release_synchronization": {key: group(methods == key, predictions) for key in sync_methods},
            "fold_support": fold_receipts, "external_template_delay_ms": fixed.delay_samples * 1000 / PPG_RATE,
            "archive_clock_timing_diagnostic": {
                "fixed_delay_ms_for_beat_synchronization": ARCHIVE_ASSUMED_DELAY_MS,
                "coarse_uses_original_ptt_delay": True, "delay_selected_using_held_out_ecg": False,
                "all_four_models_share_sample_support": True, "overall": group(all_rows, timing_models),
                "by_release_synchronization": {key: group(methods == key, timing_models) for key in sync_methods},
            }}


def _run(args):
    manifest_bytes = args.source_manifest.read_bytes()
    lock_bytes = args.lock.read_bytes()
    manifest, lock = map(json.loads, (manifest_bytes, lock_bytes))
    fit, calibration, test = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    roster = fit | calibration
    table, people, timelines = load_features(args.development_features)
    if (fit & calibration or roster & test or set(people) != roster or (len(fit), len(calibration), len(test)) != (5, 5, 25)
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]):
        raise ValueError("Use only the locked original development cohort")
    _inventory(manifest, roster | test)
    files = {row["key"]: row for row in manifest["files"]}
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    eligible = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    fit_labels = np.where(eligible, grid.table["stable_label"], -1)
    assignments = _subject_folds(fit_labels, grid.subjects, FOLDS, SEED)
    outer = {person: int(assignments[grid.subjects == person][0]) for person in roster}
    specimen = json.loads(args.receipt.read_bytes())
    extras = _supplemental_rows(args.supplement, specimen, sha256(lock_bytes).hexdigest(),
                               args.development_raw, args.stride_raw, roster)
    rows = []
    for person in sorted(roster):
        declared = _stable_rows(args.development_raw, args.development_features, person) + extras[person]
        if not declared:
            continue
        if len({row[0] for row in declared}) != len(declared):
            raise ValueError("The expanded waveform study repeats an original clock input")
        with ExitStack() as stack:
            inputs = [_open_source(stack, files[f"{person}_{signal}.mat"], args.source_dir, args.cache_dir)
                      for signal in ("ECG", "PPG")]
            subject = stack.enter_context(load_subject(*inputs, subject_id=person, include_acceleration=False))
            rows.extend(_read_native(subject, declared))
    coverage = Counter((row[3], row[2]) for row in rows)
    exact = {("original", 0): int(np.sum(fit_labels == 0)), ("original", 1): int(np.sum(fit_labels == 1)),
             ("supplemental", 0): 66, ("supplemental", 1): 43}
    if coverage != exact or len(rows) != 223 or sum(exact["original", y] for y in (0, 1)) != 114:
        raise ValueError("The original 114 fitting rows and the 109 stable supplements must all supply paired signals")
    per_window_peak_counts = [len(row[9]) for row in rows]
    args.private_pair_output.parent.mkdir(parents=True, exist_ok=True)
    with args.private_pair_output.open("xb") as stream:
        np.savez_compressed(stream, subjects=[row[0] for row in rows], start_s=[row[1] for row in rows],
                            stable_label=[row[2] for row in rows], origin=[row[3] for row in rows], sync_method=[row[4] for row in rows],
                            ppg_phase_seconds=[row[5] for row in rows], ppg_core=np.stack([row[6] for row in rows]),
                            ecg_500hz=np.stack([row[7] for row in rows]), ecg_ppg_clock=np.stack([row[8] for row in rows]),
                            native_qrs_float_indices=np.concatenate([row[9] for row in rows]),
                            native_qrs_offsets=np.r_[0, np.cumsum(per_window_peak_counts)], source_protocol_sha256=sha256(lock_bytes).hexdigest())
    print(f"Staged and verified {len(rows)} exact paired native windows", flush=True)
    comparison = _evaluate(rows, outer, args.ptt_dir)
    return {"scope": "Development-only paired wrist PPG to ECG waveform benchmark",
            "source_doi": manifest["doi"], "source_inventory_sha256": sha256(manifest_bytes).hexdigest(),
            "original_protocol_sha256": sha256(lock_bytes).hexdigest(), "supplement_sha256": specimen["extra_bundle_sha256"],
            "private_paired_manifest_sha256": sha256(args.private_pair_output.read_bytes()).hexdigest(),
            "locked_participants": len(roster), "paired_participants": len({row[0] for row in rows}),
            "waveform_rate_hz": PPG_RATE, "reference_ecg_native_hz": int(ECG_FS), "window_duration_seconds": WINDOW,
            "trim_seconds_each_edge": EDGE / PPG_RATE, "beat_match_tolerance_ms": 100, "ridge_penalty": RIDGE,
            "external_template_ptt_subjects": PTT_SUBJECTS, "fold_seed": SEED,
            "signal_provenance": {origin: {name: coverage[origin, label] for label, name in ((0, "non_AF"), (1, "AF"))}
                                  for origin in ("original", "supplemental")},
            "native_ppg_phase_ms_counts": dict(Counter(str(round(row[5] * 1000, 2)) for row in rows)),
            "verified_500Hz_qrs_annotations_in_windows": sum(per_window_peak_counts),
            "independent_waveforms_or_predictions": 0, "network_requests": 0, **comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = ("source-manifest", "source-dir", "cache-dir", "development-raw", "development-features", "stride-raw",
                "stride-features", "lock", "supplement", "receipt", "ptt-dir", "private-pair-output", "output")
    for name in required:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_pair_output.exists() or args.output.resolve() == args.private_pair_output.resolve():
        parser.error("Use distinct new public aggregate and private pair outputs")
    try:
        _write_json(args.output, _run(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Native wrist waveform study failed: {error}\n")


if __name__ == "__main__":
    main()
