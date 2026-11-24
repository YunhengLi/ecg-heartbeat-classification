"""Compare causal PPG quality features on the original grouped wrist AF folds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import audit_route, fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_independent_false_audit import summarize_false_alerts
from .af_wrist_template_study import (
    CANDIDATE_C, DEFAULT_C, ROUTE, _calibration_status, fit_template, select_regularization, verify_fit_support,
)
from .af_wrist_train import load_features
from .af_wrist_transition_metrics import compare_pre_onset_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare, _held_out_replay
from .features import ECG_FEATURE_NAMES, ppg_features


_IRREGULARITY_COLUMNS = [ECG_FEATURE_NAMES.index(name) for name in
                         ("successive_interval_rmssd_over_median", "successive_interval_pnn50")]
_ROW_FIELDS = ("subject_id", "start_s", "run_id", "run_kind", "reference_valid")


def quality_interaction_features(template, quality):
    """Allow the impact of two rhythm-irregularity features to vary with PPG quality."""
    interactions = quality[:, None] * template[:, _IRREGULARITY_COLUMNS]
    return np.column_stack((template, quality, interactions))


def _archive_quality(raw_path, feature_path, *, stride_seconds):
    with (np.load(raw_path, allow_pickle=False) as raw,
          np.load(feature_path, allow_pickle=False) as feature):
        if (str(raw["subject_id"].item()) != raw_path.stem or raw["ppg_fs"].item() != PPG_RATE
                or any(not np.array_equal(raw[key], feature[key]) for key in _ROW_FIELDS)):
            raise ValueError("Raw and cached quality must refer to the same participant and decision rows")
        current = feature[ROUTE].copy()
        if stride_seconds != WINDOW_SECONDS:
            end = raw["decision_end_s"]
            if (raw["decision_stride_seconds"].item() != stride_seconds
                    or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                    or not np.array_equal(end - WINDOW_SECONDS, raw["input_start_s"])
                    or not np.array_equal(end - stride_seconds, raw["start_s"])):
                raise ValueError("Quality must use the locked trailing PPG input")
            current[raw["input_status"] == "unknown"] = np.nan
        quality = feature["quality"]
        core = raw["ppg"][:, PPG_RATE:-PPG_RATE]
        if quality.shape != (len(current),) or core.shape != (len(current), WINDOW_SECONDS * PPG_RATE):
            raise ValueError("Quality values must align with the thirty-second PPG windows")
        rows = np.isfinite(current).all(axis=1)
        recalculated = np.array([ppg_features(pulse, PPG_RATE)[1] for pulse in core[rows]])
        if not np.allclose(quality[rows], recalculated, rtol=1e-10, atol=1e-12):
            raise ValueError("Cached quality disagrees with the trailing native PPG")
        return current, quality


def build_quality_matrices(grid, development_raw, development_features, stride_raw, stride_features):
    """Return original, quality-only and quality-interaction matrices in grid row order."""
    quality_blocks, feature_blocks, people = [], [], []
    for seconds, raw_dir, feature_dir in ((WINDOW_SECONDS, development_raw, development_features),
                                          (grid.seconds, stride_raw, stride_features)):
        for subject in np.unique(grid.subjects):
            name = f"{subject}.npz"
            current, quality = _archive_quality(raw_dir / name, feature_dir / name, stride_seconds=seconds)
            quality_blocks.append(quality)
            feature_blocks.append(current)
            people.extend([subject] * len(current))
    base = grid.table[ROUTE]
    if (not np.array_equal(people, grid.subjects)
            or not np.array_equal(np.concatenate(feature_blocks), base, equal_nan=True)):
        raise ValueError("Cached quality order must match the locked model rows")
    quality = np.concatenate(quality_blocks)
    shared = _shared_features(grid.table)
    if not ((quality[shared] >= 0) & (quality[shared] <= 1)).all():
        raise ValueError("Every shared model row needs a finite quality value in [0, 1]")
    return {"baseline": base, "quality": np.column_stack((base, quality)),
            "quality_interactions": quality_interaction_features(base, quality)}


def _replay_and_audit(grid, evidence, scores, cutoffs, decisions):
    replay = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    audited, cases = audit_route(grid, 1, scores, cutoffs, decisions, evidence, replay[0])
    false_alerts = summarize_false_alerts(grid, scores - cutoffs, 0.0, cases["false_alerts"], evidence)
    return replay, {"false_alert_audit": false_alerts, "missed_reasons": audited["missed_reasons"]}


def _compare_with_pre_onset(baseline, candidate, lock):
    return {**_compare(baseline, candidate, lock),
            "pre_onset_comparison": compare_pre_onset_alert_runs(baseline[1], candidate[1])}


def evaluate_quality(grid, lock, previous, matrices, evidence):
    fixed_fits, fixed_replays, fixed_results = {}, {}, {}
    for name, matrix in matrices.items():
        fitted = fit_template(grid, matrix, lock, DEFAULT_C)
        if fixed_fits:
            verify_fit_support(fixed_fits["baseline"], fitted)
        scores, cutoffs, decisions = fold_arrays(fitted, fitted["routes"][ROUTE])
        replay, audit = _replay_and_audit(grid, evidence, scores, cutoffs, decisions)
        if name == "baseline" and any(previous[key] != value for key, value in replay[0].items()):
            raise ValueError("The fixed C=1 template baseline must reproduce the clock study")
        fixed_fits[name], fixed_replays[name] = fitted, replay
        comparison = {"summary": replay[0]} if name == "baseline" else _compare_with_pre_onset(
            fixed_replays["baseline"], replay, lock,
        )
        fixed_results[name] = {**comparison, **audit, "inner_calibration_status_counts": _calibration_status(fitted)}
        print(f"{name} at C=1: {replay[0]['episodes_detected']}/{replay[0]['complete_episodes']} detections, "
              f"{replay[0]['wrist_false_alerts']} wrist false starts", flush=True)

    tuned_replays, tuned_results = {}, {}
    for name in ("baseline", "quality"):
        fits = {DEFAULT_C: fixed_fits[name]}
        for value in CANDIDATE_C:
            if value != DEFAULT_C:
                candidate = fit_template(grid, matrices[name], lock, value)
                verify_fit_support(fits[DEFAULT_C], candidate)
                fits[value] = candidate
        scores, cutoffs, selection, status = select_regularization(fits)
        decisions = np.where(np.isfinite(scores - cutoffs), scores >= cutoffs, -1).astype(np.int8)
        default_scores, _, default_decisions = fold_arrays(fits[DEFAULT_C], fits[DEFAULT_C]["routes"][ROUTE])
        if (not np.array_equal(np.isfinite(scores), np.isfinite(default_scores))
                or not np.array_equal(decisions >= 0, default_decisions >= 0)):
            raise ValueError("Inner selection changed the default candidate's score availability")
        replay, audit = _replay_and_audit(grid, evidence, scores, cutoffs, decisions)
        tuned_replays[name] = replay
        comparison = {"summary": replay[0]} if name == "baseline" else _compare_with_pre_onset(
            tuned_replays["baseline"], replay, lock,
        )
        tuned_results[name] = {**comparison, **audit, "selected_C_counts": selection,
                               "inner_calibration_status_counts": status}
        print(f"{name} with inner-selected C: {replay[0]['episodes_detected']}/{replay[0]['complete_episodes']} "
              f"detections, {replay[0]['wrist_false_alerts']} wrist false starts", flush=True)

    return {
        "scope": "Development-only causal PPG quality comparison on the original participant folds",
        "development_participants": len(np.unique(grid.subjects)),
        "outer_folds": fixed_fits["baseline"]["n_splits"],
        "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds,
        "target_episode_sensitivity": lock["target_sensitivity"], "sensitivity_tolerance": lock["sensitivity_tolerance"],
        "verified_shared_quality_rows": int(_shared_features(grid.table).sum()),
        "fixed_regularization": {"C": DEFAULT_C, **fixed_results},
        "nested_regularization_followup": {
            "timing": "Specified after observing the fixed-C comparisons",
            "candidate_C": list(CANDIDATE_C), **tuned_results,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, independent = (_subject_ids(lock[name]) for name in
                                         ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & independent or set(subjects) != fit | calibration
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use only the original locked development cohort and clock study")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        matrices = build_quality_matrices(grid, args.development_raw, args.development_features,
                                          args.stride_raw, args.stride_features)
        evidence = load_input_evidence(grid, args.stride_raw, args.stride_features)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        _write_json(args.output, evaluate_quality(grid, lock, previous, matrices, evidence))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF quality study failed: {error}\n")


if __name__ == "__main__":
    main()
