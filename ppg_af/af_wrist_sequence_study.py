"""Compare ordered pulse context and an order-invariant control in grouped AF CV."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_sequence_features import build_sequence_features
from .af_wrist_template_study import (
    CANDIDATE_C, DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, select_regularization,
)
from .af_wrist_train import load_features
from .af_wrist_transition_metrics import compare_pre_onset_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _compare, _held_out_replay, _IMMEDIATE


def _verify_fit_support(baseline, candidate):
    if not np.array_equal(baseline["fold_assignment"], candidate["fold_assignment"]):
        raise ValueError("Temporal context changed an outer participant fold")
    original, augmented = (item["routes"][ROUTE] for item in (baseline, candidate))
    if (not np.array_equal(np.isfinite(original["scores"]), np.isfinite(augmented["scores"]))
            or not np.array_equal(original["decisions"] >= 0, augmented["decisions"] >= 0)):
        raise ValueError("Temporal context changed held-out score or alert availability")
    fields = ("fold", "train_subjects", "test_subjects", "inner_splits", "fit_af_windows",
              "fit_non_af_windows", "held_out_finite_scores")
    for before, after in zip(original["folds"], augmented["folds"], strict=True):
        if any(before[key] != after[key] for key in fields):
            raise ValueError("Temporal context changed the original fold's training support")


def _comparison(baseline, candidate, lock):
    return {**_compare(baseline, candidate, lock),
            "pre_onset_comparison": compare_pre_onset_alert_runs(baseline[1], candidate[1])}


def _choose_regularization(grid, matrix, lock, default_fit):
    fits = {DEFAULT_C: default_fit}
    for value in CANDIDATE_C:
        if value == DEFAULT_C:
            continue
        fitted = fit_template(grid, matrix, lock, value)
        _verify_fit_support(default_fit, fitted)
        fits[value] = fitted
    scores, cutoffs, selected, status = select_regularization(fits)
    replay = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    return replay, {"selected_C_counts": selected, "inner_calibration_status_counts": status}


def evaluate_sequence(grid, lock, previous, chronological, unordered, support):
    original_matrix = grid.table[ROUTE]
    if (chronological.ndim != 2 or chronological.shape != unordered.shape
            or len(chronological) != len(original_matrix) or chronological.shape[1] == 0):
        raise ValueError("The ordered and unordered models must receive matching context columns")

    baseline_fit = fit_template(grid, original_matrix, lock, DEFAULT_C)
    baseline = _replay(grid, baseline_fit)
    if any(previous[key] != value for key, value in baseline[0].items()):
        raise ValueError("The original template ECG baseline failed to reproduce")

    candidates, replays = {}, {}
    tuned_baseline, tuned_baseline_details = _choose_regularization(grid, original_matrix, lock, baseline_fit)
    tuned_candidates, tuned_replays = {}, {}
    for name, context in (("order_invariant", unordered), ("chronological", chronological)):
        matrix = np.column_stack((original_matrix, context))
        fitted = fit_template(grid, matrix, lock, DEFAULT_C)
        _verify_fit_support(baseline_fit, fitted)
        replay = _replay(grid, fitted)
        candidates[name] = {**_comparison(baseline, replay, lock),
                            "inner_calibration_status_counts": _calibration_status(fitted)}
        replays[name] = replay
        summary = replay[0]
        print(f"{name}: {summary['episodes_detected']}/{summary['complete_episodes']} detections, "
              f"{summary['wrist_false_alerts']} wrist false starts", flush=True)
        tuned, details = _choose_regularization(grid, matrix, lock, fitted)
        tuned_candidates[name] = {**_comparison(tuned_baseline, tuned, lock), **details}
        tuned_replays[name] = tuned
        print(f"{name} with inner-selected C: {tuned[0]['episodes_detected']}/{tuned[0]['complete_episodes']} "
              f"detections, {tuned[0]['wrist_false_alerts']} wrist false starts", flush=True)

    return {
        "scope": "Development-only PPG sequence comparison on the original participant folds",
        "development_participants": len(np.unique(grid.subjects)),
        "outer_folds": baseline_fit["n_splits"],
        "feature_window_seconds": WINDOW_SECONDS,
        "decision_stride_seconds": grid.seconds,
        "logistic_C": DEFAULT_C,
        "target_episode_sensitivity": lock["target_sensitivity"],
        "sensitivity_tolerance": lock["sensitivity_tolerance"],
        "context_support": support,
        "baseline": {**baseline[0], "inner_calibration_status_counts": _calibration_status(baseline_fit)},
        **candidates,
        "chronological_vs_order_invariant": _comparison(replays["order_invariant"], replays["chronological"], lock),
        "nested_regularization_followup": {
            "C_values": list(CANDIDATE_C),
            "baseline": {**tuned_baseline[0], **tuned_baseline_details},
            **tuned_candidates,
            "chronological_vs_order_invariant": _comparison(tuned_replays["order_invariant"],
                                                             tuned_replays["chronological"], lock),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments = ("development-raw", "development-features", "stride-raw", "stride-features", "lock",
                 "clock-results", "output")
    for name in arguments:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in
                                      ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(subjects) != fit | calibration
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use the original locked development cohort and clock study")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        chronological, unordered, support = build_sequence_features(
            grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features,
        )
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        results = evaluate_sequence(grid, lock, previous, chronological, unordered, support)
        _write_json(args.output, results)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF sequence study failed: {error}\n")


if __name__ == "__main__":
    main()
