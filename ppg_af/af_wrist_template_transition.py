"""Test ECG-verified onset examples on the original template development folds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_template_study import DEFAULT_C, ROUTE, fit_template
from .af_wrist_train import load_features
from .af_wrist_transition_examples import select_transition_examples
from .af_wrist_transition_metrics import compare_pre_onset_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare, _held_out_replay


def _replay(grid, fitted):
    scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
    return _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)


def _calibration_counts(folds):
    return dict(Counter(fold["operating_point"]["status"] for fold in folds))


def evaluate_transition(grid, original_timelines, raw_dir, lock, previous):
    labels, support = select_transition_examples(grid, original_timelines, raw_dir)
    if not np.isin((0, 1), labels).all():
        raise ValueError("A two-class onset experiment needs verified supplemental examples of both rhythms")
    print(f"Selected {np.count_nonzero(labels == 0)} pre-onset and {np.count_nonzero(labels == 1)} "
          "early-AF training windows", flush=True)

    matrix = grid.table[ROUTE]
    original_fit = fit_template(grid, matrix, lock, DEFAULT_C)
    baseline = _replay(grid, original_fit)
    if any(previous[key] != value for key, value in baseline[0].items()):
        raise ValueError("The original development baseline did not reproduce")
    candidate_fit = fit_template(grid, matrix, lock, DEFAULT_C, supplemental_fit_labels=labels)
    if not np.array_equal(original_fit["fold_assignment"], candidate_fit["fold_assignment"]):
        raise ValueError("The supplemental examples changed the original participant split")
    for key in ("scores", "decisions"):
        before_values = original_fit["routes"][ROUTE][key]
        after_values = candidate_fit["routes"][ROUTE][key]
        before_available = np.isfinite(before_values) if key == "scores" else before_values >= 0
        after_available = np.isfinite(after_values) if key == "scores" else after_values >= 0
        if not np.array_equal(before_available, after_available):
            raise ValueError("The supplemental examples changed the held-out score support")
    before_folds, after_folds = (fit["routes"][ROUTE]["folds"] for fit in (original_fit, candidate_fit))
    for before, after in zip(before_folds, after_folds, strict=True):
        if before["inner_splits"] != after["inner_splits"]:
            raise ValueError("The supplemental examples changed the number of supported inner folds")
    candidate = _replay(grid, candidate_fit)

    return {
        "scope": "Development-only onset-example ablation on the original grouped folds",
        "development_participants": len(np.unique(grid.subjects)), "outer_folds": original_fit["n_splits"],
        "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds, "logistic_C": DEFAULT_C,
        "target_episode_sensitivity": lock["target_sensitivity"], "sensitivity_tolerance": lock["sensitivity_tolerance"],
        "supplemental_training_support": support,
        "additional_fit_windows_by_outer_fold": [
            {"AF": after["fit_af_windows"] - before["fit_af_windows"],
             "non_AF": after["fit_non_af_windows"] - before["fit_non_af_windows"]}
            for before, after in zip(before_folds, after_folds, strict=True)
        ],
        "baseline": {**baseline[0], "inner_calibration_status_counts": _calibration_counts(before_folds)},
        "transition_training": {**_compare(baseline, candidate, lock),
                                "inner_calibration_status_counts": _calibration_counts(after_folds)},
        "pre_onset_comparison": compare_pre_onset_alert_runs(baseline[1], candidate[1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-features", "stride-raw", "stride-features", "lock", "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        base, subjects, timelines = load_features(args.development_features)
        fit, calibration, reserved = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(subjects) != (fit | calibration)
                or base[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use only the original locked development cohort and clock study")
        grid = load_grid(base, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        _write_json(args.output, evaluate_transition(grid, timelines, args.stride_raw, lock, previous))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Template transition study failed: {error}\n")
    print("Compared onset examples on the original development folds")


if __name__ == "__main__":
    main()
