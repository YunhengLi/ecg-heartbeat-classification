"""Test a recent timing feature and nested logistic regularization for template ECG."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_cv import train_oof
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features
from .af_wrist_operating import select_scored_threshold
from .af_wrist_template_recent import append_recent_template_feature
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _compare, _held_out_replay, _IMMEDIATE


ROUTE = "pulse_template"
DEFAULT_C = 1.0
CANDIDATE_C = (0.1, DEFAULT_C, 10.0)


def fit_template(grid, matrix, lock, regularization, *, participant_weighting=False, supplemental_fit_labels=None,
                 seed=2026):
    shared = _shared_features(grid.table)
    if not np.isfinite(matrix[shared]).all():
        raise ValueError("Template variants must retain every originally scoreable row")
    features = np.where(shared[:, None], matrix, np.nan)
    fit_rows = shared & (grid.table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        full = np.full(len(grid.subjects), np.nan)
        full[rows] = scores
        evaluate, pool = scoring_policy(grid, full, set(grid.subjects[rows]), _IMMEDIATE)
        return select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                       max_thresholds=max(3, len(pool) + 1))

    return train_oof({ROUTE: features}, grid.table["stable_label"], grid.subjects, n_splits=5, inner_splits=3,
                     seed=seed, train_eligible=fit_rows, calibration_selector=calibrate,
                     regularization=regularization, participant_weighting=participant_weighting,
                     supplemental_fit_labels=supplemental_fit_labels)


def verify_fit_support(baseline, candidate):
    """Ensure a feature variant preserves the original nested fit and score footprint."""
    if not np.array_equal(baseline["fold_assignment"], candidate["fold_assignment"]):
        raise ValueError("The candidate changed an outer participant fold")
    original, augmented = (item["routes"][ROUTE] for item in (baseline, candidate))
    if (not np.array_equal(np.isfinite(original["scores"]), np.isfinite(augmented["scores"]))
            or not np.array_equal(original["decisions"] >= 0, augmented["decisions"] >= 0)):
        raise ValueError("The candidate changed held-out score or alert availability")
    fields = ("fold", "train_subjects", "test_subjects", "inner_splits", "fit_af_windows",
              "fit_non_af_windows", "held_out_finite_scores")
    for before, after in zip(original["folds"], augmented["folds"], strict=True):
        if any(before[key] != after[key] for key in fields):
            raise ValueError("The candidate changed the original fold's training support")


def rank_calibration(option, regularization):
    """Rank a cutoff and C using calibration metrics, with C=1 preferred on ties."""
    preference = (regularization != DEFAULT_C, regularization)
    if option["threshold"] is None:
        return (2, 0, 0, *preference)
    sensitivity = option["validation"]["episode_metrics"]["complete_episode_recall"]
    false_starts = option["validation"]["burden_metrics"]["false_alert_starts"]
    if option["status"] == "target_met":
        return (0, false_starts, -sensitivity, *preference)
    return (1, -sensitivity, false_starts, *preference)


def select_regularization(fits):
    """Select each outer fold's C and cutoff using inner validation metrics only."""
    baseline = fits[DEFAULT_C]
    assignments = baseline["fold_assignment"]
    scores = np.full(len(assignments), np.nan)
    cutoffs = np.full(len(assignments), np.nan)
    counts, status = Counter(), Counter()
    for fitted in fits.values():
        if not np.array_equal(fitted["fold_assignment"], assignments):
            raise ValueError("All regularization candidates must use identical outer folds")

    for fold in range(baseline["n_splits"]):
        options = {value: fitted["routes"][ROUTE]["folds"][fold] for value, fitted in fits.items()}
        if any(option["fold"] != fold for option in options.values()):
            raise ValueError("Candidate calibration results must align by outer fold")
        chosen = min(options, key=lambda value: rank_calibration(options[value]["operating_point"], value))
        point = options[chosen]["operating_point"]
        rows = assignments == fold
        scores[rows] = fits[chosen]["routes"][ROUTE]["scores"][rows]
        if point["threshold"] is not None:
            cutoffs[rows] = point["threshold"]
        counts[str(chosen)] += 1
        status[point["status"]] += 1
    return scores, cutoffs, dict(sorted(counts.items())), dict(sorted(status.items()))


def _replay(grid, fitted):
    scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
    return _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)


def _calibration_status(fitted):
    counts = Counter(fold["operating_point"]["status"] for fold in fitted["routes"][ROUTE]["folds"])
    return dict(sorted(counts.items()))


def run_study(grid, lock, previous, augmented, recent_support):
    original = grid.table[ROUTE]
    if augmented.shape != (len(original), original.shape[1] + 1) or not np.array_equal(
        augmented[:, :-1], original, equal_nan=True,
    ):
        raise ValueError("The recent-feature candidate must retain the original feature columns")

    baseline_fit = fit_template(grid, original, lock, DEFAULT_C)
    baseline = _replay(grid, baseline_fit)
    if any(previous[key] != value for key, value in baseline[0].items()):
        raise ValueError("The original template ECG result did not reproduce")
    candidate_fit = fit_template(grid, augmented, lock, DEFAULT_C)
    comparison = _compare(baseline, _replay(grid, candidate_fit), lock)
    summary = comparison["summary"]
    passed_goal = (summary["episode_sensitivity"] >= lock["target_sensitivity"]
                   and summary["wrist_false_alerts"] <= baseline[0]["wrist_false_alerts"])
    print(f"Recent feature detected {summary['episodes_detected']}/{summary['complete_episodes']} episodes "
          f"with {summary['wrist_false_alerts']} false starts", flush=True)

    report = {
        "scope": "Nested participant-held-out development evaluation",
        "development_participants": len(np.unique(grid.subjects)), "outer_folds": baseline_fit["n_splits"],
        "decision_stride_seconds": STRIDE_SECONDS, "feature_window_seconds": WINDOW_SECONDS,
        "target_episode_sensitivity": lock["target_sensitivity"], "sensitivity_tolerance": lock["sensitivity_tolerance"],
        "baseline": {**baseline[0], "inner_calibration_status_counts": _calibration_status(baseline_fit)},
        "recent_feature": {**comparison, "input_support": recent_support, "met_target_without_extra_false_starts": passed_goal,
                           "inner_calibration_status_counts": _calibration_status(candidate_fit)},
    }
    if not passed_goal:
        fits = {DEFAULT_C: baseline_fit}
        for value in CANDIDATE_C:
            if value != DEFAULT_C:
                fits[value] = fit_template(grid, original, lock, value)
                print(f"Completed inner cutoff search at C={value:g} on the original features", flush=True)
        scores, cutoffs, selection_counts, status = select_regularization(fits)
        tuned = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
        report["nested_regularization"] = {**_compare(baseline, tuned, lock), "candidate_C": list(CANDIDATE_C),
                                           "selected_outer_folds_by_C": selection_counts,
                                           "inner_calibration_status_counts": status}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features",
                 "ptt-dir", "lock", "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, reserved = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(subjects) != (fit | calibration)
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use the locked development cohort and 10-second clock study")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        matrix, support = append_recent_template_feature(
            grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features,
            ptt_dir=args.ptt_dir,
        )
        report = run_study(grid, lock, previous, matrix, support)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Template ECG study failed: {error}\n")


if __name__ == "__main__":
    main()
