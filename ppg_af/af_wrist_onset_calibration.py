"""Compare wrist-only and pre-onset-aware cutoffs using the same inner scores."""

import argparse
from functools import cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_cv import train_oof
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import _replay_and_audit, build_quality_matrices
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status
from .af_wrist_train import load_features
from .af_wrist_transition_metrics import compare_pre_onset_alert_runs, pre_onset_alert_burden
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare


def _distinct_burden(metrics):
    diagnostics = metrics["alert_diagnostics"]
    if (diagnostics["wrist_false_starts"] != metrics["burden_metrics"]["false_alert_starts"]
            or diagnostics["complete_episodes"] != metrics["episode_metrics"]["complete_reference_episodes"]):
        raise ValueError("The diagnostic and original scorer must agree on wrist starts and complete episodes")
    return diagnostics["distinct_false_starts"]


def _fit(grid, matrix, lock):
    shared = _shared_features(grid.table)
    if not np.isfinite(matrix[shared]).all():
        raise ValueError("Calibration must retain all original model inputs")
    features = np.where(shared[:, None], matrix, np.nan)
    fit_rows = shared & (grid.table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        full = np.full(len(grid.subjects), np.nan)
        full[rows] = scores
        evaluator, pool = scoring_policy(grid, full, set(grid.subjects[rows]), _IMMEDIATE,
                                         alert_diagnostics=pre_onset_alert_burden)
        # The same actual inner predictions, cutoffs and causal replays feed
        # both choices. Only the burden ranking is changed.
        evaluate = cache(evaluator)
        options = {"target_sensitivity": lock["target_sensitivity"], "max_thresholds": max(3, len(pool) + 1)}
        baseline = select_scored_threshold(pool, evaluate, **options)
        onset = select_scored_threshold(pool, evaluate, **options, burden_rank=_distinct_burden)
        if any(baseline[key] != onset[key] for key in ("status", "threshold_candidates")):
            raise ValueError("Pre-onset calibration must preserve goal feasibility and the threshold pool")
        if baseline["status"] == "target_not_met" and (
            baseline["validation"]["episode_metrics"]["complete_episode_recall"]
            != onset["validation"]["episode_metrics"]["complete_episode_recall"]
        ):
            raise ValueError("Both rules must maximize sensitivity below target")
        return {**baseline, "onset_aware": onset}

    return train_oof({ROUTE: features}, grid.table["stable_label"], grid.subjects, n_splits=5, inner_splits=3,
                     seed=2026, train_eligible=fit_rows, calibration_selector=calibrate, regularization=DEFAULT_C)


def _calibration_summary(choice):
    validation = choice["validation"]
    return {"status": choice["status"],
            "episode_sensitivity": validation["episode_metrics"]["complete_episode_recall"],
            **validation["alert_diagnostics"]}


def evaluate_onset_calibration(grid, lock, matrix, evidence, previous):
    fit = _fit(grid, matrix, lock)
    route = fit["routes"][ROUTE]
    scores, baseline_cutoffs, baseline_decisions = fold_arrays(fit, route)
    baseline, baseline_audit = _replay_and_audit(grid, evidence, scores, baseline_cutoffs, baseline_decisions)
    if any(previous["summary"][key] != value for key, value in baseline[0].items()):
        raise ValueError("The published fixed-C quality-interaction baseline must reproduce")

    candidate_cutoffs = np.full(len(scores), np.nan)
    calibrations = []
    for fold in route["folds"]:
        before = fold["operating_point"]
        after = before["onset_aware"]
        if after["threshold"] is not None:
            candidate_cutoffs[fit["fold_assignment"] == fold["fold"]] = after["threshold"]
        calibrations.append({"fold": fold["fold"] + 1, "threshold_candidates": before["threshold_candidates"],
                             "cutoff_changed": after["threshold"] != before["threshold"],
                             "wrist_only": _calibration_summary(before), "onset_aware": _calibration_summary(after)})
    candidate_decisions = np.where(np.isfinite(scores - candidate_cutoffs), scores >= candidate_cutoffs, -1).astype(np.int8)
    if not np.array_equal(baseline_decisions >= 0, candidate_decisions >= 0):
        raise ValueError("The altered cutoff objective changed decision availability")
    candidate, candidate_audit = _replay_and_audit(grid, evidence, scores, candidate_cutoffs, candidate_decisions)

    before_burden = pre_onset_alert_burden(baseline[2], baseline[1])
    after_burden = pre_onset_alert_burden(candidate[2], candidate[1])
    onset = compare_pre_onset_alert_runs(baseline[1], candidate[1])
    if (before_burden["pre_onset_false_starts"] != onset["pre_onset_starts"]["baseline"]
            or after_burden["pre_onset_false_starts"] != onset["pre_onset_starts"]["candidate"]
            or before_burden["pre_onset_false_starts"] != previous["pre_onset_comparison"]["pre_onset_starts"]["candidate"]
            or before_burden["wrist_false_starts"] != baseline[0]["wrist_false_alerts"]
            or after_burden["wrist_false_starts"] != candidate[0]["wrist_false_alerts"]):
        raise ValueError("Combined burden must reconcile with the published and paired event diagnostics")

    return {
        "scope": "Exploratory development-only cutoff objective on fixed quality-interaction predictions",
        "development_participants": len(np.unique(grid.subjects)),
        "outer_folds": fit["n_splits"], "outer_seed": 2026, "logistic_C": DEFAULT_C,
        "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds,
        "target_episode_sensitivity": lock["target_sensitivity"],
        "inner_calibration_status_counts": _calibration_status(fit),
        "outer_folds_with_different_cutoffs": sum(item["cutoff_changed"] for item in calibrations),
        "inner_calibration_by_outer_fold": calibrations,
        "wrist_only": {"summary": baseline[0], **baseline_audit, "combined_burden": before_burden},
        "onset_aware": {**_compare(baseline, candidate, lock), **candidate_audit, "combined_burden": after_burden},
        "pre_onset_comparison": onset,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        quality = json.loads(args.quality_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, independent = (_subject_ids(lock[name]) for name in
                                         ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & independent or set(subjects) != fit | calibration
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()
                or quality["fixed_regularization"]["C"] != DEFAULT_C):
            raise ValueError("Use the locked development cohort and original quality-interaction comparison")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        matrices = build_quality_matrices(grid, args.development_raw, args.development_features,
                                          args.stride_raw, args.stride_features)
        evidence = load_input_evidence(grid, args.stride_raw, args.stride_features)
        previous = quality["fixed_regularization"]["quality_interactions"]
        report = evaluate_onset_calibration(grid, lock, matrices["quality_interactions"], evidence, previous)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF pre-onset calibration failed: {error}\n")
    print(f"Compared objectives on {report['outer_folds']} held-out development folds", flush=True)


if __name__ == "__main__":
    main()
