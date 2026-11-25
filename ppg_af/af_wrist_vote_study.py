"""Compare immediate and two-of-three alerting on the 10-second RDDM grid."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, load_grid, predecessor_decisions, scoring_policy, scoring_runs
from .af_wrist_cv import train_oof
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_history import apply_alert_history
from .af_wrist_holdout import _shared_features
from .af_wrist_operating import score_decisions, select_scored_threshold
from .af_wrist_train import load_features
from .af_wrist_vote import apply_two_of_three_alerts
from .af_wrist_vote_compare import compare_alert_runs


STRIDE_SECONDS = 10
_IMMEDIATE = partial(apply_alert_history, confirmation_windows=1)


def _select_matched(evaluated, baseline, tolerance):
    if baseline["threshold"] is None or not evaluated:
        return {"status": "insufficient_reference", "threshold": None, "validation": None}
    baseline_sensitivity = baseline["validation"]["episode_metrics"]["complete_episode_recall"]

    def rank(candidate):
        cutoff, metrics = candidate
        sensitivity = metrics["episode_metrics"]["complete_episode_recall"]
        difference = abs(sensitivity - baseline_sensitivity)
        false_alerts = metrics["burden_metrics"]["false_alert_starts"]
        if difference <= tolerance + 1e-12:
            return (0, false_alerts, difference, -sensitivity, -cutoff)
        return (1, difference, false_alerts, -sensitivity, -cutoff)

    cutoff, validation = min(evaluated.items(), key=rank)
    status = "within_tolerance" if rank((cutoff, validation))[0] == 0 else "outside_tolerance"
    return {"threshold": float(cutoff), "status": status, "validation": validation}


def fit_comparison(grid, lock):
    shared = _shared_features(grid.table)
    features = np.where(shared[:, None], grid.table["rddm"], np.nan)
    fit_rows = shared & (grid.table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        full = np.full(len(grid.subjects), np.nan)
        full[rows] = scores
        subjects = set(grid.subjects[rows])
        evaluate, pool = scoring_policy(grid, full, subjects, _IMMEDIATE)
        options = {"target_sensitivity": lock["target_sensitivity"], "max_thresholds": max(3, len(pool) + 1)}
        baseline = select_scored_threshold(pool, evaluate, **options)
        evaluate_vote, _ = scoring_policy(grid, full, subjects, apply_two_of_three_alerts)
        evaluated = {}

        def record(cutoff):
            metrics = evaluate_vote(cutoff)
            evaluated[cutoff] = metrics
            return metrics

        target = select_scored_threshold(pool, record, **options)
        matched = (_select_matched(evaluated, baseline, lock["sensitivity_tolerance"])
                   if target["threshold"] is not None else target)
        return {**baseline, "vote_target": target, "vote_matched": matched}

    return train_oof({"rddm": features}, grid.table["stable_label"], grid.subjects, n_splits=5, inner_splits=3,
                     seed=2026, train_eligible=fit_rows, calibration_selector=calibrate)


def _fold_cutoffs(fitted, variant):
    assignments = fitted["fold_assignment"]
    cutoffs = np.full(len(assignments), np.nan)
    for fold in fitted["routes"]["rddm"]["folds"]:
        options = fold["operating_point"]
        selected = options if variant == "immediate" else options.get(variant, {})
        if selected.get("threshold") is not None:
            cutoffs[assignments == fold["fold"]] = selected["threshold"]
    return cutoffs


def _held_out_replay(grid, scores, cutoffs, replay):
    known = np.isfinite(scores) & np.isfinite(cutoffs)
    decisions = np.where(known, scores >= cutoffs, -1).astype(np.int8)
    history = predecessor_decisions(grid, decisions)
    monitor, episodes = scoring_runs(grid, np.where(known, decisions, np.nan), set(grid.subjects))
    wrist, monitor_unknown = replay((run.decisions(0.5) for run in monitor), prior_decisions=history)
    reference, episode_unknown = replay((run.decisions(0.5) for run in episodes), prior_decisions=history)
    metrics = score_decisions(wrist, episode_runs=reference)
    summary = _summarize({**metrics, "monitor_blocks_with_unknown_cutoff_state": monitor_unknown,
                          "episode_blocks_with_unknown_cutoff_state": episode_unknown})
    return summary, reference, wrist


def _compare(baseline, candidate, lock):
    before, before_episodes, before_wrist = baseline
    after, after_episodes, after_wrist = candidate
    support = ("complete_episodes", "wrist_ecg_observed_seconds", "wrist_non_af_seconds")
    if any(before[key] != after[key] for key in support):
        raise ValueError("Candidates and baseline must contain identical reference exposure")
    difference = after["episode_sensitivity"] - before["episode_sensitivity"]
    return {"summary": after, "paired_changes": compare_alert_runs(before_episodes, after_episodes,
                                                                  before_wrist, after_wrist),
            "sensitivity_change": difference,
            "within_sensitivity_tolerance": abs(difference) <= lock["sensitivity_tolerance"] + 1e-12,
            "both_reach_sensitivity_target": min(before["episode_sensitivity"], after["episode_sensitivity"])
            >= lock["target_sensitivity"]}


def _fold_summary(fitted, variant):
    folds = fitted["routes"]["rddm"]["folds"]
    options = [fold["operating_point"] if variant == "immediate" else fold["operating_point"].get(variant, {})
               for fold in folds]
    statuses = sorted({option.get("status", "unavailable") for option in options})
    return {status: sum(option.get("status", "unavailable") == status for option in options) for status in statuses}


def compare_modes(grid, lock, previous):
    fitted = fit_comparison(grid, lock)
    scores = fitted["routes"]["rddm"]["scores"]
    immediate_cutoffs = _fold_cutoffs(fitted, "immediate")
    baseline = _held_out_replay(grid, scores, immediate_cutoffs, _IMMEDIATE)
    for key, value in baseline[0].items():
        if previous["rddm"][key] != value:
            raise ValueError(f"The immediate RDDM baseline failed to reproduce: {key}")
    fixed = _held_out_replay(grid, scores, immediate_cutoffs, apply_two_of_three_alerts)
    retuned = {}
    for variant in ("vote_target", "vote_matched"):
        retuned[variant] = {**_compare(baseline, _held_out_replay(grid, scores, _fold_cutoffs(fitted, variant),
                                                                apply_two_of_three_alerts), lock),
                            "inner_fold_calibration_status_counts": _fold_summary(fitted, variant)}
    return {"scope": "Nested participant-held-out development comparison",
            "development_participants": len(np.unique(grid.subjects)), "outer_folds": fitted["n_splits"],
            "decision_stride_seconds": STRIDE_SECONDS, "feature_window_seconds": WINDOW_SECONDS,
            "target_episode_sensitivity": lock["target_sensitivity"], "sensitivity_tolerance": lock["sensitivity_tolerance"],
            "rddm_immediate": {**baseline[0], "inner_fold_calibration_status_counts": _fold_summary(fitted, "immediate")},
            "rddm_vote_fixed_cutoffs": _compare(baseline, fixed, lock),
            "rddm_vote_recalibrated": retuned, "template_immediate_reference": previous["pulse_template"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-features", "stride-raw", "stride-features", "lock", "clock-results", "output"):
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
        previous = clock["warm_10s"]["nested_development_cv"]["1"]
        results = compare_modes(grid, lock, previous)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF voting comparison failed: {error}\n")
    print(f"Compared two-of-three confirmation on {results['development_participants']} development participants")


if __name__ == "__main__":
    main()
