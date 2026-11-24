"""Replay an alert reset after two consecutive negative wrist decisions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_history import apply_negative_rearm
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import _compare_with_pre_onset, build_quality_matrices
from .af_wrist_template_study import DEFAULT_C, ROUTE, fit_template
from .af_wrist_train import load_features
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE, _select_matched, STRIDE_SECONDS


SEEDS = (2026, 2030)
MODELS = ("baseline", "quality_interactions")
CALIBRATED_POLICIES = ("target", "matched_immediate")


def _calibrate_inner(grid, lock):
    def calibrate(scores, training_rows):
        timeline_scores = np.full(len(grid.subjects), np.nan)
        timeline_scores[training_rows] = scores
        participants = set(grid.subjects[training_rows])
        immediate, pool = scoring_policy(grid, timeline_scores, participants, _IMMEDIATE)
        rearming, _ = scoring_policy(grid, timeline_scores, participants, apply_negative_rearm)
        settings = {"target_sensitivity": lock["target_sensitivity"], "max_thresholds": max(3, len(pool) + 1)}
        reference = select_scored_threshold(pool, immediate, **settings)
        evaluated = {}

        def record(point):
            report = rearming(point)
            evaluated[point] = report
            return report

        target = select_scored_threshold(pool, record, **settings)
        matched = (_select_matched(evaluated, reference, lock["sensitivity_tolerance"])
                   if target["threshold"] is not None else target)
        return {**reference, "target": target, "matched_immediate": matched}

    return calibrate


def _replay_recalibrated(grid, fitted, scores, original_cutoffs, baseline, lock):
    results = {}
    assignment = fitted["fold_assignment"]
    for policy in CALIBRATED_POLICIES:
        cutoffs = np.full(len(scores), np.nan)
        statuses = Counter()
        for fold in fitted["routes"][ROUTE]["folds"]:
            choice = fold["operating_point"].get(policy, {"status": "unavailable", "threshold": None})
            statuses[choice["status"]] += 1
            if choice["threshold"] is not None:
                cutoffs[assignment == fold["fold"]] = choice["threshold"]
        if not np.array_equal(np.isfinite(cutoffs), np.isfinite(original_cutoffs)):
            raise ValueError("Recalibration must preserve the same set of scoreable folds")
        candidate = _held_out_replay(grid, scores, cutoffs, apply_negative_rearm)
        results[policy] = {**_compare_with_pre_onset(baseline, candidate, lock),
                           "inner_calibration_status_counts": dict(statuses)}
    return results


def compare_rearming(grid, matrices, lock, original, *, recalibrate=False):
    results = {}
    calibration = _calibrate_inner(grid, lock) if recalibrate else None
    for seed in SEEDS:
        arms = {}
        for name in MODELS:
            fitted = fit_template(grid, matrices[name], lock, DEFAULT_C, seed=seed, calibration_selector=calibration)
            scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
            baseline = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
            if seed == SEEDS[0] and baseline[0] != original[name]["summary"]:
                raise ValueError("The original immediate alert study must reproduce before the policy comparison")
            candidate = _held_out_replay(grid, scores, cutoffs, apply_negative_rearm)
            for before_runs, after_runs in zip(baseline[1:], candidate[1:], strict=True):
                for before, after in zip(before_runs, after_runs, strict=True):
                    if (not np.array_equal(before.decisions, after.decisions)
                            or np.any((after.alert_start_status == 1) & (before.alert_start_status != 1))):
                        raise ValueError("Rearming may only suppress new starts on the unchanged decision clock")
            arms[name] = {"immediate": baseline[0], "two_negative_rearm": _compare_with_pre_onset(baseline, candidate, lock)}
            print(f"{seed} {name}: {baseline[0]['episodes_detected']} to {candidate[0]['episodes_detected']} detections, "
                  f"{baseline[0]['wrist_false_alerts']} to {candidate[0]['wrist_false_alerts']} false starts", flush=True)
            if recalibrate:
                alternatives = _replay_recalibrated(grid, fitted, scores, cutoffs, baseline, lock)
                arms[name]["recalibrated_two_negative_rearm"] = alternatives
                for policy, comparison in alternatives.items():
                    row = comparison["summary"]
                    print(f"{seed} {name} recalibrated {policy}: {row['episodes_detected']}/{row['complete_episodes']} "
                          f"detections, {row['wrist_false_alerts']} false starts", flush=True)
        results[str(seed)] = arms
    return {"scope": "Fixed-cutoff development replay after two consecutive observed negative decisions",
            "development_participants": len(np.unique(grid.subjects)), "outer_folds": 5,
            "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
            "threshold_recalibration": recalibrate, "negative_decisions_to_rearm": 2, "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--recalibrate", action="store_true", help="Compare additional thresholds selected only inside training folds")
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock, clock = json.loads(lock_bytes), json.loads(args.clock_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(subjects) != fit | calibration
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use only the original locked wrist development cohort and clock study")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        matrices = build_quality_matrices(grid, args.development_raw, args.development_features,
                                          args.stride_raw, args.stride_features)
        original = json.loads(args.quality_results.read_bytes())["fixed_regularization"]
        _write_json(args.output, compare_rearming(grid, matrices, lock, original, recalibrate=args.recalibrate))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF alert rearming study failed: {error}\n")


if __name__ == "__main__":
    main()
