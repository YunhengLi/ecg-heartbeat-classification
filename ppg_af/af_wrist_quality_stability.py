"""Check a fixed quality-aware AF candidate across development fold seeds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_quality_study import build_quality_matrices
from .af_wrist_sequence_study import _comparison
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE


SEEDS = (2026, 2027, 2028, 2029, 2030)
FIXED_CUTOFF = 0.5


def _participant_partition(subjects, assignments, fold_count):
    return frozenset(frozenset(subjects[assignments == fold]) for fold in range(fold_count))


def _fixed_replay(grid, fitted):
    scores, original_cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
    cutoffs = np.where(np.isfinite(original_cutoffs), FIXED_CUTOFF, np.nan)
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    evaluated = evaluation[np.isfinite(scores[evaluation]) & np.isfinite(cutoffs[evaluation])]
    counts = {"scoreable_evaluation_decisions": len(evaluated),
              "positive_at_fold_cutoff": int(np.sum(scores[evaluated] >= original_cutoffs[evaluated])),
              "positive_at_fixed_cutoff": int(np.sum(scores[evaluated] >= FIXED_CUTOFF))}
    return _held_out_replay(grid, scores, cutoffs, _IMMEDIATE), counts


def evaluate_stability(grid, matrix, lock, original_summary):
    trials, partitions = [], set()
    for seed in SEEDS:
        baseline_fit = fit_template(grid, matrix["baseline"], lock, DEFAULT_C, seed=seed)
        candidate_fit = fit_template(grid, matrix["quality_interactions"], lock, DEFAULT_C, seed=seed)
        verify_fit_support(baseline_fit, candidate_fit)
        baseline, candidate = _replay(grid, baseline_fit), _replay(grid, candidate_fit)
        fixed_baseline, before_counts = _fixed_replay(grid, baseline_fit)
        fixed_candidate, after_counts = _fixed_replay(grid, candidate_fit)
        if seed == 2026 and any(original_summary[key] != value for key, value in baseline[0].items()):
            raise ValueError("The published baseline for the original fold seed failed to reproduce")
        partitions.add(_participant_partition(grid.subjects, baseline_fit["fold_assignment"], baseline_fit["n_splits"]))
        trials.append({
            "seed": seed,
            "baseline": {**baseline[0], "inner_calibration_status_counts": _calibration_status(baseline_fit)},
            "quality_interactions": {**_comparison(baseline, candidate, lock),
                                     "inner_calibration_status_counts": _calibration_status(candidate_fit)},
            "fixed_cutoff_diagnostic": {
                "baseline": {**fixed_baseline[0], "evaluation_decisions": before_counts},
                "quality_interactions": {**_comparison(fixed_baseline, fixed_candidate, lock),
                                         "evaluation_decisions": after_counts},
            },
        })
        print(f"Seed {seed}: template {baseline[0]['episodes_detected']}/{baseline[0]['complete_episodes']}, "
              f"{baseline[0]['wrist_false_alerts']} false starts; quality interactions "
              f"{candidate[0]['episodes_detected']}/{candidate[0]['complete_episodes']}, "
              f"{candidate[0]['wrist_false_alerts']} false starts", flush=True)

    baseline = [trial["baseline"] for trial in trials]
    candidate = [trial["quality_interactions"]["summary"] for trial in trials]
    summary = {}
    for label, values in (("baseline", baseline), ("quality_interactions", candidate)):
        summary[label] = {
            "episode_detection_range": [min(row["episodes_detected"] for row in values),
                                        max(row["episodes_detected"] for row in values)],
            "wrist_false_start_range": [min(row["wrist_false_alerts"] for row in values),
                                       max(row["wrist_false_alerts"] for row in values)],
            "seeds_at_sensitivity_target": sum(row["episode_sensitivity"] >= lock["target_sensitivity"] for row in values),
        }
    wins = sum(
        after["episodes_detected"] >= before["episodes_detected"]
        and after["wrist_false_alerts"] <= before["wrist_false_alerts"]
        and (after["episodes_detected"] > before["episodes_detected"]
             or after["wrist_false_alerts"] < before["wrist_false_alerts"])
        for before, after in zip(baseline, candidate, strict=True)
    )
    return {
        "scope": "Fixed-model split sensitivity on the same wrist development participants",
        "development_participants": len(np.unique(grid.subjects)),
        "outer_folds": baseline_fit["n_splits"],
        "logistic_C": DEFAULT_C,
        "target_episode_sensitivity": lock["target_sensitivity"],
        "diagnostic_fixed_cutoff": FIXED_CUTOFF,
        "distinct_outer_participant_partitions": len(partitions),
        "seeds_with_improvement_and_no_tradeoff": wins,
        "summary": summary,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock",
                 "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_raw = args.lock.read_bytes()
        lock = json.loads(lock_raw)
        clock = json.loads(args.clock_results.read_bytes())
        table, people, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in
                                      ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(people) != (fit | calibration)
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_raw).hexdigest()):
            raise ValueError("Use the original locked development cohort and clock study")
        grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        matrix = build_quality_matrices(grid, args.development_raw, args.development_features,
                                        args.stride_raw, args.stride_features)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        report = evaluate_stability(grid, matrix, lock, previous)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF split sensitivity failed: {error}\n")


if __name__ == "__main__":
    main()
