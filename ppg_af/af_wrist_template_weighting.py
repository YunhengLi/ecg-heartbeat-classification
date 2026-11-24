"""Compare participant-balanced template ECG fitting on locked development folds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import audit_route, fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_template_study import DEFAULT_C, ROUTE, fit_template
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare, _held_out_replay


def _training_support(grid):
    labels = grid.table["stable_label"]
    eligible = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    counts = {}
    for label, name in ((1, "AF"), (0, "non_AF")):
        subject_counts = np.unique(grid.subjects[eligible & (labels == label)], return_counts=True)[1]
        counts[name] = {"training_rows": int(subject_counts.sum()), "participants": len(subject_counts),
                        "training_rows_from_two_largest_contributors": int(np.sort(subject_counts)[-2:].sum())}
    return counts


def evaluate_weighting(grid, lock, evidence, previous):
    matrix = grid.table[ROUTE]
    fits, replays, audits = {}, {}, {}
    for name in ("usual_class_balance", "participant_within_class"):
        fitted = fit_template(grid, matrix, lock, DEFAULT_C, participant_weighting=name == "participant_within_class")
        route = fitted["routes"][ROUTE]
        scores, cutoffs, decisions = fold_arrays(fitted, route)
        replay = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
        audit, _ = audit_route(grid, 1, scores, cutoffs, decisions, evidence, replay[0])
        fits[name], replays[name] = fitted, replay
        audits[name] = {
            "missed_reasons": audit["missed_reasons"],
            "misses_with_confirmed_state_during_af": audit["misses_with_confirmed_state_during_af"],
            "inner_calibration_status_counts": dict(Counter(
                fold["operating_point"]["status"] for fold in route["folds"]
            )),
        }
    baseline, candidate = (replays[name] for name in ("usual_class_balance", "participant_within_class"))
    if any(previous[key] != value for key, value in baseline[0].items()):
        raise ValueError("The previous development result must reproduce before comparing weighting")
    if not np.array_equal(fits["usual_class_balance"]["fold_assignment"],
                          fits["participant_within_class"]["fold_assignment"]):
        raise ValueError("The classifier comparison must preserve the same participant folds")
    return {
        "scope": "Development-only ablation following the independent error audit",
        "development_participants": len(np.unique(grid.subjects)),
        "outer_folds": fits["usual_class_balance"]["n_splits"],
        "logistic_C": DEFAULT_C, "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds,
        "target_episode_sensitivity": lock["target_sensitivity"], "sensitivity_tolerance": lock["sensitivity_tolerance"],
        "training_support": _training_support(grid),
        "baseline": baseline[0],
        "participant_weighting": _compare(baseline, candidate, lock),
        "event_audit_by_classifier": audits,
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
        fit, calibration, independent = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & independent or set(subjects) != (fit | calibration)
                or base[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use only the locked development cohort and original clock comparison")
        grid = load_grid(base, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        evidence = load_input_evidence(grid, args.stride_raw, args.stride_features)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        _write_json(args.output, evaluate_weighting(grid, lock, evidence, previous))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Template ECG weighting study failed: {error}\n")
    print("Compared participant weighting on the locked development folds")


if __name__ == "__main__":
    main()
