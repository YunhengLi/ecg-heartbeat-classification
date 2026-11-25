"""Compare 10-second and 30-second measured ECG windows on one decision clock."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .af_wrist_clock_study import load_grid, predecessor_decisions, scoring_runs
from .af_wrist_cv import _nested_folds, _subject_folds
from .af_wrist_failure_events import audit_events
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_measured_ecg import _DOI, measured_features
from .af_wrist_sequence_study import _comparison
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS


SHORT_SECONDS = 10
SEED = 2026
FOLDS = 5
INNER_FOLDS = 3


def _verify_original_groups(grid, original, common):
    stable = grid.table["stable_label"]
    original_fit_scope = grid.table["run_kind"] != "enriched_episode"
    labels = [np.where(available & original_fit_scope, stable, -1) for available in (original, common)]
    outer = [_subject_folds(rows, grid.subjects, FOLDS, SEED) for rows in labels]
    if not np.array_equal(*outer):
        raise ValueError("Shared ECG support changed the original outer participant groups")
    for fold in range(FOLDS):
        train = np.flatnonzero(outer[0] != fold)
        left, right = (_nested_folds(train, rows, grid.subjects, {ROUTE: rows >= 0}, INNER_FOLDS, SEED + fold + 1)
                       for rows in labels)
        if not np.array_equal(left[0], right[0]) or left[1:] != right[1:]:
            raise ValueError("Shared ECG support changed the original inner participant groups")


def _diagnostics(grid, fitted, replay, seconds, scopes):
    scores, cutoffs, decisions = fold_arrays(fitted, fitted["routes"][ROUTE])
    available = np.isfinite(scores) & np.isfinite(cutoffs)
    stable = scopes["stable_fitting"]
    stable = stable[np.isfinite(scores[stable])]
    labels = grid.table["stable_label"][stable]
    predecessor = predecessor_decisions(grid, decisions)
    _, runs = scoring_runs(grid, np.where(available, decisions, np.nan), set(grid.subjects))
    audit = audit_events((run.decisions(0.5) for run in runs), replay[1], confirmation_windows=1,
                         prior_decisions=predecessor, feature_window_s=seconds)
    complete = audit["complete_episodes"]
    if (len(complete), sum(case["detected"] for case in complete)) != (replay[0]["complete_episodes"],
                                                                     replay[0]["episodes_detected"]):
        raise ValueError("Episode diagnostics must reconcile with the original scorer")
    return {
        "pooled_held_out_stable_window_ROC_AUC": float(roc_auc_score(labels, scores[stable])),
        "stable_AF_windows": int(np.count_nonzero(labels == 1)),
        "stable_non_AF_windows": int(np.count_nonzero(labels == 0)),
        "inner_calibration_status_counts": _calibration_status(fitted),
        "positive_evaluation_decisions": int(np.count_nonzero(decisions[scopes["evaluation"]] == 1)),
        "missed_episode_mechanisms": dict(sorted(Counter(case["primary"] for case in complete
                                                         if not case["detected"]).items())),
        "already_active_misses_with_positive_in_episode": sum(
            case["primary"] == "already_active_at_onset" and case["positive_count"] > 0 for case in complete
        ),
    }


def evaluate_context(grid, lock, inputs, previous):
    shared = _shared_features(grid.table)
    matrices = {seconds: matrix for seconds, (matrix, _) in inputs.items()}
    support = {seconds: shared & np.isfinite(matrix).all(axis=1) for seconds, matrix in matrices.items()}
    original = support[WINDOW_SECONDS]
    common = original & support[SHORT_SECONDS]
    _verify_original_groups(grid, original, common)
    gate = np.where(common[:, None], grid.table[ROUTE], np.nan)
    comparison_grid = replace(grid, table={**grid.table, ROUTE: gate})
    stable = np.flatnonzero((grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode"))
    scopes = {"evaluation": np.concatenate([block.rows for block in grid.timelines]),
              "preceding_history": np.concatenate(list(grid.history_rows.values())), "stable_fitting": stable}
    models = {seconds: fit_template(comparison_grid, matrix, lock, DEFAULT_C, seed=SEED)
              for seconds, matrix in matrices.items()}
    verify_fit_support(models[WINDOW_SECONDS], models[SHORT_SECONDS])
    replays = {seconds: _replay(comparison_grid, fit) for seconds, fit in models.items()}
    baseline, candidate = replays[WINDOW_SECONDS], replays[SHORT_SECONDS]
    footprint_unchanged = np.array_equal(common, original)
    if footprint_unchanged and baseline[0] != previous["measured_ecg_on_common_footprint"]["summary"]:
        raise ValueError("Identical support must reproduce the original 30-second measured ECG result")

    return {
        "scope": "Exploratory same-clock measured ECG feature-window comparison",
        "development_participants": len(np.unique(grid.subjects)), "seed": SEED, "outer_folds": FOLDS,
        "logistic_C": DEFAULT_C, "decision_stride_seconds": grid.seconds,
        "support_on_original_shared_PPG_rows": {str(seconds): status for seconds, (_, status) in inputs.items()},
        "common_footprint_unchanged_from_30s_reference": footprint_unchanged,
        "support_by_scope": {name: {"original_shared_PPG_rows": int(shared[rows].sum()),
                                    "ECG_30s": int(original[rows].sum()),
                                    "ECG_10s": int(support[SHORT_SECONDS][rows].sum()),
                                    "both": int(common[rows].sum())} for name, rows in scopes.items()},
        "ECG_30s": {"summary": baseline[0],
                    "diagnostics": _diagnostics(comparison_grid, models[WINDOW_SECONDS], baseline, WINDOW_SECONDS, scopes)},
        "ECG_10s": {**_comparison(baseline, candidate, lock),
                    "diagnostics": _diagnostics(comparison_grid, models[SHORT_SECONDS], candidate, SHORT_SECONDS, scopes)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("source-inventory", "source-dir", "cache-dir", "development-raw", "development-features", "stride-raw",
             "stride-features", "lock", "clock-results", "measured-control-results", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        inventory = json.loads(args.source_inventory.read_bytes())
        previous = json.loads(args.measured_control_results.read_bytes())
        table, participants, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[name]) for name in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(participants) != fit | calibration
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest() or inventory["doi"] != _DOI):
            raise ValueError("Use the pinned v6 inventory and original development cohort and clock")
        grid = load_grid(table, participants, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        source = {entry["key"]: entry for entry in inventory["files"]}
        inputs = {seconds: measured_features(grid, source, args.source_dir, args.cache_dir, args.development_raw,
                                             args.stride_raw, feature_window_seconds=seconds)
                  for seconds in (WINDOW_SECONDS, SHORT_SECONDS)}
        report = evaluate_context(grid, lock, inputs, previous)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Measured ECG context comparison failed: {error}\n")
    before, after = (report[f"ECG_{seconds}s"]["summary"] for seconds in (WINDOW_SECONDS, SHORT_SECONDS))
    print(f"Measured ECG 30s: {before['episodes_detected']}/{before['complete_episodes']} detections, "
          f"{before['wrist_false_alerts']} wrist false starts. 10s: {after['episodes_detected']}/"
          f"{after['complete_episodes']} detections, {after['wrist_false_alerts']} wrist false starts.")


if __name__ == "__main__":
    main()
