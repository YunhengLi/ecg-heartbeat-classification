"""Freeze fixed-split template ECG classifiers before independent evaluation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, load_grid, scoring_policy
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import MONITOR_KIND, _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_template_study import CANDIDATE_C, DEFAULT_C, ROUTE, rank_calibration
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE


def freeze_template(grid, protocol, protocol_digest, *, feature_version):
    """Return a private model lock and an aggregate report from development only."""
    fit, calibration, roster = (_subject_ids(protocol[key])
                                for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & roster or set(grid.subjects) != (fit | calibration)
            or grid.seconds != STRIDE_SECONDS or feature_version != protocol[FEATURE_VERSION_KEY]):
        raise ValueError("Use the original frozen development split and 10-second feature version")
    labels = grid.table["stable_label"]
    shared = _shared_features(grid.table)
    fit_rows = (np.isin(grid.subjects, sorted(fit)) & shared & (labels >= 0)
                & (grid.table["run_kind"] != "enriched_episode"))
    if set(labels[fit_rows]) != {0, 1}:
        raise ValueError("The fitting participants must contain usable rows for both rhythms")
    calibration_rows = np.isin(grid.subjects, sorted(calibration)) & shared
    matrix = grid.table[ROUTE]
    candidates = {}
    for regularization in CANDIDATE_C:
        model = _fit_model(matrix[fit_rows], labels[fit_rows], regularization=regularization)
        scores = _predict(matrix, calibration_rows, model)
        evaluate, pool = scoring_policy(grid, scores, calibration, _IMMEDIATE)
        option = select_scored_threshold(pool, evaluate, target_sensitivity=protocol["target_sensitivity"],
                                         max_thresholds=max(3, len(pool) + 1))
        if option["threshold"] is None:
            raise ValueError("No supported calibration cutoff was available for this development split")
        candidates[regularization] = {"logistic_C": regularization, "model": model, "cutoff": option["threshold"],
                                      "calibration": option}

    selected = min(candidates, key=lambda value: rank_calibration(candidates[value]["calibration"], value))
    baseline, tuned = candidates[DEFAULT_C], candidates[selected]
    report = {
        "scope": "Fixed-split development freeze before independent evaluation",
        "fit_participants": len(fit), "calibration_participants": len(calibration),
        "planned_independent_participants": len(roster), "decision_stride_seconds": grid.seconds,
        "feature_window_seconds": WINDOW_SECONDS, "target_episode_sensitivity": protocol["target_sensitivity"],
        "sensitivity_tolerance": protocol["sensitivity_tolerance"],
        "fit_windows": {"AF": int(np.sum(fit_rows & (labels == 1))),
                        "non_AF": int(np.sum(fit_rows & (labels == 0)))},
        "baseline_C": DEFAULT_C, "selected_C": selected,
        "frozen_predictions_identical": baseline["model"] == tuned["model"] and baseline["cutoff"] == tuned["cutoff"],
        "calibration_candidates": [
            {"C": value, "status": candidate["calibration"]["status"],
             "threshold_candidates": candidate["calibration"]["threshold_candidates"],
             **_summarize(candidate["calibration"]["validation"])}
            for value, candidate in candidates.items()
        ],
    }
    lock = {
        "lock_version": 1, "source_protocol_sha256": protocol_digest,
        FEATURE_VERSION_KEY: feature_version, "route": ROUTE,
        "window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds,
        "monitor_kind": MONITOR_KIND, "alert_policy": "causal_immediate_with_predecessor_state",
        "target_sensitivity": protocol["target_sensitivity"], "sensitivity_tolerance": protocol["sensitivity_tolerance"],
        "fit_subjects": sorted(fit), "calibration_subjects": sorted(calibration), "test_roster": sorted(roster),
        "models": {"baseline": baseline, "tuned": tuned},
    }
    return lock, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-features", "stride-raw", "stride-features", "protocol-lock", "output-lock", "output-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if len({args.protocol_lock.resolve(), args.output_lock.resolve(), args.output_report.resolve()}) != 3:
        parser.error("Use distinct protocol, private model lock, and aggregate report paths")
    try:
        protocol_bytes = args.protocol_lock.read_bytes()
        protocol = json.loads(protocol_bytes)
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, roster = (_subject_ids(protocol[key])
                                    for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if fit & calibration or (fit | calibration) & roster or set(subjects) != (fit | calibration):
            raise ValueError("The feature directory must contain exactly the locked development participants")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        lock, report = freeze_template(grid, protocol, sha256(protocol_bytes).hexdigest(),
                                       feature_version=table[FEATURE_VERSION_KEY])
        _write_json(args.output_lock, lock)
        _write_json(args.output_report, report)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Template ECG freeze failed: {error}\n")
    print("Froze baseline and tuned template ECG models using development data")


if __name__ == "__main__":
    main()
