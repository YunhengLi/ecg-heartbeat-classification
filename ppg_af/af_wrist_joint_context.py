"""Test joint short and long wrist pulse-template interval inputs at fixed decision times."""

import argparse
import json
from pathlib import Path

import numpy as np

from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_context import short_template_features
from .af_wrist_template_study import DEFAULT_C, _calibration_status, _replay, fit_template, verify_fit_support


def evaluate(args):
    grid, quality, _, lock, _, _ = load_motion_development(args)
    previous = json.loads(args.context_results.read_bytes())
    prior_confirmation = json.loads(args.confirmation_results.read_bytes())
    short, status, _ = short_template_features(
        grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features,
        args.ptt_dir, lock[FEATURE_VERSION_KEY],
    )
    long = quality["baseline"]
    shared = _shared_features(grid.table)
    if not np.array_equal(np.isfinite(short).all(axis=1), shared):
        raise ValueError("Short and long inputs must retain the full locked scoring footprint")
    matrices = {"short_template": short, "joint_context": np.column_stack((long, short))}
    outcomes = {}
    for seed in SEEDS:
        fitted = fit_template(grid, long, lock, DEFAULT_C, seed=seed)
        baseline = _replay(grid, fitted)
        verified = prior_confirmation["comparisons"][str(seed)]["plain_template"]["original"]
        if baseline[0] != verified or (seed == SEEDS[0] and baseline[0] != previous["original_30s_full_support"]):
            raise ValueError("The full 30-second baseline must reproduce before fusion")
        routes = {"long_template": {"summary": baseline[0], "calibration_status": _calibration_status(fitted)}}
        for name, features in matrices.items():
            candidate = fit_template(grid, features, lock, DEFAULT_C, seed=seed)
            verify_fit_support(fitted, candidate)
            replay = _replay(grid, candidate)
            if seed == SEEDS[0] and name == "short_template" and replay[0] != previous["pulse_template_10s_common"]["summary"]:
                raise ValueError("The published 10-second baseline must reproduce before fusion")
            routes[name] = {**_compare_with_pre_onset(baseline, replay, lock), "calibration_status": _calibration_status(candidate)}
            print(f"{seed} {name}: {replay[0]['episodes_detected']}/{replay[0]['complete_episodes']}, "
                  f"{replay[0]['wrist_false_alerts']} false starts", flush=True)
        outcomes[str(seed)] = routes
    return {"scope": "Joint 10-second and 30-second plain wrist pulse templates on matched decision times",
            "outer_folds": 5, "inner_folds": 3, "short_support": status, "comparisons": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "ptt-dir", "context-results", "confirmation-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
