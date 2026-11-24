"""Test whether direct PPG spectral and interval features complement the template."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, scoring_policy
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_operating import select_scored_threshold
from .af_wrist_vote_study import _IMMEDIATE
from .features import ECG_FEATURE_NAMES, PPG_FEATURE_NAMES


SPECTRAL_ROUTE = "quality_template_plus_ppg_spectrum"


def freeze_spectral(args):
    grid, quality, _, lock, lock_digest, _ = load_motion_development(args)
    fit, calibration = (_subject_ids(lock[name]) for name in ("fit_subjects", "calibration_subjects"))
    labels = grid.table["stable_label"]
    shared = _shared_features(grid.table)
    training = np.isin(grid.subjects, sorted(fit)) & shared & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    reference = np.isin(grid.subjects, sorted(calibration)) & shared
    counts = {"AF": int(np.count_nonzero(training & (labels == 1))), "non_AF": int(np.count_nonzero(training & (labels == 0)))}
    if counts != {"AF": 21, "non_AF": 22}:
        raise ValueError("Train on exactly the original frozen set of 43 stable fitting windows")
    matrix = np.column_stack((quality["quality_interactions"], grid.table["direct_ppg"][:, len(ECG_FEATURE_NAMES):]))
    model = _fit_model(matrix[training], labels[training], regularization=1.0)
    scores = _predict(matrix, reference, model)
    score, pool = scoring_policy(grid, scores, calibration, _IMMEDIATE)
    point = select_scored_threshold(pool, score, target_sensitivity=lock["target_sensitivity"], max_thresholds=max(3, len(pool) + 1))
    if point["threshold"] is None:
        raise ValueError("The new spectral comparator requires a development-calibrated cutoff")
    return {"route": SPECTRAL_ROUTE, "source_protocol_sha256": lock_digest, FEATURE_VERSION_KEY: lock[FEATURE_VERSION_KEY],
            "fit_participants": len(fit), "calibration_participants": len(calibration), "fit_windows": counts,
            "frozen_model": {"model": model, "cutoff": point["threshold"]},
            "calibration": {"status": point["status"], **_summarize(point["validation"])}}


def evaluate(args):
    grid, quality, _, lock, _, original = load_motion_development(args)
    historical = json.loads(args.clock_results.read_bytes())["warm_10s"]["nested_development_cv"]["1"]
    direct = grid.table["direct_ppg"]
    template = quality["quality_interactions"]
    if direct.shape != (len(template), len(PPG_FEATURE_NAMES)):
        raise ValueError("Expected the original direct PPG intervals and two spectral features on the same clock")
    spectral = direct[:, len(ECG_FEATURE_NAMES):]
    comparisons = _compare(grid, {"quality_interactions": template, "original_direct_ppg": direct,
                                  SPECTRAL_ROUTE: np.column_stack((template, spectral)),
                                  "quality_template_plus_all_ppg": np.column_stack((template, direct))}, lock, original)
    reproduction = comparisons["2026"]["original_direct_ppg"]["summary"]
    if any(historical["direct_ppg"][key] != value for key, value in reproduction.items()):
        raise ValueError("The unchanged direct PPG model must reproduce its original held-out summary")
    return {"scope": "Controlled early fusion of direct PPG and template-derived AF features",
            "additional_spectral_features": PPG_FEATURE_NAMES[len(ECG_FEATURE_NAMES):],
            "additional_full_direct_features": PPG_FEATURE_NAMES, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--freeze-spectral", action="store_true", help="Write the private development-only model lock")
    args = parser.parse_args()
    _write_json(args.output, freeze_spectral(args) if args.freeze_spectral else evaluate(args))
    if args.freeze_spectral:
        print(f"Spectral model lock SHA-256 {sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
