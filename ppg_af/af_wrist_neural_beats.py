"""Compare AF features from a source-frozen PPG network and template ECG."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .af_wrist_beat_cnn import EDGE, SAMPLES, load_move, predict, reference_targets, standardize, train
from .af_wrist_features import PPG_RATE
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_quality_study import quality_interaction_features
from .features import ECG_FEATURE_NAMES, _interval_features


def peak_intervals(predictions):
    """Keep the same beat-count, RR variability, and sufficient-support features."""
    supported = np.r_[np.zeros(EDGE, dtype=bool), np.ones(SAMPLES - 2 * EDGE, dtype=bool), np.zeros(EDGE, dtype=bool)]
    features = np.stack([_interval_features(peaks, PPG_RATE, supported) for peaks in predictions])
    no_intervals = features[:, ECG_FEATURE_NAMES.index("valid_interval_fraction")] <= 0
    return features, no_intervals


def evaluate(args):
    grid, quality, paths, lock, _, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)
    source, _ = load_move(args.move_dir)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source_input = standardize(source.pulse, 8.0).astype(np.float32)[:, None]
    network = train(source_input, reference_targets(source.reference), source.participants, args.device)
    new_peaks = predict(network, standardize(cores[shared], 8.0).astype(np.float32)[:, None], args.device)
    converted, unsupported = peak_intervals(new_peaks)
    intervals = quality["baseline"].copy()
    selected = np.flatnonzero(shared)
    intervals[selected[~unsupported]] = converted[~unsupported]
    if not np.array_equal(np.isfinite(intervals).all(axis=1), np.isfinite(quality["baseline"]).all(axis=1)):
        raise ValueError("Source beat features must preserve original wrist input coverage")
    neural = quality_interaction_features(intervals, quality["quality"][:, -1])
    matrices = {"quality_interactions": quality["quality_interactions"], "neural_intervals": neural,
                "template_and_neural_intervals": np.column_stack((quality["quality_interactions"], intervals))}
    return {"scope": "Develop AF feature classifiers using a source-frozen neural beat estimator",
            "source_participants": int(len(np.unique(source.participants))), "source_windows": len(source.pulse),
            "wrist_ppg_rows": len(selected), "rows_with_neural_intervals": int((~unsupported).sum()),
            "fallback_template_rows": int(unsupported.sum()),
            "comparisons": _compare(grid, matrices, lock, original)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("move-dir", "development-raw", "development-features", "stride-raw", "stride-features", "lock",
                 "clock-results", "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
