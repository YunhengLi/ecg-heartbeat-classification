"""Score released RDDM inference with the authors' prepared 4-second test arrays."""

from collections.abc import Callable
from pathlib import Path

import neurokit2 as nk
import numpy as np
from sklearn.preprocessing import minmax_scale


AUTHOR_DATASETS = ("WESAD", "CAPNO", "DALIA", "BIDMC", "MIMIC-AFib")
SAMPLE_RATE = 128
WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS


def _read_windows(path: Path) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except ValueError as exc:
        raise ValueError(f"Expected a numeric NumPy array in {path}") from exc

    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError(f"Expected a .npy array in {path}")
    if values.dtype.kind not in "iuf" or not values.size or values.size % WINDOW_SAMPLES:
        raise ValueError(f"Expected nonempty real-valued windows of {WINDOW_SAMPLES} samples in {path}")
    return values.reshape(-1, WINDOW_SAMPLES)


def _scale_windows(segments: np.ndarray) -> np.ndarray:
    segments = np.nan_to_num(segments.astype(np.float32, copy=True), copy=False)
    return minmax_scale(segments, feature_range=(-1, 1), axis=1)


def _frechet_distance(target: np.ndarray, generated: np.ndarray) -> float:
    """Gaussian waveform Fréchet distance used in the upstream evaluator.

    The nuclear norm of the cross-Gram matrix is the covariance square-root
    trace. Computing singular values keeps rank-deficient batches stable.
    """
    target_mean = target.mean(axis=0)
    generated_mean = generated.mean(axis=0)
    mean_difference = target_mean - generated_mean

    divisor = np.sqrt(len(target) - 1)
    target_centered = (target - target_mean) / divisor
    generated_centered = (generated - generated_mean) / divisor
    if len(target_centered) > target.shape[1]:
        target_centered = np.linalg.qr(target_centered, mode="r")
        generated_centered = np.linalg.qr(generated_centered, mode="r")

    covariance_trace = np.sum(target_centered**2) + np.sum(generated_centered**2)
    covariance_sqrt_trace = np.linalg.svd(
        target_centered @ generated_centered.T, compute_uv=False
    ).sum()
    distance = mean_difference @ mean_difference + covariance_trace - 2 * covariance_sqrt_trace
    return max(0.0, float(distance))


def evaluate_author_split(
    data_dir: Path,
    dataset: str,
    predict: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    seed: int = 31,
) -> dict:
    """Score PPG-only predictions against the authors' prepared test ECG.

    The caller supplies the already-split numeric arrays. The author repository
    applies window min-max scaling before its NeuroKit filters. This evaluator
    uses a local seeded shuffle, while the authors used a shuffled DataLoader.
    The Fréchet result averages non-singleton batches equally, so its value can
    change with the batch size and batch order. RMSE includes every window.
    """
    if dataset not in AUTHOR_DATASETS:
        raise ValueError(f"Unsupported author dataset: {dataset}. Choose from {', '.join(AUTHOR_DATASETS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    folder = Path(data_dir) / dataset
    ecg = _read_windows(folder / "ecg_test_4sec.npy")
    ppg = _read_windows(folder / "ppg_test_4sec.npy")
    if len(ecg) != len(ppg):
        raise ValueError(f"{dataset} must have the same number of ECG and PPG windows")

    order = np.random.default_rng(seed).permutation(len(ecg))
    squared_error = 0.0
    distances = []
    singleton_batches = 0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        inputs = np.asarray([
            nk.ppg_clean(window, sampling_rate=SAMPLE_RATE) for window in _scale_windows(ppg[indices])
        ], dtype=np.float32)
        targets = np.asarray([
            nk.ecg_clean(window, sampling_rate=SAMPLE_RATE, method="pantompkins1985")
            for window in _scale_windows(ecg[indices])
        ], dtype=np.float32)
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("Preprocessing produced nonfinite ECG or PPG samples")

        generated = np.asarray(predict(inputs), dtype=np.float64)
        if generated.shape != targets.shape or not np.isfinite(generated).all():
            raise ValueError(f"Predictor must return finite waveform values with shape {targets.shape}")

        target_values = targets.astype(np.float64)
        squared_error += float(np.square(generated - target_values).sum())
        if len(indices) > 1:
            distances.append(_frechet_distance(target_values, generated))
        else:
            singleton_batches += 1

    return {
        "scope": "Four-second waveform inference on user-supplied author-prepared test arrays",
        "dataset": dataset,
        "windows": len(ecg),
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "batch_size": batch_size,
        "shuffle_seed": seed,
        "shuffle": "numpy.default_rng permutation",
        "rmse": float(np.sqrt(squared_error / ecg.size)),
        "mean_batch_frechet_distance": float(np.mean(distances)) if distances else None,
        "fd_batches": len(distances),
        "fd_singleton_batches_excluded": singleton_batches,
    }
