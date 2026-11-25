"""A delayed-input ridge baseline for offline PPG-to-ECG experiments."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _relative(signal: np.ndarray, percentile: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional signal window")
    present = np.isfinite(values)
    relative = np.full(values.shape, np.nan)
    if not present.any():
        return relative

    observed = values[present]
    magnitude = np.max(np.abs(observed))
    observed = observed / magnitude if magnitude > 0 else observed
    centered = observed - np.median(observed)
    scale = np.percentile(np.abs(centered), percentile)
    if scale < np.finfo(float).eps:
        scale = max(np.max(np.abs(centered)), np.finfo(float).eps)
    relative[present] = centered / scale
    return relative


def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    """Center ECG and divide by its 98th percentile absolute deviation.

    Relative amplitude has no physical units. ECG polarity is retained.
    """
    return _relative(ecg, percentile=98)


def _lags(fs: float) -> np.ndarray:
    if not np.isfinite(fs) or fs < 25:
        raise ValueError("Waveform sampling rate must be finite and at least 25 Hz")
    step = max(1, round(0.032 * fs))
    return np.arange(-round(0.32 * fs), round(0.48 * fs) + 1, step)


def _design(ppg: np.ndarray, lags: np.ndarray, fs: float):
    pulse = _relative(ppg, percentile=95)
    present = np.isfinite(pulse)
    radius = max(1, round(0.036 * fs))
    before = -lags[0] + radius + 1
    after = lags[-1] + radius + 1
    centers = np.arange(before, len(pulse) - after)
    missing = np.r_[0, np.cumsum(~present)]
    centers = centers[missing[centers + after + 1] == missing[centers - before]]
    if not len(centers) or not np.any(pulse[present]):
        return np.empty(0, dtype=int), np.empty((0, 3 * len(lags)))

    pulse = np.clip(np.nan_to_num(pulse, nan=0.0), -8.0, 8.0)
    smooth = gaussian_filter1d(pulse, sigma=0.012 * fs, radius=radius)
    slope = np.gradient(smooth) * fs
    taps = centers[:, None] + lags
    raw, signed_slope = smooth[taps], slope[taps]
    return centers, np.concatenate((raw, signed_slope, np.abs(signed_slope)), axis=1)


@dataclass(frozen=True)
class WaveformStats:
    fs: float
    count: int
    feature_sum: np.ndarray
    target_sum: float
    feature_product: np.ndarray
    feature_target: np.ndarray


def waveform_stats(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], fs: float = 125.0
) -> WaveformStats:
    """Collect ridge moments one paired window at a time."""
    lags = _lags(fs)
    width = 3 * len(lags)
    count = 0
    feature_sum = np.zeros(width)
    target_sum = 0.0
    feature_product = np.zeros((width, width))
    feature_target = np.zeros(width)

    for ppg, ecg in pairs:
        target = normalize_ecg(ecg)
        if len(ppg) != len(target):
            raise ValueError("PPG and ECG must have aligned lengths")
        centers, design = _design(ppg, lags, fs)
        labels = target[centers]
        valid = np.isfinite(labels)
        design, labels = design[valid], labels[valid]
        count += len(labels)
        feature_sum += design.sum(axis=0)
        target_sum += labels.sum()
        feature_product += design.T @ design
        feature_target += design.T @ labels

    return WaveformStats(fs, count, feature_sum, target_sum, feature_product, feature_target)


@dataclass(frozen=True)
class WaveformModel:
    fs: float
    coefficients: np.ndarray
    intercept: float

    def predict(self, ppg: np.ndarray) -> np.ndarray:
        """Estimate relative ECG, leaving invalid context and edges as NaN."""
        centers, design = _design(ppg, _lags(self.fs), self.fs)
        prediction = np.full(len(ppg), np.nan)
        prediction[centers] = design @ self.coefficients + self.intercept
        return prediction


def fit_waveform_from_stats(
    stats: Iterable[WaveformStats], ridge: float = 0.01
) -> WaveformModel:
    """Fit with sufficient statistics collected only from training subjects."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Ridge penalty must be finite and positive")
    training = list(stats)
    if not training or sum(s.count for s in training) == 0:
        raise ValueError("No matched ECG samples with complete PPG context")
    fs = training[0].fs
    if any(s.fs != fs for s in training):
        raise ValueError("Training subjects must share a sampling rate")

    # Normalize each subject's moments before pooling so long recordings do
    # not outweigh short recordings. One aggregate statistics object is also valid.
    training = [s for s in training if s.count]
    center = np.mean([s.feature_sum / s.count for s in training], axis=0)
    target_center = np.mean([s.target_sum / s.count for s in training])
    covariance = np.mean([s.feature_product / s.count for s in training], axis=0)
    covariance -= np.outer(center, center)
    cross = np.mean([s.feature_target / s.count for s in training], axis=0)
    cross -= center * target_center
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = covariance / np.outer(scale, scale)
    standardized.flat[::len(scale) + 1] += ridge
    coefficients = np.linalg.solve(standardized, cross / scale) / scale
    intercept = target_center - center @ coefficients
    return WaveformModel(fs, coefficients, float(intercept))


def fit_waveform(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], fs: float = 125.0, ridge: float = 0.01
) -> WaveformModel:
    """Fit a single model from aligned training windows."""
    return fit_waveform_from_stats((waveform_stats(pairs, fs),), ridge)
