"""A generic ECG beat template timed using only PPG pulses at inference."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .features import _bandpass, _plateaus, _prepare, ecg_peaks
from .waveform import normalize_ecg


def _beat_limits(fs: float) -> tuple[int, int]:
    if not np.isfinite(fs) or fs < 25:
        raise ValueError("Waveform sampling rate must be finite and at least 25 Hz")
    return round(0.24 * fs), round(0.40 * fs)


def _pulse_peaks(ppg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    prepared = _prepare(ppg, fs, minimum_fs=25)
    supported = np.zeros(len(ppg), dtype=bool)
    if prepared is None:
        return np.empty(0, dtype=int), supported

    values, present = prepared
    usable = present & ~_plateaus(values, present, fs)
    boundaries = np.flatnonzero(np.diff(usable, prepend=False, append=False))
    guard = round(0.24 * fs)
    peaks = []
    for start, stop in zip(boundaries[::2], boundaries[1::2], strict=True):
        if stop - start < 2 * fs:
            continue
        pulse = _bandpass(values[start:stop], fs, 0.4, 8.0)
        lower, upper = np.percentile(pulse, (5, 95))
        candidates, _ = find_peaks(
            pulse,
            distance=max(1, round(0.25 * fs)),
            prominence=max(np.finfo(float).eps, 0.18 * (upper - lower)),
            width=max(1, round(0.05 * fs)),
        )
        supported[start + guard:stop - guard] = True
        peaks.extend(start + candidates)

    pulse_peaks = np.asarray(peaks, dtype=int)
    return pulse_peaks[supported[pulse_peaks]], supported


@dataclass(frozen=True)
class TemplateStats:
    fs: float
    delay_histogram: np.ndarray
    representative_beat: np.ndarray | None


def template_stats(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], fs: float = 125.0
) -> TemplateStats:
    """Collect pulse-delay evidence and a signed median ECG beat for one subject."""
    before, after = _beat_limits(fs)
    offsets = np.arange(-before, after + 1)
    maximum_delay = round(0.50 * fs)
    histogram = np.zeros(maximum_delay + 1)
    traces = []
    for ppg, ecg in pairs:
        target = normalize_ecg(ecg)
        if len(ppg) != len(target):
            raise ValueError("PPG and ECG must have aligned lengths")
        pulses, _ = _pulse_peaks(ppg, fs)
        references = ecg_peaks(ecg, fs)
        delays = pulses[None, :] - references[:, None]
        candidates = delays[(delays >= 0) & (delays <= maximum_delay)]
        histogram += np.bincount(candidates, minlength=len(histogram))
        complete = references[(references >= before) & (references + after < len(target))]
        beat_windows = target[complete[:, None] + offsets]
        traces.append(beat_windows[np.isfinite(beat_windows).all(axis=1)])

    beats = np.concatenate(traces) if traces else np.empty((0, len(offsets)))
    representative = np.median(beats, axis=0) if len(beats) else None
    return TemplateStats(fs, histogram, representative)


@dataclass(frozen=True)
class TemplateModel:
    fs: float
    delay_samples: int
    beat_template: np.ndarray

    def predict(self, ppg: np.ndarray) -> np.ndarray:
        """Place learned beats on the original sample grid using a fixed delay."""
        before, after = _beat_limits(self.fs)
        pulses, supported = _pulse_peaks(ppg, self.fs)
        length = len(ppg)
        prediction = np.full(length, np.nan)
        if not len(pulses):
            return prediction

        indices = np.arange(length)
        left_context = indices + self.delay_samples - after
        right_context = indices + self.delay_samples + before
        inside = (left_context >= 0) & (right_context < length)
        gaps = np.r_[0, np.cumsum(~supported)]
        eligible = indices[inside & (
            gaps[np.clip(right_context + 1, 0, length)]
            == gaps[np.clip(left_context, 0, length)]
        )]
        prediction[eligible] = 0.0

        centers = pulses - self.delay_samples
        midpoints = (centers[:-1] + centers[1:] + 1) // 2
        left_limits = np.maximum(0, np.r_[0, midpoints])
        right_limits = np.minimum(length, np.r_[midpoints, length])
        for center, left_limit, right_limit in zip(
            centers, left_limits, right_limits, strict=True
        ):
            start = max(left_limit, center - before)
            stop = min(right_limit, center + after + 1)
            if stop > start:
                offsets = slice(start - center + before, stop - center + before)
                valid = np.isfinite(prediction[start:stop])
                prediction[start:stop] = np.where(valid, self.beat_template[offsets], np.nan)
        return prediction


def fit_template_from_stats(stats: Iterable[TemplateStats]) -> TemplateModel:
    """Learn a global pulse delay and generic signed beat from training subjects."""
    subjects = list(stats)
    if not subjects:
        raise ValueError("At least one training subject is required")
    fs = subjects[0].fs
    before, after = _beat_limits(fs)
    if any(s.fs != fs for s in subjects):
        raise ValueError("Training subjects must share a sampling rate")

    histograms = [
        s.delay_histogram / s.delay_histogram.sum()
        for s in subjects if s.delay_histogram.any()
    ]
    beats = [s.representative_beat for s in subjects if s.representative_beat is not None]
    if not histograms or not beats:
        raise ValueError("No paired training pulses or complete ECG beats")
    scores = gaussian_filter1d(np.mean(histograms, axis=0), sigma=0.04 * fs, mode="constant")
    delay = int(np.argmax(scores))
    template = np.median(beats, axis=0)

    # Taper the generic waveform to zero outside the learned beat extent.
    taper = min(round(0.06 * fs), before, after)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, taper, endpoint=False)) ** 2
    template[:taper] *= ramp
    template[-taper:] *= ramp[::-1]
    return TemplateModel(fs, delay, template)
