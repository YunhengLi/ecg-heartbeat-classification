"""Train pulse-to-R timing on annotated data and shift frozen ECG estimates."""

from operator import index

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .features import ecg_peaks
from .template import _pulse_peaks


CHUNK_SECONDS = 4.0
MAX_SHIFT_SECONDS = 0.40
MATCH_SECONDS = 0.05
MAX_DELAY_SECONDS = 0.50


def pulse_delay_histogram(ppg, marks, fs):
    """Count candidate delays from each marked R peak to later PPG peaks."""
    ppg = np.asarray(ppg, dtype=float)
    marks = np.asarray(marks)
    if ppg.ndim != 2 or ppg.shape != marks.shape:
        raise ValueError("PPG and R labels must be aligned 2-D window arrays")
    _check_rate(fs)
    max_delay = round(MAX_DELAY_SECONDS * fs)
    histogram = np.zeros(max_delay + 1, dtype=np.int64)
    for pulse_window, reference_window in zip(ppg, marks, strict=True):
        pulse_peaks, _ = _pulse_peaks(pulse_window, fs)
        references = np.flatnonzero(reference_window > 0.5)
        delays = pulse_peaks[:, None] - references[None, :]
        candidates = delays[(delays >= 0) & (delays <= max_delay)]
        histogram += np.bincount(candidates, minlength=len(histogram))
    return histogram


def fit_pulse_delay(histograms, fs):
    """Fit a delay in samples, giving each training subject equal weight."""
    _check_rate(fs)
    expected_shape = (round(MAX_DELAY_SECONDS * fs) + 1,)
    normalized = []
    for histogram in histograms:
        histogram = np.asarray(histogram, dtype=float)
        if histogram.shape != expected_shape or not np.isfinite(histogram).all() or np.any(histogram < 0):
            raise ValueError("Invalid pulse-delay histogram")
        total = histogram.sum()
        if total:
            normalized.append(histogram / total)
    if not normalized:
        raise ValueError("No paired training pulses and annotated R peaks")
    votes = gaussian_filter1d(np.mean(normalized, axis=0), sigma=0.04 * fs, mode="constant")
    return int(np.argmax(votes))


def timing_scoring_mask(n_samples, fs, *, chunk_seconds=CHUNK_SECONDS, max_shift_seconds=MAX_SHIFT_SECONDS):
    """Reserve the same border for all methods, regardless of the chosen shifts."""
    n_samples = index(n_samples)
    if n_samples < 0:
        raise ValueError("Sample count must be nonnegative")
    chunk_size, limit = _chunk_parameters(fs, chunk_seconds, max_shift_seconds)
    mask = np.zeros(n_samples, dtype=bool)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        if stop - start > 2 * limit:
            mask[start + limit:stop - limit] = True
    return mask


def correct_timing(
    ppg, generated, delay_samples, fs, *, chunk_seconds=CHUNK_SECONDS,
    max_shift_seconds=MAX_SHIFT_SECONDS, match_seconds=MATCH_SECONDS,
):
    """Translate each chunk using PPG-derived R anchors and generated ECG only.

    A positive shift delays all waveform samples in the chunk. Vacated samples
    are NaN. A chunk stays in place when fewer than two beats can be matched.
    """
    ppg = np.asarray(ppg, dtype=float)
    generated = np.asarray(generated, dtype=float)
    delay_samples = index(delay_samples)
    if ppg.ndim != 1 or generated.shape != ppg.shape:
        raise ValueError("PPG and generated ECG must be aligned 1-D arrays")
    chunk_size, limit = _chunk_parameters(fs, chunk_seconds, max_shift_seconds)
    if not 0 <= delay_samples <= round(MAX_DELAY_SECONDS * fs):
        raise ValueError("Pulse delay is outside the fitted range")
    if not np.isfinite(match_seconds) or match_seconds <= 0:
        raise ValueError("Match tolerance must be positive and finite")
    tolerance = match_seconds * fs

    pulses, _ = _pulse_peaks(ppg, fs)
    anchors = pulses - delay_samples
    estimates = ecg_peaks(generated, fs)
    result = generated.copy()
    shifts = []
    for start in range(0, len(generated), chunk_size):
        stop = min(start + chunk_size, len(generated))
        left, right = start + limit, stop - limit
        reference = anchors[(anchors >= left) & (anchors < right)]
        peaks = estimates[(estimates >= start) & (estimates < stop)]
        shift = _choose_shift(reference, peaks, left, right, limit, tolerance)
        shifts.append(shift)
        if shift > 0:
            result[start:start + shift] = np.nan
            result[start + shift:stop] = generated[start:stop - shift]
        elif shift < 0:
            result[stop + shift:stop] = np.nan
            result[start:stop + shift] = generated[start - shift:stop]
    return result, np.asarray(shifts, dtype=int)


def _check_rate(fs):
    if not np.isfinite(fs) or fs < 25:
        raise ValueError("Waveform sampling rate must be finite and at least 25 Hz")


def _chunk_parameters(fs, chunk_seconds, max_shift_seconds):
    _check_rate(fs)
    if not np.isfinite(chunk_seconds) or not np.isfinite(max_shift_seconds) or max_shift_seconds < 0:
        raise ValueError("Chunk length and maximum shift must be finite, with a nonnegative shift")
    chunk_size = round(chunk_seconds * fs)
    limit = round(max_shift_seconds * fs)
    if chunk_size <= 2 * limit:
        raise ValueError("A chunk must be longer than twice the maximum shift")
    return chunk_size, limit


def _choose_shift(anchors, peaks, left, right, limit, tolerance):
    if len(anchors) < 2 or len(peaks) < 2:
        return 0

    best_shift = 0
    best_score = (1, 0.0, 0)
    for shift in range(-limit, limit + 1):
        moved = peaks + shift
        supported = moved[(moved >= left) & (moved < right)]
        matches, total_error = _match_cost(anchors, supported, tolerance)
        if matches >= 2:
            score = (matches, -(total_error / matches + 0.1 * abs(shift)), -abs(shift))
            if score > best_score:
                best_shift = shift
                best_score = score
    return best_shift


def _match_cost(anchors, peaks, tolerance):
    i, j = 0, 0
    matches, total_error = 0, 0
    while i < len(anchors) and j < len(peaks):
        error = peaks[j] - anchors[i]
        if error < -tolerance:
            j += 1
        elif error > tolerance:
            i += 1
        else:
            matches += 1
            total_error += abs(error)
            i += 1
            j += 1
    return matches, total_error
