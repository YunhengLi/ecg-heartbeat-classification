"""Window features for offline, ECG-guided PPG rhythm experiments."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


ECG_FEATURE_NAMES = (
    "detected_beats_per_second",
    "median_interval_s",
    "interval_iqr_over_median",
    "interval_cv",
    "successive_interval_rmssd_over_median",
    "successive_interval_pnn50",
    "valid_interval_fraction",
)

PPG_FEATURE_NAMES = ECG_FEATURE_NAMES + (
    "dominant_pulse_frequency_hz",
    "pulse_band_entropy",
)


def _prepare(signal: np.ndarray, fs: float, minimum_fs: float):
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional signal window")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if fs < minimum_fs or len(values) < 2 * fs:
        return None

    present = np.isfinite(values)
    observed = values[present]
    if len(observed) < 2 or np.all(observed == observed[0]):
        return None

    # Keep feature extraction invariant to the units and DC offset of the source.
    observed = observed / np.max(np.abs(observed))
    observed -= np.median(observed)
    if not np.all(present):
        observed = np.interp(np.arange(len(values)), np.flatnonzero(present), observed)
    return observed, present


def _bandpass(values, fs, low, high):
    filter_sections = butter(
        3, (low, min(high, 0.45 * fs)), fs=fs, btype="bandpass", output="sos"
    )
    padlen = min(len(values) - 1, 3 * (2 * len(filter_sections) + 1))
    return sosfiltfilt(filter_sections, values, padlen=padlen)


def _interval_features(peaks, fs, present, *, verified_peaks=None):
    features = np.zeros(len(ECG_FEATURE_NAMES))
    observed_peaks = present[peaks]
    if verified_peaks is not None:
        if verified_peaks.shape != peaks.shape or verified_peaks.dtype != bool:
            raise ValueError("Verified peak flags must match detected peaks")
        verified_peaks = verified_peaks[observed_peaks]
    peaks = peaks[observed_peaks]
    features[0] = len(peaks) * fs / len(present)
    intervals = np.diff(peaks) / fs
    if not len(intervals):
        return features

    # Exclude intervals spanning substantial signal gaps, including before
    # computing successive differences so gaps cannot create fictitious pairs.
    missing = np.r_[0, np.cumsum(~present)]
    samples_between = np.diff(peaks)
    valid = np.diff(missing[peaks]) <= 0.1 * samples_between
    if verified_peaks is not None:
        valid &= verified_peaks[:-1] & verified_peaks[1:]
    features[-1] = np.mean(valid)
    usable = intervals[valid]
    if not len(usable):
        return features

    lower, median, upper = np.percentile(usable, (25, 50, 75))
    features[1:4] = median, (upper - lower) / median, np.std(usable) / np.mean(usable)
    differences = np.diff(intervals)[valid[:-1] & valid[1:]]
    if len(differences):
        features[4] = np.sqrt(np.mean(np.square(differences))) / median
        features[5] = np.mean(np.abs(differences) > 0.05)
    return features


def _plateaus(values, present, fs):
    observed_pairs = present[:-1] & present[1:]
    flat = observed_pairs & (values[:-1] == values[1:])
    boundaries = np.flatnonzero(np.diff(flat, prepend=False, append=False))
    starts, ends = boundaries[::2], boundaries[1::2]
    sustained = ends - starts >= 0.5 * fs
    changes = np.zeros(len(values) + 1, dtype=int)
    changes[starts[sustained]] += 1
    changes[ends[sustained] + 1] -= 1
    return np.cumsum(changes[:-1]) > 0


def ppg_features(signal: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Extract pulse timing and spectral features with a heuristic quality score.

    The score multiplies observed sample coverage, the fraction free of sustained
    plateaus and the fraction of energy in the 0.25–8 Hz band. It does not use
    pulse interval regularity and cannot distinguish in-band motion from pulses.
    """
    prepared = _prepare(signal, fs, minimum_fs=4.0)
    if prepared is None:
        return np.zeros(len(PPG_FEATURE_NAMES)), 0.0

    values, present = prepared
    pulse = _bandpass(values, fs, 0.4, 8.0)
    lower, upper = np.percentile(pulse, (5, 95))
    peaks, _ = find_peaks(
        pulse, distance=max(1, round(0.25 * fs)),
        prominence=max(np.finfo(float).eps, 0.18 * (upper - lower)),
        width=max(1, round(0.05 * fs)),
    )

    frequencies, power = welch(values, fs=fs, nperseg=min(len(values), round(8 * fs)))
    in_band = (frequencies >= 0.5) & (frequencies <= 8.0)
    band_power = power[in_band]
    total = power.sum()
    band_total = band_power.sum()
    peak_frequency = entropy = 0.0
    if band_total > 0:
        peak_frequency = frequencies[in_band][np.argmax(band_power)]
        weights = band_power[band_power > 0] / band_total
        if len(band_power) > 1:
            entropy = -np.sum(weights * np.log(weights)) / np.log(len(band_power))

    pulse_and_nearby = (frequencies >= 0.25) & (frequencies <= 8.0)
    spectral_fraction = power[pulse_and_nearby].sum() / total if total > 0 else 0.0
    usable = present & ~_plateaus(values, present, fs)
    quality = np.mean(usable) * spectral_fraction
    features = np.r_[_interval_features(peaks, fs, usable), peak_frequency, entropy]
    return features, float(np.clip(quality, 0.0, 1.0))


def _ecg_peaks(values, fs):
    qrs = _bandpass(values, fs, 5.0, 20.0)
    energy = uniform_filter1d(np.square(np.gradient(qrs)), size=max(1, round(0.10 * fs)))
    threshold = max(4 * np.median(energy), 0.12 * np.percentile(energy, 98), np.finfo(float).eps)
    candidates, _ = find_peaks(
        energy, distance=max(1, round(0.25 * fs)), height=threshold, prominence=threshold / 2,
    )

    search_radius = max(1, round(0.07 * fs))
    peaks = []
    for candidate in candidates:
        start = max(0, candidate - search_radius)
        end = min(len(qrs), candidate + search_radius + 1)
        peaks.append(start + np.argmax(np.abs(qrs[start:end])))
    return np.asarray(peaks, dtype=int)


def ecg_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Locate ECG peaks using the offline detector shared by the features."""
    prepared = _prepare(signal, fs, minimum_fs=15.0)
    if prepared is None:
        return np.empty(0, dtype=int)
    values, present = prepared
    peaks = _ecg_peaks(values, fs)
    return peaks[present[peaks]]


def ecg_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract ECG interval features using a simplified offline QRS detector."""
    prepared = _prepare(signal, fs, minimum_fs=15.0)
    if prepared is None:
        return np.zeros(len(ECG_FEATURE_NAMES))
    values, present = prepared
    return _interval_features(_ecg_peaks(values, fs), fs, present)
