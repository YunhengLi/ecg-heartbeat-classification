"""Score waveform and beat timing without aligning predictions to test ECG."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .features import ecg_peaks


@dataclass(frozen=True)
class BeatScore:
    reference_count: int
    predicted_count: int
    errors_ms: np.ndarray
    interval_errors_ms: np.ndarray


def match_beats(reference, predicted, fs, tolerance_ms=100.0):
    """Match sorted events in order, with at most one prediction per reference."""
    first, second = 0, 0
    times, intervals = [], []
    previous = (-2, -2)
    tolerance = tolerance_ms * fs / 1000
    while first < len(reference) and second < len(predicted):
        error = predicted[second] - reference[first]
        if error < -tolerance:
            second += 1
        elif error > tolerance:
            first += 1
        else:
            times.append(abs(error) * 1000 / fs)
            if previous == (first - 1, second - 1):
                expected = reference[first] - reference[first - 1]
                observed = predicted[second] - predicted[second - 1]
                intervals.append(abs(observed - expected) * 1000 / fs)
            previous = first, second
            first += 1
            second += 1
    return BeatScore(len(reference), len(predicted), np.asarray(times), np.asarray(intervals))


def _peak_score(reference, predicted, eligible, fs):
    """Avoid matching events across separate supported stretches of a window."""
    edges = np.flatnonzero(np.diff(eligible, prepend=False, append=False))
    scores = []
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        observed = reference[(reference >= start) & (reference < stop)]
        estimated = predicted[(predicted >= start) & (predicted < stop)]
        scores.append(match_beats(observed, estimated, fs))
    return BeatScore(
        sum(s.reference_count for s in scores),
        sum(s.predicted_count for s in scores),
        _gather(scores, "errors_ms"),
        _gather(scores, "interval_errors_ms"),
    )


def _gather(scores, field):
    return np.concatenate([getattr(s, field) for s in scores]) if scores else np.empty(0)


def _median(values):
    valid = [value for value in values if value is not None]
    return float(np.median(valid)) if len(valid) else None


def summarize_beats(scores):
    matches = _gather(scores, "errors_ms")
    intervals = _gather(scores, "interval_errors_ms")
    n_ref = sum(s.reference_count for s in scores)
    n_pred = sum(s.predicted_count for s in scores)
    return {
        "reference_peaks": n_ref,
        "predicted_peaks": n_pred,
        "matched_peaks": len(matches),
        "precision": float(len(matches) / n_pred) if n_pred else None,
        "recall": float(len(matches) / n_ref) if n_ref else None,
        "f1": float(2 * len(matches) / (n_ref + n_pred)) if n_ref + n_pred else None,
        "median_abs_timing_error_ms": _median(matches),
        "matched_consecutive_intervals": len(intervals),
        "median_abs_rr_error_ms": _median(intervals),
    }


def score_group(target, annotations, ppg, waveforms, auxiliary_logits, fs, trim_seconds=1.0):
    """Use the same support for all methods and track support before intersecting."""
    edge = round(trim_seconds * fs)
    measured = np.isfinite(target) & np.isfinite(ppg)
    measured[:, :edge] = False
    measured[:, -edge:] = False
    eligible = measured.copy()
    for estimated in waveforms.values():
        eligible &= np.isfinite(estimated)

    reference_peaks = [np.flatnonzero(window > 0.5) for window in annotations]
    references = list(zip(reference_peaks, eligible, strict=True))

    def score_peaks(predicted_peaks):
        return summarize_beats([
            _peak_score(reference, predicted, valid, fs)
            for (reference, valid), predicted in zip(references, predicted_peaks, strict=True)
        ])

    def fraction(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    models = {}
    for name, estimated in waveforms.items():
        mae, correlation = [], []
        for reference, prediction, mask in zip(target, estimated, eligible, strict=True):
            if not np.any(mask):
                continue
            actual, forecast = reference[mask], prediction[mask]
            mae.append(float(np.mean(np.abs(actual - forecast))))
            if len(actual) > 1 and np.std(actual) > 0 and np.std(forecast) > 0:
                correlation.append(float(np.corrcoef(actual, forecast)[0, 1]))
        models[name] = {
            "relative_mae": float(np.mean(mae)) if mae else None,
            "median_pearson_r": _median(correlation),
            "correlation_windows": len(correlation),
            "own_support_fraction": fraction((measured & np.isfinite(estimated)).sum(), measured.sum()),
            "waveform_beats": score_peaks([ecg_peaks(wave, fs) for wave in estimated]),
        }

    def auxiliary_peaks(logits):
        return [find_peaks(np.where(np.isfinite(row), row, -np.inf), height=0,
                           distance=round(0.25 * fs))[0] for row in logits]

    zero_mae = [np.mean(np.abs(row[mask])) for row, mask in zip(target, eligible, strict=True) if mask.any()]
    return {
        "windows": len(target),
        "common_support_fraction": fraction(eligible.sum(), len(target) * (target.shape[1] - 2 * edge)),
        "zero_waveform_relative_mae": float(np.mean(zero_mae)) if zero_mae else None,
        "models": models,
        "measured_ecg_detector": score_peaks([ecg_peaks(wave, fs) for wave in target]),
        "auxiliary_peak_heads": {name: score_peaks(auxiliary_peaks(logits))
                                 for name, logits in auxiliary_logits.items()},
    }
