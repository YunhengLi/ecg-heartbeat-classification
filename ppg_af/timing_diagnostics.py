"""Describe generated beat timing on the measured ECG time grid."""

import numpy as np

from .features import ecg_peaks


TOLERANCES_MS = (100, 200)


def _signed_match_errors_ms(reference, predicted, fs, tolerance_ms):
    """Apply the benchmark's ordered one-to-one matching and keep the sign."""
    tolerance = tolerance_ms * fs / 1000
    ref_index = pred_index = 0
    errors = []
    while ref_index < len(reference) and pred_index < len(predicted):
        offset = predicted[pred_index] - reference[ref_index]
        if offset < -tolerance:
            pred_index += 1
        elif offset > tolerance:
            ref_index += 1
        else:
            errors.append(offset * 1000 / fs)
            ref_index += 1
            pred_index += 1
    return errors


def _summarize_matches(errors, reference_count, prediction_count):
    errors = np.asarray(errors)
    matched = len(errors)
    total = reference_count + prediction_count
    return {
        "matched_peaks": matched,
        "unmatched_reference_peaks": reference_count - matched,
        "unmatched_predicted_peaks": prediction_count - matched,
        "f1": 2 * matched / total if total else None,
        "median_signed_error_ms": float(np.median(errors)) if matched else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if matched else None,
        "early_pct": float(100 * np.mean(errors < 0)) if matched else None,
        "exact_pct": float(100 * np.mean(errors == 0)) if matched else None,
        "late_pct": float(100 * np.mean(errors > 0)) if matched else None,
    }


def _diagnose_model(reference_peaks, waveforms, eligible, fs):
    reference_count = prediction_count = 0
    count_errors = []
    timing_errors = {tolerance: [] for tolerance in TOLERANCES_MS}
    for reference, waveform, mask in zip(reference_peaks, waveforms, eligible, strict=True):
        if not mask.any():
            continue
        predicted = ecg_peaks(waveform, fs)
        ref_in_window = int(mask[reference].sum())
        pred_in_window = int(mask[predicted].sum())
        reference_count += ref_in_window
        prediction_count += pred_in_window
        count_errors.append(pred_in_window - ref_in_window)

        # The matcher must not pair peaks on opposite sides of a masked gap.
        edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            observed = reference[(reference >= start) & (reference < stop)]
            estimated = predicted[(predicted >= start) & (predicted < stop)]
            for tolerance, errors in timing_errors.items():
                errors.extend(_signed_match_errors_ms(observed, estimated, fs, tolerance))

    return {
        "reference_peaks": reference_count,
        "predicted_peaks": prediction_count,
        "mean_signed_peak_count_error_per_window": float(np.mean(count_errors)) if count_errors else None,
        "mean_absolute_peak_count_error_per_window": float(np.mean(np.abs(count_errors))) if count_errors else None,
        "by_tolerance_ms": {
            str(tolerance): _summarize_matches(errors, reference_count, prediction_count)
            for tolerance, errors in timing_errors.items()
        },
    }


def diagnose_timing(target, annotations, ppg, waveforms, fs, *, trim_seconds=1.0):
    """Score all forecasts on their common finite interior using annotated R peaks.

    Timing errors are predicted minus annotated time. Unmatched counts depend
    on the tolerance. A displaced beat can lead to one unmatched reference and
    one unmatched prediction, so the counts do not identify a failure cause.
    The 200 ms tolerance is a secondary diagnostic for timing sensitivity.
    """
    target, annotations, ppg = map(np.asarray, (target, annotations, ppg))
    if target.ndim != 2 or target.shape != annotations.shape or target.shape != ppg.shape:
        raise ValueError("Target, annotations and PPG must share shape (windows, samples)")
    if not waveforms:
        raise ValueError("At least one waveform method is required")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(trim_seconds) or trim_seconds < 0:
        raise ValueError("Sampling rate must be positive and trim seconds must be nonnegative")

    edge = round(trim_seconds * fs)
    eligible = np.isfinite(target) & np.isfinite(ppg)
    if edge:
        eligible[:, :edge] = False
        eligible[:, -edge:] = False
    forecasts = {name: np.asarray(values) for name, values in waveforms.items()}
    for values in forecasts.values():
        if values.shape != target.shape:
            raise ValueError("Each waveform method must share the target shape")
        eligible &= np.isfinite(values)

    reference_peaks = [np.flatnonzero(window > 0.5) for window in annotations]
    support = len(target) * max(0, target.shape[1] - 2 * edge)
    return {
        "windows": len(target),
        "windows_with_scoring_support": int(eligible.any(axis=1).sum()),
        "support_fraction": float(eligible.sum() / support) if support else None,
        "timing_error_sign": "predicted minus annotated milliseconds",
        "models": {
            name: _diagnose_model(reference_peaks, values, eligible, fs)
            for name, values in forecasts.items()
        },
    }
