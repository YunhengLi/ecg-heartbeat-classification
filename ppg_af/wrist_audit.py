"""Inspect paired wrist windows before training or scoring an AF classifier."""

from dataclasses import asdict, dataclass

import numpy as np

from .features import ppg_features


WINDOW_SECONDS = 30
LABEL_MARGIN_SECONDS = 1.0


@dataclass(frozen=True)
class WristWindowAudit:
    start_s: float
    reference: str
    af_seconds: float | None
    ppg_observed_fraction: float
    ecg_observed_fraction: float
    ppg_quality: float
    acceleration_rms: float | None

    def to_dict(self):
        return asdict(self)


def _sample_width(rate, name):
    if not np.isfinite(rate) or rate <= 0 or not np.isclose(rate, round(rate)):
        raise ValueError(f"Expected a positive integer nominal {name} sampling rate")
    return round(rate) * WINDOW_SECONDS


def _ordered_intervals(intervals, name):
    rows = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if (not np.isfinite(rows).all() or np.any(rows[:, 1] <= rows[:, 0])
            or np.any(rows[1:, 0] < rows[:-1, 1])):
        raise ValueError(f"Expected sorted nonoverlapping {name} intervals")
    if len(rows) < 2:
        return rows
    merged = [rows[0].tolist()]
    for lo, hi in rows[1:]:
        if lo == merged[-1][1]:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return np.asarray(merged)


def _overlap(intervals, start, end):
    return float(np.maximum(0, np.minimum(intervals[:, 1], end) - np.maximum(intervals[:, 0], start)).sum())


def _covered(intervals, start, end):
    return bool(np.any((intervals[:, 0] <= start) & (intervals[:, 1] >= end)))


def _activity_rms(acceleration):
    if acceleration is None:
        return None
    observed = acceleration[np.isfinite(acceleration).all(axis=1)]
    if len(observed) < 2:
        return None
    centered = observed - np.median(observed, axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def audit_windows(
    *, start_s, ppg, ppg_fs, ecg, ecg_fs, af_intervals_s, reference_intervals_s,
    acceleration=None, acceleration_fs=None,
):
    """Audit 30-second windows on a common ECG clock.

    AF and reference intervals are half-open, in seconds on the same clock as
    ``start_s``. A pure AF or non-AF window also needs that rhythm over a
    one-second margin on each side. Its source annotations may extend beyond
    the supplied signal slice. Mixed, transitional and unscorable references
    remain explicit. Acceleration RMS is in the supplied acceleration units.
    """
    if not np.isfinite(start_s):
        raise ValueError("The paired window needs a finite start time on the ECG clock")
    ppg, ecg = np.asarray(ppg, dtype=float), np.asarray(ecg, dtype=float)
    if ppg.ndim != 1 or ecg.ndim != 1:
        raise ValueError("PPG and reference ECG must be one-dimensional")
    ppg_width, ecg_width = _sample_width(ppg_fs, "PPG"), _sample_width(ecg_fs, "ECG")
    if abs(len(ppg) / ppg_fs - len(ecg) / ecg_fs) > max(1 / ppg_fs, 1 / ecg_fs) + 1e-9:
        raise ValueError("The paired PPG and ECG slices must span the same time")

    af = _ordered_intervals(af_intervals_s, "AF")
    reference = _ordered_intervals(reference_intervals_s, "reference")
    if acceleration is not None:
        acceleration = np.asarray(acceleration, dtype=float)
        if acceleration.ndim != 2 or acceleration.shape[1] != 3 or acceleration_fs is None:
            raise ValueError("Acceleration needs three columns and a sampling rate")
        acc_width = _sample_width(acceleration_fs, "acceleration")
        if abs(len(ppg) / ppg_fs - len(acceleration) / acceleration_fs) > max(1 / ppg_fs, 1 / acceleration_fs) + 1e-9:
            raise ValueError("PPG and acceleration slices must span the same time")

    count = min(len(ppg) // ppg_width, len(ecg) // ecg_width)
    results = []
    for index in range(count):
        start = start_s + index * WINDOW_SECONDS
        end = start + WINDOW_SECONDS
        pulse = ppg[index * ppg_width : (index + 1) * ppg_width]
        voltage = ecg[index * ecg_width : (index + 1) * ecg_width]
        act = None if acceleration is None else acceleration[index * acc_width : (index + 1) * acc_width]
        ppg_present, ecg_present = np.isfinite(pulse), np.isfinite(voltage)
        _, quality = ppg_features(pulse, ppg_fs)

        label = "unscorable"
        af_seconds = None
        if ecg_present.all() and _covered(reference, start, end):
            af_seconds = _overlap(af, start, end)
            lo, hi = start - LABEL_MARGIN_SECONDS, end + LABEL_MARGIN_SECONDS
            label = "transition"
            if _covered(reference, lo, hi):
                if _covered(af, lo, hi):
                    label = "af"
                elif _overlap(af, lo, hi) == 0:
                    label = "non_af"

        results.append(WristWindowAudit(start, label, af_seconds, float(ppg_present.mean()),
                                        float(ecg_present.mean()), quality, _activity_rms(act)))
    return results
