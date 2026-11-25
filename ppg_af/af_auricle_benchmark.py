"""Check pulse timing using separate ear and finger optical measures."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .af_auricle_stage import CASES_PER_COHORT, CHANNELS, COHORTS, RATE, RELEASE_SHA, STARTS_SECONDS
from .af_periwave_benchmark import pulse_view, reference_intervals
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_pulse_consensus import PNN50, RMSSD, VALID, corroborated_peaks
from .features import _bandpass, _interval_features


MAX_LAG = PPG_RATE // 4
ECG_AUDIT_RATE = 250
METHODS = ("finger", "ear", "unshifted_dual", "optical_calibrated_dual")


def optical_lag(primary, secondary):
    first, second = (_bandpass(view[0], PPG_RATE, 0.4, 8.0) for view in (primary, secondary))
    width = len(first)
    target = first[MAX_LAG:-MAX_LAG]
    lags = range(-MAX_LAG, MAX_LAG + 1)
    scores = [np.corrcoef(target, second[MAX_LAG + lag:width - MAX_LAG + lag])[0, 1] for lag in lags]
    if not np.isfinite(scores).all():
        return None
    chosen = max(lags, key=lambda lag: (scores[lag + MAX_LAG], -abs(lag)))
    return chosen, float(scores[chosen + MAX_LAG])


def dual_intervals(primary, secondary, lag):
    _, finger, available = primary
    _, ear, valid_ear = secondary
    in_ear = finger + lag
    common = (in_ear >= 0) & (in_ear < len(valid_ear))
    verified = corroborated_peaks(finger, ear - lag)
    verified &= common & valid_ear[np.clip(in_ear, 0, len(valid_ear) - 1)]
    return _interval_features(finger, PPG_RATE, available, verified_peaks=verified)


def evaluate(directory):
    sources, coverage = Counter(), Counter()
    records, delays = [], []
    members = set()
    for participant, path in enumerate(sorted(directory.glob("*.npz"))):
        with np.load(path, allow_pickle=False) as stored:
            wave, cohort, member = stored["waveforms"], stored["cohort"].item(), stored["record_digest"].item()
            if (stored["release_sha"].item() != RELEASE_SHA or cohort not in COHORTS or member in members
                    or not np.array_equal(stored["starts_seconds"], STARTS_SECONDS)
                    or wave.shape != (len(STARTS_SECONDS), len(CHANNELS), WINDOW_SECONDS * RATE)):
                raise ValueError("Use only the recorded nonoverlapping Auricle collection plan")
        sources[cohort] += 1
        members.add(member)
        early = [pulse_view(row, sampling_rate=RATE) for row in wave[0, 1:]]
        lag_info = optical_lag(*early) if all(view is not None for view in early) else None
        if lag_info is not None:
            delays.append(lag_info)
        for row in wave[1:]:
            reference = reference_intervals(row[0], sampling_rate=RATE, xqrs_rate=ECG_AUDIT_RATE)
            if reference is None:
                continue
            coverage["consensus_ecg"] += 1
            finger, ear = [pulse_view(channel, sampling_rate=RATE) for channel in row[1:]]
            routes = {}
            for name, optical in zip(METHODS[:2], (finger, ear), strict=True):
                if optical is not None:
                    routes[name] = _interval_features(optical[1], PPG_RATE, optical[2])
            if finger is not None and ear is not None:
                routes["unshifted_dual"] = dual_intervals(finger, ear, 0)
                if lag_info is not None:
                    routes["optical_calibrated_dual"] = dual_intervals(finger, ear, lag_info[0])
            accepted = {name: values for name, values in routes.items() if values[VALID] > 0}
            coverage.update(accepted.keys())
            if set(accepted) == set(METHODS):
                records.append((cohort, participant, reference, accepted))
    if sources != dict.fromkeys(COHORTS, CASES_PER_COHORT) or not records:
        raise ValueError("The comparison needs the complete balanced roster and shared ECG and PPG support")

    def summarize(rows):
        if not rows:
            return {"paired_windows": 0}
        bpm, actual = np.asarray([row[2] for row in rows]).T
        results = {}
        for name in METHODS:
            values = np.stack([row[3][name] for row in rows])
            errors = np.abs(values[:, RMSSD] + values[:, PNN50] - actual)
            results[name] = {"median_absolute_irregularity_error": float(np.median(errors)),
                             "mean_absolute_irregularity_error": float(np.mean(errors)),
                             "median_absolute_heart_rate_error_bpm": float(np.median(np.abs(60 / values[:, 1] - bpm))),
                             "median_confirmed_interval_fraction": float(np.median(values[:, VALID]))}
        return {"paired_windows": len(rows), "participants": len({row[1] for row in rows}), "methods": results}

    return {"scope": "Ear and finger corroboration on separate healthy and heart-failure waveform samples",
            "release_sha256": RELEASE_SHA, "participants_by_cohort": dict(sources), "scheduled_evaluation_windows": sum(sources.values()) * 2,
            "reference_peak_rate_hz": RATE, "ecg_cross_check_rate_hz": ECG_AUDIT_RATE,
            "coverage": dict(coverage), "available_optical_calibration": len(delays),
            "median_ear_minus_finger_lag_ms": float(np.median([lag for lag, _ in delays]) * 1000 / PPG_RATE),
            "median_optical_calibration_correlation": float(np.median([value for _, value in delays])),
            "all": summarize(records), "by_cohort": {cohort: summarize([row for row in records if row[0] == cohort]) for cohort in COHORTS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args.staged))
