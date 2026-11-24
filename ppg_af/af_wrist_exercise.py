"""Stress frozen wrist classifiers and heart-rate estimates on public exercise PPG."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_events import AFRun
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_operating import _NO_ALERT_THRESHOLD
from .af_wrist_quality_retrospective import _read_pinned, VARIANTS
from .af_wrist_quality_study import quality_interaction_features
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_study import _IMMEDIATE, STRIDE_SECONDS
from .features import ECG_FEATURE_NAMES, _bandpass, ecg_features, ppg_features


SOURCE = "https://physionet.org/content/wrist/1.0.0/"
SOURCE_RATE = 256
RELEASE_SHA = "3d187a65462656a830072f536363f44c7e48c05fc251fbf3025c23c3f11e732e"
PROTOCOL_SHA = "db3fd9297f112986ae34d04804daee6bbaa8f875b690ecbb0ccfb22055262e68"
BASELINE_SHA = "f7533f83303e511c51365747f0d6aff8add57850d3afa6cc57f7383466437438"
VARIANTS_SHA = "5dc5eddca865828f89aeaf262b98fce6b5b702ae8e29f5542f22c0051e494759"
ACTIVITIES = ("walk", "run", "low_resistance_bike", "high_resistance_bike")
MAX_SOURCE_GAP_SECONDS = 0.10
HIGH_QUALITY = 0.90
LARGE_ERROR_BPM = 10.0
MOTION_CHANNELS = tuple(f"wrist_{sensor}_{axis}" for sensor in ("gyro", "low_noise_accelerometer") for axis in "xyz")
MOTION_LAG_SECONDS = (-0.25, -0.125, 0.0, 0.125, 0.25)
MOTION_RIDGE = 0.10


def verify_release(directory):
    checksums = (directory / "SHA256SUMS.txt").read_bytes()
    if sha256(checksums).hexdigest() != RELEASE_SHA:
        raise ValueError("Use the published PhysioNet exercise version 1.0.0 checksum list")
    expected = {name: digest for digest, name in (line.split() for line in checksums.decode().splitlines())}
    for name, digest in expected.items():
        if sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Exercise file does not match the release: {name}")
    names = (directory / "RECORDS").read_text(encoding="ascii").split()
    keys = [name.split("_", 1) for name in names]
    if (len(set(names)) != 19 or len(names) != 19 or len({person for person, _ in keys}) != 8
            or any(activity not in ACTIVITIES for _, activity in keys)
            or any(f"{name}.{extension}" not in expected for name in names for extension in ("hea", "dat", "atr"))):
        raise ValueError("Exercise release must include all 19 paired records from eight participants")
    return names


def clocked_window(pulse, times, end):
    """Resample only the samples available by the scheduled decision time."""
    start = end - WINDOW_SECONDS
    clock = start + np.arange(WINDOW_SECONDS * SOURCE_RATE) / SOURCE_RATE
    first = max(0, np.searchsorted(times, start, side="right") - 1)
    stop = np.searchsorted(times, end, side="left")
    observed = times[first:stop]
    if (len(observed) < 2 or observed[0] > start or clock[-1] - observed[-1] > MAX_SOURCE_GAP_SECONDS
            or np.any(np.diff(observed) > MAX_SOURCE_GAP_SECONDS)):
        return np.full(WINDOW_SECONDS * PPG_RATE, np.nan)
    uniform = np.interp(clock, observed, pulse[first:stop])
    ratio = Fraction(PPG_RATE, SOURCE_RATE)
    return resample_poly(uniform, ratio.numerator, ratio.denominator, padtype="line")


def _bpm(features):
    median_interval = features[:, 1]
    return np.divide(60.0, median_interval, out=np.full(len(features), np.nan), where=median_interval > 0)


def remove_linear_motion(pulse, motion):
    """Project out wrist-motion components estimated entirely inside one window."""
    lags = np.rint(np.asarray(MOTION_LAG_SECONDS) * PPG_RATE).astype(int)
    padding = int(np.max(np.abs(lags)))
    predictors = np.column_stack([_bandpass(axis, PPG_RATE, 0.4, 8.0) for axis in motion.T])
    padded = np.pad(predictors, ((padding, padding), (0, 0)), mode="edge")
    matrix = np.column_stack([padded[padding + lag:padding + lag + len(pulse)] for lag in lags])
    matrix -= np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    matrix /= np.where(scale > np.finfo(float).eps, scale, 1.0)
    target = _bandpass(pulse - np.median(pulse), PPG_RATE, 0.4, 8.0)
    weights = np.linalg.solve(matrix.T @ matrix + (MOTION_RIDGE * len(pulse)) * np.eye(matrix.shape[1]),
                              matrix.T @ target)
    return pulse - matrix @ weights


def _replay(scores, cutoff):
    decisions = np.where(np.isfinite(scores), scores >= cutoff, -1).astype(np.int8)
    grid_start = WINDOW_SECONDS - STRIDE_SECONDS
    run = AFRun(grid_start, STRIDE_SECONDS, np.empty((0, 2)), decisions)
    (replayed,), _ = _IMMEDIATE((run,), prior_decisions=lambda _: ())
    return decisions, replayed.alert_start_status


def evaluate_record(directory, name, template, models):
    import wfdb

    from .af_wrist_sampling_stress import _extract

    stem = str(directory / name)
    signal = wfdb.rdrecord(stem, channel_names=["chest_ecg", "wrist_ppg",
                                               "sample_times_for_all_signals_apart_from_ecg", *MOTION_CHANNELS])
    annotations = wfdb.rdann(stem, "atr")
    ecg, pulse, times = signal.p_signal[:, :3].T
    if signal.fs != SOURCE_RATE or not np.isfinite(ecg).all() or set(annotations.symbol) - {"N", '"'}:
        raise ValueError("Expected published 256 Hz ECG and the source's manually marked R peaks")
    beats = annotations.sample[np.array(annotations.symbol) == "N"]
    if (len(beats) < 2 or beats[0] < 0 or beats[-1] >= len(ecg) or np.any(np.diff(beats) <= 0)):
        raise ValueError("Reference R peaks must be sorted and inside the ECG clock")
    finite = np.isfinite(pulse) & np.isfinite(times)
    native_rows = np.flatnonzero(finite)
    times = np.unwrap(times[finite], period=60.0)
    if (len(times) < 2 or native_rows[0] != 0 or np.any(np.diff(native_rows) != 1)
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Expected strictly ordered Shimmer timestamps followed by optional padding")
    times -= times[0]
    ends = np.arange(WINDOW_SECONDS, len(ecg) // SOURCE_RATE + 1, STRIDE_SECONDS)
    pulse = pulse[finite]
    cores = np.stack([clocked_window(pulse, times, end) for end in ends])
    supported = np.isfinite(cores).all(axis=1)
    template_features = np.full((len(ends), len(ECG_FEATURE_NAMES)), np.nan)
    quality = np.full(len(ends), np.nan)
    direct_hr = np.full(len(ends), np.nan)
    filtered_hr = np.full(len(ends), np.nan)
    if np.any(supported):
        extracted = _extract(cores[supported], template)
        template_features[supported] = extracted["baseline"]
        direct, qualities = zip(*(ppg_features(core, PPG_RATE) for core in cores[supported]), strict=True)
        quality[supported] = qualities
        direct_hr[supported] = _bpm(np.stack(direct))
        inertial = signal.p_signal[finite, 3:]
        for row in np.flatnonzero(supported):
            motion = np.column_stack([clocked_window(axis, times, ends[row]) for axis in inertial.T])
            if np.isfinite(motion).all():
                filtered, _ = ppg_features(remove_linear_motion(cores[row], motion), PPG_RATE)
                filtered_hr[row] = _bpm(filtered[None, :]).item()
    template_hr = _bpm(template_features)
    starts = (ends - WINDOW_SECONDS) * SOURCE_RATE
    stops = ends * SOURCE_RATE
    ecg_hr = _bpm(np.stack([ecg_features(ecg[start:stop], SOURCE_RATE) for start, stop in zip(starts, stops, strict=True)]))
    reference_hr = np.full(len(ends), np.nan)
    for row, (first, stop) in enumerate(np.searchsorted(beats, np.column_stack((starts, stops)))):
        if stop - first >= 2:
            reference_hr[row] = 60.0 * SOURCE_RATE / np.median(np.diff(beats[first:stop]))
    matrices = {"baseline": template_features,
                "quality": np.column_stack((template_features, quality)),
                "quality_interactions": quality_interaction_features(template_features, quality)}
    scoreable = np.isfinite(template_features).all(axis=1)
    alerts = {route: _replay(_predict(matrices[route], scoreable, locked["model"]), locked["cutoff"])
              for route, locked in models.items()}
    person, activity = name.split("_", 1)
    return {"person": person, "activity": activity, "raw_seconds": len(ecg) / SOURCE_RATE,
            "supported": supported, "quality": quality, "reference_hr": reference_hr,
            "heart_rate": {"direct_ppg": direct_hr, "template_ecg": template_hr, "measured_ecg": ecg_hr},
            "motion_filtered_hr": filtered_hr, "alerts": alerts}


def summarize(records):
    reference = np.concatenate([record["reference_hr"] for record in records])
    quality = np.concatenate([record["quality"] for record in records])
    routes = records[0]["heart_rate"]
    estimates = {name: np.concatenate([record["heart_rate"][name] for record in records]) for name in routes}
    paired = np.isfinite(reference) & np.logical_and.reduce([np.isfinite(values) for values in estimates.values()])
    high_quality = paired & (quality >= HIGH_QUALITY)
    error = {}
    for name, values in estimates.items():
        absolute = np.abs(values - reference)
        error[name] = {"paired_mae_bpm": float(np.mean(absolute[paired])) if paired.any() else None,
                       "paired_median_absolute_error_bpm": float(np.median(absolute[paired])) if paired.any() else None,
                       "available_windows": int(np.isfinite(absolute).sum()),
                       "high_quality_windows_above_10_bpm_error": int(np.sum(high_quality & (absolute > LARGE_ERROR_BPM)))}
    duration = len(reference) * STRIDE_SECONDS
    notifications = {}
    for name in records[0]["alerts"]:
        states = [record["alerts"][name] for record in records]
        decisions, starts = (np.concatenate(items) for items in zip(*states, strict=True))
        definite = int(np.sum(starts == 1))
        notifications[name] = {"scoreable_decisions": int(np.sum(decisions >= 0)),
                               "positive_decisions": int(np.sum(decisions == 1)),
                               "definite_starts": definite, "possible_initial_starts": int(np.sum(starts == -1)),
                               "starts_after_unavailable_decisions": sum(int(np.sum((decision[:-1] < 0) & (status[1:] == 1)))
                                                                          for decision, status in states),
                               "definite_starts_per_decision_clock_hour": definite * 3600 / duration}
    motion_filtered = np.concatenate([record["motion_filtered_hr"] for record in records])
    motion_paired = np.isfinite(reference) & np.isfinite(estimates["direct_ppg"]) & np.isfinite(motion_filtered)
    original_error = np.abs(estimates["direct_ppg"][motion_paired] - reference[motion_paired])
    filtered_error = np.abs(motion_filtered[motion_paired] - reference[motion_paired])
    motion_detail = {"paired_windows": int(motion_paired.sum()),
                     "before_mae_bpm": float(np.mean(original_error)) if len(original_error) else None,
                     "after_mae_bpm": float(np.mean(filtered_error)) if len(filtered_error) else None,
                     "error_decreased_windows": int(np.sum(filtered_error < original_error)),
                     "error_increased_windows": int(np.sum(filtered_error > original_error))}
    return {"participants": len({record["person"] for record in records}), "records": len(records),
            "source_ecg_seconds": sum(record["raw_seconds"] for record in records), "decision_clock_seconds": duration,
            "scheduled_windows": len(reference), "ppg_supported_windows": sum(int(record["supported"].sum()) for record in records),
            "paired_heart_rate_windows": int(paired.sum()), "paired_high_quality_windows": int(high_quality.sum()),
            "heart_rate": error, "motion_filter": motion_detail, "notifications": notifications}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("exercise-dir", "ptt-dir", "protocol-lock", "baseline-lock", "variants-lock", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    try:
        from .af_wrist_smoke import fit_ptt_calibration

        names = verify_release(args.exercise_dir)
        protocol = _read_pinned(args.protocol_lock, PROTOCOL_SHA)
        baseline, *_ = _locked_model(_read_pinned(args.baseline_lock, BASELINE_SHA), protocol)
        variants = _read_pinned(args.variants_lock, VARIANTS_SHA)
        if (variants["source_protocol_sha256"] != PROTOCOL_SHA
                or variants[FEATURE_VERSION_KEY] != protocol[FEATURE_VERSION_KEY]):
            raise ValueError("Quality variants must come from the same development lock")
        models = {"baseline": baseline, **{name: variants["models"][name] for name in VARIANTS}}
        if any(not np.isfinite(model["cutoff"]) or not 0 <= model["cutoff"] <= _NO_ALERT_THRESHOLD for model in models.values()):
            raise ValueError("All frozen notification thresholds must be valid")
        _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
        records = [evaluate_record(args.exercise_dir, name, template, models) for name in names]
        _write_json(args.output, {"scope": "Independent device/activity heart-rate and notification stress test",
                                  "source": SOURCE, "source_manifest_sha256": RELEASE_SHA,
                                  "baseline_model_lock_sha256": BASELINE_SHA, "quality_model_lock_sha256": VARIANTS_SHA,
                                  "native_ppg_and_ecg_hz": SOURCE_RATE, "model_ppg_hz": PPG_RATE,
                                  "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": STRIDE_SECONDS,
                                  "maximum_interpolated_gap_seconds": MAX_SOURCE_GAP_SECONDS,
                                  "high_quality_minimum": HIGH_QUALITY, "diagnostic_error_bpm": LARGE_ERROR_BPM,
                                  "motion_filter_lag_seconds": MOTION_LAG_SECONDS, "motion_filter_ridge": MOTION_RIDGE,
                                  "all": summarize(records), "by_activity": {
                                      activity: summarize([record for record in records if record["activity"] == activity])
                                      for activity in ACTIVITIES}, "af_episode_labels_available": False})
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Wrist exercise stress test failed: {error}\n")


if __name__ == "__main__":
    main()
