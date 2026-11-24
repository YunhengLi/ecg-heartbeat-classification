"""Check the fixed wrist spectral heart-rate estimator on ScientISST MOVE."""

import argparse
from collections import Counter
from datetime import date, datetime
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import warnings

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_exercise import HEART_RATE_BAND_HZ, spectral_heart_rate
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import ecg_peaks, ppg_features


SOURCE = "https://physionet.org/content/scientisst-move-biosignals/1.0.1/"
RELEASE_SHA = "f8fddf94f7da24e97edc6f60da7d7bc03d21551a265fd511bd106d55b74124ee"
ACTIVITIES = ("baseline", "lift", "greetings", "gesticulate", "jumps", "walk_before", "run", "walk_after")
ESTIMATORS = ("ppg_intervals", "ppg_spectral_peak", "ppg_spectrum_weighted_by_motion_coherence")
PPG_SOURCE_RATE = 64
MOTION_SOURCE_RATE = 32
ECG_SOURCE_RATE = 500
MAX_ACTIVITY_CLOCK_SPREAD_S = 0.002
MAX_HEADER_CLOCK_DIFFERENCE_S = 1.0
PEAK_MATCH_TOLERANCE_S = 0.10
MIN_MATCH_FRACTION = 0.80
MAX_REFERENCE_HEART_RATE_DIFFERENCE_BPM = 5.0


def verify_release(directory):
    checksums = (directory / "SHA256SUMS.txt").read_bytes()
    if sha256(checksums).hexdigest() != RELEASE_SHA:
        raise ValueError("Use the published ScientISST MOVE 1.0.1 checksum list")
    counts = Counter()
    for line in checksums.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        path = directory / name
        if path.name in ("scientisst_chest.edf", "empatica.edf"):
            if sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"EDF file does not match the release: {name}")
            counts[path.name] += 1
    if counts != {"scientisst_chest.edf": 17, "empatica.edf": 16}:
        raise ValueError("The original release must contain 17 chest and 16 E4 recordings")
    subjects = sorted(path.parent for path in directory.glob("*/scientisst_chest.edf"))
    if len(subjects) != 17:
        raise ValueError("Use the original 17 ScientISST MOVE participant folders")
    return subjects


def activity_clock(chest, wrist):
    """Anchor independent device clocks with shared source activity annotations."""
    intervals, offsets = [], []
    for activity in ACTIVITIES:
        chest_events = [a for a in chest.annotations if a.text == activity and a.duration and a.duration > 0]
        wrist_events = [a for a in wrist.annotations if a.text == activity and a.duration and a.duration > 0]
        if len(chest_events) != len(wrist_events):
            continue
        for first, second in zip(chest_events, wrist_events, strict=True):
            if abs(first.duration - second.duration) > 0.010:
                continue
            intervals.append((activity, first.onset, first.onset + first.duration))
            offsets.append(second.onset - first.onset)
    if len(offsets) < 2 or np.ptp(offsets) > MAX_ACTIVITY_CLOCK_SPREAD_S:
        return None
    offset = float(np.median(offsets))
    checks = datetime.combine(date(2000, 1, 1), chest.starttime) - datetime.combine(date(2000, 1, 1), wrist.starttime)
    coarse_offset = checks.total_seconds()
    if abs((offset - coarse_offset + 43_200) % 86_400 - 43_200) > MAX_HEADER_CLOCK_DIFFERENCE_S:
        return None
    return offset, intervals


def _nearest_fraction(reference, other):
    index = np.searchsorted(other, reference)
    before = other[np.maximum(index - 1, 0)]
    after = other[np.minimum(index, len(other) - 1)]
    return float(np.mean(np.minimum(np.abs(before - reference), np.abs(after - reference)) <= PEAK_MATCH_TOLERANCE_S))


def consensus_heart_rate(chest_signals):
    """Return the gel-electrode rate when both simultaneous traces agree."""
    if not np.isfinite(chest_signals).all():
        return np.nan
    gel, dry = [ecg_peaks(signal, ECG_SOURCE_RATE) / ECG_SOURCE_RATE for signal in chest_signals]
    if min(len(gel), len(dry)) < 3:
        return np.nan
    rates = [60 / np.median(np.diff(peaks)) for peaks in (gel, dry)]
    if min(_nearest_fraction(gel, dry), _nearest_fraction(dry, gel)) < MIN_MATCH_FRACTION:
        return np.nan
    if abs(rates[0] - rates[1]) > MAX_REFERENCE_HEART_RATE_DIFFERENCE_BPM:
        return np.nan
    return float(rates[0])


def _clip(values, onset, rate):
    first = int(np.ceil(onset * rate - 1e-9))
    return values[..., first:first + WINDOW_SECONDS * rate]


def _resample(values, source_rate):
    ratio = Fraction(PPG_RATE, source_rate)
    return resample_poly(values, ratio.numerator, ratio.denominator, axis=-1, padtype="line")


def _signals(record, names, rate):
    lookup = {signal.label: signal for signal in record.signals}
    selected = [lookup[name] for name in names]
    if any(signal.sampling_frequency != rate for signal in selected):
        raise ValueError("Use the released native sampling rates")
    return np.stack([signal.data for signal in selected])


def evaluate_person(directory):
    import edfio

    wrist_path = directory / "empatica.edf"
    if not wrist_path.exists():
        return "missing_wrist_file", [], 0
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        chest = edfio.read_edf(directory / "scientisst_chest.edf")
        wrist = edfio.read_edf(wrist_path)
    partial_records = sum("Incomplete data record at the end" in str(notice.message) for notice in notices)
    if len(notices) != partial_records or chest.reserved != "EDF+C" or wrist.reserved != "EDF+C":
        raise ValueError("Unexpected EDF reader warning or unconforming recording type")
    clock = activity_clock(chest, wrist)
    if clock is None:
        return "missing_shared_activity_clock", [], partial_records
    offset, intervals = clock
    ecg = _signals(chest, ("ecg:gel", "ecg:dry"), ECG_SOURCE_RATE)
    ppg = _signals(wrist, ("ppg:wrist",), PPG_SOURCE_RATE)[0]
    accel = _signals(wrist, tuple(f"acc_e4:{axis}" for axis in "xyz"), MOTION_SOURCE_RATE)
    rows = []
    for activity, start, stop in intervals:
        start = max(start, 0, -offset)
        stop = min(stop, chest.duration, wrist.duration - offset)
        first = int(np.ceil((start + WINDOW_SECONDS) / STRIDE_SECONDS)) * STRIDE_SECONDS
        for end in range(first, int(np.floor(stop)) + 1, STRIDE_SECONDS):
            onset = end - WINDOW_SECONDS
            reference = consensus_heart_rate(_clip(ecg, onset, ECG_SOURCE_RATE))
            optical = _clip(ppg, onset + offset, PPG_SOURCE_RATE)
            movement = _clip(accel, onset + offset, MOTION_SOURCE_RATE)
            estimates = (np.nan,) * len(ESTIMATORS)
            valid_wrist = (len(optical) == WINDOW_SECONDS * PPG_SOURCE_RATE and movement.shape[-1] == WINDOW_SECONDS * MOTION_SOURCE_RATE
                           and np.isfinite(optical).all() and np.isfinite(movement).all() and np.ptp(optical) > 0)
            if np.isfinite(reference) and valid_wrist:
                core = _resample(optical, PPG_SOURCE_RATE)
                motion = _resample(movement, MOTION_SOURCE_RATE).T
                features, _ = ppg_features(core, PPG_RATE)
                interval_rate = 60 / features[1] if features[1] > 0 else np.nan
                estimates = (interval_rate, *spectral_heart_rate(core, motion))
            rows.append((activity, reference, bool(valid_wrist), estimates))
    return "clock_verified", rows, partial_records


def _summary(evaluated, activity=None):
    people = set()
    scheduled = reference_windows = wrist_windows = paired_windows = 0
    errors_by_person = []
    for index, (_, rows, _) in enumerate(evaluated):
        matching = [row for row in rows if activity is None or row[0] == activity]
        if not matching:
            continue
        scheduled += len(matching)
        reference = np.array([row[1] for row in matching])
        estimates = np.array([row[3] for row in matching])
        reference_windows += int(np.isfinite(reference).sum())
        wrist_windows += sum(row[2] for row in matching)
        paired = np.isfinite(reference) & np.isfinite(estimates).all(axis=1)
        if paired.any():
            people.add(index)
            paired_windows += int(paired.sum())
            errors_by_person.append(np.abs(reference[paired, None] - estimates[paired]))
    outcomes = {}
    if errors_by_person:
        errors = np.concatenate(errors_by_person)
        participant_means = np.stack([np.mean(group, axis=0) for group in errors_by_person])
        changes = participant_means[:, -1] - participant_means[:, 0]
        outcomes = {"estimators": {name: {"pooled_mae_bpm": float(np.mean(errors[:, col])),
                                          "pooled_median_error_bpm": float(np.median(errors[:, col])),
                                          "participant_mean_mae_bpm": float(np.mean(participant_means[:, col]))}
                                 for col, name in enumerate(ESTIMATORS)},
                    "participants_with_lower_mae_after_motion_weighting": int((changes < 0).sum())}
    return {"participants_with_paired_estimates": len(people), "scheduled_activity_windows": scheduled,
            "windows_with_ecg_consensus": reference_windows, "windows_with_wrist_inputs": wrist_windows,
            "paired_windows": paired_windows, **outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluated = [evaluate_person(directory) for directory in verify_release(args.move_dir)]
        _write_json(args.output, {"scope": "Fixed wrist spectral estimator on an independent activity cohort",
                                  "source": SOURCE, "source_manifest_sha256": RELEASE_SHA,
                                  "source_participant_status": dict(Counter(status for status, _, _ in evaluated)),
                                  "files_with_incomplete_final_edf_record": sum(missing for _, _, missing in evaluated),
                                  "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": STRIDE_SECONDS,
                                  "ppg_hz": PPG_SOURCE_RATE, "accelerometer_hz": MOTION_SOURCE_RATE, "ecg_hz": ECG_SOURCE_RATE,
                                  "minimum_ecg_peak_agreement_fraction": MIN_MATCH_FRACTION,
                                  "ecg_peak_match_tolerance_seconds": PEAK_MATCH_TOLERANCE_S,
                                  "max_ecg_heart_rate_difference_bpm": MAX_REFERENCE_HEART_RATE_DIFFERENCE_BPM,
                                  "heart_rate_search_band_hz": HEART_RATE_BAND_HZ,
                                  "all": _summary(evaluated),
                                  "by_activity": {activity: _summary(evaluated, activity) for activity in ACTIVITIES}})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist activity validation failed: {error}\n")


if __name__ == "__main__":
    main()
