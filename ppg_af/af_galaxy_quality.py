"""Measure transfer of the wrist PPG quality boundary to paired commercial sensors."""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from .af_periwave_benchmark import reference_intervals
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_pulse_consensus import PNN50, RMSSD, VALID
from .af_wrist_quality_confirmation import QUALITY_CUTOFF
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_context import _pulse_features
from .features import ppg_features


DOI = "10.5281/zenodo.14635823"
ARCHIVE_MD5 = "d9a48bfcd07928fb22b96dc40cf2e7d4"
PARTICIPANTS = tuple(f"P{number:02}" for number in range(1, 25))
ACTIVITIES = ("baseline", "keyboard-typing", "mobile-typing", "standing", "walking", "jogging", "running")
LEAD_IN_SECONDS = 10
POLAR_RATE, E4_RATE, GALAXY_RATE = 130, 64, 25
CLOCK_OFFSET_SECONDS = 9 * 3600
LARGE_ERROR = 0.20
SENSORS = ("galaxy_25hz", "e4_64hz", "e4_simulated_25hz")
ROUTES = tuple(f"{sensor}_{method}" for sensor in SENSORS for method in ("template", "direct"))
MEMBERS = ("Event.csv", "PolarH10/ECG.csv", "E4/BVP.csv", "GalaxyWatch/PPG.csv")


def from_archive(archive, name, columns):
    with archive.open(name) as member:
        return pd.read_csv(member, usecols=columns)


def windows(events):
    for name in ACTIVITIES:
        activity = events[events["session"] == name]
        if activity.empty:
            continue
        stamps, directions = activity["timestamp"].to_numpy(dtype=np.int64), activity["status"].str.upper().tolist()
        if len(stamps) != 2 or directions != ["ENTER", "EXIT"] or stamps[0] >= stamps[1]:
            raise ValueError(f"Ambiguous activity boundary for {name}")
        start, stop = stamps
        for offset in range(LEAD_IN_SECONDS, (stop - start) // 1000 - WINDOW_SECONDS + 1, WINDOW_SECONDS):
            yield name, start / 1000 + offset


def clock(table, clock_column, signal_column, scale, offset=0, polarity=1):
    time = table[clock_column].to_numpy(dtype=float) * scale - offset
    values = polarity * table[signal_column].to_numpy(dtype=float)
    if not np.isfinite(time).all() or np.any(np.diff(time) < 0):
        raise ValueError("Published clocks must contain finite, ordered times")
    bad = ~np.isfinite(values)
    bad[1:] |= np.diff(time) == 0
    return time[~bad], values[~bad]


def uniform(stream, start, source_rate, required_rate):
    time, signal = stream
    first, stop = np.searchsorted(time, [start, start + WINDOW_SECONDS], side="left")
    local_time, values = time[first:stop] - start, signal[first:stop]
    maximum_gap = 3 / source_rate
    if (len(values) < 0.95 * WINDOW_SECONDS * source_rate or local_time[0] > maximum_gap
            or WINDOW_SECONDS - local_time[-1] > maximum_gap or np.max(np.diff(local_time)) > maximum_gap):
        return None
    grid = np.arange(WINDOW_SECONDS * required_rate) / required_rate
    return np.interp(grid, local_time, values)


def optical_metrics(signal, reference, template):
    if np.ptp(signal) <= 0:
        return {}
    direct, quality = ppg_features(signal, PPG_RATE)
    bpm, irregularity = reference
    metrics = {}
    for name, features in (("template", _pulse_features(template, signal)), ("direct", direct[:VALID + 1])):
        if features[VALID] > 0 and np.isfinite(features).all():
            error = features[RMSSD] + features[PNN50] - irregularity
            metrics[name] = quality, abs(error), error, abs(60 / features[1] - bpm)
    return metrics


def aligned_windows(archive, participant, coverage):
    prefix = f"Dataset/{participant}/"
    if not all(prefix + name in archive.namelist() for name in MEMBERS):
        coverage["participants_missing_one_or_more_members"] += 1
        return
    event, ecg_path, e4_path, galaxy_path = (prefix + name for name in MEMBERS)
    rows = list(windows(from_archive(archive, event, ("timestamp", "session", "status"))))
    polar = clock(from_archive(archive, ecg_path, ("phoneTimestamp", "ecg")), "phoneTimestamp", "ecg", 1e-3,
                  CLOCK_OFFSET_SECONDS)
    e4 = clock(from_archive(archive, e4_path, ("timestamp", "value")), "timestamp", "value", 1e-6)
    galaxy = clock(from_archive(archive, galaxy_path, ("timestamp", "ppg")), "timestamp", "ppg", 1e-3, polarity=-1)
    coverage["participants_with_all_members"] += 1
    for activity, start in rows:
        coverage["scheduled_windows"] += 1
        ecg = uniform(polar, start, POLAR_RATE, POLAR_RATE)
        if ecg is None:
            continue
        coverage["uniform_ecg_windows"] += 1
        reference = reference_intervals(ecg, sampling_rate=POLAR_RATE, xqrs_rate=250)
        if reference is None:
            continue
        coverage["ecg_consensus_windows"] += 1
        e4_values = uniform(e4, start, E4_RATE, E4_RATE)
        galaxy_values = uniform(galaxy, start, GALAXY_RATE, GALAXY_RATE)
        pulses = {}
        if galaxy_values is not None:
            pulses[SENSORS[0]] = resample_poly(galaxy_values, PPG_RATE, GALAXY_RATE, padtype="line")
        if e4_values is not None:
            pulses[SENSORS[1]] = resample_poly(e4_values, PPG_RATE, E4_RATE, padtype="line")
            limited = resample_poly(e4_values, GALAXY_RATE, E4_RATE, padtype="line")
            pulses[SENSORS[2]] = resample_poly(limited, PPG_RATE, GALAXY_RATE, padtype="line")
        yield activity, start, reference, pulses


def measure(archive, participant, template, coverage):
    result = []
    for activity, _, reference, pulses in aligned_windows(archive, participant, coverage):
        signals = {f"{name}_{method}": metrics for name, values in pulses.items()
                   for method, metrics in optical_metrics(values, reference, template).items()}
        coverage.update(signals.keys())
        if len(signals) == len(ROUTES):
            result.append((activity, signals))
    return result


def summarize(records):
    if not records:
        return {"windows": 0, "participants": 0}
    participants = np.asarray([row[0] for row in records])
    methods = {}
    for name in ROUTES:
        quality, absolute, signed, heart_rate_error = np.asarray([row[2][name] for row in records]).T
        low = quality < QUALITY_CUTOFF
        severe = absolute >= LARGE_ERROR
        within = []
        for person in np.unique(participants):
            group = participants == person
            below, above = group & low, group & ~low
            if below.any() and above.any():
                within.append(absolute[below].mean() - absolute[above].mean())
        methods[name] = {
            "median_ppg_quality": float(np.median(quality)), "lower_quality_windows": int(low.sum()),
            "median_absolute_irregularity_error": float(np.median(absolute)), "median_irregularity_bias": float(np.median(signed)),
            "median_absolute_heart_rate_error_bpm": float(np.median(heart_rate_error)),
            "median_absolute_error_lower_quality": float(np.median(absolute[low])) if low.any() else None,
            "median_absolute_error_higher_quality": float(np.median(absolute[~low])) if (~low).any() else None,
            "participants_with_both_quality_strata": len(within),
            "participant_mean_error_difference_low_minus_other": float(np.mean(within)) if within else None,
            "large_error_windows": int(severe.sum()), "large_error_windows_flagged_lower_quality": int((severe & low).sum()),
            "quality_roc_auc_for_large_error": float(roc_auc_score(severe, -quality)) if severe.any() and (~severe).any() else None,
        }
    return {"windows": len(records), "participants": len(np.unique(participants)), "methods": methods}


def evaluate(path, ptt_dir):
    _, template = fit_ptt_calibration(ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    coverage = Counter()
    records = []
    with ZipFile(path) as archive:
        for number, person in enumerate(PARTICIPANTS, 1):
            results = measure(archive, person, template, coverage)
            records.extend((person, activity, signals) for activity, signals in results)
            print(f"Scored {len(results)} matched windows on participant {number}", flush=True)
    if coverage["participants_with_all_members"] != 23 or not records:
        raise ValueError("The released archive must supply all four signals and clock files for 23 of 24 participants")
    return {"scope": "External cross-device optical quality and interval reliability without clinical AF labels",
            "source_doi": DOI, "declared_archive_md5": ARCHIVE_MD5, "scheduled_participants": len(PARTICIPANTS),
            "window_seconds": WINDOW_SECONDS, "quality_cutoff": QUALITY_CUTOFF, "large_irregularity_error": LARGE_ERROR,
            "coverage": dict(coverage), "all_common": summarize(records),
            "by_activity": {activity: summarize([row for row in records if row[1] == activity]) for activity in ACTIVITIES}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args.archive, args.ptt_dir))
