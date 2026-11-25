"""Test wrist quality models on a separate exercise recording source."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_galaxy_motion_quality import SENSORS, matched_inputs
from .af_galaxy_quality import LARGE_ERROR
from .af_wrist_exercise import MOTION_CHANNELS, SOURCE_RATE, clocked_window, verify_release
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_motion_quality import optical_motion_features
from .af_wrist_pulse_consensus import PNN50, RMSSD, VALID
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_context import _pulse_features
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import _interval_features


def exercise_rows(directory, name, template):
    import wfdb

    prefix = str(directory / name)
    inertial_channels = MOTION_CHANNELS[-3:]
    channels = ["wrist_ppg", "sample_times_for_all_signals_apart_from_ecg", *inertial_channels]
    record = wfdb.rdrecord(prefix, channel_names=channels)
    reference = wfdb.rdann(prefix, "atr")
    if record.fs != SOURCE_RATE or record.sig_name != channels or set(reference.symbol) - {"N", '"'}:
        raise ValueError("Use the official exercise physiology and manually marked ECG reference clocks")
    samples = np.asarray(record.p_signal)
    beats = reference.sample[np.array(reference.symbol) == "N"]
    if len(beats) < 3 or np.any(np.diff(beats) <= 0) or beats[0] < 0 or beats[-1] >= len(samples):
        raise ValueError("The exercise reference beats must be ordered on the recorded clock")
    present = np.isfinite(samples[:, :2]).all(axis=1)
    positions = np.flatnonzero(present)
    paired = samples[present]
    times = np.unwrap(paired[:, 1], period=60.0)
    if not len(times) or positions[0] != 0 or not np.all(np.diff(positions) == 1) or np.any(np.diff(times) <= 0):
        raise ValueError("Exercise PPG and motion require a common, continuous wrist sampling clock")
    times -= times[0]
    duration = len(samples) // SOURCE_RATE
    ends = range(WINDOW_SECONDS, duration + 1, STRIDE_SECONDS)
    window_size = WINDOW_SECONDS * SOURCE_RATE
    mask = np.ones(window_size, dtype=bool)
    people, rows, errors = [], [], []
    for end in ends:
        start_sample, end_sample = (end - WINDOW_SECONDS) * SOURCE_RATE, end * SOURCE_RATE
        start_beat, end_beat = np.searchsorted(beats, (start_sample, end_sample))
        if end_beat - start_beat < 3:
            continue
        base = _interval_features(beats[start_beat:end_beat] - start_sample, SOURCE_RATE, mask)
        pulse = clocked_window(paired[:, 0], times, end)
        motion = np.column_stack([clocked_window(axis, times, end) for axis in paired[:, 2:].T])
        if not np.isfinite(pulse).all() or not np.isfinite(motion).all() or np.ptp(pulse) <= 0:
            continue
        optical = _pulse_features(template, pulse)
        if optical[VALID] <= 0:
            continue
        error = abs(optical[RMSSD] + optical[PNN50] - base[RMSSD] - base[PNN50])
        people.append(name.split("_", 1)[0])
        rows.append(optical_motion_features(pulse, motion))
        errors.append(error)
    return people, rows, errors, len(ends)


def metrics(errors, scores, people):
    labels = errors >= LARGE_ERROR
    within, rank = [], []
    for person in np.unique(people):
        selected = people == person
        if len(np.unique(labels[selected])) == 2:
            within.append(float(roc_auc_score(labels[selected], scores[selected])))
        if np.ptp(errors[selected]) > 0 and np.ptp(scores[selected]) > 0:
            rank.append(float(spearmanr(errors[selected], scores[selected]).statistic))
    both = len(np.unique(labels)) == 2
    correlation = float(spearmanr(errors, scores).statistic) if np.ptp(errors) > 0 and np.ptp(scores) > 0 else None
    return {"pooled_ROC_AUC": float(roc_auc_score(labels, scores)) if both else None,
            "average_precision": float(average_precision_score(labels, scores)) if both else None,
            "mean_within_participant_ROC_AUC": float(np.mean(within)) if within else None,
            "participants_with_both_error_classes": len(within),
            "exploratory_pooled_continuous_error_spearman": correlation,
            "exploratory_mean_within_participant_continuous_error_spearman": float(np.mean(rank)) if rank else None}


def evaluate(args):
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    training, x_train, y_train, _ = matched_inputs(args.galaxy_archive, args.galaxy_acceleration_archive, template)
    if len(training) != 618 or len(np.unique(training)) != 23:
        raise ValueError("Use the same 618 published matched windows from the two training devices")
    records = verify_release(args.exercise_dir)
    test = [exercise_rows(args.exercise_dir, name, template) for name in records]
    people, x, error = (np.concatenate(values) for values in zip(*(record[:3] for record in test), strict=True))
    comparison = {"quality_heuristic": metrics(error, -x[:, 0], people)}
    for device, name in enumerate(SENSORS[:2]):
        comparison[name] = {}
        for width, method in ((2, "optical_quality_and_entropy"), (4, "optical_and_acceleration")):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
            model.fit(x_train[:, device, :width], y_train[:, device])
            comparison[name][method] = metrics(error, model.predict_proba(x[:, :width])[:, 1], people)
    return {"scope": "Frozen GalaxyPPG quality models on an independently collected Shimmer exercise dataset",
            "training_participants": len(np.unique(training)), "training_windows_per_device": len(training),
            "test_records": len(records), "test_participants": len(np.unique(people)),
            "test_scheduled_windows": sum(record[3] for record in test), "test_common_windows": len(people),
            "test_large_irregularity_errors": int(np.sum(error >= LARGE_ERROR)), "irregularity_error_threshold": LARGE_ERROR,
            "test_irregularity_error_min_median_max": np.quantile(error, (0, 0.5, 1)).tolist(),
            "comparisons": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("galaxy-archive", "galaxy-acceleration-archive", "exercise-dir", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
