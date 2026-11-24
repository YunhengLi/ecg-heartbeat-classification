"""Test participant-held-out signal reliability and cross-device transfer."""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_galaxy_quality import (
    DOI, GALAXY_RATE, LARGE_ERROR, PARTICIPANTS, SENSORS, aligned_windows, clock, from_archive, optical_metrics, uniform,
)
from .af_wrist_exercise import HEART_RATE_BAND_HZ, motion_weighted_spectrum
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _write_json
from .af_wrist_smoke import fit_ptt_calibration
from .features import ppg_features


AXES = ("x", "y", "z")
DEVICES = (("GalaxyWatch", GALAXY_RATE, 1e-3), ("E4", 32, 1e-6))
FOLDS = 5


def read_motion(archive, participant):
    streams = []
    for folder, rate, scale in DEVICES:
        filename = f"Dataset/{participant}/{folder}/ACC.csv"
        if filename not in archive.namelist():
            return None
        frame = from_archive(archive, filename, ("timestamp", *AXES))
        streams.append(([clock(frame, "timestamp", name, scale) for name in AXES], rate))
    return streams


def motion_window(stream, rate, start):
    axes = [uniform(axis, start, rate, rate) for axis in stream]
    if any(axis is None for axis in axes):
        return None
    return resample_poly(np.column_stack(axes), PPG_RATE, rate, axis=0, padtype="line")


def sensor_features(pulse, motion):
    ppg, quality = ppg_features(pulse, PPG_RATE)
    frequencies, raw, retained = motion_weighted_spectrum(pulse, motion)
    support = (frequencies >= HEART_RATE_BAND_HZ[0]) & (frequencies <= HEART_RATE_BAND_HZ[1])
    total = raw[support].sum()
    residual = float(retained[support].sum() / total) if total > 0 else 0.0
    magnitude = np.linalg.norm(motion, axis=1)
    center = np.median(magnitude)
    motion_variation = np.median(np.abs(magnitude - center)) / max(center, np.finfo(float).eps)
    return quality, ppg[-1], residual, np.log1p(100 * motion_variation)


def matched_inputs(path, acceleration_path, template):
    coverage = Counter()
    participants, features, labels = [], [], []
    with ZipFile(path) as archive, ZipFile(acceleration_path) as motion:
        for person in PARTICIPANTS:
            native_motion = read_motion(motion, person)
            if native_motion is None:
                coverage["participants_missing_bilateral_acceleration"] += 1
                continue
            for _, start, reference, pulses in aligned_windows(archive, person, coverage):
                if not all(name in pulses for name in SENSORS[:2]):
                    continue
                pulse_values = [pulses[name] for name in SENSORS[:2]]
                movement_values = [motion_window(signal, rate, start) for signal, rate in native_motion]
                if any(values is None for values in movement_values):
                    continue
                errors = [optical_metrics(pulse, reference, template).get("template") for pulse in pulse_values]
                if any(result is None for result in errors):
                    continue
                participants.append(person)
                features.append([sensor_features(pulse, movement) for pulse, movement in zip(pulse_values, movement_values, strict=True)])
                labels.append([result[1] >= LARGE_ERROR for result in errors])
            print(f"Checked matched windows for {len(set(participants))} participants", flush=True)
    return np.asarray(participants), np.asarray(features), np.asarray(labels), coverage


def evaluate_models(people, x, y):
    assigned = tuple(GroupKFold(n_splits=FOLDS).split(x, groups=people))

    def performance(target, scores):
        return {"roc_auc": float(roc_auc_score(target, scores)),
                "average_precision": float(average_precision_score(target, scores))}

    comparisons = {}
    for testing, sensor in enumerate(SENSORS[:2]):
        target = y[:, testing]
        results = {"n_large_errors": int(target.sum()), "quality_heuristic": performance(target, -x[:, testing, 0])}
        for training, routing in ((testing, "same_device"), (1 - testing, "other_device")):
            outcomes = {}
            for width, method in ((2, "optical_quality_and_entropy"), (4, "optical_and_acceleration")):
                predictions = np.full(len(people), np.nan)
                for train, test in assigned:
                    train_labels = y[train, training]
                    if set(people[train]) & set(people[test]) or len(np.unique(train_labels)) != 2:
                        raise ValueError("All grouped fits require separate participants and both error labels")
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
                    model.fit(x[train, training, :width], train_labels)
                    predictions[test] = model.predict_proba(x[test, testing, :width])[:, 1]
                if not np.isfinite(predictions).all():
                    raise ValueError("Each common row requires one prediction from participant-held-out training")
                outcomes[method] = performance(target, predictions)
            results[routing] = outcomes
        comparisons[sensor] = results
    return comparisons


def evaluate(args):
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    people, x, y, coverage = matched_inputs(args.archive, args.acceleration_archive or args.archive, template)
    if len(np.unique(people)) < FOLDS or not np.isfinite(x).all():
        raise ValueError("Both wearable accelerometers must provide finite matched inputs in each participant group")
    return {"scope": "Grouped reliability ranking on two GalaxyPPG devices with native wrist motion",
            "source_doi": DOI, "outer_participant_groups": FOLDS, "large_irregularity_error": LARGE_ERROR,
            "matched_windows": len(people), "participants": len(np.unique(people)), "coverage": dict(coverage),
            "comparisons": evaluate_models(people, x, y)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--acceleration-archive", type=Path)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
