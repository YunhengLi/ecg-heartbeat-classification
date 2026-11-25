"""Compare matched stable and recent-onset ECG training positives on Holter data."""

import argparse
from pathlib import Path

import numpy as np

from .af_events import _coverage
from .af_shdb_alerts import CANDIDATES, FIT_CAP, FOLDS, SEED, SENSITIVITY_TOLERANCE, STRIDE, TARGET
from .af_shdb_alerts import load_holters, person_groups, replay_record, summarize
from .af_wrist_holdout import _fit_model, _predict, _write_json
from .af_wrist_operating import score_decisions, select_scored_threshold
from .af_wrist_vote_study import _select_matched


MAX_ONSET_SECONDS = 20.0
PRE_ONSET_SECONDS = 20.0
MODES = ("matched_stable_training", "recent_onset_training")


def recent_onset_rows(record):
    ends = record.start + STRIDE * np.arange(1, len(record.stable) + 1)
    edges = np.r_[record.start, ends]
    bounds = np.flatnonzero(np.diff(record.reference, prepend=False, append=False)).reshape(-1, 2)
    complete, _ = _coverage(record.episodes, edges[bounds])
    positive_rows = []
    for onset, offset in record.episodes[complete]:
        first, upper = np.searchsorted(ends, [onset, min(offset, onset + MAX_ONSET_SECONDS)], side="right")
        # A full twenty seconds of reference non-AF before a known onset defines the training cohort.
        before_onset = onset - PRE_ONSET_SECONDS
        left = np.searchsorted(ends, before_onset, side="right")
        ending_before = np.searchsorted(edges, onset, side="left")
        previous_af = (record.episodes[:, 0] < onset) & (record.episodes[:, 1] > before_onset)
        if (before_onset < record.start or previous_af.any() or not record.reference[left:ending_before].all()):
            continue
        observed = np.flatnonzero(record.available[first:upper] & record.reference[first:upper])
        if len(observed):
            positive_rows.append(first + observed[0])
    return np.asarray(positive_rows, dtype=int)


def _sample(values, count):
    if not count:
        return values[:0]
    return values[np.linspace(0, len(values) - 1, count, dtype=int)]


def matched_training(holters):
    groups = {}
    for record in holters:
        positive_rows = recent_onset_rows(record)
        stable = record.available & (record.stable == 1)
        negative = record.available & (record.stable == 0)
        joined = (record.features[0, stable], record.features[0, positive_rows], record.features[0, negative])
        groups.setdefault(record.participant, []).append(joined)
    stable, onset, negatives = [], [], []
    positive_participants = 0
    for visits in groups.values():
        left, right, non_af = (np.concatenate(items) for items in zip(*visits, strict=True))
        positives = min(len(left), len(right), FIT_CAP)
        positive_participants += positives > 0
        stable.append(_sample(left, positives))
        onset.append(_sample(right, positives))
        negatives.append(_sample(non_af, min(len(non_af), FIT_CAP)))
    left, right, non_af = map(np.concatenate, (stable, onset, negatives))
    if not len(left) or not len(non_af):
        raise ValueError("Each fitting group needs matched stable and early AF examples and negative reference windows")
    labels = np.r_[np.ones(len(left), dtype=int), np.zeros(len(non_af), dtype=int)]
    return {MODES[0]: np.vstack((left, non_af)), MODES[1]: np.vstack((right, non_af))}, labels, positive_participants


def evaluate(holters):
    outer = person_groups(holters, FOLDS, SEED)
    combined = {name: [] for name in (*MODES, "recent_onset_at_matched_calibration")}
    receipt = []
    for fold in range(FOLDS):
        training = [record for record in holters if outer[record.participant] != fold]
        held = [record for record in holters if outer[record.participant] == fold]
        inner = person_groups(training, 4, SEED + fold + 1)
        calibration = [record for record in training if inner[record.participant] == 0]
        fitted = [record for record in training if inner[record.participant] != 0]
        if (set(record.participant for record in fitted) & set(record.participant for record in held + calibration)
                or set(record.participant for record in calibration) & set(record.participant for record in held)):
            raise ValueError("Onset models must never fit or calibrate on their held-out groups")
        features, labels, donors = matched_training(fitted)
        settings = {"positive_fitting_participants": donors, "positive_fitting_windows_per_model": int(labels.sum()),
                    "non_af_fitting_windows_per_model": len(labels) - int(labels.sum()), "calibration": {}}
        calibrations = {}
        for mode, matrix in features.items():
            model = _fit_model(matrix, labels)
            validation = [(record, _predict(record.features[0], record.available, model)) for record in calibration]
            scores = np.concatenate([probability for _, probability in validation])
            grid = {}

            def score(cutoff):
                result = score_decisions([replay_record(record, probability, cutoff) for record, probability in validation])
                grid[cutoff] = result
                return result

            point = select_scored_threshold(scores, score, target_sensitivity=TARGET, max_thresholds=CANDIDATES)
            if point["threshold"] is None:
                raise ValueError("The common calibration groups need AF episode and ECG time support")
            calibrations[mode] = point
            settings["calibration"][mode] = point["status"]
            held_scores = [(record, _predict(record.features[0], record.available, model)) for record in held]
            combined[mode].extend(replay_record(record, probability, point["threshold"]) for record, probability in held_scores)
            if mode == MODES[1]:
                matched = _select_matched(grid, calibrations[MODES[0]], SENSITIVITY_TOLERANCE)
                settings["calibration"]["recent_onset_at_matched_calibration"] = matched["status"]
                combined["recent_onset_at_matched_calibration"].extend(
                    replay_record(record, probability, matched["threshold"]) for record, probability in held_scores)
        receipt.append(settings)
        print(f"Finished matched early and stable ECG training in fold {fold + 1}/{FOLDS}", flush=True)
    return {"scope": "Grouped Holter alert comparison for stable versus first recent-onset positive windows",
            "positive_onset_window_seconds": MAX_ONSET_SECONDS, "verified_pre_onset_seconds": PRE_ONSET_SECONDS,
            "grouped_participants": len(outer), "fit_receipts": receipt,
            "held_out_results": {name: summarize(runs) for name, runs in combined.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-dir", "feature-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(load_holters(args.annotations_dir, args.feature_dir)))
