"""Calibrate future AF-time probabilities on separate Holter participants."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .af_events import _intervals, _union
from .af_shdb_alerts import load_holters
from .af_shdb_ecg import STRIDE
from .af_shdb_state_calibration import grouped_score_inputs
from .af_state import annotated_seconds
from .af_wrist_holdout import _write_json


def time_targets(recordings):
    frames = []
    for item, raw in recordings:
        available_probability = np.r_[np.nan, raw[:-1]]
        keep = np.isfinite(available_probability) & item.reference
        edges = item.start + np.arange(len(raw) + 1) * STRIDE
        truth = annotated_seconds(edges, _union(_intervals(item.episodes))) / STRIDE
        frames.append((available_probability[keep], truth[keep], np.full(np.count_nonzero(keep), item.participant)))
    return (np.concatenate(column) for column in zip(*frames, strict=True))


def logits(scores):
    return logit(np.clip(scores, 1e-6, 1 - 1e-6))[:, None]


def fit_calibrators(scores, truth):
    x = logits(scores)
    logistic = LogisticRegression(C=1.0, max_iter=1000)
    # A partly positive bin contributes its exact fraction to each binary outcome.
    logistic.fit(np.concatenate((x, x)), np.repeat([0, 1], len(x)), sample_weight=np.r_[1 - truth, truth])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, truth)
    return logistic, isotonic


def summarize(scores, truth, people):
    brier = truth * np.square(1 - scores) + (1 - truth) * np.square(scores)
    per_person = np.asarray([np.mean((scores - truth)[people == person]) for person in np.unique(people)])
    return {"mean_brier_score_by_duration": float(brier.mean()), "actual_af_time_fraction": float(truth.mean()),
            "estimated_af_time_fraction": float(scores.mean()),
            "mean_absolute_participant_burden_error": float(np.mean(np.abs(per_person))),
            "median_absolute_participant_burden_error": float(np.median(np.abs(per_person)))}


def evaluate(holters):
    outcomes = []
    for fold, _, validation, held in grouped_score_inputs(holters):
        train_scores, train_truth, training = time_targets(validation)
        test_scores, test_truth, testing = time_targets(held)
        if set(training) & set(testing):
            raise ValueError("Time-probability calibration and testing participants must be disjoint")
        logistic, isotonic = fit_calibrators(train_scores, train_truth)
        outcomes.append((test_truth, testing, test_scores, logistic.predict_proba(logits(test_scores))[:, 1],
                         isotonic.predict(test_scores)))
        print(f"Calibrated future AF-time probabilities in fold {fold + 1}", flush=True)
    truth, people, raw, logistic, isotonic = (np.concatenate(column) for column in zip(*outcomes, strict=True))
    return {"scope": "Calibrate 10-second causal ECG AF-duration probabilities without changing alert cutoffs",
            "scoreable_seconds": len(truth) * STRIDE, "participants": len(np.unique(people)),
            "results": {name: summarize(values, truth, people) for name, values in
                        (("uncalibrated", raw), ("logit_calibrated", logistic), ("isotonic_calibrated", isotonic))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-dir", "feature-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(load_holters(args.annotations_dir, args.feature_dir)))
