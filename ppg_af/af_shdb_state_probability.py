"""Calibrate future AF-time probabilities on separate Holter participants."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .af_shdb_alerts import load_holters
from .af_shdb_ecg import STRIDE
from .af_shdb_state_calibration import grouped_score_inputs
from .af_state_probability import fit_logit_map, logits, summarize_probabilities, time_targets
from .af_wrist_holdout import _write_json
from .af_wrist_operating import AFScoredRun


def holter_time_targets(recordings):
    runs = (AFScoredRun(item.start, STRIDE, item.episodes, raw, item.reference, subject_id=item.participant)
            for item, raw in recordings)
    return time_targets(runs, STRIDE)


def evaluate(holters):
    outcomes = []
    for fold, _, validation, held in grouped_score_inputs(holters):
        train_scores, train_truth, training = holter_time_targets(validation)
        test_scores, test_truth, testing = holter_time_targets(held)
        if set(training) & set(testing):
            raise ValueError("Time-probability calibration and testing participants must be disjoint")
        logistic = fit_logit_map(train_scores, train_truth)
        isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(train_scores, train_truth)
        outcomes.append((test_truth, testing, test_scores, logistic.predict_proba(logits(test_scores))[:, 1],
                         isotonic.predict(test_scores)))
        print(f"Calibrated future AF-time probabilities in fold {fold + 1}", flush=True)
    truth, people, raw, logistic, isotonic = (np.concatenate(column) for column in zip(*outcomes, strict=True))
    return {"scope": "Calibrate 10-second causal ECG AF-duration probabilities without changing alert cutoffs",
            "scoreable_seconds": len(truth) * STRIDE, "participants": len(np.unique(people)),
            "results": {name: summarize_probabilities(values, truth, people) for name, values in
                        (("uncalibrated", raw), ("logit_calibrated", logistic), ("isotonic_calibrated", isotonic))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-dir", "feature-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(load_holters(args.annotations_dir, args.feature_dir)))
