"""Compare ECG time-state calibration with the original episode objective."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .af_shdb_alerts import (
    CANDIDATES, FOLDS, ROUTES, SEED, TARGET, _columns, load_holters, person_groups, replay_record, stable_training, summarize,
)
from .af_state import state_durations, state_summary
from .af_wrist_holdout import _fit_model, _predict, _write_json
from .af_wrist_operating import score_decisions, select_scored_threshold


def participant_state(runs, members):
    totals = defaultdict(lambda: np.zeros(6))
    for (_, record), values in state_durations(runs).items():
        totals[members[record]] += values
    return state_summary(totals)


def state_rank(metrics):
    sensitivity = metrics["af_time_sensitivity"]
    false_rate = metrics["non_af_time_false_positive_rate"]
    if sensitivity is None or false_rate is None:
        raise ValueError("An operating point needs both annotated AF and non-AF time")
    return sensitivity - false_rate, -false_rate


def grouped_score_inputs(holters):
    """Yield original 10-second probabilities and separate calibration and test people."""
    outer = person_groups(holters, FOLDS, SEED)
    indices = ROUTES["10"]
    for fold in range(FOLDS):
        development = [item for item in holters if outer[item.participant] != fold]
        held = [item for item in holters if outer[item.participant] == fold]
        inner = person_groups(development, 4, SEED + fold + 1)
        validation = [item for item in development if inner[item.participant] == 0]
        fitting = [item for item in development if inner[item.participant] != 0]
        groups = [{item.participant for item in split} for split in (fitting, validation, held)]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("ECG recordings from a participant must stay together across all splits")
        x, y = stable_training(fitting)
        model = _fit_model(_columns(x, indices), y, regularization=1.0)
        calibration_scores, held_scores = (
            [(item, _predict(_columns(item.features, indices), item.available, model)) for item in recordings]
            for recordings in (validation, held)
        )
        yield fold, [len(group) for group in groups], calibration_scores, held_scores


def evaluate(holters, published):
    members = {item.record: item.participant for item in holters}
    evaluated = {"episode_target": [], "causal_state_balanced": []}
    calibration = []
    for fold, group_sizes, validation_scores, held_scores in grouped_score_inputs(holters):
        state = {}

        def score(cutoff):
            runs = [replay_record(item, probability, cutoff) for item, probability in validation_scores]
            state[cutoff] = participant_state(runs, members)
            return score_decisions(runs)

        candidates = np.concatenate([scores for _, scores in validation_scores])
        reference = select_scored_threshold(candidates, score, target_sensitivity=TARGET, max_thresholds=CANDIDATES)
        original = reference["threshold"]
        if original is None:
            raise ValueError("The original episode cutoff needs complete validation episodes")
        state_choice = max(state, key=lambda value: (*state_rank(state[value]), value))
        calibration.append({"participants_fit_calibrate_and_held": group_sizes,
                            "thresholds_compared": len(state), "episode_target_status": reference["status"],
                            "episode_cutoff_state_balanced_accuracy": (1 + state_rank(state[original])[0]) / 2,
                            "state_cutoff_state_balanced_accuracy": (1 + state_rank(state[state_choice])[0]) / 2})
        for item, probability in held_scores:
            evaluated["episode_target"].append(replay_record(item, probability, original))
            evaluated["causal_state_balanced"].append(replay_record(item, probability, state_choice))
        print(f"Completed state and episode calibration comparison for fold {fold + 1}/{FOLDS}", flush=True)

    outcomes = {name: {"alerts": summarize(runs), "causal_state": participant_state(runs, members)}
                for name, runs in evaluated.items()}
    expected = {key: value for key, value in published["horizons_seconds"]["10"].items()
                if key != "participant_weighted_stable_auc"}
    if outcomes["episode_target"]["alerts"] != expected:
        raise ValueError("The original continuous ECG episode calibration must reproduce exactly")
    return {"scope": "Compare choosing cutoffs for causal AF time state and new episode alerts on identical measured ECG scores",
            "participants": len(set(members.values())), "recordings": len(holters), "folds": calibration,
            "input_seconds": 10, "results": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-dir", "feature-dir", "alert-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(load_holters(args.annotations_dir, args.feature_dir),
                                      json.loads(args.alert_results.read_bytes())))
