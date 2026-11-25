"""Compare short and long measured ECG intervals on disjoint Holter participants."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .af_events import _coverage, _event_at_decision, AFRun
from .af_shdb_ecg import HORIZONS, STRIDE, inventory
from .af_shdb_stage import MANIFEST_SHA
from .af_wrist_clock_study import _summarize
from .af_wrist_cv import _subject_folds, _within_rhythm_participant_weights
from .af_wrist_holdout import _fit_model, _predict, _write_json
from .af_wrist_operating import score_decisions, select_scored_threshold
from .af_wrist_vote_study import _select_matched


FOLDS = 5
SEED = 2026
FIT_CAP = 120
TARGET = 0.80
CANDIDATES = 41
SENSITIVITY_TOLERANCE = 0.10
ROUTES = {"10": (0,), "30": (1,), "10_and_30": (0, 1)}


@dataclass(frozen=True)
class Holter:
    record: str
    participant: str
    features: np.ndarray
    stable: np.ndarray
    reference: np.ndarray
    episodes: np.ndarray
    start: float
    available: np.ndarray


def load_holters(annotations_dir, feature_dir):
    roster, _ = inventory(annotations_dir)
    result = []
    for record, participant in roster.items():
        with np.load(feature_dir / f"{record}.npz", allow_pickle=False) as archive:
            if archive["release_sha"].item() != MANIFEST_SHA or archive["group"].item() != participant:
                raise ValueError("Each feature archive must match the original Holter participant")
            x = archive["features"]
            ends = archive["ends_s"]
            observed = archive["reference_valid"]
            labels = archive["stable_label"]
            if (x.shape != (len(HORIZONS), len(ends), 7) or observed.shape != ends.shape or labels.shape != ends.shape
                    or ends[0] != max(HORIZONS) or np.any(np.diff(ends) != STRIDE)):
                raise ValueError("ECG intervals must match the fixed 10-second reference clock")
            result.append(Holter(record, participant, x, labels, observed, archive["af_intervals_s"],
                                 ends[0] - STRIDE, np.isfinite(x).all(axis=(0, 2))))
    return result


def person_groups(holters, count, seed):
    groups, labels = [], []
    for item in holters:
        available = item.stable[item.available]
        labels.extend(sorted(set(available[available >= 0])) or [-1])
        groups.extend([item.participant] * max(1, len(set(available[available >= 0]))))
    groups = np.asarray(groups)
    assignment = _subject_folds(np.asarray(labels), groups, count, seed)
    return dict(zip(groups, assignment, strict=True))


def stable_training(holters):
    by_person = {}
    for item in holters:
        take = item.available & (item.stable >= 0)
        by_person.setdefault(item.participant, []).append((item.features[:, take], item.stable[take]))
    features, labels = [], []
    for visits in by_person.values():
        x = np.concatenate([item[0] for item in visits], axis=1)
        y = np.concatenate([item[1] for item in visits])
        for label in (0, 1):
            positions = np.flatnonzero(y == label)
            if len(positions) > FIT_CAP:
                positions = positions[np.linspace(0, len(positions) - 1, FIT_CAP, dtype=int)]
            features.append(x[:, positions])
            labels.append(y[positions])
    return np.concatenate(features, axis=1), np.concatenate(labels)


def replay_record(item, probability, cutoff):
    available = np.isfinite(probability)
    decisions = np.where(available, probability >= cutoff, -1).astype(np.int8)
    positive = decisions == 1
    starts = (positive & ~np.r_[False, positive[:-1]]).astype(np.int8)
    if starts[0] == 1:
        starts[0] = -1
    # The sampled clock starts anew in each recording. Model groups still use participant IDs.
    return AFRun(item.start, STRIDE, item.episodes, decisions, item.reference, available, item.record, starts)


def event_strata(runs):
    references = Counter()
    detected = Counter()
    for run in runs:
        ends = run.start_s + STRIDE * np.arange(1, len(run.decisions) + 1)
        edges = np.r_[run.start_s, ends]
        boundaries = np.flatnonzero(np.diff(run.reference_valid, prepend=False, append=False)).reshape(-1, 2)
        complete, _ = _coverage(run.af_intervals_s, edges[boundaries])
        starts = ends[(run.alert_start_status == 1) & run.reference_valid]
        indices = np.unique(_event_at_decision(starts, run.af_intervals_s))
        found = np.isin(np.arange(len(complete)), indices)
        duration = np.diff(run.af_intervals_s, axis=1).ravel()
        for name, mask in (("under_30_seconds", duration < 30), ("at_least_30_seconds", duration >= 30)):
            references[name] += int(np.sum(complete & mask))
            detected[name] += int(np.sum(complete & mask & found))
    return {name: {"complete_episodes": count, "detected": detected[name]} for name, count in references.items()}


def summarize(runs):
    evaluation = score_decisions(runs)
    metrics = _summarize({**evaluation, "monitor_blocks_with_unknown_cutoff_state": len(runs),
                          "episode_blocks_with_unknown_cutoff_state": len(runs)})
    strata = event_strata(runs)
    if sum(row["complete_episodes"] for row in strata.values()) != metrics["complete_episodes"]:
        raise ValueError("Duration strata must reconcile with the complete episode total")
    return {"complete_episodes": metrics["complete_episodes"], "episodes_detected": metrics["episodes_detected"],
            "episode_sensitivity": metrics["episode_sensitivity"], "false_starts": metrics["wrist_false_alerts"],
            "ecg_observed_hours": metrics["wrist_ecg_observed_seconds"] / 3600,
            "false_starts_per_ecg_hour": metrics["false_alerts_per_observed_wrist_hour"],
            "median_delay_seconds": metrics["median_delay_seconds_in_detected_episodes"], "duration_strata": strata}


def opportunity(runs):
    reachable = 0
    for run in runs:
        ends = run.start_s + STRIDE * np.arange(1, len(run.decisions) + 1)
        edges = np.r_[run.start_s, ends]
        boundaries = np.flatnonzero(np.diff(run.reference_valid, prepend=False, append=False)).reshape(-1, 2)
        complete, _ = _coverage(run.af_intervals_s, edges[boundaries])
        available = run.reference_valid & run.ppg_available
        visited = np.unique(_event_at_decision(ends[available], run.af_intervals_s))
        reachable += int(np.sum(complete & np.isin(np.arange(len(complete)), visited)))
    return reachable


def _columns(features, indices):
    return np.column_stack([features[index] for index in indices])


def evaluate(holters, preflight):
    outer = person_groups(holters, FOLDS, SEED)
    records = {name: [] for name in ROUTES}
    stable = {name: [] for name in ROUTES}
    matched_records = {name: [] for name in ROUTES if name != "30"}
    details = []
    for fold in range(FOLDS):
        train = [item for item in holters if outer[item.participant] != fold]
        test = [item for item in holters if outer[item.participant] == fold]
        inner = person_groups(train, 4, SEED + fold + 1)
        calibration = [item for item in train if inner[item.participant] == 0]
        fit = [item for item in train if inner[item.participant] != 0]
        groups = [set(item.participant for item in block) for block in (fit, calibration, test)]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Both recordings from the same person must stay on one side of each fit and cutoff")
        features, labels = stable_training(fit)
        counts = np.bincount(labels, minlength=2)
        if np.min(counts) == 0:
            raise ValueError("Each training fold needs both observed rhythm classes")
        settings = {"fit_participants": len(groups[0]), "calibration_participants": len(groups[1]), "held_participants": len(groups[2]),
                    "fit_af_windows": int(counts[1]), "fit_non_af_windows": int(counts[0]), "calibration": {}}
        tested_options = {}
        for name, indices in ROUTES.items():
            model = _fit_model(_columns(features, indices), labels, regularization=1.0)
            validation = [(item, _predict(_columns(item.features, indices), item.available, model)) for item in calibration]
            evaluation = {}

            def score(cutoff):
                outcome = score_decisions(tuple(replay_record(item, probability, cutoff) for item, probability in validation))
                evaluation[cutoff] = outcome
                return outcome

            pool = np.concatenate([probability for _, probability in validation])
            operating = select_scored_threshold(pool, score, target_sensitivity=TARGET, max_thresholds=CANDIDATES)
            if operating["threshold"] is None:
                raise ValueError("Every calibration fold needs enough ECG time and complete AF episodes")
            settings["calibration"][name] = {"status": operating["status"], "threshold_candidates": operating["threshold_candidates"]}
            tested = []
            for item in test:
                probability = _predict(_columns(item.features, indices), item.available, model)
                records[name].append(replay_record(item, probability, operating["threshold"]))
                tested.append((item, probability))
                keep = item.available & (item.stable >= 0)
                stable[name].append((item.stable[keep], probability[keep], np.full(int(keep.sum()), item.participant)))
            tested_options[name] = operating, evaluation, tested
        settings["relative_to_30_second_calibration"] = {}
        for name in matched_records:
            _, evaluation, tested = tested_options[name]
            operating = _select_matched(evaluation, tested_options["30"][0], SENSITIVITY_TOLERANCE)
            if operating["threshold"] is None:
                raise ValueError("The matched sensitivity comparison needs a supported calibration cutoff")
            settings["relative_to_30_second_calibration"][name] = operating["status"]
            matched_records[name].extend(replay_record(item, probability, operating["threshold"]) for item, probability in tested)
        details.append(settings)
        print(f"Completed continuous measured ECG participant fold {fold + 1}/{FOLDS}", flush=True)

    outputs = {}
    for name, runs in records.items():
        summary = summarize(runs)
        if summary["complete_episodes"] != preflight["clinical_record_totals"]["complete_af_episodes"]:
            raise ValueError("Every independently staged complete AF episode must stay in the benchmark")
        y, score, participants = (np.concatenate(column) for column in zip(*stable[name], strict=True))
        weights = _within_rhythm_participant_weights(y, participants)
        outputs[str(name)] = {**summary, "participant_weighted_stable_auc": float(roc_auc_score(y, score, sample_weight=weights))}
    return {"scope": "Continuous AF event research reference using measured Holter R-peak intervals only",
            "participants": len(outer), "recordings": len(holters), "folds": details, "target_episode_sensitivity": TARGET,
            "decision_stride_seconds": STRIDE, "horizons_seconds": outputs,
            "complete_episodes_with_available_in_episode_decision": opportunity(records["30"]),
            "matched_compared_with_30_second_reference": {name: summarize(runs) for name, runs in matched_records.items()},
            "matched_sensitivity_tolerance": SENSITIVITY_TOLERANCE}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations-dir", "feature-dir", "preflight", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_bytes())
    _write_json(args.output, evaluate(load_holters(args.annotations_dir, args.feature_dir), preflight))
