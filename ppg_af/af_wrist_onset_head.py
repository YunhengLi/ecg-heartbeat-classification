"""Test onset-specific AF training on two fixed nested development partitions."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import ScoringGrid, load_grid, scoring_policy
from .af_wrist_cv import _fit_and_predict, _nested_folds, _subject_folds
from .af_wrist_data_preflight import _inventory
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import build_quality_matrices, _compare_with_pre_onset
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template
from .af_wrist_train import load_features
from .af_wrist_transition_examples import select_transition_examples
from .af_wrist_transition_native_study import _new_features
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE


FIT_SEEDS = (2026, 2030)
FAMILY = "quality_interactions"


@dataclass(frozen=True)
class OnsetInputs:
    grid: ScoringGrid
    features: np.ndarray
    new_features: np.ndarray
    lock: dict
    lock_hash: str
    pair_hash: str
    subjects: np.ndarray
    original_labels: np.ndarray
    original_stable_mask: np.ndarray
    onset_labels: np.ndarray
    row_kinds: np.ndarray
    original_transition_support: dict


def training_support(rows, score_rows, subjects, onset, kinds):
    if set(subjects[rows]) & set(subjects[score_rows]):
        raise ValueError("Training and scoring participants must remain disjoint")
    return {"positive_windows": int(np.sum(onset[rows] == 1)),
            "positive_participants": len(set(subjects[rows][onset[rows] == 1])),
            "negative_windows": int(np.sum(onset[rows] == 0)),
            "stable_AF_control_windows": int(np.sum(kinds[rows] == "stable_AF")),
            "stable_non_AF_control_windows": int(np.sum(kinds[rows] == "stable_non_AF")),
            "pre_onset_control_windows": int(np.sum(kinds[rows] == "pre_onset"))}


def exact_splits(seed, original_labels, subjects, stable, n_original):
    assignment = _subject_folds(original_labels, subjects, 5, seed)
    original_assignment = _subject_folds(original_labels[:n_original], subjects[:n_original], 5, seed)
    if not np.array_equal(assignment[:n_original], original_assignment):
        raise ValueError("The two new rows changed an original participant fold")
    for fold in range(5):
        train, score = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        mask = {ROUTE: stable}
        inner, count, supported = _nested_folds(train, original_labels, subjects, mask, 3, seed + fold + 1)
        original_train = train[train < n_original]
        orig_inner, orig_count, orig_supported = _nested_folds(original_train, original_labels[:n_original],
            subjects[:n_original], {ROUTE: stable[:n_original]}, 3, seed + fold + 1)
        if count != orig_count or supported != orig_supported or ROUTE not in supported or not np.array_equal(
                inner[train < n_original], orig_inner):
            raise ValueError("The two native rows changed an original calibration split")
        yield fold, train, score, inner, count


def prepare(args):
    lock_bytes = args.lock.read_bytes()
    lock_hash = sha256(lock_bytes).hexdigest()
    lock = json.loads(lock_bytes)
    source_bytes = args.source_manifest.read_bytes()
    source = json.loads(source_bytes)
    receipt = json.loads(args.receipt.read_bytes())
    table, people, timelines = load_features(args.development_features)
    fit, calibration, test = (_subject_ids(lock[k]) for k in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & test or set(people) != fit | calibration
            or (len(fit), len(calibration), len(test)) != (5, 5, 25) or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or receipt["source_inventory_sha256"] != sha256(source_bytes).hexdigest()):
        raise ValueError("Use only the original locked development participants")
    _inventory(source, fit | calibration | test)
    entries = {item["key"]: item for item in source["files"]}
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    shared = _shared_features(grid.table)
    features = build_quality_matrices(grid, args.development_raw, args.development_features,
                                     args.stride_raw, args.stride_features)[FAMILY]
    transition_labels, support = select_transition_examples(grid, timelines, args.development_raw)
    if (np.sum(transition_labels == 0), np.sum(transition_labels == 1)) != (10, 16) or not shared[transition_labels >= 0].all():
        raise ValueError("Use the exact original 26 ECG-supported onset examples")
    new_people, new_labels, new_matrices, new_hash = _new_features(args, receipt, grid, transition_labels, lock_hash, entries)
    new_features = new_matrices[FAMILY]
    stable = shared & (grid.table["run_kind"] != "enriched_episode") & (grid.table["stable_label"] >= 0)
    if np.any(stable & (transition_labels >= 0)) or not np.isfinite(features[shared]).all() or not np.isfinite(new_features).all():
        raise ValueError("Keep the original stable and transition fitting evidence separate")
    onset = np.where(stable, 0, transition_labels)
    kind = np.full(len(stable), "unlabeled", dtype="<U16")
    kind[stable & (grid.table["stable_label"] == 0)] = "stable_non_AF"
    kind[stable & (grid.table["stable_label"] == 1)] = "stable_AF"
    kind[transition_labels == 0] = "pre_onset"
    kind[transition_labels == 1] = "early_AF"
    return OnsetInputs(grid, features, new_features, lock, lock_hash, new_hash, np.r_[grid.subjects, new_people],
                       np.r_[np.where(stable, grid.table["stable_label"], -1), -1, -1], np.r_[stable, False, False],
                       np.r_[onset, new_labels], np.r_[kind, ["pre_onset", "early_AF"]], support)


def support_report(prepared):
    subjects, onset, kind = prepared.subjects, prepared.onset_labels, prepared.row_kinds
    trials = []
    for seed in FIT_SEEDS:
        outer = []
        for fold, train, score, inner, count in exact_splits(
                seed, prepared.original_labels, subjects, prepared.original_stable_mask, len(prepared.grid.subjects)):
            current = training_support(train[onset[train] >= 0], score, subjects, onset, kind)
            inner_fits = []
            for subfold in range(count):
                fit_rows = train[(inner != subfold) & (onset[train] >= 0)]
                inner_fits.append(training_support(fit_rows, train[inner == subfold], subjects, onset, kind))
            outer.append({"fold": fold, "outer_support": current, "inner_support": inner_fits})
        inner = [r for fold in outer for r in fold["inner_support"]]
        trials.append({"seed": seed, "outer_fits": len(outer), "inner_fits": len(inner),
                       "outer_fits_without_early_AF": sum(row["outer_support"]["positive_windows"] == 0 for row in outer),
                       "inner_fits_without_early_AF": sum(row["positive_windows"] == 0 for row in inner),
                       "minimum_outer_early_AF_windows": min(row["outer_support"]["positive_windows"] for row in outer),
                       "minimum_inner_early_AF_windows": min(row["positive_windows"] for row in inner), "folds": outer})
    supported = all(not (trial["outer_fits_without_early_AF"] or trial["inner_fits_without_early_AF"] or any(
        row["negative_windows"] == 0 for fold in trial["folds"] for row in [fold["outer_support"], *fold["inner_support"]]))
                    for trial in trials)
    return {"scope": "Development support audit for a dedicated new-onset quality-interaction head",
            "source_protocol_sha256": prepared.lock_hash, "native_transition_pair_sha256": prepared.pair_hash,
            "original_transition_support": prepared.original_transition_support, "early_AF_windows": int(np.sum(onset == 1)),
            "early_AF_participants": len(set(subjects[onset == 1])), "non_onset_control_windows": int(np.sum(onset == 0)),
            "stable_AF_controls": int(np.sum(kind == "stable_AF")), "stable_non_AF_controls": int(np.sum(kind == "stable_non_AF")),
            "pre_onset_non_AF_controls": int(np.sum(kind == "pre_onset")), "eligible_for_exact_nested_candidate": supported,
            "trials": trials, "independent_predictions": 0}


def evaluate(prepared, seed, previous):
    grid, lock, subjects, labels = prepared.grid, prepared.lock, prepared.subjects, prepared.onset_labels
    n = len(grid.subjects)
    shared = _shared_features(grid.table)
    matrix = np.vstack((np.where(shared[:, None], prepared.features, np.nan), prepared.new_features))
    if not np.isfinite(matrix[labels >= 0]).all():
        raise ValueError("Every eligible onset/control training row needs features")
    baseline_model = fit_template(grid, prepared.features, lock, DEFAULT_C, seed=seed)
    baseline = _replay(grid, baseline_model)
    if baseline[0] != previous:
        raise ValueError("Published quality-interaction replay did not reproduce")
    baseline_scores, baseline_cutoffs, _ = fold_arrays(baseline_model, baseline_model["routes"][ROUTE])
    scores, cutoffs = np.full(n, np.nan), np.full(n, np.nan)
    status = Counter()
    fits = 0

    def train_and_predict(train, score):
        if set(subjects[train]) & set(subjects[score]) or set(labels[train]) != {0, 1}:
            raise ValueError("Onset training requires both labels and excludes scoring participants")
        if np.any(score >= n):
            raise ValueError("The independently staged pair is reserved for fitting only")
        return _fit_and_predict(matrix, labels, train, score, regularization=DEFAULT_C)

    for fold, train, held, inner, count in exact_splits(
            seed, prepared.original_labels, subjects, prepared.original_stable_mask, n):
        score_rows = held[held < n]
        if not np.array_equal(score_rows, np.flatnonzero(baseline_model["fold_assignment"] == fold)):
            raise ValueError("Candidate and baseline must share outer held-out membership")
        fit_rows = train[labels[train] >= 0]
        scores[score_rows] = train_and_predict(fit_rows, score_rows)
        inner_scores = np.full(n, np.nan)
        for inner_fold in range(count):
            inner_fit = train[(inner != inner_fold) & (labels[train] >= 0)]
            inner_test = train[(inner == inner_fold) & (train < n)]
            inner_scores[inner_test] = train_and_predict(inner_fit, inner_test)
        original_training = train[train < n]
        if not np.array_equal(np.flatnonzero(np.isfinite(inner_scores)), original_training[shared[original_training]]):
            raise ValueError("Onset calibration must use every originally scoreable training participant row")
        evaluate_threshold, pool = scoring_policy(grid, inner_scores, set(grid.subjects[original_training]), _IMMEDIATE)
        point = select_scored_threshold(pool, evaluate_threshold, target_sensitivity=lock["target_sensitivity"],
                                        max_thresholds=max(3, len(pool) + 1))
        if point["threshold"] is not None:
            cutoffs[score_rows] = point["threshold"]
        status[point["status"]] += 1
        fits += 1 + count
    if (not np.array_equal(np.isfinite(scores), np.isfinite(baseline_scores))
            or not np.array_equal(np.isfinite(cutoffs), np.isfinite(baseline_cutoffs))):
        raise ValueError("The onset model must preserve fixed score and cutoff support")
    result = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    print(f"{seed}: onset head detected {result[0]['episodes_detected']}/{result[0]['complete_episodes']}, "
          f"with {result[0]['wrist_false_alerts']} false starts", flush=True)
    return {"seed": seed, "outer_and_inner_fits": fits, "quality_event_baseline": baseline[0],
            "quality_event_baseline_calibration_status": _calibration_status(baseline_model),
            "dedicated_onset_head": _compare_with_pre_onset(baseline, result, lock),
            "onset_head_calibration_status": dict(status)}


def _run(args):
    prepared = prepare(args)
    support = support_report(prepared)
    trials = []
    if support["eligible_for_exact_nested_candidate"]:
        receipt_bytes = args.stability_receipt.read_bytes()
        previous = {row["seed"]: row["quality_interactions"]["summary"] for row in json.loads(receipt_bytes)["trials"]}
        for seed in FIT_SEEDS:
            trial = evaluate(prepared, seed, previous[seed])
            trials.append({"scope": "Dedicated onset-vs-non-onset head on original grouped development folds",
                           "logistic_C": DEFAULT_C, "original_quality_stability_report_sha256": sha256(receipt_bytes).hexdigest(),
                           "independent_predictions": 0, **trial})
    else:
        print("At least one original fitting split lacks early AF or controls. Candidate fits skipped.", flush=True)
    return {"scope": "Fixed development onset-head support audit and controlled comparison", "support_audit": support, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = ("source-manifest", "source-dir", "cache-dir", "development-raw", "development-features",
                "development-preroll", "stride-raw", "stride-features", "lock", "pair", "receipt",
                "protected-supplement", "ptt-dir", "stability-receipt", "output")
    for argument in required:
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new aggregate output")
    try:
        _write_json(args.output, _run(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Dedicated onset head failed: {error}\n")


if __name__ == "__main__":
    main()
