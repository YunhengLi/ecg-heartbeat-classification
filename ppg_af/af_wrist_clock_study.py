"""Compare causal alert history and decision strides on the same development periods."""

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_alert_study import _summary
from .af_wrist_cv import train_oof
from .af_wrist_features import FEATURE_VERSION_KEY, ROUTES, WINDOW_SECONDS, read_feature_version
from .af_wrist_history import apply_alert_history
from .af_wrist_holdout import MONITOR_KIND, _predict, _shared_features
from .af_wrist_operating import AFScoredRun, _evaluate_scopes, _remaining_non_af_time, select_scored_threshold
from .af_wrist_train import load_features


SCORED_KINDS = ("monitor", "monitor_wear", "enriched_episode")


@dataclass(frozen=True)
class ScoringGrid:
    seconds: int
    table: dict
    subjects: np.ndarray
    timelines: tuple
    history_rows: dict
    unknown_source: np.ndarray


def load_grid(base_table, subjects, timelines, raw_dir, feature_dir, *, seconds):
    """Append scored-only history and finer cells to the original fitting rows."""
    keys = ("reference_valid", "ppg_available", "stable_label", "run_kind", *ROUTES)
    chunks = {key: [base_table[key]] for key in keys}
    people, unknown = [subjects], [np.zeros(len(subjects), dtype=bool)]
    original = {(timeline.subject, timeline.start): timeline for timeline in timelines if timeline.kind in SCORED_KINDS}
    history_rows, blocks = {}, {}
    offset = len(subjects)
    finer = seconds != WINDOW_SECONDS
    for subject in np.unique(subjects):
        with np.load(raw_dir / f"{subject}.npz", allow_pickle=False) as raw, np.load(
            feature_dir / f"{subject}.npz", allow_pickle=False,
        ) as features:
            if str(raw["subject_id"].item()) != subject or any(
                not np.array_equal(raw[key], features[key]) for key in ("subject_id", "start_s", "run_id", "run_kind")
            ):
                raise ValueError(f"Raw and feature clocks disagree for participant {subject}")
            if read_feature_version(features) != base_table[FEATURE_VERSION_KEY]:
                raise ValueError("Use the same feature version for baseline, history and stride scores")
            starts, run_starts = raw["start_s"], raw["run_start_s"]
            kinds = raw["run_kind"]
            evaluation = raw["evaluation"] if finer else np.zeros(len(starts), dtype=bool)
            source_unknown = raw["input_status"] == "unknown" if finer else raw["preroll_status"] == "cache_missing"
            values = {name: features[name].copy() for name in ROUTES}
            for matrix in values.values():
                matrix[source_unknown] = np.nan
            raw_available = raw["ppg_available"] if finer else features["ppg_available"]
            for key, value in {
                **values, "reference_valid": raw["reference_valid"], "ppg_available": raw_available,
                "stable_label": np.full(len(starts), -1), "run_kind": kinds,
            }.items():
                chunks[key].append(value)
            people.append(np.full(len(starts), subject))
            unknown.append(source_unknown)
            for start in np.unique(run_starts):
                key = (subject, float(start))
                if key not in original or key in blocks:
                    raise ValueError(f"Unexpected or repeated evaluation block: {key}")
                parent = original[key]
                local = np.flatnonzero(run_starts == start)
                if not np.all(kinds[local] == parent.kind):
                    raise ValueError(f"Evaluation block scope has changed: {key}")
                prior = local[~evaluation[local]]
                prior = prior[np.argsort(starts[prior])]
                expected_ends = np.arange(start - 60, start + 1, seconds)
                if not np.array_equal(starts[prior] + seconds, expected_ends) or np.any(raw["reference_valid"][prior]):
                    raise ValueError(f"History must end at the evaluation cutoff and have no scoring reference: {key}")
                history_rows[key] = prior + offset
                if not finer:
                    blocks[key] = parent
                    continue
                scored = local[evaluation[local]]
                scored = scored[np.argsort(starts[scored])]
                stop = start + len(parent.rows) * WINDOW_SECONDS
                if (not np.array_equal(starts[scored], np.arange(start, stop, seconds))
                        or not np.array_equal(raw["reference_valid"][scored],
                                              np.repeat(base_table["reference_valid"][parent.rows], WINDOW_SECONDS // seconds))):
                    raise ValueError(f"The finer grid changed the ECG reference exposure: {key}")
                blocks[key] = replace(parent, rows=scored + offset)
            offset += len(starts)
    if set(blocks) != set(original):
        raise ValueError("History and stride must cover the original scheduled evaluation blocks")
    table = {key: np.concatenate(parts) for key, parts in chunks.items()}
    return ScoringGrid(seconds, table, np.concatenate(people), tuple(blocks.values()), history_rows, np.concatenate(unknown))


def scoring_runs(grid, probabilities, people):
    """Map row-aligned scores to the two existing evaluation scopes."""
    monitored, episodes = [], []
    for block in grid.timelines:
        if block.subject not in people:
            continue
        rows = block.rows
        run = AFScoredRun(block.start, grid.seconds, block.episodes, probabilities[rows],
                          grid.table["reference_valid"][rows], grid.table["ppg_available"][rows], block.subject)
        episodes.append(run)
        if block.kind == MONITOR_KIND:
            monitored.append(run)
    return tuple(monitored), tuple(episodes)


def predecessor_decisions(grid, decisions):
    """Look up chronological binary or unavailable decisions before a block."""
    def history(run):
        rows = grid.history_rows[(run.subject_id, run.start_s)]
        return [None if grid.unknown_source[row] else int(decisions[row]) for row in rows]

    return history


def _policy(grid, probabilities, people, confirmation_windows):
    monitored, episodes = scoring_runs(grid, probabilities, people)

    coverage = tuple(run.decisions(2.0) for run in monitored)
    non_af = _remaining_non_af_time(coverage)
    analyzable_non_af = _remaining_non_af_time(coverage, analyzable_only=True)

    def evaluate(threshold):
        decisions = np.where(np.isfinite(probabilities), probabilities >= threshold, -1).astype(np.int8)
        history = predecessor_decisions(grid, decisions)

        monitor_decisions, monitor_unknown = apply_alert_history(
            (run.decisions(threshold) for run in monitored), confirmation_windows=confirmation_windows,
            prior_decisions=history,
        )
        episode_decisions, episode_unknown = apply_alert_history(
            (run.decisions(threshold) for run in episodes), confirmation_windows=confirmation_windows,
            prior_decisions=history,
        )
        result = _evaluate_scopes(monitor_decisions, episode_decisions, allow_daily_rate=False,
                                  non_af_seconds=non_af, analyzable_non_af_seconds=analyzable_non_af)
        return {**result, "monitor_blocks_with_unknown_cutoff_state": monitor_unknown,
                "episode_blocks_with_unknown_cutoff_state": episode_unknown}

    pool = [run.scores for run in episodes] + [
        probabilities[grid.history_rows[(run.subject_id, run.start_s)]] for run in episodes
    ]
    return evaluate, np.concatenate(pool) if pool else np.empty(0)


def _summarize(metrics):
    return {**_summary(metrics),
            "monitor_blocks_with_unknown_cutoff_state": metrics["monitor_blocks_with_unknown_cutoff_state"],
            "episode_blocks_with_unknown_cutoff_state": metrics["episode_blocks_with_unknown_cutoff_state"],
            "possibly_detected_episodes": metrics["episode_metrics"]["complete_episodes_with_definite_or_possible_alert"],
            "possible_non_af_alert_timestamps": metrics["burden_metrics"]["ambiguous_non_af_alert_start_timestamps"]}


def fit_nested_grid(grid, lock, confirmation_windows):
    """Fit and calibrate on the same participant folds as the clock comparison."""
    shared = _shared_features(grid.table)
    matrices = {name: np.where(shared[:, None], grid.table[name], np.nan) for name in ROUTES}
    eligible = shared & (grid.table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        full = np.full(len(grid.subjects), np.nan)
        full[rows] = scores
        evaluate, pool = _policy(grid, full, set(grid.subjects[rows]), confirmation_windows)
        return select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                       max_thresholds=max(3, len(pool) + 1))

    return train_oof(matrices, grid.table["stable_label"], grid.subjects, n_splits=5, inner_splits=3,
                     seed=2026, train_eligible=eligible, calibration_selector=calibrate)


def compare_grid(grid, lock, previous):
    def rules(confirmation_windows):
        return 1 if confirmation_windows == 1 else WINDOW_SECONDS // grid.seconds + 1

    calibration = set(lock["calibration_subjects"])
    shared = _shared_features(grid.table)
    in_calibration = np.isin(grid.subjects, sorted(calibration))
    fixed, recalibrated, grouped = {}, {}, {}
    for logical_rule in (1, 2):
        count = rules(logical_rule)
        label = str(logical_rule)
        fixed[label], recalibrated[label] = {}, {}
        for name in ROUTES:
            scores = _predict(grid.table[name], shared & in_calibration, lock["routes"][name]["model"])
            evaluate, pool = _policy(grid, scores, calibration, count)
            cutoff = previous["recalibrated_thresholds"][name][label]["threshold"]
            fixed[label][name] = {"threshold": cutoff, **_summarize(evaluate(cutoff))}
            selected = select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                              max_thresholds=max(3, len(pool) + 1))
            recalibrated[label][name] = {"threshold": selected["threshold"], "status": selected["status"],
                                        "threshold_candidates": selected["threshold_candidates"],
                                        **_summarize(selected["validation"])}

        fitted = fit_nested_grid(grid, lock, count)
        grouped[label] = {}
        for name, route in fitted["routes"].items():
            # Test folds use their own threshold, including for the predecessor rows.
            margins = route["scores"].copy()
            for fold in route["folds"]:
                rows = fitted["fold_assignment"] == fold["fold"]
                cutoff = fold["operating_point"].get("threshold")
                margins[rows] = margins[rows] >= cutoff if cutoff is not None else np.nan
                margins[rows & ~np.isfinite(route["scores"])] = np.nan
            evaluate, _ = _policy(grid, margins, set(grid.subjects), count)
            grouped[label][name] = {**_summarize(evaluate(0.5)), "outer_folds": len(route["folds"]),
                                    "inner_calibrations_at_target": sum(fold["operating_point"]["status"] == "target_met"
                                                                        for fold in route["folds"])}
    return {"stride_seconds": grid.seconds, "feature_window_seconds": WINDOW_SECONDS,
            "confirmation_windows": {str(rule): rules(rule) for rule in (1, 2)},
            "fixed_previous_cutoffs": fixed, "recalibrated_cutoffs": recalibrated,
            "nested_development_cv": grouped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-features", "history-raw", "history-features", "lock", "previous-confirmation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stride-raw", type=Path)
    parser.add_argument("--stride-features", type=Path)
    args = parser.parse_args()
    if (args.stride_raw is None) != (args.stride_features is None):
        parser.error("Supply both stride archives and stride features")
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        previous = json.loads(args.previous_confirmation.read_text(encoding="utf-8"))
        digest = sha256(lock_bytes).hexdigest()
        table, subjects, timelines = load_features(args.development_features)
        development = set(lock["fit_subjects"]) | set(lock["calibration_subjects"])
        if (previous["original_lock_sha256"] != digest or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or set(subjects) != development
                or set(subjects) & set(lock["test_roster"])):
            raise ValueError("Use the frozen models, cutoffs, and participants from the original development study")
        history = load_grid(table, subjects, timelines, args.history_raw, args.history_features, seconds=WINDOW_SECONDS)
        report = {"scope": "Development only; same evaluation exposure with causal alert history",
                  "original_lock_sha256": digest, "development_participants": len(np.unique(subjects)),
                  "warm_30s": compare_grid(history, lock, previous)}
        if args.stride_raw is not None:
            finer = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=10)
            report["warm_10s"] = compare_grid(finer, lock, previous)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF clock study failed: {error}\n")
    print(f"Evaluated {report['development_participants']} development participants with causal alert history")


if __name__ == "__main__":
    main()
