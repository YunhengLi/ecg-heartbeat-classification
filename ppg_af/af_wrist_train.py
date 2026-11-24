"""Train and compare AF classifiers with nested participant-held-out folds."""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .af_events import AFRun
from .af_wrist_cv import train_oof
from .af_wrist_features import ROUTES, WINDOW_SECONDS
from .af_wrist_operating import AFScoredRun, compare_operating_points, score_decisions, select_operating_point


@dataclass(frozen=True)
class Timeline:
    subject: str
    kind: str
    rows: np.ndarray
    start: float
    episodes: np.ndarray


def load_features(directory):
    fields = ("stable_label", "reference_valid", "ppg_available", "run_kind", *ROUTES)
    columns = {name: [] for name in fields}
    subjects, timelines = [], []
    seen = set()
    offset = 0
    for path in sorted(directory.glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path, allow_pickle=False) as source:
            subject = str(source["subject_id"].item())
            if subject in seen:
                raise ValueError(f"A participant has more than one feature archive in {directory}")
            seen.add(subject)
            start = source["start_s"]
            run_ids = source["run_id"]
            kinds = source["run_kind"]
            full_episodes = source["af_intervals_s"]
            for run_id in np.unique(run_ids):
                selected = np.flatnonzero(run_ids == run_id)
                rows = selected[np.argsort(start[selected])]
                if len(np.unique(kinds[rows])) != 1 or not np.allclose(np.diff(start[rows]), WINDOW_SECONDS):
                    raise ValueError(f"Each timeline run must be contiguous and have one kind in {path.name}")
                lo = start[rows[0]]
                hi = start[rows[-1]] + WINDOW_SECONDS
                episodes = full_episodes[(full_episodes[:, 0] < hi) & (full_episodes[:, 1] > lo)]
                timelines.append(Timeline(subject, str(kinds[rows[0]]), rows + offset, float(lo), episodes))
            for name in fields:
                columns[name].append(source[name])
            subjects.extend([subject] * len(start))
            offset += len(start)
    if not subjects:
        raise ValueError("The feature directory contains no complete participant files")
    table = {name: np.concatenate(values) for name, values in columns.items()}
    return table, np.asarray(subjects), timelines


def _timeline_runs(timelines, subjects, predictions, table, *, scored, monitor_kind):
    scopes = {"monitor": [], "monitor_wear": [], "enriched_episode": []}
    factory = AFScoredRun if scored else AFRun
    for timeline in timelines:
        if timeline.subject not in subjects or timeline.kind not in scopes:
            continue
        rows = timeline.rows
        scopes[timeline.kind].append(factory(
            timeline.start, WINDOW_SECONDS, timeline.episodes, predictions[rows],
            reference_valid=table["reference_valid"][rows], ppg_available=table["ppg_available"][rows],
            subject_id=timeline.subject,
        ))
    return scopes[monitor_kind], scopes["monitor"] + scopes["monitor_wear"] + scopes["enriched_episode"]


def _stable_operating_point(labels, scores, target):
    eligible = np.isfinite(scores) & (labels >= 0)
    positive = np.sort(scores[eligible & (labels == 1)])
    negative = scores[eligible & (labels == 0)]
    if not len(positive) or not len(negative):
        return {"threshold": None, "af_windows": len(positive), "non_af_windows": len(negative)}
    threshold = float(positive[-int(np.ceil(target * len(positive)))])
    return {"threshold": threshold, "af_windows": len(positive), "non_af_windows": len(negative),
            "sensitivity": float(np.mean(positive >= threshold)),
            "false_positive_rate": float(np.mean(negative >= threshold))}


def _score_window_comparison(route, assignments, labels):
    scores = route["scores"]
    valid = np.isfinite(scores) & (labels >= 0)
    result = {"scored_af_windows": int(np.sum(valid & (labels == 1))),
              "scored_non_af_windows": int(np.sum(valid & (labels == 0))),
              "auroc": None, "average_precision": None}
    if np.unique(labels[valid]).size == 2:
        result["auroc"] = float(roc_auc_score(labels[valid], scores[valid]))
        result["average_precision"] = float(average_precision_score(labels[valid], scores[valid]))
    decisions = np.full(len(labels), -1, dtype=np.int8)
    missing = 0
    for history in route["folds"]:
        threshold = history["operating_point"].get("window_calibration", {}).get("threshold")
        if threshold is None:
            missing += 1
            continue
        in_fold = np.isfinite(scores) & (assignments == history["fold"])
        decisions[in_fold] = scores[in_fold] >= threshold
    classified, positive = decisions >= 0, decisions == 1
    af, non_af = classified & (labels == 1), classified & (labels == 0)
    result.update(
        folds_without_calibrated_window_threshold=missing,
        calibrated_af_windows=int(af.sum()), calibrated_non_af_windows=int(non_af.sum()),
        true_positive_windows=int(np.sum(af & positive)), false_positive_windows=int(np.sum(non_af & positive)),
        sensitivity=float(np.mean(positive[af])) if af.any() else None,
        false_positive_rate=float(np.mean(positive[non_af])) if non_af.any() else None,
    )
    return result, decisions


def _compare_window_points(points, *, target, tolerance):
    sensitivities = [row["sensitivity"] for row in points.values()]
    supports = {(row["calibrated_af_windows"], row["calibrated_non_af_windows"]) for row in points.values()}
    result = {"status": "incomplete_calibration", "lowest_false_positive_rate_routes": []}
    if any(row["folds_without_calibrated_window_threshold"] for row in points.values()):
        return result
    if len(supports) != 1 or any(row["sensitivity"] is None or row["false_positive_rate"] is None for row in points.values()):
        return {**result, "status": "insufficient_shared_window_support"}
    if max(sensitivities) - min(sensitivities) > tolerance + 1e-12:
        return {**result, "status": "held_out_window_sensitivities_differ"}
    if min(sensitivities) + tolerance + 1e-12 < target:
        return {**result, "status": "held_out_window_sensitivity_below_target"}
    lowest = min(row["false_positive_rate"] for row in points.values())
    return {
        "status": "comparable_held_out_window_sensitivity",
        "lowest_false_positive_rate_routes": [name for name, row in points.items() if row["false_positive_rate"] == lowest],
    }


def evaluate(table, subjects, timelines, *, folds, inner_folds, seed, target, tolerance, monitor_kind="monitor"):
    labels = table["stable_label"]
    availability = {name: np.isfinite(table[name]).all(axis=1) for name in ROUTES}
    shared = np.logical_and.reduce(list(availability.values()))
    features = {name: np.where(shared[:, None], table[name], np.nan) for name in ROUTES}
    train_eligible = shared & (table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        on_original_grid = np.full(len(labels), np.nan)
        on_original_grid[rows] = scores
        monitor, episodes = _timeline_runs(timelines, set(subjects[rows]), on_original_grid, table,
                                           scored=True, monitor_kind=monitor_kind)
        return {**select_operating_point(monitor, episode_runs=episodes, target_sensitivity=target),
                "window_calibration": _stable_operating_point(labels[rows], scores, target)}

    training = train_oof(features, labels, subjects, n_splits=folds, inner_splits=inner_folds,
                         seed=seed, train_eligible=train_eligible, calibration_selector=calibrate)
    test_metrics, window_metrics, window_calibrated_alerts, calibration = {}, {}, {}, {}
    for name, route in training["routes"].items():
        monitor, episodes = _timeline_runs(timelines, set(subjects), route["decisions"], table,
                                           scored=False, monitor_kind=monitor_kind)
        test_metrics[name] = score_decisions(monitor, episode_runs=episodes)
        window_metrics[name], decisions = _score_window_comparison(route, training["fold_assignment"], labels)
        window_monitor, _ = _timeline_runs(timelines, set(subjects), decisions, table,
                                            scored=False, monitor_kind=monitor_kind)
        window_calibrated_alerts[name] = score_decisions(window_monitor)["burden_metrics"]
        calibration[name] = route["folds"]
    target_met = {name: all(fold["operating_point"]["status"] == "target_met" for fold in history)
                  for name, history in calibration.items()}

    def compare(names):
        return compare_operating_points({name: test_metrics[name] for name in names}, target_sensitivity=target,
                                        sensitivity_tolerance=tolerance,
                                        validation_target_met={name: target_met[name] for name in names})

    monitor_rows = table["run_kind"] == monitor_kind
    return {
        "dataset": "AFPPGECG v6 (10.5281/zenodo.23921320)",
        "participants": len(np.unique(subjects)),
        "participants_with_eligible_af": len(np.unique(subjects[train_eligible & (labels == 1)])),
        "participants_with_eligible_non_af": len(np.unique(subjects[train_eligible & (labels == 0)])),
        "windows_by_kind": {name: int(np.sum(table["run_kind"] == name)) for name in np.unique(table["run_kind"])},
        "monitor_kind": monitor_kind,
        "monitoring_blocks": sum(timeline.kind == monitor_kind for timeline in timelines),
        "calendar_sample_windows": int(np.sum(table["run_kind"] == "monitor")),
        "calendar_sample_ppg_available_windows": int(np.sum((table["run_kind"] == "monitor") & table["ppg_available"])),
        "monitoring_reference_seconds": int(np.sum(monitor_rows & table["reference_valid"])) * WINDOW_SECONDS,
        "raw_monitoring_feature_windows_by_route": {name: int(np.sum(monitor_rows & values))
                                                    for name, values in availability.items()},
        "shared_monitoring_feature_windows": int(np.sum(monitor_rows & shared)),
        "outer_folds": folds, "max_inner_folds": inner_folds, "seed": seed,
        "operating_point": compare(ROUTES),
        "pairwise_operating_points": {f"{left}_vs_{right}": compare((left, right))
                                      for left, right in combinations(ROUTES, 2)},
        "test_alert_metrics": test_metrics,
        "test_stable_window_metrics": window_metrics,
        "stable_window_operating_point": _compare_window_points(window_metrics, target=target, tolerance=tolerance),
        "pairwise_stable_window_operating_points": {
            f"{left}_vs_{right}": _compare_window_points({name: window_metrics[name] for name in (left, right)},
                                                          target=target, tolerance=tolerance)
            for left, right in combinations(ROUTES, 2)
        },
        "test_alert_burden_at_window_thresholds": window_calibrated_alerts,
        "fold_calibration": calibration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-sensitivity", type=float, default=0.8)
    parser.add_argument("--sensitivity-tolerance", type=float, default=0.1)
    parser.add_argument("--monitor-kind", choices=("monitor", "monitor_wear"), default="monitor")
    args = parser.parse_args()
    if not 0 < args.target_sensitivity <= 1 or not 0 <= args.sensitivity_tolerance <= 1:
        parser.error("Sensitivity target must be in (0, 1] and tolerance in [0, 1]")
    try:
        table, subjects, timelines = load_features(args.features)
        result = evaluate(table, subjects, timelines, folds=args.folds, inner_folds=args.inner_folds,
                          seed=args.seed, target=args.target_sensitivity, tolerance=args.sensitivity_tolerance,
                          monitor_kind=args.monitor_kind)
    except ValueError as error:
        parser.exit(1, f"AF evaluation failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluated {result['participants']} participants over {result['outer_folds']} folds")
    print(f"Operating point: {result['operating_point']['status']}")


if __name__ == "__main__":
    main()
