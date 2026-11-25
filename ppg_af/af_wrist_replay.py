"""Audit complete held-out episodes with the recorded thresholds and training split."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .af_events import evaluate_af
from .af_wrist_cv import train_oof
from .af_wrist_features import FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION, PPG_RATE, ROUTES, WINDOW_SECONDS
from .af_wrist_train import _timeline_runs, load_features


EPISODE_KINDS = {"monitor", "monitor_wear", "enriched_episode"}
TIME_TOLERANCE_S = 1e-6
EPISODE_CHECKS = ("complete_reference_episodes", "partial_reference_episodes",
                  "detected_complete_episodes", "complete_episodes_with_positive_decision",
                  "detected_partial_episodes", "true_alert_starts", "false_alert_starts")
BURDEN_CHECKS = ("windows", "ecg_reference_seconds", "analyzable_seconds", "true_alert_starts", "false_alert_starts")


def decision_mask(scores, thresholds):
    """Apply each subject's original fold threshold, leaving unscored rows at -1."""
    present = np.isfinite(scores) & np.isfinite(thresholds)
    return np.where(present, scores >= thresholds, -1).astype(np.int8)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if merged and start <= merged[-1][1] + TIME_TOLERANCE_S:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def verify_counts(table, timelines, subjects, decisions, report):
    for route in ROUTES:
        monitoring, all_runs = _timeline_runs(timelines, subjects, decisions[route], table,
                                             scored=False, monitor_kind=report["monitor_kind"])
        for runs, section, fields in ((all_runs, "episode_metrics", EPISODE_CHECKS),
                                      (monitoring, "burden_metrics", BURDEN_CHECKS)):
            actual = evaluate_af(runs)
            expected = report["test_alert_metrics"][route][section]
            for field in fields:
                if actual[field] != expected[field]:
                    raise ValueError(f"Replay mismatch for {route} {section} {field}")


def alert_starts(timelines, decisions):
    result = np.zeros(len(decisions), dtype=bool)
    previous_end, previously_positive = -np.inf, False
    for timeline in timelines:
        same_run = abs(timeline.start - previous_end) <= TIME_TOLERANCE_S
        rows = timeline.rows
        positive = decisions[rows] == 1
        result[rows] = positive & ~np.r_[same_run and previously_positive, positive[:-1]]
        previous_end = timeline.start + WINDOW_SECONDS * len(rows)
        previously_positive = bool(positive[-1])
    return result


def summarize_route(route, rows, table, availability, shared, thresholds, scores, decisions, starts):
    scored = scores[route][rows]
    finite = np.isfinite(scored)
    applied = thresholds[route][rows]
    ppg = table["ppg_available"][rows]
    positive = decisions[route][rows] == 1
    new_alert = starts[route][rows]
    if new_alert.any():
        status = "new_alert_during_episode"
    elif len(rows) == 0:
        status = "no_ecg_valid_decision_endpoint"
    elif not ppg.any():
        status = "no_complete_raw_ppg_window"
    elif not availability[route][rows].any():
        status = "no_route_features"
    elif not shared[rows].any():
        status = "no_shared_feature_window"
    elif not finite.any():
        status = "no_held_out_score"
    elif not np.isfinite(applied).any():
        status = "missing_calibrated_threshold"
    elif np.all(applied > 1):
        status = "calibration_selected_no_alert_policy"
    elif positive.any():
        status = "positive_streak_started_before_episode"
    else:
        status = "scores_below_threshold"
    return {
        "status": status,
        "complete_ppg_windows": int(ppg.sum()),
        "route_feature_windows": int(availability[route][rows].sum()),
        "scored_windows": int(finite.sum()),
        "positive_windows": int(positive.sum()),
        "new_alerts": int(new_alert.sum()),
        "max_score": float(scored[finite].max()) if finite.any() else None,
        "calibrated_threshold": float(applied[0]) if len(applied) and np.isfinite(applied[0]) else None,
    }


def flat_raw_windows(features_dir, waveforms_dir, prefixes):
    constant = []
    for prefix in prefixes:
        with (np.load(waveforms_dir / f"{prefix}.npz", allow_pickle=False) as raw,
              np.load(features_dir / f"{prefix}.npz", allow_pickle=False) as features):
            for key in ("subject_id", "start_s", "run_id", "run_kind"):
                if not np.array_equal(raw[key], features[key]):
                    raise ValueError("Waveform archive and feature archive rows differ")
            if raw["ppg_fs"].item() != PPG_RATE:
                raise ValueError("Waveform archive has an unexpected PPG sampling frequency")
            ppg = raw["ppg"][:, PPG_RATE:-PPG_RATE]
            complete = np.isfinite(ppg).all(axis=1)
            if not np.array_equal(complete, features["ppg_available"]):
                raise ValueError("Raw waveform availability differs from recorded feature metadata")
            constant.append((np.ptp(ppg, axis=1) == 0) & complete)
    return np.concatenate(constant)


def replay(features_dir, report, waveforms_dir=None):
    expected_subjects = sorted(report["source_cohort"]["public_record_file_prefixes"])
    actual_subjects = sorted(f.stem for f in features_dir.glob("*.npz") if not f.name.endswith(".partial.npz"))
    if actual_subjects != expected_subjects:
        raise ValueError("Feature cohort must match the recorded prior participants exactly")
    table, subjects, timelines = load_features(features_dir)
    if table[FEATURE_VERSION_KEY] != report.get(FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION):
        raise ValueError("Feature version must match the recorded AF report")
    constant = None if waveforms_dir is None else flat_raw_windows(features_dir, waveforms_dir, expected_subjects)
    labels = table["stable_label"]
    availability = {route: np.isfinite(table[route]).all(axis=1) for route in ROUTES}
    shared = np.logical_and.reduce(list(availability.values()))
    model_features = {route: np.where(shared[:, None], table[route], np.nan) for route in ROUTES}
    train_eligible = shared & (table["run_kind"] != "enriched_episode")
    fitted = train_oof(model_features, labels, subjects, train_eligible=train_eligible,
                       n_splits=report["outer_folds"], inner_splits=report["max_inner_folds"], seed=report["seed"])
    assignments = fitted["fold_assignment"]
    thresholds, scores, decisions = {}, {}, {}
    for route in ROUTES:
        thresholds[route] = np.full(len(labels), np.nan)
        scores[route] = fitted["routes"][route]["scores"]
        for fold, reproduced in zip(report["fold_calibration"][route], fitted["routes"][route]["folds"], strict=True):
            for key in ("fold", "train_subjects", "test_subjects", "fit_af_windows", "fit_non_af_windows", "held_out_finite_scores"):
                if fold[key] != reproduced[key]:
                    raise ValueError(f"Fold replay mismatch for {route} {key}")
            threshold = fold["operating_point"]["threshold"]
            if threshold is not None:
                thresholds[route][assignments == fold["fold"]] = threshold
        decisions[route] = decision_mask(scores[route], thresholds[route])

    timelines = [t for t in timelines if t.kind in EPISODE_KINDS]
    verify_counts(table, timelines, set(subjects), decisions, report)
    by_subject = defaultdict(list)
    for timeline in timelines:
        by_subject[timeline.subject].append(timeline)

    episodes = []
    for subject, runs in sorted(by_subject.items()):
        runs = sorted(runs, key=lambda t: t.start)
        rows = np.concatenate([r.rows for r in runs])
        ends = np.concatenate([r.start + WINDOW_SECONDS * np.arange(1, len(r.rows) + 1) for r in runs])
        valid = table["reference_valid"][rows]
        coverage = merge_intervals(zip(ends[valid] - WINDOW_SECONDS, ends[valid], strict=True))
        events = merge_intervals(ep for run in runs for ep in run.episodes)
        starts = {route: alert_starts(runs, decisions[route]) for route in ROUTES}
        for onset, offset in events:
            if not any(start <= onset and offset <= end for start, end in coverage):
                continue
            # Decisions reflect ECG state just before the endpoint, as in the benchmark.
            eligible = rows[valid & (onset < ends) & (ends <= offset)]
            fold = int(assignments[rows[0]])
            episodes.append({
                "episode_index": len(episodes) + 1,
                "held_out_fold": fold,
                "ecg_valid_decision_endpoints": int(len(eligible)),
                "shared_feature_endpoints": int(shared[eligible].sum()),
                "flat_raw_ppg_endpoints": int(constant[eligible].sum()) if constant is not None else None,
                "routes": {route: summarize_route(route, eligible, table, availability, shared, thresholds, scores, decisions, starts)
                           for route in ROUTES},
            })

    for route in ROUTES:
        expected = report["test_alert_metrics"][route]["episode_metrics"]
        if len(episodes) != expected["complete_reference_episodes"]:
            raise ValueError("Independent reference episode count did not match")
        if sum(e["routes"][route]["new_alerts"] > 0 for e in episodes) != expected["detected_complete_episodes"]:
            raise ValueError("Independent detected episode count did not match")
        if sum(e["routes"][route]["positive_windows"] > 0 for e in episodes) != expected["complete_episodes_with_positive_decision"]:
            raise ValueError("Independent positive episode count did not match")
    return {
        "scope": "Retrospective diagnostic replay of the recorded participant-held-out pilot",
        "participants": int(np.unique(subjects).size),
        "complete_reference_episodes": len(episodes),
        "fold_thresholds": "Recorded prior thresholds, reapplied without retuning",
        "benchmark_counts_reproduced": True,
        "episode_failure_counts": {route: dict(Counter(e["routes"][route]["status"] for e in episodes)) for route in ROUTES},
        "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--waveforms", type=Path, help="Optional staged inputs for flatline diagnostics")
    ap.add_argument("--expected", type=Path, required=True, help="Prior grouped-results.json with frozen thresholds")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = replay(args.features, json.loads(args.expected.read_text(encoding="utf-8")), args.waveforms)
    except (OSError, ValueError, KeyError) as error:
        ap.exit(1, f"Episode replay failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Replayed {result['complete_reference_episodes']} complete episodes from {result['participants']} participants")


if __name__ == "__main__":
    main()
