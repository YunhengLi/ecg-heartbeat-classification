"""Align archived participant-held-out template scores to AF onset."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_events import _union
from .af_wrist_clock_study import load_grid
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import audit_route, fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features
from .af_wrist_train import load_features
from .features import ECG_FEATURE_NAMES


EPS = 1e-6


def overlap(intervals, left, right):
    return float(np.maximum(0, np.minimum(right, intervals[:, 1]) - np.maximum(left, intervals[:, 0])).sum())


def observed_af(episodes, coverage):
    pairs = [(max(a, c), min(b, d)) for a, b in episodes for c, d in coverage if max(a, c) < min(b, d)]
    return _union(np.asarray(pairs, dtype=float).reshape((-1, 2)))


def finite(value):
    return float(value) if np.isfinite(value) else None


def _referenced_timeline(grid, scores, cutoffs, raw_dir):
    by_subject = defaultdict(dict)
    for block in grid.timelines:
        for index, row in enumerate(block.rows):
            if not grid.table["reference_valid"][row]:
                continue
            end = float(block.start + (index + 1) * grid.seconds)
            timeline = by_subject[block.subject]
            if end in timeline:
                raise ValueError("Expected one referenced decision per participant and timestamp")
            timeline[end] = {"row": int(row), "margin": finite(scores[row] - cutoffs[row])}
    for subject, timeline in by_subject.items():
        with np.load(raw_dir / f"{subject}.npz", allow_pickle=False) as raw:
            intervals = _union(raw["af_intervals_s"])
        covered = _union(np.asarray([(end - grid.seconds, end) for end in timeline], dtype=float))
        verified_af = observed_af(intervals, covered)
        for end, decision in timeline.items():
            decision.update(
                trailing_observed_af_seconds=overlap(verified_af, end - WINDOW_SECONDS, end),
                trailing_reference_fully_observed=overlap(covered, end - WINDOW_SECONDS, end) >= WINDOW_SECONDS - EPS,
            )
    return by_subject


def episodes_analysis(cases, timelines):
    results = []
    for event in cases["complete_episodes"]:
        subject, onset, end = (event[key] for key in ("subject_id", "start_s", "end_s"))
        rows = []
        for when, row in sorted(timelines[subject].items()):
            if onset - WINDOW_SECONDS - EPS <= when <= end:
                seconds_target_af = max(0.0, min(when, end) - max(when - WINDOW_SECONDS, onset))
                rows.append({**row, "since_onset_s": when - onset, "seconds_target_af": seconds_target_af,
                             "phase": "pre" if when <= onset else "af"})
        in_episode = [d for d in rows if d["phase"] == "af"]
        if len(in_episode) != event["decision_count"]:
            raise ValueError("An onset trajectory lost part of the verified episode")
        scorable = [d for d in in_episode if d["margin"] is not None]
        maximum = max(scorable, key=lambda d: d["margin"], default=None)
        pure = [d for d in scorable if d["seconds_target_af"] >= WINDOW_SECONDS - EPS]
        pre_ref = [d for d in rows if d["phase"] == "pre" and d["margin"] is not None]
        pre_verified_non_af = [d for d in pre_ref if d["trailing_reference_fully_observed"] and
                               d["trailing_observed_af_seconds"] <= EPS]
        best_pre_ref = max(pre_ref, key=lambda d: d["margin"], default=None)
        best_pre_non_af = max(pre_verified_non_af, key=lambda d: d["margin"], default=None)
        first_alert = next((d for d in in_episode
                            if event["detected"] and abs(d["since_onset_s"] - event["first_alert_delay_s"]) <= EPS),
                           None)
        if event["detected"] and first_alert is None:
            raise ValueError("An attributed alert does not align to the verified episode")
        results.append({
            "detected": event["detected"], "active_at_onset": event["active_at_onset"],
            "scored_in_episode": len(scorable), "scoreable_pure_30_s": len(pure), "best_in_episode": maximum,
            "best_pre_reference": best_pre_ref, "best_pre_full_non_af": best_pre_non_af,
            "first_alert": first_alert, "trajectory": rows,
        })
    return results


def median(values):
    return float(np.median(values)) if values else None


def trajectory_bins(group):
    bins = []
    cadence = 10
    for left in range(-30, 60, cadence):
        right = left + cadence
        scores = [
            row for episode in group for row in episode["trajectory"]
            if left < row["since_onset_s"] <= right and row["margin"] is not None
        ]
        bins.append({
            "onset_offset_left_s_exclusive": left, "onset_offset_right_s_inclusive": right,
            "episodes_with_scored_reference_endpoint": len(scores),
            "median_margin": median([row["margin"] for row in scores]),
            "median_target_af_seconds_in_30_s_input": median([row["seconds_target_af"] for row in scores]),
            "fully_observed_trailing_reference_endpoints": sum(row["trailing_reference_fully_observed"] for row in scores),
        })
    return bins


def summarize(results, false_starts):
    groups = {"detected": [x for x in results if x["detected"]],
              "scorable_misses": [x for x in results if not x["detected"] and x["scored_in_episode"]],
              "unscorable_misses": [x for x in results if not x["scored_in_episode"]]}
    summary = {}
    for key, group in groups.items():
        best = [x["best_in_episode"] for x in group if x["best_in_episode"] is not None]
        same_case_reference = [x for x in group if x["best_pre_reference"] and x["best_in_episode"]]
        same_case_non_af = [x for x in group if x["best_pre_full_non_af"] and x["best_in_episode"]]
        alerts = [x["first_alert"] for x in group if x["first_alert"] is not None]
        summary[key] = {
            "episodes": len(group),
            "scoreable_pure_30_s_episodes": sum(bool(x["scoreable_pure_30_s"]) for x in group),
            "best_af_score_in_pure_30_s_episodes": sum(x["seconds_target_af"] >= WINDOW_SECONDS - EPS for x in best),
            "active_at_onset": sum(x["active_at_onset"] is True for x in group),
            "comparable_pre_onset_reference_score": len(same_case_reference),
            "pre_onset_reference_peak_at_least_af_peak": sum(
                x["best_pre_reference"]["margin"] >= x["best_in_episode"]["margin"] for x in same_case_reference
            ),
            "comparable_pre_onset_full_non_af_score": len(same_case_non_af),
            "pre_onset_full_non_af_peak_at_least_af_peak": sum(
                x["best_pre_full_non_af"]["margin"] >= x["best_in_episode"]["margin"] for x in same_case_non_af
            ),
            "median_best_in_episode_margin": median([x["margin"] for x in best]),
            "median_af_seconds_at_best_in_episode": median([x["seconds_target_af"] for x in best]),
            "median_af_seconds_at_first_alert": median([x["seconds_target_af"] for x in alerts]),
            "first_alert_before_30_s_in_af": sum(x["seconds_target_af"] < WINDOW_SECONDS - EPS for x in alerts),
            "median_seconds_from_onset_to_first_alert": median([x["since_onset_s"] for x in alerts]),
            "trajectory_bins": trajectory_bins(group),
        }
    summary["false_starts"] = {
        "count": len(false_starts),
        "with_fully_observed_trailing_reference": sum(x["trailing_reference_fully_observed"] for x in false_starts),
        "with_observed_af_in_trailing_reference": sum(x["trailing_observed_af_seconds"] > EPS for x in false_starts),
        "median_score_minus_cutoff": median([x["margin"] for x in false_starts]),
    }
    return summary


def _fold_coefficients(grid, scores, fold_assignment):
    shared = _shared_features(grid.table)
    labels = grid.table["stable_label"]
    matrix = grid.table["pulse_template"]
    fit = shared & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    if fold_assignment.shape != scores.shape or scores.shape != labels.shape:
        raise ValueError("The fold assignments and scores must align with the template feature rows")
    coefficients = {}
    reconstructed = np.full(len(scores), np.nan)
    for fold in np.unique(fold_assignment):
        selected = fit & (fold_assignment != fold)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=1000))
        classifier.fit(matrix[selected], labels[selected])
        held_out = shared & (fold_assignment == fold)
        if np.any(held_out):
            reconstructed[held_out] = classifier.predict_proba(matrix[held_out])[:, 1]
        scaler = classifier.named_steps["standardscaler"]
        logistic = classifier.named_steps["logisticregression"]
        coefficients[fold] = logistic.coef_[0] / scaler.scale_
    if not np.allclose(reconstructed, scores, rtol=1e-12, atol=1e-14, equal_nan=True):
        raise ValueError("Rebuilt baseline fold models do not reproduce the archived held-out template scores")
    return matrix, coefficients


def _paired_logit_contributions(grid, scores, fold_assignment, episodes):
    matrix, coefficients = _fold_coefficients(grid, scores, fold_assignment)
    delta, lower_count, largest_decrease = [], np.zeros(len(ECG_FEATURE_NAMES), dtype=int), Counter()
    for episode in episodes:
        pre = episode["best_pre_full_non_af"]
        af = episode["best_in_episode"]
        if episode["detected"] or pre is None or af is None:
            continue
        pre_row, af_row = pre["row"], af["row"]
        fold = fold_assignment[af_row]
        if fold != fold_assignment[pre_row]:
            raise ValueError("A within-participant attribution must use a single held-out fold")
        values = (matrix[af_row] - matrix[pre_row]) * coefficients[fold]
        lower_count += values < 0
        largest_decrease[ECG_FEATURE_NAMES[np.argmin(values)]] += 1
        delta.append(values)
    return {
        "eligible_scored_misses": len(delta),
        "baseline_fold_scores_reproduced": True,
        "median_within_af_minus_pre_onset_logit": median([float(x.sum()) for x in delta]),
        "median_contribution_change_by_feature": {
            name: median([float(x[index]) for x in delta]) for index, name in enumerate(ECG_FEATURE_NAMES)
        },
        "negative_contribution_change_counts": dict(zip(ECG_FEATURE_NAMES, map(int, lower_count), strict=True)),
        "most_negative_contribution_counts": dict(sorted(largest_decrease.items())),
    }


def analyze_template_onset(grid, scores, cutoffs, decisions, fold_assignment, raw_dir, feature_dir, expected):
    """Return aggregate onset diagnostics for the existing 10-second template baseline.

    Baseline scores and folds come from the balanced, scaled logistic model
    with C=1. Attribution compares each missed episode's peak in AF with its
    strongest scored and fully verified non-AF window up to 30 seconds before
    onset, using the existing fold model. No participant-level rows are returned.
    """
    if grid.seconds != 10:
        raise ValueError("Template onset trajectories require the locked 10-second clock")
    quality = load_input_evidence(grid, raw_dir, feature_dir)
    _, cases = audit_route(grid, 1, scores, cutoffs, decisions, quality, expected)
    by_subject = _referenced_timeline(grid, scores, cutoffs, raw_dir)
    outcomes = episodes_analysis(cases, by_subject)
    false_starts = [by_subject[event["subject_id"]][event["time_s"]] for event in cases["false_alerts"]]
    report = summarize(outcomes, false_starts)
    report["paired_logit_attribution"] = _paired_logit_contributions(grid, scores, fold_assignment, outcomes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-features", "stride-raw", "stride-features", "lock", "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from .af_wrist_template_study import fit_template

    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        table, subjects, timelines = load_features(args.development_features)
        fit, calibration, reserved = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(subjects) != (fit | calibration)
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
            raise ValueError("Use the locked development cohort and 10-second clock study")
        grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=10)
        expected = clock["warm_10s"]["nested_development_cv"]["1"]["pulse_template"]
        fitted = fit_template(grid, grid.table["pulse_template"], lock, 1.0)
        route = fitted["routes"]["pulse_template"]
        report = analyze_template_onset(grid, *fold_arrays(fitted, route), fitted["fold_assignment"],
                                        args.stride_raw, args.stride_features, expected)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Template onset diagnostic failed: {error}\n")
    print("Reproduced the 10-second template baseline and analyzed its onset trajectories")


if __name__ == "__main__":
    main()
