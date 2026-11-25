"""Audit out-of-fold episode misses and false alarms for every alert configuration."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_events import _event_at_decision, _grid, _intervals, _union
from .af_wrist_clock_study import fit_nested_grid, load_grid, predecessor_decisions, scoring_runs
from .af_wrist_failure_events import audit_events
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_features import FEATURE_VERSION_KEY, ROUTES, WINDOW_SECONDS
from .af_wrist_history import apply_alert_history
from .af_wrist_holdout import _shared_features
from .af_wrist_train import load_features


RECENT_SECONDS = 30


def _median(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def _support(episode, evidence, shared_inputs, stride):
    statuses, pure_af = [], False
    for start, row in episode["decision_refs"]:
        key = (episode["subject_id"], start)
        status = evidence[key]["status"][row]
        statuses.append(str(status))
        endpoint = start + (row + 1) * stride
        if endpoint - WINDOW_SECONDS >= episode["start_s"] and shared_inputs[key][row]:
            pure_af = True
    episode["input_status_counts"] = dict(sorted(Counter(statuses).items()))
    episode["pure_af_input_at_decision"] = pure_af


def _summarize(episodes, false_starts, monitoring, evidence):
    missed = [episode for episode in episodes if not episode["detected"]]
    negatives = []
    subject_intervals = defaultdict(list)
    for run in monitoring:
        subject_intervals[run.subject_id].append(_intervals(run.af_intervals_s))
    subject_intervals = {subject: _union(np.concatenate(intervals))
                         for subject, intervals in subject_intervals.items()}
    for run in monitoring:
        edges, decisions, reference, _ = _grid(run)
        non_af = _event_at_decision(edges[1:], subject_intervals[run.subject_id]) < 0
        quality = evidence[(run.subject_id, run.start_s)]["quality"]
        negatives.extend(quality[reference & non_af & (decisions >= 0)])

    persists, short, censored = 0, 0, 0
    resets = Counter()
    for event in false_starts:
        if event["active_duration_s"] >= RECENT_SECONDS:
            persists += 1
        elif event["right_censored"]:
            censored += 1
        else:
            short += 1
        age = event["seconds_since_previous_reset"]
        if age is not None and age <= RECENT_SECONDS:
            resets[event["previous_reset_kind"]] += 1
        event["input_quality"] = float(evidence[(event["subject_id"], event["run_start_s"])]["quality"][event["row_index"]])
    return {
        "complete_episodes": len(episodes),
        "detected": len(episodes) - len(missed),
        "missed_reasons": dict(sorted(Counter(e["primary"] for e in missed).items())),
        "misses_with_confirmed_state_during_af": sum(e["active_count"] > 0 for e in missed),
        "misses_with_positive_raw_decision": sum(e["positive_count"] > 0 for e in missed),
        "misses_with_all_trailing_inputs_flat": sum(
            set(e["input_status_counts"]) == {"flatline"} for e in missed
        ),
        "median_best_score_minus_cutoff_for_below_cutoff_misses": _median(
            e["max_margin"] for e in missed if e["primary"] == "below_cutoff"
        ),
        "false_alerts": {
            "starts": len(false_starts),
            "one_positive_decision_followed_by_reset": sum(
                e["positive_streak_decisions"] == 1 and not e["right_censored"] for e in false_starts
            ),
            "duration_below_30_s": short,
            "duration_at_least_30_s": persists,
            "duration_censored_before_30_s": censored,
            "with_reference_af_in_trailing_input": sum(e["af_in_trailing_input_s"] > 1e-6 for e in false_starts),
            "within_30_s_before_annotated_af": sum(
                e["next_af_onset_in_s"] is not None and e["next_af_onset_in_s"] <= RECENT_SECONDS
                for e in false_starts
            ),
            "restart_within_30_s_after_negative": resets["negative"],
            "restart_within_30_s_after_abstention": resets["unavailable"],
            "median_score_minus_cutoff": _median(e["alert_margin"] for e in false_starts),
            "median_ppg_quality_at_false_start": _median(e["input_quality"] for e in false_starts),
            "median_ppg_quality_at_scored_non_af_monitor_decisions": _median(negatives),
        },
    }


def audit_route(grid, confirmation, scores, cutoffs, route_decisions, evidence, expected):
    margins = scores - cutoffs
    decisions = np.where(np.isfinite(margins), margins >= 0, -1).astype(np.int8)
    if not np.array_equal(decisions, route_decisions):
        raise ValueError("Replayed predictions must match the fitted outer-fold decisions")
    binary = np.where(decisions < 0, np.nan, decisions)
    monitor, episodes = scoring_runs(grid, binary, set(grid.subjects))
    history = predecessor_decisions(grid, decisions)
    margin_runs = {(block.subject, block.start): margins[block.rows] for block in grid.timelines}

    def trace(runs):
        raw = tuple(run.decisions(0.5) for run in runs)
        confirmed, _ = apply_alert_history(raw, confirmation_windows=confirmation, prior_decisions=history)
        events = audit_events(raw, confirmed, confirmation_windows=confirmation, prior_decisions=history,
                             margins=margin_runs, feature_window_s=WINDOW_SECONDS)
        return events, raw

    clinical, _ = trace(episodes)
    monitoring, monitoring_raw = trace(monitor)
    complete = clinical["complete_episodes"]
    shared = _shared_features(grid.table)
    shared_inputs = {(block.subject, block.start): shared[block.rows] for block in grid.timelines}
    for episode in complete:
        _support(episode, evidence, shared_inputs, grid.seconds)
    false_starts = monitoring["false_alerts"]
    summary = _summarize(complete, false_starts, monitoring_raw, evidence)
    summary["partial_episodes"] = clinical["partial_episodes"]
    delays = _median(e["first_alert_delay_s"] for e in complete if e["detected"])
    if (summary["complete_episodes"], summary["detected"], summary["false_alerts"]["starts"], delays) != (
        expected["complete_episodes"], expected["episodes_detected"], expected["wrist_false_alerts"],
        expected["median_delay_seconds_in_detected_episodes"],
    ):
        raise ValueError("Event attributions disagree with the audited clock study counts or delays")
    return summary, {"complete_episodes": complete, "false_alerts": false_starts}


def fold_arrays(fitted, route):
    cutoffs = np.full(len(fitted["fold_assignment"]), np.nan)
    for fold in route["folds"]:
        cutoff = fold["operating_point"]["threshold"]
        if cutoff is not None:
            cutoffs[fitted["fold_assignment"] == fold["fold"]] = cutoff
    return route["scores"], cutoffs, route["decisions"]


def _paired(cases):
    identities, primary = defaultdict(list), Counter()
    per_route, per_stride = defaultdict(set), defaultdict(dict)
    false_starts = 0
    for stride, by_rules in cases.items():
        for rule, routes in by_rules.items():
            for name, cases_for_route in routes.items():
                false_starts += len(cases_for_route["false_alerts"])
                for episode in cases_for_route["complete_episodes"]:
                    key = (episode["subject_id"], episode["start_s"], episode["end_s"])
                    identities[key].append(episode)
                    per_stride[stride][key] = episode["pure_af_input_at_decision"]
                    if episode["detected"]:
                        per_route[name].add(key)
                    else:
                        primary[episode["primary"]] += 1
    settings = sum(len(routes) for by_rules in cases.values() for routes in by_rules.values())
    if any(len(group) != settings for group in identities.values()):
        raise ValueError("A real episode must be represented once in every configuration")
    outcomes = Counter(sum(e["detected"] for e in group) for group in identities.values())
    never = [group for group in identities.values() if not any(e["detected"] for e in group)]
    return {
        "configurations": settings,
        "unique_complete_episodes": len(identities),
        "episode_configuration_observations": len(identities) * settings,
        "false_alert_configuration_observations": false_starts,
        "detected_at_least_once_by_route": {name: len(per_route[name]) for name in ROUTES},
        "episodes_by_number_of_configurations_that_detected_them": dict(sorted(outcomes.items())),
        "never_detected_episode_reason_observations": dict(sorted(Counter(e["primary"] for group in never for e in group).items())),
        "episodes_with_all_trailing_decision_inputs_flat": sum(
            all(set(e["input_status_counts"]) == {"flatline"} for e in group) for group in identities.values()
        ),
        "episodes_shorter_than_30_s": sum(end - start < WINDOW_SECONDS for _, start, end in identities),
        "episodes_with_a_scoreable_wholly_af_input_by_cadence": {
            stride: sum(flags.values()) for stride, flags in per_stride.items()
        },
        "missed_episode_configuration_observations_by_reason": dict(sorted(primary.items())),
    }


def audit_grids(grids, lock, previous, *, folded=None):
    results, private = {}, {}
    for grid, evidence in grids:
        stride = str(grid.seconds)
        prior = previous[f"warm_{stride}s"]["nested_development_cv"]
        results[stride], private[stride] = {}, {}
        for key, confirmation in (("1", 1), ("2", WINDOW_SECONDS // grid.seconds + 1)):
            label = "immediate" if key == "1" else "confirm_30_s"
            if folded is None:
                fitted = fit_nested_grid(grid, lock, confirmation)
                arrays = {name: fold_arrays(fitted, route) for name, route in fitted["routes"].items()}
            else:
                arrays = folded(grid.seconds, confirmation)
            results[stride][label], private[stride][label] = {}, {}
            for name, values in arrays.items():
                summary, cases = audit_route(grid, confirmation, *values, evidence, prior[key][name])
                results[stride][label][name], private[stride][label][name] = summary, cases
            print(f"Audited all three routes at {stride}s cadence, {label}", flush=True)
    return {"scope": "Nested development out-of-fold event audit", "configurations": results,
            "paired_analysis": _paired(private)}, private


def load_study(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    previous = json.loads(args.clock_results.read_text(encoding="utf-8"))
    table, subjects, timelines = load_features(args.development_features)
    fit, calibration, test = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & test or set(subjects) != (fit | calibration)
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or previous["original_lock_sha256"] != sha256(lock_bytes).hexdigest()):
        raise ValueError("Audit the original locked development cohort and clock study")
    grids = []
    for stride, grid_raw, grid_features, raw, features in (
        (30, args.history_raw, args.history_features, args.development_raw, args.development_features),
        (10, args.stride_raw, args.stride_features, args.stride_raw, args.stride_features),
    ):
        grid = load_grid(table, subjects, timelines, grid_raw, grid_features, seconds=stride)
        grids.append((grid, load_input_evidence(grid, raw, features)))
    return grids, lock, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "history-raw", "history-features",
                 "stride-raw", "stride-features", "lock", "clock-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report, _ = audit_grids(*load_study(args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF event audit failed: {error}\n")


if __name__ == "__main__":
    main()
