"""Evaluate the single frozen template ECG model on audited independent recordings."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .af_events import evaluate_af
from .af_wrist_alert_study import _summary
from .af_wrist_clock_study import load_grid, scoring_runs
from .af_wrist_features import FEATURE_VERSION_KEY, ROUTES, WINDOW_SECONDS
from .af_wrist_holdout import MONITOR_KIND, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_operating import _NO_ALERT_THRESHOLD, _remaining_non_af_time, score_decisions
from .af_wrist_stride import STRIDE
from .af_wrist_template_coverage import _subject_support, _sum_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20261124
ROUTE = "pulse_template"


def _archives(directory):
    return {
        path.stem: path for path in directory.glob("*.npz")
        if not path.name.endswith(".partial.npz")
    }


def _locked_model(lock, protocol):
    if (lock["lock_version"] != 1 or lock["window_seconds"] != WINDOW_SECONDS
            or lock["decision_stride_seconds"] != STRIDE or lock["route"] != ROUTE
            or lock["monitor_kind"] != MONITOR_KIND
            or lock["alert_policy"] != "causal_immediate_with_predecessor_state"):
        raise ValueError("Use the frozen 10-second template ECG protocol")
    shared_fields = (FEATURE_VERSION_KEY, "window_seconds", "monitor_kind",
                     "target_sensitivity", "sensitivity_tolerance")
    if any(lock[field] != protocol[field] for field in shared_fields):
        raise ValueError("The model lock and original study protocol do not agree")
    groups = []
    for role in ("fit_subjects", "calibration_subjects", "test_roster"):
        people = _subject_ids(lock[role])
        if people != _subject_ids(protocol[role]):
            raise ValueError("The frozen participant groups differ from the source protocol")
        groups.append(people)
    fit, calibration, roster = groups
    if fit & calibration or (fit | calibration) & roster:
        raise ValueError("Independent participants must be disjoint from development")
    baseline, tuned = (lock["models"][key] for key in ("baseline", "tuned"))
    model_fields = ("logistic_C", "model", "cutoff")
    if any(baseline[field] != tuned[field] for field in model_fields):
        raise ValueError("This evaluation expects one unique frozen template model and cutoff")
    cutoff = baseline["cutoff"]
    if not math.isfinite(cutoff) or not 0 <= cutoff <= _NO_ALERT_THRESHOLD:
        raise ValueError("The frozen cutoff is outside the supported probability range")
    return baseline, roster, len(fit), len(calibration)


def _match_coverage(actual, expected):
    for key, value in actual.items():
        prior = expected[key]
        matched = (math.isclose(value, prior, rel_tol=0, abs_tol=1e-8)
                   if isinstance(value, float) else value == prior)
        if not matched:
            raise ValueError(f"The scoring grid changed audited coverage for {key}")


def _audited_stride(base_archives, stride_raw, stride_features, coverage, roster, version):
    found = set(base_archives)
    raw_archives, feature_archives = _archives(stride_raw), _archives(stride_features)
    if not found or not found <= roster or set(raw_archives) != found or set(feature_archives) != found:
        raise ValueError("The three archives must contain the same subset of the locked independent roster")
    _match_coverage({
        "prespecified_roster_size": len(roster),
        "decoded_roster_members": len(found),
        "roster_members_without_decoded_archives": len(roster - found),
        "feature_window_seconds": WINDOW_SECONDS,
    }, coverage)

    reports = []
    for member in sorted(found):
        with (np.load(base_archives[member], allow_pickle=False) as base,
              np.load(raw_archives[member], allow_pickle=False) as raw,
              np.load(feature_archives[member], allow_pickle=False) as features):
            if str(base["subject_id"].item()) != member or str(raw["subject_id"].item()) != member:
                raise ValueError("An archive filename does not match its participant")
            aligned = ("subject_id", "start_s", "decision_end_s", "input_start_s", "run_start_s", "run_id",
                       "run_kind", "evaluation", "reference_valid", "ppg_available", "input_status", "af_intervals_s")
            if any(not np.array_equal(raw[key], features[key]) for key in aligned):
                raise ValueError("Independent raw and feature archives must preserve the same timeline")
            if (str(base[FEATURE_VERSION_KEY].item()) != version
                    or str(features[FEATURE_VERSION_KEY].item()) != version
                    or not np.array_equal(base["af_intervals_s"], raw["af_intervals_s"])
                    or raw["decision_stride_seconds"].item() != STRIDE
                    or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                    or not np.array_equal(raw["start_s"] + STRIDE, raw["decision_end_s"])
                    or not np.array_equal(raw["decision_end_s"] - WINDOW_SECONDS, raw["input_start_s"])):
                raise ValueError("Independent feature version, ECG annotations, or causal window timing changed")
            statuses = raw["input_status"]
            if not np.isin(statuses, ("available", "unavailable", "unknown")).all():
                raise ValueError("Independent inputs must have a recognized availability status")
            available = (statuses == "available") & raw["ppg_available"]
            for name in ROUTES:
                matrix = features[name]
                if matrix.ndim != 2 or len(matrix) != len(available):
                    raise ValueError("Independent feature rows must align to the decision grid")
                if np.any(np.isfinite(matrix).all(1) & ~available):
                    raise ValueError("A shared-gate feature is present without its native PPG input")
            reports.append(_subject_support(member, raw, STRIDE))
    support = _sum_support(reports, STRIDE)
    _match_coverage(support, coverage["ten_second_grid"])
    if not support["complete_reference_episodes"] or not support["monitoring_observed_non_af_seconds"]:
        raise ValueError("Independent scoring needs complete reference episodes and observed non-AF wrist time")
    return support


def _finite_interval(values, valid):
    chosen = values[valid]
    if len(chosen) < 2:
        return None
    lower, upper = np.quantile(chosen, [0.025, 0.975]).tolist()
    return [lower, upper] if lower < upper else None


def _participant_uncertainty(reference, wrist, people):
    episode_by_person, wrist_by_person = defaultdict(list), defaultdict(list)
    for run in reference:
        episode_by_person[run.subject_id].append(run)
    for run in wrist:
        wrist_by_person[run.subject_id].append(run)
    contributions = []
    for member in sorted(people):
        episodes = evaluate_af(episode_by_person[member])
        runs = wrist_by_person.get(member, ())
        burden = evaluate_af(runs) if runs else {}
        contributions.append((episodes["detected_complete_episodes"], episodes["complete_reference_episodes"],
                              burden.get("false_alert_starts", 0), burden.get("ecg_reference_seconds", 0),
                              _remaining_non_af_time(runs, analyzable_only=True)))
    counts = np.asarray(contributions, dtype=float)
    episode_people = int(np.count_nonzero(counts[:, 1]))
    monitoring_people = int(np.count_nonzero(counts[:, 3]))
    with_episodes = counts[:, 1] > 0
    person_recall = counts[with_episodes, 0] / counts[with_episodes, 1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(people), size=(BOOTSTRAP_REPETITIONS, len(people)))
    detected, complete, false_alerts, exposure, analyzable_non_af = counts[draws].sum(1).T
    recall = np.divide(detected, complete, out=np.zeros_like(complete), where=complete > 0)
    false_rate = np.divide(3600 * false_alerts, exposure, out=np.zeros_like(exposure), where=exposure > 0)
    return {
        "method": "Participant-cluster percentile bootstrap, 95%, descriptive",
        "replicates": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "participants_with_complete_episodes": episode_people,
        "participants_with_ecg_observed_wrist_time": monitoring_people,
        "mean_episode_sensitivity_across_participants": float(person_recall.mean()) if episode_people else None,
        "maximum_complete_episodes_from_one_participant": int(counts[:, 1].max()),
        "episode_sensitivity_interval": _finite_interval(recall, complete > 0) if episode_people >= 2 else None,
        "false_alerts_per_observed_wrist_hour_interval": (
            _finite_interval(false_rate, (exposure > 0) & (analyzable_non_af > 0)) if monitoring_people >= 2 else None
        ),
    }


def load_independent_inputs(lock, protocol, independent_features, stride_raw, stride_features, coverage):
    """Load the immutable feature grid and verify its independent reference coverage."""
    model, roster, fit_count, calibration_count = _locked_model(lock, protocol)
    originals = _archives(independent_features)
    coverage_support = _audited_stride(originals, stride_raw, stride_features, coverage, roster, lock[FEATURE_VERSION_KEY])
    base, subjects, timelines = load_features(independent_features)
    people = set(originals)
    if set(subjects) != people or base[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]:
        raise ValueError("Independent features must match the audited roster and frozen feature version")
    grid = load_grid(base, subjects, timelines, stride_raw, stride_features, seconds=STRIDE)
    scored_rows = np.concatenate([block.rows for block in grid.timelines])
    replay_rows = np.concatenate([scored_rows, *grid.history_rows.values()])
    available = np.zeros(len(grid.subjects), dtype=bool)
    shared = _shared_features(grid.table)
    available[replay_rows] = shared[replay_rows]
    return grid, model, available, coverage_support, roster, fit_count, calibration_count


def evaluate_independent(lock, protocol, independent_features, stride_raw, stride_features, coverage):
    """Audit eligibility before applying a single frozen classifier and cutoff."""
    grid, model, available, coverage_support, roster, fit_count, calibration_count = load_independent_inputs(
        lock, protocol, independent_features, stride_raw, stride_features, coverage,
    )
    people = set(grid.subjects)
    scored_rows = np.concatenate([block.rows for block in grid.timelines])
    evaluation_scores = int(available[scored_rows].sum())

    # An all-positive eligibility probe measures opportunity before any model inference.
    candidate_monitor, candidate_episodes = scoring_runs(grid, np.where(available, 1.0, np.nan), people)
    raw_support = score_decisions(
        (run.decisions(0.5) for run in candidate_monitor),
        episode_runs=(run.decisions(0.5) for run in candidate_episodes),
    )
    episode_support, monitor_support = (raw_support[key] for key in ("episode_metrics", "burden_metrics"))
    _match_coverage({
        "complete_reference_episodes": episode_support["complete_reference_episodes"],
        "partial_reference_episodes": episode_support["partial_reference_episodes"],
        "reference_observed_seconds": episode_support["ecg_reference_seconds"],
        "monitoring_observed_seconds": monitor_support["ecg_reference_seconds"],
        "monitoring_observed_non_af_seconds": monitor_support["ecg_non_af_seconds"],
    }, coverage_support)
    supported_episodes = episode_support["complete_episodes_with_positive_decision"]
    if supported_episodes > coverage_support["complete_reference_episodes_with_usable_raw_input_endpoint"]:
        raise ValueError("Feature-supported episodes exceed the audited native PPG opportunity")

    probabilities = _predict(grid.table[ROUTE], available, model["model"])
    summary, reference, wrist = _held_out_replay(grid, probabilities, np.full(len(probabilities), model["cutoff"]), _IMMEDIATE)
    return {
        "scope": "Single frozen model on the audited independent cohort",
        FEATURE_VERSION_KEY: lock[FEATURE_VERSION_KEY],
        "route": ROUTE,
        "logistic_C": model["logistic_C"],
        "feature_window_seconds": WINDOW_SECONDS,
        "decision_stride_seconds": STRIDE,
        "alert_policy": lock["alert_policy"],
        "fit_participants": fit_count,
        "calibration_participants": calibration_count,
        "planned_independent_participants": len(roster),
        "evaluated_independent_participants": len(people),
        "participants_with_shared_feature_endpoint": len(set(grid.subjects[scored_rows[available[scored_rows]]])),
        "independent_participants_without_decoded_archives": len(roster - people),
        "planned_independent_roster_complete": people == roster,
        "target_episode_sensitivity": lock["target_sensitivity"],
        "independent_sensitivity_target_met": summary["episode_sensitivity"] >= lock["target_sensitivity"],
        "calibration": {"status": model["calibration"]["status"], **_summary(model["calibration"]["validation"])},
        "independent": {
            **summary,
            "scored_evaluation_cells": evaluation_scores,
            "scored_predecessor_cells": int(available.sum()) - evaluation_scores,
            "partial_reference_episodes": coverage_support["partial_reference_episodes"],
            "all_evaluation_ecg_observed_seconds": coverage_support["reference_observed_seconds"],
            "complete_episodes_with_usable_raw_ppg_endpoint": (
                coverage_support["complete_reference_episodes_with_usable_raw_input_endpoint"]
            ),
            "complete_episodes_with_shared_feature_endpoint": supported_episodes,
            "episode_sensitivity_support_ceiling": supported_episodes / summary["complete_episodes"],
            "wrist_analyzable_seconds": monitor_support["analyzable_seconds"],
            "wrist_analyzable_non_af_seconds": monitor_support["analyzable_non_af_seconds"],
        },
        "participant_cluster_uncertainty": _participant_uncertainty(reference, wrist, people),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "protocol-lock", "independent-features", "stride-raw", "stride-features", "coverage-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Use a new output path to preserve the recorded independent result")
        lock_bytes = args.lock.read_bytes()
        protocol_bytes = args.protocol_lock.read_bytes()
        coverage_bytes = args.coverage_report.read_bytes()
        lock = json.loads(lock_bytes)
        if sha256(protocol_bytes).hexdigest() != lock["source_protocol_sha256"]:
            raise ValueError("The frozen model must match the original protocol lock hash")
        report = evaluate_independent(lock, json.loads(protocol_bytes), args.independent_features,
                                      args.stride_raw, args.stride_features, json.loads(coverage_bytes))
        report["model_lock_sha256"] = sha256(lock_bytes).hexdigest()
        report["coverage_report_sha256"] = sha256(coverage_bytes).hexdigest()
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Independent template ECG evaluation failed: {error}\n")
    print(f"Evaluated {report['evaluated_independent_participants']} independent participants with the frozen model")


if __name__ == "__main__":
    main()
