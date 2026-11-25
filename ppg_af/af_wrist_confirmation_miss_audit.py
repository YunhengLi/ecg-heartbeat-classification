"""Attribute episode misses before and after frozen optical-quality confirmation."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import audit_route
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_independent_miss_audit import summarize_missed_episodes
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_template_independent import _match_coverage
from .af_wrist_template_study import ROUTE
from .af_wrist_vote_study import _IMMEDIATE, _held_out_replay


CONFIRMATION_RESULTS_SHA = "4b2e3e5eb90b5e49690b668b517be7d70aff02bffea020ece626c020ee7e7388"
BASELINE_AUDIT_SHA = "d9799f1be40407ed800c1935511ac2fab0f8c866a12be67e290c7d6d4a8653af"


def _outcome(case):
    if case["detected"]:
        return "detected"
    if case["possible_start_count"]:
        return "ambiguous_start"
    if not case["decision_count"]:
        return "no_decision_endpoint"
    if not case["score_count"]:
        statuses = set(case["input_status_counts"])
        if statuses == {"flatline"}:
            return "no_score_with_flat_ppg"
        if "finite_nonflat" not in statuses:
            return "no_score_with_unavailable_ppg"
        return "no_score_with_finite_ppg"
    if not case["positive_count"]:
        return "all_in_episode_scores_below_cutoff"
    if case["active_at_onset"] is True and case["active_count"]:
        return "carried_alert_with_positive_in_episode"
    return "other_positive_without_definite_new_start"


def _audit(grid, scores, cutoffs, evidence, expected):
    replay, reference, _ = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    _match_coverage(replay, expected)
    decisions = np.where(np.isfinite(scores), scores >= cutoffs, -1).astype(np.int8)
    _, cases = audit_route(grid, 1, scores, cutoffs, decisions, evidence, expected)
    complete = cases["complete_episodes"]
    return complete, decisions, {
        "frozen_replay": replay,
        "miss_attribution": summarize_missed_episodes(complete, reference),
        "episode_outcomes": dict(sorted(Counter(map(_outcome, complete)).items())),
        "complete_episodes_with_ambiguous_in_episode_start": sum(
            case["possible_start_count"] > 0 for case in complete
        ),
        "complete_episodes_with_unknown_onset_state": sum(case["active_at_onset"] is None for case in complete),
    }


def _episode_key(case):
    return case["subject_id"], case["start_s"], case["end_s"]


def _paired_attribution(grid, baseline, candidate, original_decisions, confirmed_decisions, published):
    original = {_episode_key(case): case for case in baseline}
    if len(original) != len(baseline) or original.keys() != {_episode_key(case) for case in candidate}:
        raise ValueError("Confirmation must preserve the complete episode denominator")
    rows_by_block = {(block.subject, block.start): block.rows for block in grid.timelines}
    transitions, suppressed = Counter(), Counter()
    lost = gained = both = 0
    for case in candidate:
        previous = original[_episode_key(case)]
        if previous["decision_refs"] != case["decision_refs"]:
            raise ValueError("Confirmation changed the in-episode decision clock")
        transitions[(_outcome(previous), _outcome(case))] += 1
        lost += previous["detected"] and not case["detected"]
        gained += case["detected"] and not previous["detected"]
        both += case["detected"] and previous["detected"]
        if case["detected"]:
            continue
        rows = np.array([rows_by_block[case["subject_id"], start][row]
                         for start, row in case["decision_refs"]], dtype=int)
        old = original_decisions[rows]
        new = confirmed_decisions[rows]
        removed = (old == 1) & (new != 1)
        if removed.any():
            suppressed["misses_with_a_removed_in_episode_positive"] += 1
            suppressed["removed_in_episode_positives_to_negative"] += int(np.sum(removed & (new == 0)))
            suppressed["removed_in_episode_positives_to_unavailable"] += int(np.sum(removed & (new < 0)))
            if not (new == 1).any():
                suppressed["misses_with_all_original_in_episode_positives_removed"] += 1
            if previous["detected"]:
                suppressed["lost_detections_with_a_removed_in_episode_positive"] += 1
    if (lost, gained, both) != (published["lost"], published["gained"], published["detected_by_both"]):
        raise ValueError("Paired episode detections disagree with the pinned confirmation replay")
    return {
        "lost_detections": lost, "gained_detections": gained, "detected_by_both": both,
        "episode_transitions": [{"baseline": first, "confirmation": second, "episodes": count}
                                for (first, second), count in sorted(transitions.items())],
        "removed_positives_among_confirmation_misses": dict(sorted(suppressed.items())),
    }


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    previous = _read_pinned(args.confirmation_results, CONFIRMATION_RESULTS_SHA,
                            public=True)["historical_episodes_and_monitoring"]
    baseline_audit = _read_pinned(args.baseline_audit, BASELINE_AUDIT_SHA, public=True)
    grid, model, available, roster, quality, _ = historical_inputs(args, protocol, model_lock)
    evidence = load_input_evidence(grid, args.prior_inputs / "stride-raw", args.prior_inputs / "stride-features")
    scores = _predict(grid.table[ROUTE], available, model["model"])
    cutoffs = np.full(len(scores), model["cutoff"])
    predecessor, active = preceding_rows(grid)
    filtered = confirm_scores(scores, predecessor, active & (quality["quality"][:, -1] < QUALITY_CUTOFF))
    baseline, original_decisions, before = _audit(grid, scores, cutoffs, evidence, previous["baseline"])
    published = previous["conditional_confirmation"]
    candidate, confirmed_decisions, after = _audit(grid, filtered, cutoffs, evidence, published["summary"])
    if before["miss_attribution"] != baseline_audit["miss_audit"]:
        raise ValueError("The baseline miss audit does not reproduce")
    if int(np.sum(np.isfinite(scores) & ~np.isfinite(filtered))) != previous["lost_scoreable_decisions"]:
        raise ValueError("The confirmation replay changed score support relative to published results")
    paired = _paired_attribution(grid, baseline, candidate, original_decisions, confirmed_decisions,
                                 published["paired_changes"]["episodes"])
    return {
        "scope": "Retrospective miss attribution on the same 25 previously scored out-of-fit participants",
        "frozen_classifier_and_cutoff": True, "quality_boundary": QUALITY_CUTOFF, "participants": len(roster),
        "confirmation_replay_sha256": CONFIRMATION_RESULTS_SHA, "baseline_miss_audit_sha256": BASELINE_AUDIT_SHA,
        "baseline": before, "conditional_confirmation": after, "paired": paired,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report",
                "historical-results", "confirmation-results", "baseline-audit", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
