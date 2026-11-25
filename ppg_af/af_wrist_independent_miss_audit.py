"""Summarize misses without changing the frozen ECG-clock endpoint."""

from collections import Counter, defaultdict

import numpy as np

from .af_events import _TIME_EPSILON_S, _event_at_decision, _grid, _intervals, _union


def _median(values):
    return float(np.median(values)) if values else None


def _timeline(runs):
    """Trace definite starts through contiguous replay, including warm-start state."""
    intervals = _union(np.concatenate([_intervals(run.af_intervals_s) for run in runs]))
    reference_spans, new_starts, positive_origins = [], [], {}
    previous_end = -np.inf
    origin = None
    for run in sorted(runs, key=lambda item: item.start_s):
        edges, decisions, valid, _ = _grid(run)
        starts = np.asarray(run.alert_start_status)
        if starts.shape != decisions.shape or not np.isin(starts, (-1, 0, 1)).all():
            raise ValueError("Miss audit requires explicit replayed alert-start states")
        if edges[0] < previous_end - _TIME_EPSILON_S:
            raise ValueError("Runs for the same subject must not overlap")
        if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
            origin = None
        known = np.flatnonzero(np.diff(valid, prepend=False, append=False)).reshape(-1, 2)
        reference_spans.append(edges[known])
        rhythms = _event_at_decision(edges[1:], intervals)
        for row, endpoint in enumerate(edges[1:]):
            if decisions[row] != 1:
                origin = None
                continue
            if starts[row] == 1:
                phase = "ECG_AF" if rhythms[row] >= 0 else "ECG_non_AF"
                origin = float(endpoint), phase if valid[row] else "ECG_unobserved"
                if valid[row]:
                    new_starts.append(float(endpoint))
            positive_origins[(float(run.start_s), row)] = float(endpoint), origin
        previous_end = edges[-1]
    return intervals, _union(np.concatenate(reference_spans)), np.array(new_starts), positive_origins


def _mechanism(case):
    if not case["score_count"]:
        return "no_score"
    if not case["positive_count"]:
        return "all_in_episode_scores_below_cutoff"
    if case["active_at_onset"] is True and case["active_count"]:
        return "carried_alert_with_positive_in_episode"
    return "other_positive_without_definite_new_start"


def _group_counts(cases):
    missed = [case for case in cases if not case["detected"]]
    detected = len(cases) - len(missed)
    return {
        "complete_episodes": len(cases),
        "detected": detected,
        "episode_sensitivity": detected / len(cases) if cases else None,
        "missed": len(missed),
        "misses_by_mechanism": dict(sorted(Counter(_mechanism(case) for case in missed).items())),
    }


def _participant_concentration(cases):
    by_subject = defaultdict(list)
    for case in cases:
        by_subject[case["subject_id"]].append(case)
    largest = max((len(group) for group in by_subject.values()), default=0)
    dominant, others = [], []
    for group in by_subject.values():
        (dominant if len(group) == largest else others).extend(group)
    return {
        "participants_with_complete_episodes": len(by_subject),
        "participants_with_missed_episodes": sum(
            any(not case["detected"] for case in group) for group in by_subject.values()
        ),
        "maximum_complete_episodes_per_participant": largest,
        "participants_tied_for_maximum": sum(len(group) == largest for group in by_subject.values()),
        "maximum_episode_participant_group": _group_counts(dominant),
        "all_other_participants_with_complete_episodes": _group_counts(others),
    }


def summarize_missed_episodes(episodes, reference, *, feature_window_s=30):
    """Audit complete episodes with the same decision-end clock as published scoring.

    Pass complete cases from the existing event audit and replayed reference
    runs. The result contains counts, clock intervals, and score margins only.
    """
    if not np.isfinite(feature_window_s) or feature_window_s <= 0:
        raise ValueError("The audit window must be positive and finite")
    cases = tuple(episodes)
    by_subject = defaultdict(list)
    for run in reference:
        by_subject[run.subject_id].append(run)
    traces = {subject: _timeline(runs) for subject, runs in by_subject.items()}
    missed = [case for case in cases if not case["detected"]]
    scored = [case for case in missed if case["score_count"]]
    below = [case for case in missed if _mechanism(case) == "all_in_episode_scores_below_cutoff"]
    carried = [case for case in missed if _mechanism(case) == "carried_alert_with_positive_in_episode"]
    no_score = [case for case in missed if not case["score_count"]]
    below_margins = [case["max_margin"] for case in below]
    if any(margin is None or margin >= 0 or not np.isfinite(margin) for margin in below_margins):
        raise ValueError("A scored, all-below-cutoff miss must have a finite negative margin")

    origin_counts = Counter()
    pre_onset_leads, first_positive_delays = [], []
    for case in carried:
        _, _, _, positive_origins = traces[case["subject_id"]]
        first_positive = next(
            (positive_origins[(start, row)] for start, row in case["decision_refs"]
             if (start, row) in positive_origins), None,
        )
        if first_positive is None:
            raise ValueError("The case's carried positive state must be present on the replayed clock")
        endpoint, origin = first_positive
        first_positive_delays.append(endpoint - case["start_s"])
        if origin is None:
            origin_counts["already_active_at_start_of_replayed_segment"] += 1
            continue
        onset = case["start_s"]
        start, phase = origin
        if start > onset + _TIME_EPSILON_S:
            raise ValueError("An in-episode new start cannot be classified as carried from before onset")
        if abs(start - onset) <= _TIME_EPSILON_S:
            phase = "exactly_at_episode_onset"
        origin_counts[phase] += 1
        pre_onset_leads.append(onset - start)

    post_counts = Counter()
    after_offset_delays = []
    for case in scored:
        intervals, covered, new_starts, _ = traces[case["subject_id"]]
        left, right = case["end_s"], case["end_s"] + feature_window_s
        fully_observed = np.any((covered[:, 0] <= left) & (covered[:, 1] >= right))
        another_episode = np.any((intervals[:, 0] < right) & (intervals[:, 1] > left))
        if not fully_observed or another_episode:
            continue
        post_counts["with_complete_followup"] += 1
        starts = new_starts[(new_starts > left) & (new_starts <= right)]
        if len(starts):
            post_counts["with_new_start"] += 1
            after_offset_delays.append(float(starts.min() - left))

    return {
        **_group_counts(cases),
        "scored_misses": len(scored),
        "scored_misses_with_alert_active_at_onset": sum(case["active_at_onset"] is True for case in scored),
        "below_cutoff_misses_with_alert_active_at_onset": sum(case["active_at_onset"] is True for case in below),
        "scored_misses_with_wholly_af_feature_window": sum(case["pure_af_input_at_decision"] for case in scored),
        "below_cutoff_misses_with_wholly_af_feature_window": sum(
            case["pure_af_input_at_decision"] for case in below
        ),
        "no_score_misses": {
            "no_decision_endpoint": sum(not case["decision_count"] for case in no_score),
            "decision_endpoints_but_all_inputs_flat": sum(
                set(case["input_status_counts"]) == {"flatline"} for case in no_score
            ),
            "alert_active_at_onset": sum(case["active_at_onset"] is True for case in no_score),
        },
        "below_cutoff_margin": {
            "median_best_score_minus_cutoff": _median(below_margins),
            "largest_best_score_minus_cutoff": max(below_margins, default=None),
            "within_0_01_of_cutoff": sum(margin >= -0.01 for margin in below_margins),
            "within_0_05_of_cutoff": sum(margin >= -0.05 for margin in below_margins),
        },
        "carried_alerts": {
            "observed_origins": dict(sorted(origin_counts.items())),
            "median_observed_start_lead_s": _median(pre_onset_leads),
            "median_first_positive_after_onset_s": _median(first_positive_delays),
        },
        "scored_misses_followed_by_new_alert": {
            "followup_window_s": feature_window_s,
            "with_complete_ECG_non_AF_followup": post_counts["with_complete_followup"],
            "new_starts_during_complete_non_AF_followup": post_counts["with_new_start"],
            "median_new_start_delay_from_offset_s": _median(after_offset_delays),
        },
        "participant_concentration": _participant_concentration(cases),
    }
