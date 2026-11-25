"""Summarize false starts in prescheduled independent wrist monitoring."""

from collections import Counter, defaultdict

import numpy as np

from .af_events import _TIME_EPSILON_S, _event_at_decision, _intervals, _union
from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import MONITOR_KIND


def _median(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _overlap_seconds(intervals, start, end):
    return np.maximum(0, np.minimum(intervals[:, 1], end) - np.maximum(intervals[:, 0], start)).sum()


def summarize_false_alerts(grid, scores, cutoff, false_starts, evidence, *, episodes=None):
    """Aggregate detailed false-alert audit rows without exposing identifiers."""
    wrist_by_subject = defaultdict(list)
    wrist_rows = {}
    for block in grid.timelines:
        if block.kind == MONITOR_KIND:
            wrist_by_subject[block.subject].append(block)
            wrist_rows[(block.subject, block.start)] = block.rows

    references, rhythms = {}, {}
    observed_non_af_margins, observed_non_af_quality = [], []
    endpoint_scores = {}
    for subject, blocks in wrist_by_subject.items():
        af_intervals = _union(np.concatenate([_intervals(block.episodes) for block in blocks]))
        ref_spans = []
        for block in blocks:
            rows = block.rows
            edges = block.start + grid.seconds * np.arange(len(rows) + 1)
            times = edges[1:]
            ref = grid.table["reference_valid"][rows]
            good_spans = np.flatnonzero(np.diff(ref, prepend=False, append=False)).reshape(-1, 2)
            ref_spans.append(edges[good_spans])
            values = scores[rows]
            non_af = ref & np.isfinite(values) & (_event_at_decision(times, af_intervals) < 0)
            observed_non_af_margins.extend(values[non_af] - cutoff)
            observed_non_af_quality.extend(evidence[(subject, block.start)]["quality"][non_af])
            endpoint_scores.update({(subject, float(time)): value for time, value in zip(times, values, strict=True)})
        references[subject] = _union(np.concatenate(ref_spans))
        rhythms[subject] = af_intervals

    by_subject = Counter()
    persistence = Counter({"one_positive_then_reset": 0, "multiple_positives": 0, "one_positive_then_boundary": 0})
    duration = Counter({"under_30_seconds_observed": 0, "at_least_30_seconds_observed": 0,
                        "censored_before_30_seconds": 0})
    recent_resets = Counter({"negative": 0, "unavailable": 0})
    trailing_reference = Counter({"known_AF_overlap": 0, "fully_observed_non_AF": 0, "incomplete_reference": 0})
    one_positive_resets = Counter({"negative": 0, "unavailable": 0})
    margins, quality = [], []
    verified_non_af_persistence = 0
    for event in false_starts:
        subject, start = event["subject_id"], event["run_start_s"]
        key = (subject, start)
        rows = wrist_rows[key]
        row = event["row_index"]
        if (not 0 <= row < len(rows) or abs(start + (row + 1) * grid.seconds - event["time_s"]) > _TIME_EPSILON_S
                or not grid.table["reference_valid"][rows[row]]):
            raise ValueError("Every false start must belong to its prescheduled ECG-observed wrist decision")
        by_subject[subject] += 1
        margins.append(event["alert_margin"])
        quality.append(evidence[key]["quality"][row])

        if event["positive_streak_decisions"] > 1:
            persistence["multiple_positives"] += 1
        elif event["right_censored"]:
            persistence["one_positive_then_boundary"] += 1
        else:
            persistence["one_positive_then_reset"] += 1
            next_score = endpoint_scores[(subject, event["time_s"] + grid.seconds)]
            one_positive_resets["negative" if np.isfinite(next_score) else "unavailable"] += 1
        if event["active_duration_s"] >= WINDOW_SECONDS:
            duration["at_least_30_seconds_observed"] += 1
            first = event["time_s"]
            end = first + WINDOW_SECONDS
            if (_overlap_seconds(references[subject], first, end) >= WINDOW_SECONDS - _TIME_EPSILON_S
                    and _overlap_seconds(rhythms[subject], first, end) <= _TIME_EPSILON_S):
                verified_non_af_persistence += 1
        elif event["right_censored"]:
            duration["censored_before_30_seconds"] += 1
        else:
            duration["under_30_seconds_observed"] += 1

        reset_age = event["seconds_since_previous_reset"]
        if reset_age is not None and reset_age <= WINDOW_SECONDS:
            recent_resets[event["previous_reset_kind"]] += 1
        if event["af_in_trailing_input_s"] > _TIME_EPSILON_S:
            trailing_reference["known_AF_overlap"] += 1
        else:
            time = event["time_s"]
            coverage = references[subject]
            contained = (coverage[:, 0] <= time - WINDOW_SECONDS) & (coverage[:, 1] >= time)
            key_name = "fully_observed_non_AF" if contained.any() else "incomplete_reference"
            trailing_reference[key_name] += 1

    all_counts = [by_subject[subject] for subject in wrist_by_subject]
    summary = {
        "false_starts": len(false_starts),
        "positive_streak": dict(persistence),
        "duration": dict(duration),
        "active_for_first_30_seconds_continuous_ECG_non_AF": verified_non_af_persistence,
        "one_positive_reset_type": dict(one_positive_resets),
        "restart_within_30_seconds_after_reset": dict(recent_resets),
        "trailing_reference": dict(trailing_reference),
        "median_score_margin_at_false_start": _median(margins),
        "median_score_margin_at_scored_non_af_wrist_decisions": _median(observed_non_af_margins),
        "median_raw_quality_at_false_start": _median(quality),
        "median_raw_quality_at_scored_non_af_wrist_decisions": _median(observed_non_af_quality),
        "participants_with_false_starts": len(by_subject),
        "participants_with_no_false_start": len(all_counts) - len(by_subject),
        "false_starts_by_participant_count": dict(sorted(Counter(all_counts).items())),
        "largest_false_start_count_from_one_participant": max(all_counts, default=0),
        "top_three_participants_false_starts": sum(sorted(all_counts, reverse=True)[:3]),
    }
    if episodes:
        case_counts = Counter(case["subject_id"] for case in episodes)
        contributor, count = case_counts.most_common(1)[0]
        if len(case_counts) == 1 or case_counts.most_common(2)[1][1] < count:
            summary["false_starts_from_largest_episode_contributor"] = by_subject[contributor]
    return summary
