"""Attribute wrist AF episode misses and false starts on the scoring clock."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .af_events import _TIME_EPSILON_S, _coverage, _event_at_decision, _grid, _intervals, _union
from .af_wrist_history import _cutoff_counts


@dataclass(frozen=True)
class _Decision:
    time: float
    run_start: float
    row: int
    raw: int
    referenced: bool
    ppg: bool
    margin: float | None
    active_before: bool | None
    active: bool | None
    start: int
    previous_reset: tuple[float, str] | None


def _active(counts, required):
    if required not in counts:
        return False
    return True if len(counts) == 1 else None


def _last_history_reset(history, start, stride, required):
    last_reset = None
    for row, decision in enumerate(history):
        if decision in (-1, 0) and _active(_cutoff_counts(history[:row], required), required) is True:
            time = start - (len(history) - row - 1) * stride
            last_reset = (time, "unavailable" if decision == -1 else "negative")
    return last_reset


def _trace(runs, required, history_for, margins):
    segments, ref_spans = [], []
    previous_end = -np.inf
    for raw_run, confirmed in sorted(runs, key=lambda pair: pair[0].start_s):
        edges, raw, ref, ppg = _grid(raw_run)
        confirmed_edges, states, confirmed_ref, confirmed_ppg = _grid(confirmed)
        starts = np.asarray(confirmed.alert_start_status)
        if (raw_run.subject_id != confirmed.subject_id or not np.array_equal(edges, confirmed_edges)
                or not np.array_equal(ref, confirmed_ref) or not np.array_equal(ppg, confirmed_ppg)
                or starts.shape != raw.shape or not np.isin(starts, (-1, 0, 1)).all()):
            raise ValueError("Raw and replayed runs must share the clock, ECG and explicit start statuses")
        if edges[0] < previous_end - _TIME_EPSILON_S:
            raise ValueError("Runs for the same subject must not overlap")
        if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
            segments.append([])
            history = tuple(history_for(raw_run))
            counts = _cutoff_counts(history, required)
            previous_reset = _last_history_reset(history, edges[0], raw_run.window_s, required)

        key = (raw_run.subject_id, raw_run.start_s)
        margin = np.asarray(margins[key], dtype=float) if margins is not None else np.full(len(raw), np.nan)
        if margin.shape != raw.shape or np.isinf(margin).any():
            raise ValueError("Fold cutoff margins must align with run decisions")
        ref_spans.append(edges[np.flatnonzero(np.diff(ref, prepend=False, append=False)).reshape(-1, 2)])

        for row, (time, decision) in enumerate(zip(edges[1:], raw, strict=True)):
            before = _active(counts, required)
            expected_start = 0
            if decision == 1 and required - 1 in counts:
                expected_start = 1 if len(counts) == 1 else -1
            counts = {min(count + 1, required) for count in counts} if decision == 1 else {0}
            active = _active(counts, required)
            expected_state = int(active is True) if decision != -1 else -1
            if states[row] != expected_state or starts[row] != expected_start:
                raise ValueError("Replayed alert state does not match the supplied decisions and history")
            if before is True and decision in (-1, 0):
                previous_reset = (float(time), "unavailable" if decision == -1 else "negative")
            segments[-1].append(_Decision(
                float(time), float(raw_run.start_s), row, int(decision), bool(ref[row]), bool(ppg[row]),
                float(margin[row]) if np.isfinite(margin[row]) else None, before, active,
                int(starts[row]), previous_reset,
            ))
        previous_end = edges[-1]
    return segments, _union(np.concatenate(ref_spans))


def _episode(subject, start, end, decisions, onset_state):
    alert = [item.time for item in decisions if item.start == 1]
    possible = sum(item.start == -1 for item in decisions)
    scored = sum(item.raw >= 0 for item in decisions)
    positive = sum(item.raw == 1 for item in decisions)
    active = sum(item.active is True for item in decisions)
    if alert:
        primary = "detected"
    elif possible:
        primary = "possible_start"
    elif not decisions:
        primary = "no_decision_endpoint"
    elif onset_state is True:
        primary = "already_active_at_onset"
    elif not scored:
        primary = "no_score"
    elif not positive:
        primary = "below_cutoff"
    elif active:
        primary = "active_during_episode"
    else:
        primary = "confirmation_not_reached"

    margins = [item.margin for item in decisions if item.margin is not None]
    return {
        "subject_id": subject, "start_s": float(start), "end_s": float(end),
        "primary": primary, "detected": bool(alert), "active_at_onset": onset_state,
        "decision_refs": [[item.run_start, item.row] for item in decisions],
        "decision_count": len(decisions), "score_count": scored, "positive_count": positive,
        "active_count": active, "ppg_available_count": sum(item.ppg for item in decisions),
        "possible_start_count": possible, "max_margin": max(margins, default=None),
        "first_alert_delay_s": alert[0] - float(start) if alert else None,
    }


def _false_alert(subject, segment, row, episodes, positive_tail, required, feature_window_s):
    item = segment[row]
    stop_row = row + positive_tail + 1
    right_censored = stop_row >= len(segment)
    end = segment[-1 if right_censored else stop_row].time
    overlap = np.maximum(
        0, np.minimum(episodes[:, 1], item.time) - np.maximum(episodes[:, 0], item.time - feature_window_s),
    )
    af_input_s = float(overlap.sum())
    next_episode = np.searchsorted(episodes[:, 0], item.time, side="left")
    next_onset = float(episodes[next_episode, 0]) - item.time if next_episode < len(episodes) else None
    positives = required + positive_tail
    if af_input_s > _TIME_EPSILON_S:
        primary = "trailing_input_overlaps_af"
    elif positives > 1:
        primary = "sustained_positive"
    elif right_censored:
        primary = "right_censored"
    else:
        primary = "isolated_positive"

    previous_reset = item.previous_reset
    return {
        "subject_id": subject, "time_s": item.time, "run_start_s": item.run_start, "row_index": item.row,
        "primary": primary, "af_in_trailing_input_s": af_input_s, "next_af_onset_in_s": next_onset,
        "alert_margin": item.margin, "positive_streak_decisions": positives,
        "active_duration_s": end - item.time, "right_censored": right_censored,
        "previous_reset_kind": previous_reset[1] if previous_reset else None,
        "seconds_since_previous_reset": item.time - previous_reset[0] if previous_reset else None,
    }


def audit_events(raw_runs, confirmed_runs, *, confirmation_windows, prior_decisions, margins=None, feature_window_s=30):
    """Describe each complete episode and each definite reference-scored false start.

    Pass raw thresholded and replayed runs in the same order. Optional margins
    are keyed by (subject ID, run start). A false alert's active duration is a
    lower bound when the streak stays positive through the end of the supplied
    continuous segment. Call separately for all episode runs and for the
    prescheduled wrist runs that contribute to false-alert burden.
    """
    if not isinstance(confirmation_windows, (int, np.integer)) or confirmation_windows < 1:
        raise ValueError("Confirmation requires at least one positive decision")
    if not np.isfinite(feature_window_s) or feature_window_s <= 0:
        raise ValueError("Feature window duration must be positive and finite")

    groups = defaultdict(list)
    for number, (raw, confirmed) in enumerate(zip(raw_runs, confirmed_runs, strict=True)):
        group = ("subject", raw.subject_id) if raw.subject_id is not None else ("run", number)
        groups[group].append((raw, confirmed))

    complete, false_alerts, partial_count = [], [], 0
    for runs in groups.values():
        subject = runs[0][0].subject_id
        episodes = _union(np.concatenate([_intervals(raw.af_intervals_s) for raw, _ in runs]))
        segments, coverage = _trace(runs, confirmation_windows, prior_decisions, margins)
        whole, partial = _coverage(episodes, coverage)
        partial_count += int(partial.sum())
        decisions_by_episode = defaultdict(list)
        analysis_decisions = [item for segment in segments for item in segment]
        times = np.array([item.time for item in analysis_decisions])
        episode_ids = _event_at_decision(times, episodes)
        for item, episode_id in zip(analysis_decisions, episode_ids, strict=True):
            if item.referenced and episode_id >= 0:
                decisions_by_episode[episode_id].append(item)

        for episode_id in np.flatnonzero(whole):
            start, end = episodes[episode_id]
            next_row = np.searchsorted(times, start, side="right")
            onset_state = analysis_decisions[next_row].active_before if next_row < len(times) else None
            complete.append(_episode(subject, start, end, decisions_by_episode[episode_id], onset_state))

        for segment in segments:
            ids = _event_at_decision(np.array([item.time for item in segment]), episodes)
            positive_tail = 0
            for row in range(len(segment) - 1, -1, -1):
                item = segment[row]
                if item.referenced and item.start == 1 and ids[row] == -1:
                    false_alerts.append(_false_alert(subject, segment, row, episodes, positive_tail,
                                                     confirmation_windows, feature_window_s))
                positive_tail = positive_tail + 1 if item.raw == 1 else 0

    false_alerts.sort(key=lambda event: (event["subject_id"] is not None, event["subject_id"], event["time_s"]))
    return {"complete_episodes": complete, "partial_episodes": partial_count, "false_alerts": false_alerts}
