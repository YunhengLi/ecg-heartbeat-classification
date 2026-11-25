"""Compare paired AF episodes and false starts on unchanged reference clocks."""

from collections import defaultdict

import numpy as np

from .af_events import _TIME_EPSILON_S, _coverage, _event_at_decision, _grid, _intervals, _union


def _definite_starts(run, decisions):
    status = np.asarray(run.alert_start_status)
    if (status.shape != decisions.shape or not np.isin(status, (-1, 0, 1)).all()
            or np.any((status == 1) & (decisions != 1))
            or np.any((status == -1) & (decisions < 0))):
        raise ValueError("Paired replay requires explicit alert starts aligned to each decision")
    return status == 1


def _align_runs(baseline, candidate):
    groups = defaultdict(list)
    for number, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        left_edges, left_decisions, reference, left_ppg = _grid(left)
        right_edges, right_decisions, right_reference, right_ppg = _grid(right)
        episodes, right_episodes = _intervals(left.af_intervals_s), _intervals(right.af_intervals_s)
        if (left.subject_id != right.subject_id or not np.array_equal(left_edges, right_edges)
                or not np.array_equal(reference, right_reference) or not np.array_equal(left_ppg, right_ppg)
                or not np.array_equal(episodes, right_episodes)):
            raise ValueError("Replay pairs must share participants, samples, and reference labels")
        key = ("subject", left.subject_id) if left.subject_id is not None else ("run", number)
        spans = left_edges[np.flatnonzero(np.diff(reference, prepend=False, append=False)).reshape(-1, 2)]
        left_times = left_edges[1:][reference & _definite_starts(left, left_decisions)]
        right_times = right_edges[1:][reference & _definite_starts(right, right_decisions)]
        groups[key].append((left_edges, spans, episodes, left_times, right_times))
    return groups.values()


def _alert_times(samples):
    samples = sorted(samples, key=lambda sample: sample[0][0])
    for previous, current in zip(samples, samples[1:]):
        if current[0][0] < previous[0][-1] - _TIME_EPSILON_S:
            raise ValueError("Paired runs for the same participant must not overlap")
    _, spans, episodes, left, right = zip(*samples, strict=True)
    return _union(np.concatenate(spans)), _union(np.concatenate(episodes)), np.concatenate(left), np.concatenate(right)


def _first_alerts(alerts, episodes):
    episode_ids = _event_at_decision(alerts, episodes)
    detected = episode_ids >= 0
    first = np.full(len(episodes), np.inf)
    np.minimum.at(first, episode_ids[detected], alerts[detected])
    return first


def compare_alert_runs(baseline_episodes, candidate_episodes, baseline_monitor, candidate_monitor):
    """Return paired episode and false-start changes without identifiers.

    Each baseline and candidate pair must be passed in matching order with
    explicit replayed start states. Only definite starts in ECG-valid windows
    count. False-start overlap requires the same wrist participant and exact
    decision timestamp, so a shifted alert produces two unmatched starts.
    """
    complete = gained = lost = both = neither = 0
    baseline_delays, candidate_delays, shared_delays = [], [], []
    for samples in _align_runs(baseline_episodes, candidate_episodes):
        coverage, episodes, left_times, right_times = _alert_times(samples)
        whole, _ = _coverage(episodes, coverage)
        left = _first_alerts(left_times, episodes)[whole]
        right = _first_alerts(right_times, episodes)[whole]
        had_alert, has_alert = np.isfinite(left), np.isfinite(right)
        shared = had_alert & has_alert
        complete += len(left)
        gained += int((~had_alert & has_alert).sum())
        lost += int((had_alert & ~has_alert).sum())
        both += int(shared.sum())
        neither += int((~had_alert & ~has_alert).sum())
        starts = episodes[whole, 0]
        baseline_delays.extend(left[had_alert] - starts[had_alert])
        candidate_delays.extend(right[has_alert] - starts[has_alert])
        shared_delays.extend(right[shared] - left[shared])

    baseline_false = candidate_false = shared_false = 0
    for samples in _align_runs(baseline_monitor, candidate_monitor):
        _, episodes, left_times, right_times = _alert_times(samples)
        left = left_times[_event_at_decision(left_times, episodes) < 0]
        right = right_times[_event_at_decision(right_times, episodes) < 0]
        baseline_false += len(left)
        candidate_false += len(right)
        shared_false += len(np.intersect1d(left, right))

    shifts = np.asarray(shared_delays)
    return {
        "episodes": {
            "complete": complete,
            "gained": gained,
            "lost": lost,
            "detected_by_both": both,
            "detected_by_neither": neither,
            "baseline_median_delay_s": float(np.median(baseline_delays)) if baseline_delays else None,
            "candidate_median_delay_s": float(np.median(candidate_delays)) if candidate_delays else None,
            "median_delay_change_s_for_shared": float(np.median(shifts)) if both else None,
            "shared_delayed": int((shifts > _TIME_EPSILON_S).sum()),
            "shared_earlier": int((shifts < -_TIME_EPSILON_S).sum()),
            "shared_unchanged": int((np.abs(shifts) <= _TIME_EPSILON_S).sum()),
        },
        "false_starts": {
            "baseline": baseline_false,
            "candidate": candidate_false,
            "shared_exact": shared_false,
            "baseline_only_exact": baseline_false - shared_false,
            "candidate_only_exact": candidate_false - shared_false,
        },
    }
