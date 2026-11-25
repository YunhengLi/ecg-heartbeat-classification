"""Evaluate AF alerts on contiguous, ECG-annotated time windows."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from operator import index

import numpy as np


_TIME_EPSILON_S = 1e-6


@dataclass(frozen=True)
class AFRun:
    """One contiguous, nonoverlapping grid of model decisions.

    ``af_intervals_s`` contains half-open reference intervals on the same clock as
    ``start_s``. Supply original episode endpoints, even for clipped runs.
    Decisions are -1 (unavailable), 0 (negative), and 1 (positive). Keep PPG gaps
    on the grid as unavailable decisions and mark them in ``ppg_available``.
    ``reference_valid`` indicates windows with usable native ECG annotations.
    Supply a common subject ID when several runs share an ECG timeline.
    ``alert_start_status`` can provide explicit starts from a causal state
    replay. Values are 1 (definite), 0 (none), and -1 (possible, not certain).
    """

    start_s: float
    window_s: float
    af_intervals_s: np.ndarray
    decisions: np.ndarray
    reference_valid: np.ndarray | None = None
    ppg_available: np.ndarray | None = None
    subject_id: str | None = None
    alert_start_status: np.ndarray | None = None


def _intervals(values):
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        return np.empty((0, 2))
    if (result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all()
            or np.any(result[:, 1] <= result[:, 0])):
        raise ValueError("AF intervals must be finite (start, end) pairs with start < end")
    return result


def _union(intervals):
    merged = []
    for start, end in intervals[np.argsort(intervals[:, 0], kind="stable")]:
        if merged and start <= merged[-1][1] + _TIME_EPSILON_S:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def _mask(values, shape, name):
    if values is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(values)
    if result.shape != shape or not np.isin(result, (False, True)).all():
        raise ValueError(f"{name} must contain one boolean per decision")
    return result.astype(bool)


def _grid(run):
    decisions = np.asarray(run.decisions)
    if decisions.ndim != 1 or not len(decisions) or not np.isin(decisions, (-1, 0, 1)).all():
        raise ValueError("Decisions must be a nonempty vector of -1, 0 and 1")
    if not np.isfinite(run.start_s) or not np.isfinite(run.window_s) or run.window_s <= 0:
        raise ValueError("A run needs a finite start and a positive finite window duration")
    edges = run.start_s + run.window_s * np.arange(len(decisions) + 1)
    if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Window boundaries must be finite and strictly increasing")
    reference = _mask(run.reference_valid, decisions.shape, "reference_valid")
    ppg = _mask(run.ppg_available, decisions.shape, "ppg_available")
    if np.any(~ppg & (decisions >= 0)):
        raise ValueError("PPG gaps must have unavailable decisions (-1)")
    return edges, decisions, reference, ppg


def confirm_alerts(runs: Iterable[AFRun], consecutive_windows: int = 1) -> tuple[AFRun, ...]:
    """Keep a positive streak inactive until its required windows have ended."""
    required = index(consecutive_windows)
    if required < 1:
        raise ValueError("Confirmation requires at least one positive window")
    original = tuple(runs)
    if required == 1:
        return original
    if any(run.alert_start_status is not None for run in original):
        raise ValueError("Apply confirmation to raw decisions before annotating alert starts")

    by_subject = defaultdict(list)
    for position, run in enumerate(original):
        key = ("subject", run.subject_id) if run.subject_id is not None else ("run", position)
        by_subject[key].append(position)
    confirmed = list(original)
    for positions in by_subject.values():
        count, previous_end = 0, -np.inf
        for position in sorted(positions, key=lambda item: original[item].start_s):
            run = original[position]
            edges, decisions, _, _ = _grid(run)
            if edges[0] < previous_end - _TIME_EPSILON_S:
                raise ValueError("Runs for the same subject must not overlap")
            if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
                count = 0
            output = decisions.astype(np.int8, copy=True)
            for row, decision in enumerate(decisions):
                if decision == 1:
                    count += 1
                    output[row] = count >= required
                else:
                    count = 0
            confirmed[position] = replace(run, decisions=output)
            previous_end = edges[-1]
    return tuple(confirmed)


def _event_at_decision(times, episodes):
    """Look up the reference state just before each window-end timestamp."""
    if not len(episodes):
        return np.full(len(times), -1, dtype=int)
    indices = np.searchsorted(episodes[:, 0], times, side="left") - 1
    inside = (indices >= 0) & (times <= episodes[indices, 1])
    return np.where(inside, indices, -1)


def _coverage(episodes, covered):
    if not len(covered):
        absent = np.zeros(len(episodes), dtype=bool)
        return absent, absent.copy()
    indices = np.searchsorted(covered[:, 1], episodes[:, 0], side="right")
    safe = np.minimum(indices, len(covered) - 1)
    seen = (indices < len(covered)) & (covered[safe, 0] < episodes[:, 1])
    whole = seen & (covered[safe, 0] <= episodes[:, 0]) & (covered[safe, 1] >= episodes[:, 1])
    return whole, seen & ~whole


def _evaluate_subject(runs, totals, report_uncertainty):
    episodes = _union(np.concatenate([_intervals(run.af_intervals_s) for run in runs]))
    previous_end, previous_positive = -np.inf, False
    reference_spans, valid_start_times, positive_times, possible_start_times = [], [], [], []
    for run in sorted(runs, key=lambda item: item.start_s):
        edges, decisions, reference, ppg = _grid(run)
        if edges[0] < previous_end - _TIME_EPSILON_S:
            raise ValueError("Runs for the same subject must not overlap")
        known = np.flatnonzero(np.diff(reference, prepend=False, append=False)).reshape(-1, 2)
        reference_spans.append(edges[known])
        positive = decisions == 1
        same_run = abs(edges[0] - previous_end) <= _TIME_EPSILON_S and previous_positive
        starts = positive & ~np.r_[same_run, positive[:-1]]
        if run.alert_start_status is not None:
            status = np.asarray(run.alert_start_status)
            if (status.shape != decisions.shape or not np.isin(status, (-1, 0, 1)).all()
                    or np.any((status == 1) & ~positive) or np.any((status == -1) & (decisions < 0))):
                raise ValueError("Explicit alert starts must align with available decisions")
            starts = status == 1
            possible_start_times.append(edges[1:][(status == -1) & reference])
        valid_start_times.append(edges[1:][starts & reference])
        positive_times.append(edges[1:][positive & reference])
        previous_end, previous_positive = edges[-1], bool(positive[-1])

        duration = float(run.window_s)
        totals["windows"] += len(decisions)
        totals["recorded_seconds"] += len(decisions) * duration
        totals["ecg_reference_seconds"] += int(reference.sum()) * duration
        totals["ppg_available_seconds"] += int(ppg.sum()) * duration
        totals["ppg_gap_seconds"] += int((~ppg).sum()) * duration
        totals["reference_with_ppg_gap_seconds"] += int((reference & ~ppg).sum()) * duration
        totals["reference_with_ppg_abstention_seconds"] += int((reference & ppg & (decisions < 0)).sum()) * duration
        totals["analyzable_seconds"] += int((reference & (decisions >= 0)).sum()) * duration
        totals["unscored_alert_starts"] += int((starts & ~reference).sum())

    times = np.concatenate(valid_start_times)
    alert_events = _event_at_decision(times, episodes)
    first_alert = np.full(len(episodes), np.inf)
    true_starts = alert_events >= 0
    np.minimum.at(first_alert, alert_events[true_starts], times[true_starts])
    detected = np.isfinite(first_alert)
    any_positive = np.zeros(len(episodes), dtype=bool)
    positive_events = _event_at_decision(np.concatenate(positive_times), episodes)
    any_positive[positive_events[positive_events >= 0]] = True
    whole, partial = _coverage(episodes, _union(np.concatenate(reference_spans)))
    totals["complete_reference_episodes"] += int(whole.sum())
    totals["partial_reference_episodes"] += int(partial.sum())
    totals["episodes_outside_reference_coverage"] += int((~(whole | partial)).sum())
    totals["detected_complete_episodes"] += int((whole & detected).sum())
    totals["detected_partial_episodes"] += int((partial & detected).sum())
    totals["complete_episodes_with_positive_decision"] += int((whole & any_positive).sum())
    totals["partial_episodes_with_positive_decision"] += int((partial & any_positive).sum())
    totals["true_alert_starts"] += int(true_starts.sum())
    totals["false_alert_starts"] += int((~true_starts).sum())
    if report_uncertainty:
        possible_times = np.concatenate(possible_start_times) if possible_start_times else np.empty(0)
        possible_events = _event_at_decision(possible_times, episodes)
        could_detect = detected.copy()
        could_detect[possible_events[possible_events >= 0]] = True
        totals["ambiguous_ecg_valid_alert_start_timestamps"] += int(len(possible_times))
        totals["ambiguous_non_af_alert_start_timestamps"] += int((possible_events < 0).sum())
        totals["complete_episodes_with_definite_or_possible_alert"] += int((whole & could_detect).sum())
    return first_alert[whole & detected] - episodes[whole & detected, 0]


def evaluate_af(runs: Iterable[AFRun], *, representative: bool = False) -> dict:
    """Score alert starts at window end and keep clipped episodes separate.

    A positive streak emits one alert. Abstention, negative decisions, and
    gaps in the supplied timeline reset the streak. A complete reference episode
    is detected if a new alert starts during it. Runs without a subject ID are
    treated independently. Coverage and daily false-alert rates use time with a
    usable ECG reference as the denominator. Set ``representative`` only for a
    representative sequence of recorded time, never for AF-enriched samples.
    """
    if not isinstance(representative, (bool, np.bool_)):
        raise ValueError("representative must be a boolean")
    grouped = defaultdict(list)
    n_runs = 0
    report_uncertainty = False
    for n_runs, run in enumerate(runs, start=1):
        key = ("subject", run.subject_id) if run.subject_id is not None else ("run", n_runs)
        grouped[key].append(run)
        report_uncertainty |= run.alert_start_status is not None
    if not grouped:
        raise ValueError("At least one contiguous AF run is required")

    totals = defaultdict(int)
    delays = np.concatenate([_evaluate_subject(group, totals, report_uncertainty) for group in grouped.values()])
    known = totals["ecg_reference_seconds"]
    complete = totals["complete_reference_episodes"]
    return {
        "subjects_or_independent_runs": len(grouped),
        "runs": n_runs,
        **totals,
        "ecg_unavailable_seconds": totals["recorded_seconds"] - known,
        "complete_episode_recall": totals["detected_complete_episodes"] / complete if complete else None,
        "median_complete_episode_alert_delay_seconds": float(np.median(delays)) if len(delays) else None,
        "analyzable_coverage": totals["analyzable_seconds"] / known if known else None,
        "false_alerts_per_observed_day": totals["false_alert_starts"] * 86400 / known if representative and known else None,
        "representative_timeline": bool(representative),
    }
