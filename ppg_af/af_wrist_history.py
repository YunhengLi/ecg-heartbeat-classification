"""Initialize wrist AF alert state from preceding PPG decisions."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import replace
from operator import index

import numpy as np

from .af_events import AFRun, _grid, _TIME_EPSILON_S


def _grouped_positions(runs):
    grouped = defaultdict(list)
    for position, run in enumerate(runs):
        key = ("subject", run.subject_id) if run.subject_id is not None else ("run", position)
        grouped[key].append(position)
    return grouped.values()


def _cutoff_counts(history: Iterable[int | None], required: int) -> set[int]:
    counts = set(range(required + 1))
    for decision in history:
        if decision == 1:
            counts = {min(count + 1, required) for count in counts}
        elif decision in (-1, 0):
            counts = {0}
        elif decision is None:
            counts = {0} | {min(count + 1, required) for count in counts}
        else:
            raise ValueError("History decisions must be -1, 0, 1, or None")
    return counts


def apply_alert_history(
    runs: Iterable[AFRun], *, confirmation_windows: int,
    prior_decisions: Callable[[AFRun], Iterable[int | None]],
) -> tuple[tuple[AFRun, ...], int]:
    """Replay alert state on evaluation windows without adding exposure.

    Supply raw decisions in chronological order before each disjoint block.
    A known unavailable decision (-1) resets the streak. An unobserved one
    (None) may have been positive or reset it. Adjacent runs on the same
    timeline carry state across their boundary.

    Return confirmed runs in their original order and the number of blocks
    with more than one possible initial streak count. A confirmed decision
    is positive only when every feasible state is active. Alert status marks
    starts as definite (1), possible but uncertain (-1), or absent (0).
    """
    required = index(confirmation_windows)
    if required < 1:
        raise ValueError("Confirmation requires at least one positive window")

    original = tuple(runs)
    confirmed = list(original)
    unknown_blocks = 0
    for positions in _grouped_positions(original):
        previous_end = -np.inf
        counts = set()
        for position in sorted(positions, key=lambda item: original[item].start_s):
            run = original[position]
            edges, raw, _, _ = _grid(run)
            if edges[0] < previous_end - _TIME_EPSILON_S:
                raise ValueError("Runs for the same subject must not overlap")
            if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
                counts = _cutoff_counts(prior_decisions(run), required)
                unknown_blocks += len(counts) > 1

            decisions = raw.astype(np.int8, copy=True)
            starts = np.zeros(len(raw), dtype=np.int8)
            for row, decision in enumerate(raw):
                if decision != 1:
                    counts = {0}
                    continue

                if required - 1 in counts:
                    starts[row] = 1 if len(counts) == 1 else -1
                counts = {min(count + 1, required) for count in counts}
                decisions[row] = counts == {required}

            confirmed[position] = replace(run, decisions=decisions, alert_start_status=starts)
            previous_end = edges[-1]
    return tuple(confirmed), unknown_blocks


def _rearm_state(states, decision, required):
    if decision == 1:
        return {0}
    if decision == -1:
        return {required}
    advanced = {min(state + 1, required) for state in states}
    if decision == 0:
        return advanced
    if decision is None:
        return advanced | {0, required}
    raise ValueError("History decisions must be -1, 0, 1, or None")


def apply_negative_rearm(
    runs: Iterable[AFRun], *, prior_decisions: Callable[[AFRun], Iterable[int | None]],
    negative_windows: int = 2,
) -> tuple[tuple[AFRun, ...], int]:
    """Require consecutive observed negatives before rearming a new alert.

    Leave the classifier decisions and availability intact. A known unavailable
    decision clears alert state, as in the immediate policy. Unknown history
    propagates all feasible states so an ambiguous first start stays ambiguous.
    """
    required = index(negative_windows)
    if required < 1:
        raise ValueError("Rearming requires at least one negative decision")
    original = tuple(runs)
    output = list(original)
    unknown_blocks = 0
    for positions in _grouped_positions(original):
        previous_end = -np.inf
        states = set()
        for position in sorted(positions, key=lambda item: original[item].start_s):
            run = original[position]
            edges, raw, _, _ = _grid(run)
            if edges[0] < previous_end - _TIME_EPSILON_S:
                raise ValueError("Runs for the same subject must not overlap")
            if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
                states = set(range(required + 1))
                for decision in prior_decisions(run):
                    states = _rearm_state(states, decision, required)
                unknown_blocks += len(states) > 1
            starts = np.zeros(len(raw), dtype=np.int8)
            for row, decision in enumerate(raw):
                if decision == 1 and required in states:
                    starts[row] = 1 if len(states) == 1 else -1
                states = _rearm_state(states, decision, required)
            output[position] = replace(run, alert_start_status=starts)
            previous_end = edges[-1]
    return tuple(output), unknown_blocks
