"""Replay a causal two-of-three AF alert rule on the decision clock."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import replace

import numpy as np

from .af_events import AFRun, _grid, _TIME_EPSILON_S


# Each state records the preceding two votes, oldest first, and whether
# an alert was active at the preceding decision. These are all reachable
# states, including both ways a positive after a negative can end up.
_RESET = (False, False, False)
_UNKNOWN_HISTORY = frozenset((
    _RESET,
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, True, True),
))


def _advance(state, decision):
    if decision == -1:
        return _RESET
    older, previous, _ = state
    positive = decision == 1
    return previous, positive, positive and (older or previous)


def _cutoff_states(history):
    states = _UNKNOWN_HISTORY
    for decision in history:
        if decision is None:
            outcomes = (-1, 0, 1)
        elif decision in (-1, 0, 1):
            outcomes = (decision,)
        else:
            raise ValueError("History decisions must be -1, 0, 1, or None")
        states = {_advance(state, outcome) for state in states for outcome in outcomes}
    return states


def apply_two_of_three_alerts(
    runs: Iterable[AFRun], *, prior_decisions: Callable[[AFRun], Iterable[int | None]],
) -> tuple[tuple[AFRun, ...], int]:
    """Confirm on a current positive and a positive among the preceding two.

    A negative ends an active alert and occupies one place in the vote window.
    An unavailable decision (-1) clears the vote window and ends the alert.
    Supply chronological history before each disjoint run. Unknown history
    (None) allows all raw decision values. Adjacent runs on one subject's
    decision clock carry state across their shared boundary.

    Return replayed runs in input order and the count of disjoint runs with
    uncertain initial state. Alert starts are 1 when definite, -1 when
    possible, and 0 when absent. Only definite activity is marked positive.
    """
    original = tuple(runs)
    grouped = defaultdict(list)
    for position, run in enumerate(original):
        key = ("subject", run.subject_id) if run.subject_id is not None else ("run", position)
        grouped[key].append(position)

    replayed = list(original)
    unknown_blocks = 0
    for positions in grouped.values():
        previous_end = -np.inf
        states = _UNKNOWN_HISTORY
        for position in sorted(positions, key=lambda item: original[item].start_s):
            run = original[position]
            edges, raw, _, _ = _grid(run)
            if edges[0] < previous_end - _TIME_EPSILON_S:
                raise ValueError("Runs for the same subject must not overlap")
            if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
                states = _cutoff_states(prior_decisions(run))
                unknown_blocks += len(states) > 1

            decisions = raw.astype(np.int8, copy=True)
            starts = np.zeros(len(raw), dtype=np.int8)
            for row, decision in enumerate(raw):
                transitions = [(state, _advance(state, decision)) for state in states]
                starts_here = [not before[2] and after[2] for before, after in transitions]
                if any(starts_here):
                    starts[row] = 1 if all(starts_here) else -1
                states = {after for _, after in transitions}
                if decision >= 0:
                    decisions[row] = all(state[2] for state in states)

            replayed[position] = replace(run, decisions=decisions, alert_start_status=starts)
            previous_end = edges[-1]
    return tuple(replayed), unknown_blocks
