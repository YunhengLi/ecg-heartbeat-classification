"""Replay confidence-aware alert entry and release on the existing AF clock."""

from dataclasses import replace
from enum import IntEnum

import numpy as np

from .af_events import _TIME_EPSILON_S, _grid
from .af_wrist_clock_study import _summarize, scoring_runs
from .af_wrist_history import _grouped_positions
from .af_wrist_operating import score_decisions
from .af_wrist_vote_study import _IMMEDIATE, _held_out_replay


class _Level(IntEnum):
    UNAVAILABLE = -1
    NEGATIVE = 0
    SHALLOW_NEGATIVE = 1
    BORDERLINE_POSITIVE = 2
    STRONG_POSITIVE = 3


class _State(IntEnum):
    INACTIVE = 0
    PENDING_POSITIVE = 1
    ACTIVE = 2
    HELD_SHALLOW_NEGATIVE = 3


_ACTIVE_STATES = (_State.ACTIVE, _State.HELD_SHALLOW_NEGATIVE)


def _advance(state, level):
    if level == _Level.STRONG_POSITIVE:
        return _State.ACTIVE
    if level == _Level.BORDERLINE_POSITIVE:
        return _State.PENDING_POSITIVE if state == _State.INACTIVE else _State.ACTIVE
    if level == _Level.SHALLOW_NEGATIVE and state == _State.ACTIVE:
        return _State.HELD_SHALLOW_NEGATIVE
    return _State.INACTIVE


def _levels(scores, cutoffs, entry_margin, release_margin):
    known = np.isfinite(scores) & np.isfinite(cutoffs)
    levels = np.full(scores.shape, _Level.UNAVAILABLE, dtype=np.int8)
    levels[known] = _Level.NEGATIVE
    positive = known & (scores >= cutoffs)
    shallow = known & ~positive & (scores >= cutoffs - release_margin)
    levels[shallow] = _Level.SHALLOW_NEGATIVE
    levels[positive] = _Level.BORDERLINE_POSITIVE
    levels[positive & (scores >= cutoffs + entry_margin)] = _Level.STRONG_POSITIVE
    return levels


def _unknown_levels(cutoff, entry_margin, release_margin):
    available = [_Level.UNAVAILABLE]
    if np.isfinite(cutoff):
        if cutoff > 0 and cutoff - release_margin < cutoff and cutoff - release_margin <= 1:
            available.append(_Level.SHALLOW_NEGATIVE)
        if cutoff <= 1 and cutoff + entry_margin > max(cutoff, 0):
            available.append(_Level.BORDERLINE_POSITIVE)
        if cutoff + entry_margin <= 1:
            available.append(_Level.STRONG_POSITIVE)
    return available


def _history_states(levels, cutoffs, unknown, entry_margin, release_margin):
    if not len(levels):
        return {_State.INACTIVE}
    possible = _unknown_levels(cutoffs[0], entry_margin, release_margin)
    states = {_State.INACTIVE}
    if _Level.BORDERLINE_POSITIVE in possible:
        states.add(_State.PENDING_POSITIVE)
    if _Level.BORDERLINE_POSITIVE in possible or _Level.STRONG_POSITIVE in possible:
        states.add(_State.ACTIVE)
        if _Level.SHALLOW_NEGATIVE in possible:
            states.add(_State.HELD_SHALLOW_NEGATIVE)
    for level, cutoff, unobserved in zip(levels, cutoffs, unknown, strict=True):
        outcomes = _unknown_levels(cutoff, entry_margin, release_margin) if unobserved else (level,)
        states = {_advance(state, outcome) for state in states for outcome in outcomes}
    return states


def _replay_run(run, levels, states):
    output = np.full(len(levels), -1, dtype=np.int8)
    starts = np.zeros(len(levels), dtype=np.int8)
    for row, level in enumerate(levels):
        transitions = [(state, _advance(state, level)) for state in states]
        starts_here = [before not in _ACTIVE_STATES and after in _ACTIVE_STATES
                       for before, after in transitions]
        if any(starts_here):
            starts[row] = 1 if all(starts_here) else -1
        states = {after for _, after in transitions}
        if level != _Level.UNAVAILABLE:
            output[row] = all(state in _ACTIVE_STATES for state in states)
    replayed = replace(run, decisions=output, alert_start_status=starts)
    _grid(replayed)
    return replayed, states


def _validate(scores, entry_margin, release_margin):
    if any(not np.isfinite(value) or value < 0 for value in (entry_margin, release_margin)):
        raise ValueError("Entry and release margins must be finite and nonnegative")
    finite = scores[np.isfinite(scores)]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError("Alert scores must be probabilities")


def replay_score_block(run, scores, prior_scores, cutoff, *, entry_margin=0.0, release_margin=0.0):
    """Replay a disjoint block with scores ordered on its original clock.

    History scores are in chronological order. None marks unobserved history
    and NaN marks known unavailable input. For a nonzero margin, an empty
    history starts inactive. Output positives and starts require agreement
    across all feasible states. Start status -1 denotes an ambiguous start.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(run.decisions),):
        raise ValueError("Scores must align with all decisions in the block")
    history = tuple(prior_scores)
    prior = np.asarray(history, dtype=float)
    cutoff = float(cutoff)
    _validate(np.concatenate((scores, prior)), entry_margin, release_margin)
    if entry_margin == release_margin == 0:
        raw = np.where(np.isfinite(scores) & np.isfinite(cutoff), scores >= cutoff, -1).astype(np.int8)
        previous = [None if score is None else -1 if not np.isfinite(score) or not np.isfinite(cutoff)
                    else int(score >= cutoff) for score in history]
        (replayed,), unknown = _IMMEDIATE((replace(run, decisions=raw),), prior_decisions=lambda _: previous)
        return replayed, unknown

    cutoffs = np.full(len(prior), cutoff)
    prior_levels = _levels(prior, cutoffs, entry_margin, release_margin)
    states = _history_states(prior_levels, cutoffs, [score is None for score in history], entry_margin, release_margin)
    levels = _levels(scores, np.full(len(scores), cutoff), entry_margin, release_margin)
    replayed, _ = _replay_run(run, levels, states)
    return replayed, int(len(states) > 1)


def replay_alert_state(grid, scores, cutoffs, *, entry_margin=0.0, release_margin=0.0):
    """Score confidence-aware alert state on the frozen evaluation grids.

    A strong positive starts immediately. A borderline positive starts after
    another quality-filtered positive, or continues an active alert. A single
    shallow negative may hold an active alert for one decision. Gaps clear
    state and preserve their unavailable decision. Adjacent runs share state.
    """
    scores, cutoffs = np.asarray(scores, dtype=float), np.asarray(cutoffs, dtype=float)
    if scores.shape != cutoffs.shape or scores.shape != (len(grid.subjects),):
        raise ValueError("Scores and cutoffs must align with all grid rows")
    _validate(scores, entry_margin, release_margin)
    if entry_margin == release_margin == 0:
        return _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    levels = _levels(scores, cutoffs, entry_margin, release_margin)
    raw = np.where(levels == _Level.UNAVAILABLE, np.nan, levels >= _Level.BORDERLINE_POSITIVE)
    monitor, episodes = scoring_runs(grid, raw, set(grid.subjects))
    rows = {(block.subject, block.start): block.rows for block in grid.timelines}

    def replay(scored_runs):
        original = tuple(run.decisions(0.5) for run in scored_runs)
        output = list(original)
        unknown_blocks = 0
        for positions in _grouped_positions(original):
            previous_end = -np.inf
            for position in sorted(positions, key=lambda item: original[item].start_s):
                run = original[position]
                edges, _, _, _ = _grid(run)
                if edges[0] < previous_end - _TIME_EPSILON_S:
                    raise ValueError("Runs for the same subject must not overlap")
                key = run.subject_id, run.start_s
                if abs(edges[0] - previous_end) > _TIME_EPSILON_S:
                    history = grid.history_rows[key]
                    states = _history_states(levels[history], cutoffs[history], grid.unknown_source[history],
                                             entry_margin, release_margin)
                    unknown_blocks += len(states) > 1
                output[position], states = _replay_run(run, levels[rows[key]], states)
                previous_end = edges[-1]
        return tuple(output), unknown_blocks

    wrist, monitor_unknown = replay(monitor)
    reference, episode_unknown = replay(episodes)
    metrics = score_decisions(wrist, episode_runs=reference)
    summary = _summarize({**metrics, "monitor_blocks_with_unknown_cutoff_state": monitor_unknown,
                          "episode_blocks_with_unknown_cutoff_state": episode_unknown})
    return summary, reference, wrist
