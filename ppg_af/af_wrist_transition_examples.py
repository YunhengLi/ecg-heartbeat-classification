"""Select ECG-labelled onset examples for development-only template fitting."""

from collections import defaultdict

import numpy as np

from .af_events import _TIME_EPSILON_S, _coverage, _intervals, _union
from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import _shared_features


STRIDE_SECONDS = 10
PRE_ONSET_SECONDS = 30
EARLY_AF_SECONDS = 20


def _original_fit_ends(grid, timelines, shared):
    eligible = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    ends = defaultdict(set)
    for block in timelines:
        endpoints = block.start + WINDOW_SECONDS * np.arange(1, len(block.rows) + 1)
        ends[block.subject].update(endpoints[eligible[block.rows]])
    return ends


def _row_support(grid, selected):
    counts = np.sort(np.unique(grid.subjects[selected], return_counts=True)[1])
    scopes, scope_counts = np.unique(grid.table["run_kind"][selected], return_counts=True)
    return {"windows": int(counts.sum()), "participants": len(counts),
            "windows_from_largest_contributor": int(counts[-1]) if len(counts) else 0,
            "windows_from_two_largest_contributors": int(counts[-2:].sum()),
            "windows_by_scoring_scope": {str(scope): int(count) for scope, count in zip(scopes, scope_counts, strict=True)}}


def _candidate_examples(grid, blocks, shared, original_ends, native_episodes):
    edges = [block.start + grid.seconds * np.arange(len(block.rows) + 1) for block in blocks]
    cells = np.concatenate([np.column_stack((edge[:-1], edge[1:])) for edge in edges])
    rows = np.concatenate([block.rows for block in blocks])
    order = np.argsort(cells[:, 0])
    cells, rows = cells[order], rows[order]
    starts, ends = cells.T
    if (np.any(starts[1:] < ends[:-1] - _TIME_EPSILON_S)
            or np.any(grid.table["stable_label"][rows] >= 0)):
        raise ValueError("Expected disjoint 10-second scoring rows without original fitting labels")

    reference = grid.table["reference_valid"][rows]
    coverage = _union(cells[reference])
    episodes = _union(np.concatenate([_intervals(block.episodes) for block in blocks]))
    complete = episodes[_coverage(episodes, coverage)[0]]
    native_complete = native_episodes[_coverage(native_episodes, coverage)[0]]
    if not np.array_equal(complete, native_complete):
        raise ValueError("Native ECG annotations changed the scorer's complete AF episodes")

    fully_referenced = _coverage(np.column_stack((ends - WINDOW_SECONDS, ends)), coverage)[0]
    original = np.array(sorted(original_ends))
    lower = np.searchsorted(original, ends - _TIME_EPSILON_S, side="left")
    upper = np.searchsorted(original, ends + _TIME_EPSILON_S, side="right")
    duplicate = lower < upper
    use = shared[rows] & reference & fully_referenced
    duplicate_examples = set()
    candidates = []
    for onset, stop in complete:
        in_pre = np.flatnonzero(use & (ends >= onset - PRE_ONSET_SECONDS) & (ends <= onset))
        verified_pre = []
        for index in in_pre[::-1]:
            end = ends[index]
            overlap = np.minimum(end, native_episodes[:, 1]) - np.maximum(end - WINDOW_SECONDS, native_episodes[:, 0])
            if np.maximum(0.0, overlap).sum() <= _TIME_EPSILON_S:
                verified_pre.append(index)

        in_pre = np.asarray(verified_pre, dtype=int)
        in_early = np.flatnonzero(use & (starts >= onset - _TIME_EPSILON_S)
                                  & (ends <= onset + EARLY_AF_SECONDS) & (ends <= stop))
        eligible = []
        for indices in (in_pre, in_early):
            duplicate_examples.update(rows[indices[duplicate[indices]]])
            eligible.append(rows[indices[~duplicate[indices]]])
        candidates.append(tuple(eligible))
    return candidates, duplicate_examples


def select_transition_examples(grid, original_timelines, raw_dir):
    """Return row-aligned supplemental labels and an aggregate-only support report.

    Each complete ECG episode supplies at most one row per class. The
    reference-supported input ends within 30 seconds before or at onset for
    non-AF, or within the first 20 seconds of AF for the early-AF class. Early
    AF requires the last 10 seconds to be AF and may retain earlier non-AF.
    Among rows passing the original shared feature gate, take the latest pre-
    onset non-AF row and the earliest early-AF row. Skip original fitting
    endpoints and resolve reused candidates in episode order, prioritizing
    episodes with an early-AF candidate for pre-onset pairing.
    Call with the locked development grid and the original 30-second timelines.
    """
    if grid.seconds != STRIDE_SECONDS:
        raise ValueError("Transition labels require the locked 10-second scoring grid")
    shared = _shared_features(grid.table)
    original_ends = _original_fit_ends(grid, original_timelines, shared)
    blocks_by_subject = defaultdict(list)
    for block in grid.timelines:
        blocks_by_subject[block.subject].append(block)

    labels = np.full(len(grid.subjects), -1, dtype=np.int8)
    paired = pre_only = early_only = unsupported = original_duplicates = 0
    for subject, blocks in blocks_by_subject.items():
        with np.load(raw_dir / f"{subject}.npz", allow_pickle=False) as archive:
            native_episodes = _union(_intervals(archive["af_intervals_s"]))
        candidates, duplicates = _candidate_examples(grid, blocks, shared, original_ends[subject], native_episodes)
        original_duplicates += len(duplicates)
        # Give episodes with an early-AF candidate priority for shared pre-onset rows.
        for pre, early in sorted(candidates, key=lambda pair: len(pair[1]) == 0):
            negative = next((row for row in pre if labels[row] == -1), None)
            positive = next((row for row in early if labels[row] == -1), None)
            if negative is not None:
                labels[negative] = 0
            if positive is not None:
                labels[positive] = 1
            paired += negative is not None and positive is not None
            pre_only += negative is not None and positive is None
            early_only += positive is not None and negative is None
            unsupported += negative is None and positive is None

    return labels, {
        "feature_window_seconds": WINDOW_SECONDS,
        "decision_stride_seconds": grid.seconds,
        "pre_onset_search_seconds": PRE_ONSET_SECONDS,
        "early_AF_search_seconds": EARLY_AF_SECONDS,
        "minimum_trailing_AF_seconds": grid.seconds,
        "complete_reference_episodes": paired + pre_only + early_only + unsupported,
        "episodes_with_both_examples": paired,
        "episodes_with_pre_onset_example_only": pre_only,
        "episodes_with_early_AF_example_only": early_only,
        "episodes_without_examples": unsupported,
        "unique_eligible_candidates_duplicating_original_fit_endpoints": original_duplicates,
        "pre_onset_non_AF": _row_support(grid, labels == 0),
        "early_AF": _row_support(grid, labels == 1),
        "all_supplemental": _row_support(grid, labels >= 0),
    }
