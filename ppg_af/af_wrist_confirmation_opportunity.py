"""Count the AF episodes that could support an immediate or confirmed PPG decision."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .af_events import _coverage, _event_at_decision, _intervals, _union
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_spacing import LAGS
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned


DURATION_LABELS = ("up_to_10_s", "over_10_to_30_s", "over_30_to_60_s", "over_60_s")


def supports(grid, availability, quality):
    low_quality = quality < QUALITY_CUTOFF
    available = {"original": availability}
    for name, steps in LAGS.items():
        preceding, active = preceding_rows(grid, steps=steps)
        history_available = (preceding >= 0) & availability[np.maximum(preceding, 0)]
        available[name] = availability & ~(active & low_quality & ~history_available)
    return available, low_quality


def audit(grid, availability, quality, target):
    available, low_quality = supports(grid, availability, quality)
    blocks = defaultdict(list)
    for block in grid.timelines:
        blocks[block.subject].append(block)
    durations = []
    endpoints = {name: [] for name in available}
    for runs in blocks.values():
        episodes = _union(np.concatenate([_intervals(block.episodes) for block in runs]))
        rows = np.concatenate([block.rows for block in runs])
        end = np.concatenate([block.start + grid.seconds * np.arange(1, len(block.rows) + 1) for block in runs])
        reference = grid.table["reference_valid"][rows]
        spans = _union(np.column_stack((end[reference] - grid.seconds, end[reference])))
        whole, _ = _coverage(episodes, spans)
        if not whole.any():
            continue
        durations.extend(np.diff(episodes[whole], axis=1).ravel())
        event = _event_at_decision(end, episodes)
        eligible = reference & (event >= 0) & whole[np.maximum(event, 0)]
        end, event, rows = end[eligible], event[eligible], rows[eligible]
        time_since_onset = end - episodes[event, 0]
        for name, mask in available.items():
            timing = np.full((len(episodes), 3), np.inf)
            possible = mask[rows]
            started_in_episode = possible.copy()
            clean = possible & (time_since_onset >= WINDOW_SECONDS)
            if name != "original":
                lag = LAGS[name] * grid.seconds
                started_in_episode &= ~low_quality[rows] | (time_since_onset > lag)
                clean &= ~low_quality[rows] | (time_since_onset >= WINDOW_SECONDS + lag)
            for column, choice in enumerate((possible, started_in_episode, clean)):
                np.minimum.at(timing[:, column], event[choice], time_since_onset[choice])
            endpoints[name].extend(timing[whole])

    duration = np.asarray(durations)
    group = np.searchsorted([10, 30, 60], duration, side="left")
    columns = ("scoreable_endpoint", "required_prior_endpoint_within_same_AF_episode",
               "required_input_windows_wholly_within_AF")
    results = {}
    for name, rows in endpoints.items():
        times = np.asarray(rows).reshape(-1, len(columns))
        have = np.isfinite(times)
        count = {label: int(have[:, index].sum()) for index, label in enumerate(columns)}
        by_duration = {label: {"complete_episodes": int(np.sum(group == index)),
                               **{key: int(values.sum()) for key, values in zip(columns, have[group == index].T, strict=True)}}
                       for index, label in enumerate(DURATION_LABELS)}
        earliest = times[have[:, 0], 0]
        results[name] = {**count, "by_duration": by_duration,
                         "median_first_scoreable_endpoint_s": float(np.median(earliest)) if len(earliest) else None,
                         "reaches_target_under_scoreability_only": len(duration) > 0 and count[columns[0]] >= math.ceil(len(duration) * target)}
    return {"complete_episodes": len(duration), "target_needed_episodes": math.ceil(len(duration) * target), "routes": results}


def evaluate(args):
    grid, quality, _, protocol, _, _ = load_motion_development(args)
    dev = audit(grid, _shared_features(grid.table), quality["quality"][:, -1], protocol["target_sensitivity"])
    _read_pinned(args.lock, PROTOCOL_SHA)
    original_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    grid, _, available, _, quality, historical = historical_inputs(args, protocol, original_lock)
    historical = historical["independent"]
    held = audit(grid, available, quality["quality"][:, -1], protocol["target_sensitivity"])
    if (dev["complete_episodes"] != 17 or dev["routes"]["original"]["scoreable_endpoint"] != 16
            or held["complete_episodes"] != historical["complete_episodes"]
            or held["routes"]["original"]["scoreable_endpoint"]
            != historical["complete_episodes_with_shared_feature_endpoint"]):
        raise ValueError("Episode support must match the existing locked coverage counts")
    return {"scope": "Scoreability ceilings on unchanged ECG intervals, independent of alarm probabilities",
            "decision_stride_s": grid.seconds, "input_window_s": WINDOW_SECONDS, "quality_boundary": QUALITY_CUTOFF,
            "target_sensitivity": protocol["target_sensitivity"], "development": dev, "historical": held}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report",
                 "historical-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
