"""Test AF information in the order of successive pulse intervals."""

import argparse
from pathlib import Path

import numpy as np

from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_smoke import fit_ptt_calibration
from .features import _ecg_peaks, _prepare


TURN_SECONDS = 0.05
SHORT_LONG_FRACTION = 0.10
PAIR_SUM_FRACTION = 0.15


def order_features(intervals, supported):
    """Return observed features and their exact expectation under random order."""
    usable = intervals[supported]
    count = len(usable)
    center = np.median(usable) if count else 1.0
    triplets = supported[:-2] & supported[1:-1] & supported[2:]
    pairs = supported[:-1] & supported[1:]

    def short_long(first, second):
        extremes = (np.minimum(first, second) < (1 - SHORT_LONG_FRACTION) * center) & (
            np.maximum(first, second) > (1 + SHORT_LONG_FRACTION) * center)
        return extremes & (np.abs(first + second - 2 * center) <= PAIR_SUM_FRACTION * center)

    rising = np.diff(intervals) > TURN_SECONDS
    falling = np.diff(intervals) < -TURN_SECONDS
    turns = (rising[:-1] & falling[1:]) | (falling[:-1] & rising[1:])
    observed = np.array([np.mean(turns[triplets]) if triplets.any() else 0.0,
                         np.mean(short_long(intervals[:-1][pairs], intervals[1:][pairs])) if pairs.any() else 0.0])
    difference = usable[:, None] - usable[None, :]
    lower = np.sum(difference > TURN_SECONDS, axis=1)
    upper = np.sum(difference < -TURN_SECONDS, axis=1)
    null = np.array([
        np.sum(lower * (lower - 1) + upper * (upper - 1)) / (count * (count - 1) * (count - 2)) if count >= 3 else 0.0,
        np.sum(short_long(usable[:, None], usable[None, :])) / (count * (count - 1)) if count >= 2 else 0.0,
    ])
    return observed, null


def interval_order_from_peaks(peaks, present):
    peaks = peaks[present[peaks]]
    intervals = np.diff(peaks)
    missing = np.r_[0, np.cumsum(~present)]
    valid = np.diff(missing[peaks]) <= 0.10 * intervals
    return order_features(intervals / PPG_RATE, valid)


def template_order(cores, template):
    pad = 2 * PPG_RATE
    rows = np.full((len(cores), 2, 2), np.nan)
    for index, pulse in enumerate(cores):
        estimated = template.predict(np.pad(pulse, (pad, 0), mode="reflect"))[pad:]
        prepared = _prepare(estimated, PPG_RATE, minimum_fs=15.0)
        if prepared is None:
            continue
        waveform, present = prepared
        rows[index] = interval_order_from_peaks(_ecg_peaks(waveform, PPG_RATE), present)
    return rows


def evaluate(args):
    grid, quality, paths, lock, _, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    ppg = stage_cores(grid, paths)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    extracted = template_order(ppg[shared], template)
    if not np.isfinite(extracted).all():
        raise ValueError("The rhythm order comparison must preserve original template feature availability")
    aligned = np.full((len(shared), 2, 2), np.nan)
    aligned[shared] = extracted
    baseline = quality["quality_interactions"]
    matrices = {"quality_interactions": baseline,
                "ordering_expected_from_distribution": np.column_stack((baseline, aligned[:, 1])),
                "observed_interval_order": np.column_stack((baseline, aligned[:, 0]))}
    return {"scope": "Add two interval order features to template AF with a distribution-only control",
            "scoreable_windows": int(shared.sum()), "features": ["turns_above_50_ms", "balanced_short_long_pairs"],
            "median_observed": np.median(extracted[:, 0], axis=0).tolist(),
            "median_expected_after_permutation": np.median(extracted[:, 1], axis=0).tolist(),
            "comparisons": _compare(grid, matrices, lock, original)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
