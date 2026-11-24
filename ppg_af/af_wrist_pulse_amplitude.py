"""Test whether optical pulse prominence adds interval-independent wrist AF information."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import peak_prominences
from scipy.stats import rankdata

from .af_wrist_features import PPG_RATE
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_pulse_consensus import _find
from .features import _bandpass, _plateaus, _prepare


MIN_PAIRED_PULSES = 5


def pulse_amplitude(signal):
    """Pair each accepted optical prominence with the preceding optical interval."""
    prepared = _prepare(signal, PPG_RATE, minimum_fs=4.0)
    if prepared is None:
        return np.zeros(2), 0
    adjusted, present = prepared
    usable = present & ~_plateaus(adjusted, present, PPG_RATE)
    wave = _bandpass(adjusted, PPG_RATE, 0.4, 8.0)
    peaks = _find(wave)
    peaks = peaks[usable[peaks]]
    prominence, _, _ = peak_prominences(wave, peaks, wlen=2 * PPG_RATE + 1)
    distance = np.diff(peaks)
    unsupported = np.diff(np.r_[0, np.cumsum(~usable)][peaks])
    accepted = ((unsupported <= 0.10 * distance) & (distance >= PPG_RATE / 4) & (distance <= 2.5 * PPG_RATE)
                & (prominence[1:] > 0))
    intervals = distance[accepted] / PPG_RATE
    log_amplitude = np.log(prominence[1:][accepted])
    if len(intervals) < MIN_PAIRED_PULSES:
        return np.zeros(2), len(intervals)
    low, high = np.quantile(log_amplitude, (0.25, 0.75))
    ranks = [rankdata(x) for x in (intervals, log_amplitude)]
    rho = float(np.corrcoef(*ranks)[0, 1]) if all(np.ptp(values) for values in ranks) else 0.0
    return np.array([high - low, rho]), len(intervals)


def summarize(features, support):
    chosen = support >= MIN_PAIRED_PULSES
    return {"windows": len(features), "windows_with_supported_prominence": int(chosen.sum()),
            "median_log_prominence_iqr": float(np.median(features[chosen, 0])) if chosen.any() else None,
            "median_spearman_preceding_interval": float(np.median(features[chosen, 1])) if chosen.any() else None}


def evaluate(args):
    grid, quality, paths, lock, digest, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)
    values, support = zip(*(pulse_amplitude(core) for core in cores[shared]), strict=True)
    extracted = np.stack(values)
    aligned = np.full((len(shared), extracted.shape[1]), np.nan)
    aligned[shared] = extracted
    baseline = quality["quality_interactions"]
    matrices = {"quality_interactions": baseline,
                "prominence_distribution": np.column_stack((baseline, aligned[:, 0])),
                "prominence_and_interval_coupling": np.column_stack((baseline, aligned))}
    report = json.loads(args.waveform_results.read_bytes())
    _, pulse, _, rhythm, _ = read_pairs(args.waveform_pairs, report, digest, set(grid.subjects))
    pairs, pair_counts = map(np.asarray, zip(*(pulse_amplitude(core) for core in pulse), strict=True))
    return {"scope": "Pulse prominence and preceding interval coupling on the original wrist AF development folds",
            "feature_fields": ["log_prominence_interquartile_range", "preceding_interval_spearman_correlation"],
            "development_optical_summary": summarize(extracted, np.asarray(support)),
            "paired_stable_rhythm_summary": {"AF": summarize(pairs[rhythm == 1], pair_counts[rhythm == 1]),
                                             "non_AF": summarize(pairs[rhythm == 0], pair_counts[rhythm == 0])},
            "comparisons": _compare(grid, matrices, lock, original)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "waveform-pairs", "waveform-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
