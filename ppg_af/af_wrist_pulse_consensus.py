"""Assess interval reliability when pulse peaks must agree across two filters."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .af_wrist_false_start_audit import native_irregularity
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_quality_study import quality_interaction_features
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .features import ECG_FEATURE_NAMES, _bandpass, _interval_features, _plateaus, _prepare


FILTER_LIMITS = (8.0, 4.0)
AGREEMENT_MS = 80
AGREEMENT_SAMPLES = round(AGREEMENT_MS * PPG_RATE / 1000)
VALID = ECG_FEATURE_NAMES.index("valid_interval_fraction")
RMSSD = ECG_FEATURE_NAMES.index("successive_interval_rmssd_over_median")
PNN50 = ECG_FEATURE_NAMES.index("successive_interval_pnn50")


def _find(filtered):
    lower, upper = np.percentile(filtered, (5, 95))
    return find_peaks(filtered, distance=round(0.25 * PPG_RATE), width=round(0.05 * PPG_RATE),
                      prominence=max(np.finfo(float).eps, 0.18 * (upper - lower)))[0]


def corroborated_peaks(primary, secondary):
    """The matching tolerance is below half the minimum peak spacing."""
    nearby = np.searchsorted(secondary, primary - AGREEMENT_SAMPLES)
    right = np.r_[secondary, np.iinfo(np.int64).max]
    return right[nearby] <= primary + AGREEMENT_SAMPLES


def pulse_consensus(signal):
    prepared = _prepare(signal, PPG_RATE, minimum_fs=4.0)
    if prepared is None:
        empty = np.zeros(len(ECG_FEATURE_NAMES))
        return empty, empty.copy()
    values, present = prepared
    usable = present & ~_plateaus(values, present, PPG_RATE)
    wide, narrow = (_find(_bandpass(values, PPG_RATE, 0.4, high)) for high in FILTER_LIMITS)
    verified = corroborated_peaks(wide, narrow)
    raw = _interval_features(wide, PPG_RATE, usable)
    consensus = _interval_features(wide, PPG_RATE, usable, verified_peaks=verified)
    return raw, consensus


def _paired_diagnostics(args, grid, lock_digest):
    report = json.loads(args.waveform_results.read_bytes())
    _, pulse, _, labels, reference = read_pairs(args.waveform_pairs, report, lock_digest, set(grid.subjects))
    reference_score = np.asarray([native_irregularity(annotations / PPG_RATE, WINDOW_SECONDS) for annotations in reference])
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    estimated = _extract(pulse, template)["baseline"]
    raw, robust = np.stack([pulse_consensus(window) for window in pulse]).transpose(1, 0, 2)
    routes = {"template": estimated, "raw_pulse": raw, "consensus_pulse": robust}
    common = np.isfinite(reference_score) & np.logical_and.reduce([values[:, VALID] > 0 for values in routes.values()])

    def compare(group):
        paired = common & group
        if not paired.any():
            raise ValueError("Both reference rhythms need paired signal support")
        summary = {}
        for name, values in routes.items():
            score = values[paired, RMSSD] + values[paired, PNN50]
            delta = score - reference_score[paired]
            summary[name] = {"median_absolute_irregularity_error": float(np.median(np.abs(delta))),
                             "median_irregularity_bias": float(np.median(delta)),
                             "median_accepted_interval_fraction": float(np.median(values[paired, VALID]))}
        return {"stable_windows": int(group.sum()), "matched_windows": int(paired.sum()), "methods": summary}

    return {"overall": compare(np.ones(len(pulse), dtype=bool)), "stable_AF": compare(labels == 1),
            "stable_non_AF": compare(labels == 0)}


def evaluate(args):
    grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)
    raw, consensus = np.stack([pulse_consensus(window) for window in cores[shared]]).transpose(1, 0, 2)
    if not np.array_equal(raw, grid.table["direct_ppg"][shared, :len(ECG_FEATURE_NAMES)]):
        raise ValueError("The broad-band detector must reproduce the original direct pulse interval features")
    original_input = quality["baseline"]
    raw_matrix, robust_matrix = original_input.copy(), original_input.copy()
    raw_matrix[shared] = raw
    indices = np.flatnonzero(shared)
    robust = consensus[:, VALID] > 0
    robust_matrix[indices[robust]] = consensus[robust]
    q = quality["quality"][:, -1]
    variants = {"quality_interactions": quality["quality_interactions"],
                "raw_intervals_with_quality": quality_interaction_features(raw_matrix, q),
                "consensus_intervals_with_quality": quality_interaction_features(robust_matrix, q)}
    return {"scope": "Compare two-filter optical timing against native ECG and grouped AF alerts",
            "filter_high_cutoffs_hz": FILTER_LIMITS, "peak_agreement_ms": AGREEMENT_MS,
            "original_wrist_scoreable_windows": len(indices), "consensus_interval_windows": int(robust.sum()),
            "template_fallback_windows": int((~robust).sum()), "paired_stable_diagnostics": _paired_diagnostics(args, grid, lock_digest),
            "alert_comparisons": _compare(grid, variants, lock, original)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
