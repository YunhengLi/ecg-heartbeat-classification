"""Test whether a lower optical peak prominence recovers missing ECG-timed beats."""

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
from .af_wrist_native_beat_transfer import read_pairs, summarize
from .af_wrist_pulse_consensus import VALID, interval_diagnostics
from .af_wrist_quality_study import quality_interaction_features
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS
from .features import ECG_FEATURE_NAMES, _bandpass, _interval_features, _plateaus, _prepare


PROMINENCE = {"original_detector": 0.18, "half_prominence": 0.09}
GAP_ROUTE = "weak_pulse_inside_long_gap"
ROUTES = (*PROMINENCE, GAP_ROUTE)
GAP_FRACTIONS = (1.5, 2.5)
SPLIT_INTERVAL_FRACTIONS = (0.6, 1.4)


def insert_gap_peaks(strong, weak):
    """Use one observed weak candidate when it plausibly divides a long optical gap."""
    if len(strong) < 4:
        return strong
    interval = np.median(np.diff(strong))
    lower, upper = np.asarray(SPLIT_INTERVAL_FRACTIONS) * interval
    smallest, largest = np.asarray(GAP_FRACTIONS) * interval
    inserted = []
    for left, right in zip(strong[:-1], strong[1:], strict=True):
        if not smallest <= right - left <= largest:
            continue
        inside = weak[(weak >= max(left + lower, right - upper)) & (weak <= min(left + upper, right - lower))]
        if len(inside):
            inserted.append(inside[np.argmin(np.abs(2 * inside - left - right))])
    return np.sort(np.r_[strong, np.asarray(inserted, dtype=int)])


def pulse_routes(signal):
    prepared = _prepare(signal, PPG_RATE, minimum_fs=4.0)
    if prepared is None:
        return {name: (np.empty(0, dtype=int), np.zeros(len(ECG_FEATURE_NAMES))) for name in ROUTES}
    values, present = prepared
    usable = present & ~_plateaus(values, present, PPG_RATE)
    wave = _bandpass(values, PPG_RATE, 0.4, 8.0)
    lower, upper = np.percentile(wave, (5, 95))
    output = {}
    for method, cutoff in PROMINENCE.items():
        peaks = find_peaks(wave, distance=round(0.25 * PPG_RATE), width=round(0.05 * PPG_RATE),
                          prominence=max(np.finfo(float).eps, cutoff * (upper - lower)))[0]
        output[method] = peaks[usable[peaks]], _interval_features(peaks, PPG_RATE, usable)
    recovered = insert_gap_peaks(output["original_detector"][0], output["half_prominence"][0])
    output[GAP_ROUTE] = recovered, _interval_features(recovered, PPG_RATE, usable)
    return output


def compare_measured(args, grid, lock_digest):
    source = json.loads(args.waveform_results.read_bytes())
    people, pulse, clock, rhythm, reference = read_pairs(args.waveform_pairs, source, lock_digest, set(grid.subjects))
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    assumed = round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000)
    delays = np.where(clock == "beat", assumed, template.delay_samples)
    observations = [pulse_routes(row) for row in pulse]
    timings = {name: [row[name][0] - delay for row, delay in zip(observations, delays, strict=True)] for name in ROUTES}
    features = {name: np.stack([row[name][1] for row in observations]) for name in ROUTES}
    measured = np.asarray([native_irregularity(peaks / PPG_RATE, WINDOW_SECONDS) for peaks in reference])
    return {"beat_matching": summarize(reference, timings, people, clock, rhythm),
            "combined_interval_irregularity_error": interval_diagnostics(measured, rhythm, features),
            "archive_delay_used_on_beat_aligned_rows_ms": ARCHIVE_ASSUMED_DELAY_MS,
            "external_template_delay_used_on_coarse_rows_ms": 1000 * template.delay_samples / PPG_RATE}


def evaluate(args):
    grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    aligned = stage_cores(grid, paths)
    native = [pulse_routes(row) for row in aligned[shared]]
    broad = np.stack([row["original_detector"][1] for row in native])
    expected = grid.table["direct_ppg"][shared, :len(ECG_FEATURE_NAMES)]
    if not np.array_equal(broad, expected):
        raise ValueError("The original broad detector must reproduce the locked direct PPG timing exactly")
    options = {"quality_interactions": quality["quality_interactions"]}
    supports = {}
    for name in ROUTES:
        values = np.stack([row[name][1] for row in native])
        usable = values[:, VALID] > 0
        matrix = quality["baseline"].copy()
        matrix[np.flatnonzero(shared)[usable]] = values[usable]
        options[name] = quality_interaction_features(matrix, quality["quality"][:, -1])
        supports[name] = {"supported": int(usable.sum()), "template_fallback": int((~usable).sum())}
    return {"scope": "Verify a fixed lower wrist peak prominence against paired ECG and held-out AF alerts",
            "prominence_relative_to_bandpassed_5th_to_95th_percentile_range": PROMINENCE,
            "long_gap_fractions_of_median_interval": GAP_FRACTIONS,
            "required_split_fractions_of_median_interval": SPLIT_INTERVAL_FRACTIONS,
            "total_observed_peaks_inserted_in_long_gaps": sum(len(row[GAP_ROUTE][0]) - len(row["original_detector"][0]) for row in native),
            "native_wrist_waveform_comparison": compare_measured(args, grid, lock_digest), "alert_input_support": supports,
            "alert_comparisons": _compare(grid, options, lock, original)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "ptt-dir", "waveform-pairs", "waveform-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
