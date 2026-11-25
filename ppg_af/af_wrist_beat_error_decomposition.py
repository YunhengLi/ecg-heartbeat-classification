"""Separate beat timing and event count errors on sealed wrist and ECG pairs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_beat_cnn import EDGE, SAMPLES
from .af_wrist_beat_opportunity import BOOTSTRAPS, SEED, TOLERANCE_MS
from .af_wrist_exercise import PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS
from .af_wrist_weak_pulses import pulse_routes
from .afppgecg import ECG_FS
from .features import ECG_FEATURE_NAMES, _interval_features
from .tcn_metrics import match_beats


TOLERANCES_MS = (65, int(TOLERANCE_MS), 150)
START = round(EDGE * ECG_FS / PPG_RATE)
STOP = round((SAMPLES - EDGE) * ECG_FS / PPG_RATE)
ROUTES = ("original_optical", "matched_timing_repaired", "event_occurrence_repaired", "both_repaired")
METRICS = ("median_rr_ms", "normalized_rmssd", "pnn50", "combined_irregularity")
FEATURES = tuple(ECG_FEATURE_NAMES.index(name) for name in (
    "median_interval_s", "successive_interval_rmssd_over_median", "successive_interval_pnn50",
))
SUPPORT = np.ones(STOP - START, dtype=bool)


def match_indices(reference, optical, tolerance_ms):
    """Return the corresponding indices under the published chronological match rule."""
    pairs = []
    truth = observed = 0
    tolerance = tolerance_ms * ECG_FS / 1000
    while truth < len(reference) and observed < len(optical):
        delta = optical[observed] - reference[truth]
        if abs(delta) <= tolerance:
            pairs.append((truth, observed))
        truth += delta >= -tolerance
        observed += delta <= tolerance
    return np.asarray(pairs, dtype=int).reshape(-1, 2).T


def repair_sequences(reference, optical, matched_reference, matched_optical):
    corrected_times = optical.copy()
    corrected_times[matched_optical] = reference[matched_reference]
    reference_missing = np.ones(len(reference), dtype=bool)
    reference_missing[matched_reference] = False
    occurrence = np.sort(np.r_[optical[matched_optical], reference[reference_missing]])
    joint = np.sort(np.r_[corrected_times[matched_optical], reference[reference_missing]])
    if not np.array_equal(joint, reference):
        raise ValueError("The joint repair must reconstruct the reference beat sequence exactly")
    return optical, np.sort(corrected_times), occurrence, joint


def interval_statistics(peaks):
    if len(peaks) < 3 or np.any(np.diff(peaks) <= 0):
        return np.full(len(METRICS), np.nan)
    median_rr, normalized_rmssd, _ = _interval_features(peaks - START, ECG_FS, SUPPORT)[list(FEATURES)]
    # Compare integer sample differences so an interval change of exactly 50 ms is excluded.
    pnn50 = np.mean(np.abs(np.diff(peaks, n=2)) > ECG_FS * 0.05)
    return np.asarray((1000 * median_rr, normalized_rmssd, pnn50, normalized_rmssd + pnn50))


def compare_pair(reference, optical, tolerance_ms):
    matched_reference, matched_optical = match_indices(reference, optical, tolerance_ms)
    if len(matched_reference) != len(match_beats(reference, optical, ECG_FS, tolerance_ms).errors_ms):
        raise ValueError("The repair matches disagree with the published beat matcher")
    repaired = repair_sequences(reference, optical, matched_reference, matched_optical)
    statistics = np.stack([interval_statistics(peaks) for peaks in repaired])
    differences = statistics - interval_statistics(reference)
    consecutive = (np.diff(matched_reference) == 1) & (np.diff(matched_optical) == 1)
    rr_error = np.diff(optical[matched_optical]) - np.diff(reference[matched_reference])
    matched_rr_errors = np.abs(rr_error[consecutive]) * 1000 / ECG_FS
    return differences, (len(reference), len(optical), len(matched_reference)), matched_rr_errors


def paired_summary(delta, people, draws):
    estimates = np.stack([delta[people == person].mean(axis=0) for person in np.unique(people)])
    intervals = (np.quantile(estimates[draws].mean(axis=1), (0.025, 0.975), axis=0).T
                 if len(estimates) >= 3 else [None] * len(METRICS))
    return {metric: {"participant_weighted_mean": float(point),
                     "participant_bootstrap_95_interval": interval.tolist() if interval is not None else None}
            for metric, point, interval in zip(METRICS, estimates.mean(axis=0), intervals, strict=True)}


def summarize(selected, people, differences, counts, matched_rr_errors):
    common = selected & np.isfinite(differences).all(axis=(1, 2))
    truth, optical, matched = counts[selected].sum(axis=0)
    report = {"windows": int(selected.sum()), "participants": len(np.unique(people[selected])),
              "common_interval_feature_windows": int(common.sum()),
              "common_interval_feature_participants": len(np.unique(people[common])),
              "ecg_beats": int(truth), "optical_beats": int(optical), "paired_beats": int(matched),
              "unmatched_ecg_beats": int(truth - matched), "unmatched_optical_beats": int(optical - matched)}
    if not common.any():
        return report
    scores = differences[common]
    absolute = np.abs(scores)
    individuals = people[common]
    distinct = np.unique(individuals)
    draws = np.random.default_rng(SEED).integers(len(distinct), size=(BOOTSTRAPS, len(distinct)))
    intervals = np.concatenate([errors for errors, include in zip(matched_rr_errors, common, strict=True) if include])
    report["matched_consecutive_intervals"] = len(intervals)
    report["matched_consecutive_interval_median_absolute_error_ms"] = float(np.median(intervals)) if len(intervals) else None
    report["routes"] = {}
    for index, route in enumerate(ROUTES):
        report["routes"][route] = {
            "pooled_median_absolute_error": dict(zip(METRICS, np.median(absolute[:, index], axis=0).tolist(), strict=True)),
            "pooled_median_signed_error": dict(zip(METRICS, np.median(scores[:, index], axis=0).tolist(), strict=True)),
            "paired_absolute_error_change_from_original": paired_summary(absolute[:, index] - absolute[:, 0], individuals, draws),
        }
    report["occurrence_minus_timing_absolute_error"] = paired_summary(absolute[:, 2] - absolute[:, 1], individuals, draws)
    return report


def evaluate(args):
    lock_bytes = args.lock.read_bytes()
    lock_digest = sha256(lock_bytes).hexdigest()
    if lock_digest != PROTOCOL_SHA:
        raise ValueError("Use the locked original wrist protocol")
    lock = json.loads(lock_bytes)
    roster = _subject_ids(lock["fit_subjects"]) | _subject_ids(lock["calibration_subjects"])
    report = json.loads(args.waveform_results.read_bytes())
    people, ppg, clock, labels, reference = read_pairs(args.waveform_pairs, report, lock_digest, roster)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    beat_delay = round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000)
    delays = np.where(clock == "beat", beat_delay, template.delay_samples)

    def trimmed_native(peaks):
        samples = np.rint(peaks * ECG_FS / PPG_RATE).astype(np.int64)
        return samples[(samples >= START) & (samples < STOP)]

    truth = [trimmed_native(peaks) for peaks in reference]
    optical = [trimmed_native(pulse_routes(signal)["original_detector"][0] - delay)
               for signal, delay in zip(ppg, delays, strict=True)]
    beat_aligned = clock == "beat"
    masks = {"beat_aligned_all": beat_aligned, "beat_aligned_stable_af": beat_aligned & (labels == 1),
             "beat_aligned_stable_non_af": beat_aligned & (labels == 0), "coarse_clock_all": clock == "coarse"}
    comparisons = {}
    for tolerance in TOLERANCES_MS:
        differences, counts, rr_errors = zip(*(compare_pair(actual, estimated, tolerance)
                                               for actual, estimated in zip(truth, optical, strict=True)), strict=True)
        comparisons[str(tolerance)] = {name: summarize(mask, people, np.stack(differences), np.asarray(counts), rr_errors)
                                       for name, mask in masks.items()}
    return {
        "scope": "Reference-assisted diagnostic on stable development windows",
        "protocol_sha256": lock_digest,
        "paired_manifest_sha256": report["private_paired_manifest_sha256"],
        "primary_tolerance_ms": TOLERANCE_MS,
        "edge_trim_seconds": EDGE / PPG_RATE,
        "beat_aligned_archive_delay_ms": ARCHIVE_ASSUMED_DELAY_MS,
        "coarse_archive_external_ptt_delay_ms": 1000 * template.delay_samples / PPG_RATE,
        "native_comparison_sample_rate_hz": ECG_FS,
        "paired_windows": len(people), "participants": len(np.unique(people)),
        "participant_bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": SEED,
        "comparisons_by_tolerance_ms": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
