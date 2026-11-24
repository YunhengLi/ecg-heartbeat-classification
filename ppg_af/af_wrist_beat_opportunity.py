"""Audit ECG beat opportunities in wrist extrema against a matched-density null."""

import argparse
import json
from pathlib import Path

import numpy as np

from .af_wrist_beat_cnn import EDGE, SAMPLES
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _write_json
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS
from .af_wrist_weak_pulses import GAP_ROUTE, pulse_routes
from .tcn_metrics import match_beats


TOLERANCE_MS = 100.0
ROTATIONS = 32
BOOTSTRAPS = 10_000
SEED = 2026


def optical_candidates(signal):
    found = pulse_routes(signal)
    original, weak, gap = (found[key][0] for key in ("original_detector", "half_prominence", GAP_ROUTE))
    troughs = pulse_routes(-signal)["half_prominence"][0]
    return {"original_peaks": original, "weak_peaks": weak, "weak_peaks_only_in_gaps": gap,
            "weak_peaks_and_troughs": np.union1d(weak, troughs)}


def assess(reference, candidates, rotations):
    stop = SAMPLES - EDGE
    actual = reference[(reference >= EDGE) & (reference < stop)]
    optical = candidates[(candidates >= EDGE) & (candidates < stop)]
    observed = match_beats(actual, optical, PPG_RATE, TOLERANCE_MS)
    chance = np.mean([len(match_beats(actual, np.sort(EDGE + (optical - EDGE + shift) % (stop - EDGE)),
                                     PPG_RATE, TOLERANCE_MS).errors_ms) for shift in rotations])
    return len(actual), len(optical), len(observed.errors_ms), chance


def summarize_people(counts, participants):
    reference, candidates, matches, expected = counts.sum(axis=0)
    people = np.unique(participants)
    advantage = []
    for person in people:
        truth, _, observed, chance = counts[participants == person].sum(axis=0)
        if truth:
            advantage.append((observed - chance) / truth)
    difference = np.asarray(advantage)
    interval = None
    if len(difference) >= 3:
        random = np.random.default_rng(SEED)
        draws = random.integers(len(difference), size=(BOOTSTRAPS, len(difference)))
        interval = np.quantile(difference[draws].mean(axis=1), (0.025, 0.975)).tolist()
    return {"reference_beats": int(reference), "optical_candidates": int(candidates), "matched_beats": int(matches),
            "observed_candidate_recall": float(matches / reference) if reference else None,
            "density_matched_rotation_recall": float(expected / reference) if reference else None,
            "mean_within_participant_recall_advantage": float(difference.mean()) if len(difference) else None,
            "participant_bootstrap_95_interval": interval}


def evaluate(args):
    grid, _, _, _, lock_digest, _ = load_motion_development(args)
    report = json.loads(args.waveform_results.read_bytes())
    people, pulses, clock, labels, reference = read_pairs(args.waveform_pairs, report, lock_digest, set(grid.subjects))
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    assumed = round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000)
    delays = np.where(clock == "beat", assumed, template.delay_samples)
    random = np.random.default_rng(SEED)
    rotations = random.integers(PPG_RATE, SAMPLES - 2 * EDGE - PPG_RATE, size=(len(people), ROTATIONS))
    collected = {}
    for signal, actual, delay, shifts in zip(pulses, reference, delays, rotations, strict=True):
        for name, peaks in optical_candidates(signal).items():
            collected.setdefault(name, []).append(assess(actual, peaks - delay, shifts))
    timing = {name: np.asarray(rows) for name, rows in collected.items()}
    groups = {"all_pairs": np.ones(len(people), dtype=bool), "stable_AF": labels == 1, "stable_non_AF": labels == 0,
              "beat_aligned_archive": clock == "beat", "coarse_alignment": clock == "coarse"}
    comparisons = {name: {"windows": int(mask.sum()), "participants": len(np.unique(people[mask])),
                          "routes": {route: summarize_people(values[mask], people[mask]) for route, values in timing.items()}}
                   for name, mask in groups.items()}
    return {"scope": "Reference-assisted optical beat opportunities, with no deployable ECG prediction",
            "timing_tolerance_ms": TOLERANCE_MS, "rotations_per_window": ROTATIONS, "trim_seconds": EDGE / PPG_RATE,
            "assumed_release_delay_ms": ARCHIVE_ASSUMED_DELAY_MS,
            "fixed_coarse_delay_ms": 1000 * template.delay_samples / PPG_RATE,
            "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
