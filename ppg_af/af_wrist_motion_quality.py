"""Test native motion coherence as a wrist AF signal-quality feature."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_acceleration_stage import SIDECAR_FOLDERS
from .af_wrist_clock_study import load_grid
from .af_wrist_exercise import HEART_RATE_BAND_HZ, motion_weighted_spectrum
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_quality_study import _compare_with_pre_onset, build_quality_matrices
from .af_wrist_template_study import DEFAULT_C, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS


ACCELERATION_RATE = 50
SEEDS = (2026, 2030)
ARMS = ("quality_interactions", "acceleration_presence", "motion_coherence")


def residual_pulse_power(pulse, acceleration):
    """Compute the fraction of PPG power retained after motion weighting."""
    motion = resample_poly(acceleration, PPG_RATE, ACCELERATION_RATE, axis=0, padtype="line")
    frequencies, power, weighted = motion_weighted_spectrum(pulse, motion)
    keep = (frequencies >= HEART_RATE_BAND_HZ[0]) & (frequencies <= HEART_RATE_BAND_HZ[1])
    total = power[keep].sum()
    return float(np.clip(weighted[keep].sum() / total, 0, 1)) if total > 0 else 0.0


def verified_motion_rows(raw_file, feature_file, sidecar_file, lock_digest, *, stride):
    with (np.load(raw_file, allow_pickle=False) as raw, np.load(feature_file, allow_pickle=False) as original,
          np.load(sidecar_file, allow_pickle=False) as sidecar):
        common = _shared_features(original)
        if stride:
            common &= raw["input_status"] == "available"
        rows = np.flatnonzero(common)
        motion = sidecar["accel"]
        phase = sidecar["accel_minus_ppg_s"]
        cores = raw["ppg"][rows, PPG_RATE:-PPG_RATE]
        if (not np.array_equal(raw["start_s"], original["start_s"]) or not np.array_equal(rows, sidecar["rows"])
                or motion.shape != (len(rows), WINDOW_SECONDS * ACCELERATION_RATE, 3) or phase.shape != (len(rows),)
                or str(sidecar["original_lock_sha256"].item()) != lock_digest
                or int(sidecar["accel_fs"].item()) != ACCELERATION_RATE or int(sidecar["ppg_fs"].item()) != PPG_RATE
                or sha256(cores.tobytes()).hexdigest() != str(sidecar["staged_ppg_core_sha256"].item())):
            raise ValueError("Motion, waveform and feature archives must match the original scoreable decision rows")
        has_motion = np.isfinite(motion).all(axis=(1, 2))
        if (not np.array_equal(has_motion, np.isfinite(phase))
                or np.any((phase[has_motion] < -1e-9) | (phase[has_motion] > 1 / PPG_RATE + 1e-9))):
            raise ValueError("Acceleration samples must start on the native wrist grid")
    return len(common), rows, cores, motion, has_motion


def _native_features(raw_file, feature_file, sidecar_file, lock_digest, *, stride):
    count, rows, cores, motion, has_motion = verified_motion_rows(raw_file, feature_file, sidecar_file, lock_digest, stride=stride)
    result = np.zeros((count, 2))
    result[rows[has_motion], 0] = 1
    for index in np.flatnonzero(has_motion):
        result[rows[index], 1] = residual_pulse_power(cores[index], motion[index])
    return result


def _motion_matrices(grid, paths, sidecar_root, base, lock_digest):
    blocks, people = [], []
    for stride, ((raw, features), folder) in enumerate(zip(paths, SIDECAR_FOLDERS, strict=True)):
        for person in np.unique(grid.subjects):
            filename = f"{person}.npz"
            rows = _native_features(raw / filename, features / filename, sidecar_root / folder / filename,
                                    lock_digest, stride=bool(stride))
            blocks.append(rows)
            people.extend([person] * len(rows))
    if not np.array_equal(people, grid.subjects):
        raise ValueError("Recorded motion must align with the original cohort row ordering")
    motion = np.concatenate(blocks)
    shared = _shared_features(grid.table)
    available = motion[:, 0] > 0
    watch = np.zeros(len(shared), dtype=bool)
    for timeline in grid.timelines:
        if timeline.kind == "monitor_wear":
            watch[timeline.rows] = True
    if np.any(available & ~shared) or not np.any(available & shared):
        raise ValueError("Only scoreable original PPG rows may receive paired motion features")
    fit = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    support = {"shared_ppg_rows": int(shared.sum()), "shared_rows_with_motion": int(available.sum()),
               "participants_with_verified_motion": len(np.unique(grid.subjects[available])),
               "wrist_rows_with_motion": int(np.sum(available & watch)), "wrist_scoreable_rows": int(np.sum(shared & watch)),
               "original_fitting_af_rows_with_motion": int(np.sum(available & fit & (grid.table["stable_label"] == 1))),
               "original_fitting_non_af_rows_with_motion": int(np.sum(available & fit & (grid.table["stable_label"] == 0)))}
    return {ARMS[0]: base, ARMS[1]: np.column_stack((base, motion[:, 0])),
            ARMS[2]: np.column_stack((base, motion))}, support


def _compare(grid, matrices, lock, original):
    results = {}
    for seed in SEEDS:
        fits = {name: fit_template(grid, matrix, lock, DEFAULT_C, seed=seed) for name, matrix in matrices.items()}
        baseline = _replay(grid, fits[ARMS[0]])
        if seed == SEEDS[0] and baseline[0] != original:
            raise ValueError("The original quality interaction result must reproduce")
        arms = {ARMS[0]: baseline[0]}
        for name, fit in fits.items():
            if name == ARMS[0]:
                continue
            verify_fit_support(fits[ARMS[0]], fit)
            candidate = _replay(grid, fit)
            arms[name] = _compare_with_pre_onset(baseline, candidate, lock)
            print(f"{seed} {name}: {candidate[0]['episodes_detected']}/{candidate[0]['complete_episodes']} detections, "
                  f"{candidate[0]['wrist_false_alerts']} false starts", flush=True)
        results[str(seed)] = arms
    return results


def load_motion_development(args):
    lock_bytes = args.lock.read_bytes()
    lock, clock = json.loads(lock_bytes), json.loads(args.clock_results.read_bytes())
    lock_digest = sha256(lock_bytes).hexdigest()
    table, subjects, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[name]) for name in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & reserved or set(subjects) != fit | calibration
            or lock_digest != clock["original_lock_sha256"] or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]):
        raise ValueError("Use only the original locked wrist development cohort and clock study")
    grid = load_grid(table, subjects, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    quality = build_quality_matrices(grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features)
    paths = ((args.development_raw, args.development_features), (args.stride_raw, args.stride_features))
    original = json.loads(args.quality_results.read_bytes())["fixed_regularization"][ARMS[0]]["summary"]
    return grid, quality, paths, lock, lock_digest, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "acceleration-sidecars",
                 "lock", "clock-results", "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
        matrices, support = _motion_matrices(grid, paths, args.acceleration_sidecars, quality[ARMS[0]], lock_digest)
        results = _compare(grid, matrices, lock, original)
        _write_json(args.output, {"scope": "Native wrist motion and AF on the original grouped development cohort",
                                  "development_participants": len(np.unique(grid.subjects)), "outer_folds": 5,
                                  "regularization_C": DEFAULT_C, "native_acceleration_hz": ACCELERATION_RATE,
                                  "motion_coherence_band_hz": HEART_RATE_BAND_HZ, "support": support,
                                  "comparisons": results})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Native wrist motion quality study failed: {error}\n")


if __name__ == "__main__":
    main()
