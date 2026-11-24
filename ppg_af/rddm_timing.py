"""Compare frozen RDDM with a PPG-only timing correction on held-out PTT subjects."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .pulse_timing import (MATCH_SECONDS, MAX_SHIFT_SECONDS, correct_timing, fit_pulse_delay,
                           pulse_delay_histogram, timing_scoring_mask)
from .rddm import (AUTHOR_COMMIT, AUTHOR_WINDOW, DIFFUSION_STEPS, RANDOM_SEED, WEIGHT_NAMES,
                   generate_author_ecg, load_author_model, select_windows, sha256_file)
from .tcn_metrics import score_group
from .template import fit_template_from_stats, template_stats
from .timing_diagnostics import diagnose_timing
from .train_tcn import SAMPLE_RATE, WINDOW_SECONDS, make_folds, prepare_windows


FOLDS = 5
FOLD_SEED = 2026
SCORE_TRIM_SECONDS = 1.0


def fit_and_correct(windows, selected, generated):
    """Fit within each training partition and shift only its held-out forecasts."""
    delays, shapes = {}, {}
    for subject in np.unique(windows.subjects):
        indices = np.flatnonzero(windows.subjects == subject)
        pulses, targets = windows.ppg[indices], windows.ecg[indices]
        delays[subject] = pulse_delay_histogram(pulses, windows.r_peaks[indices], SAMPLE_RATE)
        shapes[subject] = template_stats(zip(pulses, targets, strict=True), fs=SAMPLE_RATE)

    corrected = np.full_like(generated, np.nan)
    template_estimate = np.full_like(generated, np.nan)
    visits = np.zeros(len(selected), dtype=int)
    folds, shift_samples = [], []
    for fold, (train, validation, test) in enumerate(make_folds(windows.subjects, FOLDS, FOLD_SEED), start=1):
        local_indices = np.flatnonzero(np.isin(selected, test))
        subjects = np.unique(windows.subjects[train])
        delay = fit_pulse_delay([delays[subject] for subject in subjects], SAMPLE_RATE)
        template = fit_template_from_stats(shapes[subject] for subject in subjects)
        fold_shifts = []
        for local in local_indices:
            pulse = windows.ppg[selected[local]]
            corrected[local], shifts = correct_timing(
                pulse, generated[local], delay, SAMPLE_RATE,
                chunk_seconds=AUTHOR_WINDOW, max_shift_seconds=MAX_SHIFT_SECONDS,
                match_seconds=MATCH_SECONDS,
            )
            template_estimate[local] = template.predict(pulse)
            fold_shifts.extend(int(shift) for shift in shifts)

        shift_samples.extend(fold_shifts)
        visits[local_indices] += 1
        folds.append({
            "fold": fold,
            "train_subjects": len(subjects),
            "validation_subjects": len(np.unique(windows.subjects[validation])),
            "test_subjects": len(np.unique(windows.subjects[test])),
            "pulse_delay_ms": delay * 1000 / SAMPLE_RATE,
            "generated_blocks": len(fold_shifts),
            "shifted_blocks": int(np.count_nonzero(fold_shifts)),
        })

    if not np.all(visits == 1):
        raise RuntimeError("Each selected window must be evaluated for exactly one held-out fold")
    counts, frequency = np.unique(shift_samples, return_counts=True)
    return corrected, template_estimate, folds, [
        {"shift_ms": int(count * 1000 / SAMPLE_RATE), "blocks": int(number)}
        for count, number in zip(counts, frequency, strict=True)
    ]


def _paired_metrics(target, annotations, ppg, forecasts):
    result = score_group(target, annotations, ppg, forecasts, {}, SAMPLE_RATE, SCORE_TRIM_SECONDS)
    # RDDM and the PTT template produce different amplitude units.
    for model in result["models"].values():
        model.pop("relative_mae")
    result.pop("zero_waveform_relative_mae")
    result.pop("auxiliary_peak_heads")
    return result


def evaluate_timing(windows, selected, generated):
    """Score one set of generated windows with and without shifts on common support."""
    if not len(selected) or generated.shape != windows.ppg[selected].shape or not np.isfinite(generated).all():
        raise ValueError("Frozen RDDM forecasts must match the selected finite PPG windows")
    corrected, template, folds, shifts = fit_and_correct(windows, selected, generated)
    forecasts = {"rddm_frozen": generated, "rddm_timing_correction": corrected, "pulse_template_ptt": template}
    target, annotations = windows.ecg[selected], windows.r_peaks[selected]
    ppg = windows.ppg[selected]
    fixed_guard = timing_scoring_mask(
        ppg.shape[1], SAMPLE_RATE, chunk_seconds=AUTHOR_WINDOW,
        max_shift_seconds=MAX_SHIFT_SECONDS,
    )
    guarded_ppg = np.where(fixed_guard, ppg, np.nan)

    def score_indices(indices):
        return {
            "subjects": len(np.unique(windows.subjects[selected[indices]])),
            **_paired_metrics(target[indices], annotations[indices], guarded_ppg[indices],
                              {name: predictions[indices] for name, predictions in forecasts.items()}),
        }

    activities = windows.activities[selected]
    return {
        "scope": "Exploratory phase-only timing correction with frozen RDDM on the previously evaluated PTT cohort",
        "dataset": "PhysioNet Pulse Transit Time PPG v1.1.0",
        "selection": "One midpoint 12-second window per subject and activity",
        "folds": folds,
        "shift_distribution": shifts,
        "original_full_support": _paired_metrics(target, annotations, ppg, {"rddm_frozen": generated}),
        "original_error_diagnostic": diagnose_timing(target, annotations, ppg, {"rddm_frozen": generated},
                                                      SAMPLE_RATE, trim_seconds=SCORE_TRIM_SECONDS),
        "paired_comparison": score_indices(np.arange(len(selected))),
        "by_activity": {str(activity): score_indices(np.flatnonzero(activities == activity))
                        for activity in np.unique(activities)},
        "paired_error_diagnostic": diagnose_timing(target, annotations, guarded_ppg, forecasts, SAMPLE_RATE,
                                                    trim_seconds=SCORE_TRIM_SECONDS),
    }


def build_report(windows, selected, original, *, hashes, batch_size, threads, seconds):
    return {
        "author_commit": AUTHOR_COMMIT,
        "checkpoint_sha256": hashes,
        "diffusion_steps": DIFFUSION_STEPS,
        "rddm_seed": RANDOM_SEED,
        "fold_seed": FOLD_SEED,
        "batch_size": batch_size,
        "threads": threads,
        "device": "cpu",
        "input_window_seconds": WINDOW_SECONDS,
        "generated_block_seconds": AUTHOR_WINDOW,
        "sample_rate_hz": SAMPLE_RATE,
        "max_shift_ms": int(MAX_SHIFT_SECONDS * 1000),
        "anchor_match_tolerance_ms": int(MATCH_SECONDS * 1000),
        "scoring_outer_trim_ms": int(SCORE_TRIM_SECONDS * 1000),
        "frozen_inference_seconds": round(seconds, 3),
        **evaluate_timing(windows, selected, original),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned author Git checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Directory with the three author checkpoint files")
    parser.add_argument("--ptt-data", type=Path, required=True, help="Directory with external PTT WFDB records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error("Thread count and batch size must be positive")

    torch.set_num_threads(args.threads)
    windows = prepare_windows(args.ptt_data, ("sit", "walk", "run"))
    selected = select_windows(windows, None)
    hashes = {name: sha256_file(args.weights / name) for name in WEIGHT_NAMES}
    models = load_author_model(args.upstream, args.weights, "cpu")
    original, _, seconds = generate_author_ecg(windows.ppg[selected], SAMPLE_RATE, models, args.batch_size, "cpu")
    del models
    report = build_report(windows, selected, original, hashes=hashes, batch_size=args.batch_size,
                          threads=args.threads, seconds=seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
