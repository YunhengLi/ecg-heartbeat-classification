"""Compare small PPG-to-ECG temporal CNNs on the pinned PERform patient folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .features import ecg_peaks
from .mimic_perform_waveform import (
    FIRST_CENTER_SECONDS, FOLDS, RATE, SEED, SOURCE_DOI, SOURCES, TRIM_SECONDS, WINDOW_SECONDS, WINDOW_STEP_SECONDS,
    _forecast, _read_records, _select_windows,
)
from .tcn_metrics import score_group
from .temporal import CONTEXT_SAMPLES, QRS_RADIUS_SAMPLES, fit_tcn, predict_tcn
from .waveform import normalize_ecg


VALIDATION_PATIENTS = 6
TRAINING = {"epochs": 8, "patience": 3, "batch_size": 32, "channels": 8, "peak_weight": 0.0}
QRS_WEIGHTS = {"tcn_waveform": 0.0, "tcn_qrs_weighted": 4.0}


def _fit_neural(ppg, target, peaks, patients, labels, threads):
    waveforms = {name: np.full(ppg.shape, np.nan, dtype=np.float32) for name in QRS_WEIGHTS}
    held_out_count = np.zeros(len(patients), dtype=int)
    support = []
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (outer_train, outer_test) in enumerate(splitter.split(np.arange(len(labels)), labels), 1):
        seed = SEED + fold
        validation_split = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PATIENTS, random_state=seed)
        train_local, validation_local = next(validation_split.split(outer_train, labels[outer_train]))
        train, validation = outer_train[train_local], outer_train[validation_local]
        if (np.intersect1d(train, validation).size or np.intersect1d(outer_train, outer_test).size
                or len(train) + len(validation) + len(outer_test) != len(labels)):
            raise RuntimeError("Each fold needs disjoint training, validation, and test patients")
        train_rows, validation_rows, test_rows = (
            np.flatnonzero(np.isin(patients, subset)) for subset in (train, validation, outer_test)
        )
        held_out_count[test_rows] += 1
        report = {
            "fold": fold,
            "training_patients": len(train), "training_af_patients": int(labels[train].sum()),
            "training_windows": len(train_rows),
            "validation_patients": len(validation), "validation_af_patients": int(labels[validation].sum()),
            "validation_windows": len(validation_rows),
            "test_patients": len(outer_test), "test_af_patients": int(labels[outer_test].sum()),
            "test_windows": len(test_rows), "models": {},
        }
        print(f"Fold {fold}/{FOLDS}: {len(train_rows)} training, {len(validation_rows)} validation, "
              f"{len(test_rows)} held-out windows", flush=True)
        for name, qrs_weight in QRS_WEIGHTS.items():
            model, history = fit_tcn(ppg, target, peaks, train_rows, validation_rows,
                                     seed=seed, threads=threads, qrs_weight=qrs_weight, **TRAINING)
            waveforms[name][test_rows], _ = predict_tcn(model, ppg[test_rows], batch_size=TRAINING["batch_size"])
            selected = next(epoch for epoch in history if epoch["selected"])
            report["models"][name] = {
                "parameters": sum(param.numel() for param in model.parameters()),
                "epochs_completed": len(history), "selected_epoch": selected["epoch"],
                "selected_training_waveform_loss": selected["train_waveform_loss"],
                "validation_waveform_loss_by_epoch": [epoch["validation_waveform_loss"] for epoch in history],
            }
            print(f"  {name}: selected epoch {selected['epoch']} of {len(history)}", flush=True)
        support.append(report)
    if not np.all(held_out_count == 1) or any(not np.isfinite(waveform).all() for waveform in waveforms.values()):
        raise RuntimeError("Each eligible window needs one finite held-out forecast from each neural model")
    return waveforms, support


def evaluate(data_dir, threads=2):
    records = _read_records(data_dir)
    ppg, ecg, patients, scheduled, excluded = _select_windows(records)
    labels = np.array([label for _, _, label in records])
    baselines, baseline_support = _forecast(ppg, ecg, patients, labels)
    peaks = np.zeros(ecg.shape, dtype=bool)
    for waveform, reference in zip(ecg, peaks, strict=True):
        reference[ecg_peaks(waveform, RATE)] = True
    target = np.stack([normalize_ecg(window) for window in ecg])
    neural, folds = _fit_neural(ppg, target, peaks, patients, labels, threads)
    for original, candidate in zip(baseline_support, folds, strict=True):
        if any(original[key] != candidate[key] for key in ("test_patients", "test_af_patients", "test_windows")):
            raise RuntimeError("The neural study changed the waveform benchmark's outer test schedule")
    forecasts = {**baselines, **neural}

    def group(selected):
        metrics = score_group(target[selected], peaks[selected], ppg[selected],
                              {name: waveform[selected] for name, waveform in forecasts.items()},
                              {}, RATE, trim_seconds=TRIM_SECONDS)
        del metrics["auxiliary_peak_heads"], metrics["measured_ecg_detector"]
        return {"patients": len(np.unique(patients[selected])), **metrics}

    return {
        "scope": "Patient-held-out offline PPG-to-ECG pilot with detector-derived ECG peak targets",
        "source_doi": SOURCE_DOI,
        "source_md5": {filename: digest for filename, digest, _, _ in SOURCES},
        "sample_rate_hz": RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_step_seconds": WINDOW_STEP_SECONDS,
        "first_center_seconds": FIRST_CENTER_SECONDS,
        "trim_seconds_per_edge": TRIM_SECONDS,
        "beat_match_tolerance_ms": 100,
        "scheduled_windows": scheduled,
        "excluded_windows": excluded,
        "eligible_windows": len(ppg),
        "settings": {**TRAINING, "qrs_weight_by_model": QRS_WEIGHTS,
                     "qrs_radius_samples": QRS_RADIUS_SAMPLES, "convolution_future_context_samples": CONTEXT_SAMPLES,
                     "threads": threads, "torch_version": str(torch.__version__), "folds": FOLDS, "seed": SEED},
        "folds": folds,
        "overall": group(slice(None)),
        "by_record_status": {"af": group(labels[patients] == 1), "non_af": group(labels[patients] == 0)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with both pinned PERform AF MAT files")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON report")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new report path.")
    report = evaluate(args.data, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Scored {report['eligible_windows']} windows with {FOLDS} patient-held-out folds")


if __name__ == "__main__":
    main()
