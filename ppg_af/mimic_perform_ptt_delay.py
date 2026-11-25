"""Isolate timing transfer when pooling PTT and MIMIC pulse templates."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .features import ecg_peaks
from .mimic_perform_waveform import (
    FIRST_CENTER_SECONDS, FOLDS, RATE, SEED, SOURCE_DOI, SOURCES, TRIM_SECONDS, WINDOW_SECONDS, WINDOW_STEP_SECONDS,
    _read_records, _select_windows,
)
from .ptt_data import ACTIVITIES, PPG_CHANNEL
from .rddm import select_windows
from .tcn_metrics import score_group
from .template import fit_template_from_stats, template_stats
from .train_tcn import NATIVE_RATE, SAMPLE_RATE, prepare_windows
from .waveform import normalize_ecg


MODELS = ("mimic_only", "joint", "joint_shape_local_delay")
PTT_PARTICIPANTS = 22


def _collect(ppg, ecg, groups):
    stats = []
    for patient in np.unique(groups):
        rows = groups == patient
        stats.append(template_stats(zip(ppg[rows], ecg[rows], strict=True), fs=RATE))
    return stats


def _ptt_stats(data_dir, window_samples):
    windows = prepare_windows(data_dir, ACTIVITIES)
    selected = select_windows(windows, None)
    participants, activities = windows.subjects[selected], windows.activities[selected]
    pairs = set(zip(participants, activities, strict=True))
    expected = PTT_PARTICIPANTS * len(ACTIVITIES)
    if (RATE != SAMPLE_RATE or windows.ppg.shape[1] != window_samples
            or len(np.unique(participants)) != PTT_PARTICIPANTS or len(selected) != expected or len(pairs) != expected):
        raise ValueError("Use the fixed PTT midpoint windows on the 125 Hz evaluation grid")
    return _collect(windows.ppg[selected], windows.ecg[selected], participants), len(selected)


def _forecast(ppg, patients, labels, source, target):
    predictions = {name: np.full(ppg.shape, np.nan) for name in MODELS}
    scored = np.zeros(len(ppg), dtype=int)
    folds = []
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(labels)), labels), 1):
        if np.intersect1d(train, test).size:
            raise RuntimeError("MIMIC fitting and evaluation patients must be disjoint")
        local_stats = [target[patient] for patient in train]
        local = fit_template_from_stats(local_stats)
        joint = fit_template_from_stats([*source, *local_stats])
        corrected = replace(joint, delay_samples=local.delay_samples)
        rows = np.isin(patients, test)
        models = dict(zip(MODELS, (local, joint, corrected), strict=True))
        for name, model in models.items():
            predictions[name][rows] = np.stack([model.predict(window) for window in ppg[rows]])
        scored[rows] += 1
        folds.append({"fold": fold, "mimic_train_patients": len(train), "mimic_test_patients": len(test),
                      "mimic_test_af_patients": int(labels[test].sum()), "mimic_test_windows": int(rows.sum()),
                      "template_delay_ms": {name: 1000 * model.delay_samples / RATE for name, model in models.items()}})
    if not np.all(scored == 1):
        raise RuntimeError("Every MIMIC window needs a single held-out forecast")
    return predictions, folds


def evaluate(mimic_data, ptt_data):
    records = _read_records(mimic_data)
    ppg, ecg, patients, scheduled, excluded = _select_windows(records)
    labels = np.array([label for _, _, label in records])
    source, ptt_windows = _ptt_stats(ptt_data, ppg.shape[1])
    target = _collect(ppg, ecg, patients)
    forecasts, folds = _forecast(ppg, patients, labels, source, target)

    references = np.zeros(ecg.shape, dtype=bool)
    for waveform, peaks in zip(ecg, references, strict=True):
        peaks[ecg_peaks(waveform, RATE)] = True
    normalized = np.stack([normalize_ecg(window) for window in ecg])

    def group(rows, models):
        metrics = score_group(normalized[rows], references[rows], ppg[rows],
                              {name: predicted[rows] for name, predicted in models.items()},
                              {}, RATE, trim_seconds=TRIM_SECONDS)
        del metrics["auxiliary_peak_heads"], metrics["measured_ecg_detector"]
        return {"patients": len(np.unique(patients[rows])), **metrics}

    groups = {"overall": slice(None), "af": labels[patients] == 1, "non_af": labels[patients] == 0}
    comparisons = {name: group(rows, forecasts) for name, rows in groups.items()}
    original = {"pulse_template": forecasts["mimic_only"]}
    original_only = {name: group(rows, original) for name, rows in groups.items()}
    support = {name: {"original_fraction": original_only[name]["common_support_fraction"],
                      "all_models_fraction": comparisons[name]["common_support_fraction"]} for name in groups}
    return {
        "scope": "Fixed training-only local delay for pooled PTT and MIMIC templates",
        "mimic_doi": SOURCE_DOI,
        "mimic_source_md5": {filename: digest for filename, digest, _, _ in SOURCES},
        "seed": SEED, "rate_hz": RATE, "window_seconds": WINDOW_SECONDS,
        "window_step_seconds": WINDOW_STEP_SECONDS, "first_center_seconds": FIRST_CENTER_SECONDS,
        "trim_seconds": TRIM_SECONDS, "beat_tolerance_ms": 100,
        "ptt_release": "PhysioNet Pulse Transit Time PPG v1.1.0",
        "ptt_ppg_channel": PPG_CHANNEL, "ptt_native_rate_hz": NATIVE_RATE,
        "ptt_training_participants": len(source), "ptt_training_windows": ptt_windows,
        "mimic_scheduled_windows": scheduled, "mimic_eligible_windows": len(ppg), "mimic_excluded_windows": excluded,
        "folds": folds, "support": support, "mimic_only_replay": original_only, "target_comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-data", type=Path, required=True, help="Directory with both pinned PERform AF MAT files")
    parser.add_argument("--ptt-data", type=Path, required=True, help="Directory with paired PTT WFDB records")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new report path.")
    report = evaluate(args.mimic_data, args.ptt_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Scored {report['mimic_eligible_windows']} windows with local and pooled template delays")


if __name__ == "__main__":
    main()
