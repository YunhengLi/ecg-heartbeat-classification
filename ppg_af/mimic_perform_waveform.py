"""Evaluate two PPG-to-ECG waveform baselines on held-out PERform patients."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold

from .features import ecg_peaks
from .tcn_metrics import score_group
from .template import fit_template_from_stats, template_stats
from .waveform import fit_waveform_from_stats, normalize_ecg, waveform_stats


SOURCE_DOI = "10.5281/zenodo.15906524"
SOURCES = (
    ("mimic_perform_af_data.mat", "38adc07e4420226a0edfe9769a23284f", 19, 1),
    ("mimic_perform_non_af_data.mat", "d1c4f93442a3b4c9cb95718ca0a57013", 16, 0),
)
RATE = 125
RECORD_SECONDS = 20 * 60
WINDOW_SECONDS = 12
WINDOW_STEP_SECONDS = 60
FIRST_CENTER_SECONDS = 30
TRIM_SECONDS = 1
FOLDS = 5
SEED = 2026
RIDGE = 0.01


def _read_records(data_dir):
    records, record_ids = {}, set()
    for filename, expected_md5, expected_count, label in SOURCES:
        path = data_dir / filename
        with path.open("rb") as stream:
            if file_digest(stream, "md5").hexdigest() != expected_md5:
                raise ValueError(f"MIMIC PERform v2.0 checksum mismatch for {filename}")
        items = loadmat(path, simplify_cells=True)["data"]
        if len(items) != expected_count:
            raise ValueError(f"Unexpected number of recordings in {filename}")
        for item in items:
            metadata = item["fix"]
            subject, record_id = str(metadata["subj_id"]), str(metadata["rec_id"])
            if subject in records or record_id in record_ids or metadata["af_status"] != label:
                raise ValueError("Expected one unique recording per patient with the declared AF status")
            ppg, ecg = item["ppg"], item["ekg"]
            pulse, target = (np.asarray(channel["v"], dtype=float) for channel in (ppg, ecg))
            expected_shape = (RECORD_SECONDS * RATE + 1,)
            if ppg["fs"] != RATE or ecg["fs"] != RATE or pulse.shape != expected_shape or target.shape != expected_shape:
                raise ValueError("Expected simultaneous 20-minute ECG and PPG on the native 125 Hz grid")
            records[subject] = pulse, target, label
            record_ids.add(record_id)
    return [records[subject] for subject in sorted(records)]


def _select_windows(records):
    centers = range(FIRST_CENTER_SECONDS, RECORD_SECONDS, WINDOW_STEP_SECONDS)
    width = WINDOW_SECONDS * RATE
    ppg, ecg, patients = [], [], []
    excluded = Counter(nonfinite=0, flat=0)
    for patient, (pulse, target, _) in enumerate(records):
        initial_count = len(ppg)
        for center in centers:
            start = center * RATE - width // 2
            p, e = pulse[start:start + width], target[start:start + width]
            if not np.isfinite(p).all() or not np.isfinite(e).all():
                excluded["nonfinite"] += 1
            elif np.ptp(p) == 0 or np.ptp(e) == 0:
                excluded["flat"] += 1
            else:
                ppg.append(p)
                ecg.append(e)
                patients.append(patient)
        if len(ppg) == initial_count:
            raise ValueError("Every patient needs at least one eligible paired waveform window")
    return np.stack(ppg), np.stack(ecg), np.array(patients), len(records) * len(centers), dict(excluded)


def _forecast(ppg, ecg, patients, labels):
    stats = []
    for patient in range(len(labels)):
        rows = patients == patient
        pulse, target = ppg[rows], ecg[rows]
        stats.append((waveform_stats(zip(pulse, target, strict=True), fs=RATE),
                      template_stats(zip(pulse, target, strict=True), fs=RATE)))

    forecasts = {model: np.full(ecg.shape, np.nan) for model in ("delay_ridge", "pulse_template")}
    fold_support = []
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for train, test in splitter.split(np.arange(len(labels)), labels):
        ridge = fit_waveform_from_stats((stats[patient][0] for patient in train), ridge=RIDGE)
        template = fit_template_from_stats(stats[patient][1] for patient in train)
        test_rows = np.isin(patients, test)
        for name, model in (("delay_ridge", ridge), ("pulse_template", template)):
            forecasts[name][test_rows] = np.stack([model.predict(pulse) for pulse in ppg[test_rows]])
        fold_support.append({"train_patients": len(train), "test_patients": len(test),
                             "test_af_patients": int(labels[test].sum()), "test_windows": int(test_rows.sum())})
    return forecasts, fold_support


def evaluate(data_dir):
    records = _read_records(data_dir)
    ppg, ecg, patients, scheduled, excluded = _select_windows(records)
    labels = np.array([label for _, _, label in records])
    forecasts, fold_support = _forecast(ppg, ecg, patients, labels)

    # The release supplies waveform pairs and record-level rhythm labels. Peak
    # locations are detector-derived references for this exploratory benchmark.
    reference = np.zeros(ecg.shape, dtype=bool)
    for waveform, peaks in zip(ecg, reference, strict=True):
        peaks[ecg_peaks(waveform, RATE)] = True
    target = np.stack([normalize_ecg(window) for window in ecg])

    def group(selected):
        metrics = score_group(target[selected], reference[selected], ppg[selected],
                              {name: waveform[selected] for name, waveform in forecasts.items()},
                              {}, RATE, trim_seconds=TRIM_SECONDS)
        del metrics["auxiliary_peak_heads"], metrics["measured_ecg_detector"]
        return {"patients": len(np.unique(patients[selected])), **metrics}

    return {
        "scope": "Patient-held-out offline waveform and detector-based beat-timing comparison",
        "source_doi": SOURCE_DOI,
        "source_md5": {filename: digest for filename, digest, _, _ in SOURCES},
        "sample_rate_hz": RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_step_seconds": WINDOW_STEP_SECONDS,
        "first_center_seconds": FIRST_CENTER_SECONDS,
        "trim_seconds_per_edge": TRIM_SECONDS,
        "beat_match_tolerance_ms": 100,
        "ridge_penalty": RIDGE,
        "folds": FOLDS,
        "seed": SEED,
        "scheduled_windows": scheduled,
        "excluded_windows": excluded,
        "eligible_windows": len(ppg),
        "fold_support": fold_support,
        "overall": group(slice(None)),
        "by_record_status": {"af": group(labels[patients] == 1), "non_af": group(labels[patients] == 0)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with both pinned PERform AF MAT files")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new report path.")
    report = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Scored {report['eligible_windows']} paired waveform windows across {report['folds']} patient folds")


if __name__ == "__main__":
    main()
