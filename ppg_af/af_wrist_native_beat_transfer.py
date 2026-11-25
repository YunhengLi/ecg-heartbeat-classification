"""Test neural beat transfer to native AFPPGECG wrist recordings by participant."""

import argparse
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from .af_wrist_beat_cnn import EDGE, SAMPLES, load_move, predict, reference_targets, standardize, train
from .af_wrist_cv import _subject_folds
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS, FOLDS, SEED
from .features import ecg_peaks
from .tcn_metrics import _peak_score, summarize_beats


WRIST_EPOCHS = 12
TUNED_METHODS = ("wrist_only_cnn", "source_initialized_wrist_cnn")


def read_pairs(path, report, lock_digest, roster):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != report["private_paired_manifest_sha256"]:
            raise ValueError("Use the pinned paired wrist waveform evaluation archive")
    with np.load(path, allow_pickle=False) as data:
        keys = ("subjects", "ppg_core", "ecg_ppg_clock", "sync_method", "stable_label", "native_qrs_offsets",
                "native_qrs_float_indices", "origin", "source_protocol_sha256")
        paired = {key: data[key] for key in keys}
    people, pulse, measured = (paired[key] for key in ("subjects", "ppg_core", "ecg_ppg_clock"))
    if (report["original_protocol_sha256"] != lock_digest or paired["source_protocol_sha256"].item() != lock_digest
            or not set(people) <= roster or people.shape != (223,) or pulse.shape != (223, SAMPLES)
            or measured.shape != pulse.shape or set(paired["sync_method"]) != {"beat", "coarse"}
            or np.count_nonzero(paired["origin"] == "original") != 114
            or not np.isfinite(pulse).all() or not np.isfinite(measured[:, EDGE:-EDGE]).all()):
        raise ValueError("The native beat benchmark must retain the original waveform support and clock")
    offsets = paired["native_qrs_offsets"]
    reference = paired["native_qrs_float_indices"]
    if (offsets.shape != (len(people) + 1,) or offsets[0] != 0 or offsets[-1] != len(reference)
            or np.any(np.diff(offsets) <= 0) or not np.isfinite(reference).all()):
        raise ValueError("Every paired native window must have its ECG beat annotations")
    peaks = [reference[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
    return people, pulse, paired["sync_method"], paired["stable_label"], peaks


def summarize(reference, predictions, people, synchronization, rhythm):
    supported = np.r_[np.zeros(EDGE, dtype=bool), np.ones(SAMPLES - 2 * EDGE, dtype=bool), np.zeros(EDGE, dtype=bool)]
    methods = {name: [_peak_score(actual, estimate, supported, PPG_RATE)
                      for actual, estimate in zip(reference, values, strict=True)] for name, values in predictions.items()}

    def group(eligible):
        return {"participants": len(np.unique(people[eligible])), "windows": int(eligible.sum()),
                "routes": {method: summarize_beats([value for value, selected in zip(values, eligible, strict=True) if selected])
                           for method, values in methods.items()}}

    return {"overall": group(np.ones(len(people), dtype=bool)),
            "by_synchronization": {name: group(synchronization == name) for name in np.unique(synchronization)},
            "by_rhythm": {name: group(rhythm == value) for name, value in (("AF", 1), ("non_AF", 0))}}


def evaluate(args):
    grid, _, _, _, lock_digest, _ = load_motion_development(args)
    report = json.loads(args.waveform_results.read_bytes())
    people, wrist, synchronization, rhythm, reference = read_pairs(args.waveform_pairs, report, lock_digest, set(grid.subjects))
    eligible = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    labels = np.where(eligible, grid.table["stable_label"], -1)
    assigned = _subject_folds(labels, grid.subjects, FOLDS, SEED)
    member = {participant: int(assigned[grid.subjects == participant][0]) for participant in np.unique(grid.subjects)}
    fold = np.asarray([member[participant] for participant in people])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    move, _ = load_move(args.move_dir)
    source_inputs = standardize(move.pulse, 8.0).astype(np.float32)[:, None]
    source = train(source_inputs, reference_targets(move.reference), move.participants, args.device)
    inputs = standardize(wrist, 8.0).astype(np.float32)[:, None]
    targets = reference_targets(reference)
    _, archived_template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    corrected = replace(archived_template, delay_samples=round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000))
    template_peaks = [ecg_peaks((corrected if clock == "beat" else archived_template).predict(pulse), PPG_RATE)
                      for pulse, clock in zip(wrist, synchronization, strict=True)]
    predictions = {"template_archive_delay": template_peaks, "source_frozen_cnn": predict(source, inputs, args.device),
                   **{name: [None] * len(inputs) for name in TUNED_METHODS}}
    folds = []
    for number in range(FOLDS):
        held = fold == number
        trained = ~held
        if not held.any() or set(people[trained]) & set(people[held]):
            raise ValueError("Native wrist fits must hold out entire participants")
        for method, initial in zip(TUNED_METHODS, (None, source), strict=True):
            model = train(inputs[trained], targets[trained], people[trained], args.device, epochs=WRIST_EPOCHS,
                          initialized=initial)
            for index, peaks in zip(np.flatnonzero(held), predict(model, inputs[held], args.device), strict=True):
                predictions[method][index] = peaks
        folds.append({"train_participants": len(np.unique(people[trained])), "held_participants": len(np.unique(people[held])),
                      "train_windows": int(trained.sum()), "held_windows": int(held.sum())})
        print(f"Finished paired wrist fold {number + 1}/{FOLDS}", flush=True)
    return {"study": "Development-only neural ECG beat timing on paired native wrist inputs", "wrist_epochs": WRIST_EPOCHS,
            "source_windows": len(move.pulse), "source_participants": len(np.unique(move.participants)),
            "beat_tolerance_ms": 100, "edge_seconds": EDGE / PPG_RATE, "held_folds": folds,
            "results": summarize(reference, predictions, people, synchronization, rhythm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("move-dir", "development-raw", "development-features", "stride-raw", "stride-features", "lock",
                 "clock-results", "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
