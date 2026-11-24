"""Compare neural waveform losses on the verified native wrist ECG pairs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .af_wrist_cv import _subject_folds
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS, FOLDS, SEED, participant_template_stats
from .tcn_metrics import score_group
from .template import fit_template_from_stats
from .temporal import fit_tcn, predict_tcn
from .waveform import normalize_ecg


EPOCHS = 12
CHANNELS = 8
BATCH_SIZE = 16
VARIANTS = {"waveform_only_tcn": (0.0, 0.0), "qrs_weighted_tcn": (0.35, 3.0)}


def evaluate(args):
    grid, _, _, _, lock_digest, _ = load_motion_development(args)
    previous = json.loads(args.waveform_results.read_bytes())
    people, pulse, synchronization, rhythm, reference = read_pairs(args.waveform_pairs, previous, lock_digest, set(grid.subjects))
    with np.load(args.waveform_pairs, allow_pickle=False) as archive:
        measured = archive["ecg_ppg_clock"]
    target = np.stack([normalize_ecg(row) for row in measured])
    peak_mask = np.zeros_like(target, dtype=np.float32)
    for row, markers in zip(peak_mask, reference, strict=True):
        indices = np.rint(markers).astype(int)
        row[indices[(indices >= 0) & (indices < len(row))]] = 1.0
    peak_mask[~np.isfinite(target)] = np.nan

    eligible = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    group = _subject_folds(np.where(eligible, grid.table["stable_label"], -1), grid.subjects, FOLDS, SEED)
    membership = {person: int(group[grid.subjects == person][0]) for person in np.unique(grid.subjects)}
    folds = np.asarray([membership[person] for person in people])
    _, fixed = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    fine = replace(fixed, delay_samples=ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE // 1000)
    predictions = {"template_archive_delay": np.stack([(fine if clock == "beat" else fixed).predict(row)
                                                       for row, clock in zip(pulse, synchronization, strict=True)])}
    for name in ("fold_local_template", *VARIANTS):
        predictions[name] = np.full_like(target, np.nan)
    auxiliary = {"qrs_weighted_tcn": np.full_like(target, np.nan)}
    stats = participant_template_stats(people, pulse, measured)
    history = []
    for fold in range(FOLDS):
        held = np.flatnonzero(folds == fold)
        available = np.unique(people[folds != fold])
        count = max(1, round(0.22 * len(available)))
        validation_people = np.random.default_rng(SEED + fold).permutation(available)[:count]
        validation = np.flatnonzero(np.isin(people, validation_people))
        train = np.flatnonzero((folds != fold) & ~np.isin(people, validation_people))
        if (not len(held) or not len(train) or not len(validation)
                or set(people[held]) & (set(people[train]) | set(people[validation]))):
            raise ValueError("TCN early stopping and scoring must use different wrist participants")
        template = fit_template_from_stats(stats[name] for name in np.unique(people[train]))
        predictions["fold_local_template"][held] = np.stack([template.predict(row) for row in pulse[held]])
        epoch_summary = {}
        for name, (peak_weight, qrs_weight) in VARIANTS.items():
            model, training = fit_tcn(pulse, target, peak_mask, train, validation, peak_weight=peak_weight,
                                      qrs_weight=qrs_weight, seed=SEED + fold, epochs=EPOCHS, channels=CHANNELS,
                                      batch_size=BATCH_SIZE, threads=args.threads)
            waveform, peaks = predict_tcn(model, pulse[held], batch_size=BATCH_SIZE)
            predictions[name][held] = waveform
            if name in auxiliary:
                auxiliary[name][held] = peaks
            epoch_summary[name] = next(row["epoch"] for row in training if row["selected"])
        history.append({"training_participants": len(np.unique(people[train])), "validation_participants": len(validation_people),
                        "held_participants": len(np.unique(people[held])), "train_windows": len(train),
                        "validation_windows": len(validation), "held_windows": len(held), "selected_epochs": epoch_summary})
        print(f"Completed paired waveform fold {fold + 1}/{FOLDS}", flush=True)

    def metrics(selected):
        return {"participants": len(np.unique(people[selected])),
                **score_group(target[selected], peak_mask[selected], pulse[selected],
                              {name: values[selected] for name, values in predictions.items()},
                              {name: logits[selected] for name, logits in auxiliary.items()}, PPG_RATE)}

    return {"scope": "Participant-held-out temporal waveform comparison on 223 verified native wrist pairs",
            "folds": history, "max_epochs": EPOCHS, "channels": CHANNELS, "training_loss_weights": VARIANTS,
            "all": metrics(np.ones(len(people), dtype=bool)),
            "by_synchronization": {name: metrics(synchronization == name) for name in np.unique(synchronization)},
            "by_rhythm": {"AF": metrics(rhythm == 1), "non_AF": metrics(rhythm == 0)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
