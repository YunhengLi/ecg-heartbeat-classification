"""Train a wrist PPG network to estimate individual ECG R-peak times."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.nn import functional as F

from . import af_wrist_exercise as exercise
from . import af_wrist_move as move
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS
from .features import _bandpass, ecg_peaks
from .tcn_metrics import _peak_score, summarize_beats
from .template import _pulse_peaks


SEED = 2026
WIDTH = 16
EPOCHS = 20
BATCH_SIZE = 48
SAMPLES = WINDOW_SECONDS * PPG_RATE
EDGE = PPG_RATE
TARGET_STD_SAMPLES = 3
POSITIVE_WEIGHT = 7
PROBABILITY_CUTOFF = 0.5
METHODS = ("fixed_delay", "ppg_cnn", "ppg_accelerometer_cnn")


@dataclass
class BeatWindows:
    pulse: np.ndarray
    motion: np.ndarray
    participants: np.ndarray
    activity: np.ndarray
    reference: list
    scheduled: int


class BeatCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input = nn.Conv1d(input_channels, WIDTH, 21, padding=10)
        self.context = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 9, dilation=d, padding=4 * d) for d in (1, 2, 4, 8))
        self.output = nn.Conv1d(WIDTH, 1, 1)

    def forward(self, signals):
        features = F.silu(self.input(signals))
        for layer in self.context:
            features = features + F.silu(layer(features))
        return self.output(features).squeeze(1)


def load_move(directory):
    import edfio

    pulse, motion, group, activity, reference = [], [], [], [], []
    counts = Counter()
    for index, participant in enumerate(move.verify_release(directory)):
        if not (participant / "empatica.edf").exists():
            continue
        with warnings.catch_warnings(record=True):
            chest = edfio.read_edf(participant / "scientisst_chest.edf")
            wrist = edfio.read_edf(participant / "empatica.edf")
        clock = move.activity_clock(chest, wrist)
        if clock is None:
            continue
        offset, intervals = clock
        ecg = move._signals(chest, ("ecg:gel", "ecg:dry"), move.ECG_SOURCE_RATE)
        optical = move._signals(wrist, ("ppg:wrist",), move.PPG_SOURCE_RATE)[0]
        acceleration = move._signals(wrist, tuple(f"acc_e4:{axis}" for axis in "xyz"), move.MOTION_SOURCE_RATE)
        used = set()
        for name, begin, end in intervals:
            begin = max(begin, 0, -offset)
            end = min(end, chest.duration, wrist.duration - offset)
            first = int(np.ceil(begin / WINDOW_SECONDS)) * WINDOW_SECONDS
            for start in range(first, int(np.floor(end)) - WINDOW_SECONDS + 1, WINDOW_SECONDS):
                if start in used:
                    raise ValueError("Source activity annotations overlap on the nonoverlapping window grid")
                used.add(start)
                counts["scheduled"] += 1
                chest_clip = move._clip(ecg, start, move.ECG_SOURCE_RATE)
                if not np.isfinite(move.consensus_heart_rate(chest_clip)):
                    continue
                counts["reference"] += 1
                ppg = move._clip(optical, start + offset, move.PPG_SOURCE_RATE)
                acc = move._clip(acceleration, start + offset, move.MOTION_SOURCE_RATE)
                if (len(ppg) != WINDOW_SECONDS * move.PPG_SOURCE_RATE
                        or acc.shape[-1] != WINDOW_SECONDS * move.MOTION_SOURCE_RATE
                        or not np.isfinite(ppg).all() or not np.isfinite(acc).all() or np.ptp(ppg) <= 0):
                    continue
                pulse.append(move._resample(ppg, move.PPG_SOURCE_RATE))
                motion.append(move._resample(acc, move.MOTION_SOURCE_RATE).T)
                group.append(index)
                activity.append(name)
                reference.append(ecg_peaks(chest_clip[0], move.ECG_SOURCE_RATE) * (PPG_RATE / move.ECG_SOURCE_RATE))
    return BeatWindows(np.stack(pulse), np.stack(motion), np.asarray(group), np.asarray(activity), reference,
                       counts["scheduled"]), counts["reference"]


def load_shimmer(directory):
    import wfdb

    pulse, motion, group, activity, reference = [], [], [], [], []
    scheduled = 0
    for name in exercise.verify_release(directory):
        stem = str(directory / name)
        record = wfdb.rdrecord(stem, channel_names=["chest_ecg", "wrist_ppg", "sample_times_for_all_signals_apart_from_ecg",
                                                   *exercise.MOTION_CHANNELS[3:]])
        labels = wfdb.rdann(stem, "atr")
        ecg, ppg, times = record.p_signal[:, :3].T
        if record.fs != exercise.SOURCE_RATE or not np.isfinite(ecg).all() or set(labels.symbol) - {"N", '"'}:
            raise ValueError("Expected the released ECG clock and manual beat annotations")
        beats = labels.sample[np.array(labels.symbol) == "N"] / record.fs
        if len(beats) < 2 or beats[0] < 0 or beats[-1] >= len(ecg) / record.fs or np.any(np.diff(beats) <= 0):
            raise ValueError("Reference beat annotations must be within the ECG recording and strictly ordered")
        finite = np.isfinite(ppg) & np.isfinite(times)
        rows = np.flatnonzero(finite)
        clock = np.unwrap(times[finite], period=60.0)
        if len(clock) < 2 or rows[0] != 0 or np.any(np.diff(rows) != 1) or np.any(np.diff(clock) <= 0):
            raise ValueError("Expected continuous Shimmer samples followed by any padding")
        clock -= clock[0]
        inertial = record.p_signal[finite, 3:].T
        person, task = name.split("_", 1)
        for end in range(WINDOW_SECONDS, len(ecg) // exercise.SOURCE_RATE + 1, WINDOW_SECONDS):
            scheduled += 1
            ppg_clip = exercise.clocked_window(ppg[finite], clock, end)
            acc = np.column_stack([exercise.clocked_window(axis, clock, end) for axis in inertial])
            if not np.isfinite(ppg_clip).all() or not np.isfinite(acc).all() or np.ptp(ppg_clip) <= 0:
                continue
            start = end - WINDOW_SECONDS
            native = beats[(beats >= start) & (beats < end)]
            pulse.append(ppg_clip)
            motion.append(acc)
            group.append(person)
            activity.append(task)
            reference.append((native - start) * PPG_RATE)
    return BeatWindows(np.stack(pulse), np.stack(motion), np.asarray(group), np.asarray(activity), reference, scheduled)


def prepare(windows):
    def standardize(rows, high):
        filtered = np.stack([_bandpass(row, PPG_RATE, 0.4, high) for row in rows])
        scale = np.std(filtered, axis=1, keepdims=True)
        return np.clip((filtered - np.mean(filtered, axis=1, keepdims=True)) / np.maximum(scale, 1e-6), -6, 6)

    optical = standardize(windows.pulse, 8.0)
    # Acceleration magnitude is insensitive to how the device axes are oriented.
    movement = standardize(np.linalg.norm(windows.motion, axis=2), 10.0)
    return np.stack((optical, movement), axis=1).astype(np.float32)


def reference_targets(references):
    expected = np.zeros((len(references), SAMPLES), dtype=np.float32)
    for row, beats in zip(expected, references, strict=True):
        indices = np.clip(np.rint(beats).astype(int), 0, SAMPLES - 1)
        row[indices] = 1.0
    smoothed = gaussian_filter1d(expected, TARGET_STD_SAMPLES, axis=1, mode="constant")
    gaussian_peak = gaussian_filter1d(np.eye(1, 2 * EDGE + 1, EDGE).ravel(), TARGET_STD_SAMPLES)[EDGE]
    return np.clip(smoothed / gaussian_peak, 0.0, 1.0).astype(np.float32)


def train(inputs, targets, participants, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = BeatCNN(inputs.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    array = torch.from_numpy(inputs).to(device)
    expected = torch.from_numpy(targets[:, EDGE:-EDGE].copy()).to(device)
    names, membership, counts = np.unique(participants, return_inverse=True, return_counts=True)
    weights = len(participants) / (len(names) * counts[membership])
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    positive = torch.tensor(POSITIVE_WEIGHT, device=device)
    for _ in range(EPOCHS):
        for rows in np.array_split(rng.permutation(len(inputs)), int(np.ceil(len(inputs) / BATCH_SIZE))):
            batch = array[rows]
            signs = torch.as_tensor(rng.choice(np.array([-1, 1], dtype=np.float32), size=(len(rows), 1)), device=device)
            batch[:, 0] *= signs
            logits = model(batch)[:, EDGE:-EDGE]
            loss = F.binary_cross_entropy_with_logits(logits, expected[rows], pos_weight=positive, reduction="none")
            optimizer.zero_grad(set_to_none=True)
            (loss.mean(dim=1) * weights[rows]).mean().backward()
            optimizer.step()
    return model.eval()


def predict(model, inputs, device):
    with torch.inference_mode():
        probabilities = [torch.sigmoid(model(torch.from_numpy(batch).to(device))).cpu().numpy()
                         for batch in np.array_split(inputs, int(np.ceil(len(inputs) / BATCH_SIZE)))]
    return [find_peaks(row, height=PROBABILITY_CUTOFF, distance=PPG_RATE // 4)[0] for row in np.concatenate(probabilities)]


def summarize(windows, predictions):
    mask = np.ones(SAMPLES, dtype=bool)
    mask[:EDGE] = mask[-EDGE:] = False
    scores = {name: [_peak_score(reference, estimated, mask, PPG_RATE)
                     for reference, estimated in zip(windows.reference, values, strict=True)] for name, values in predictions.items()}
    results = {}
    for name, values in scores.items():
        pooled = summarize_beats(values)
        people = [summarize_beats([value for value, selected in zip(values, windows.participants == person, strict=True) if selected])
                  for person in np.unique(windows.participants)]
        activity = {task: summarize_beats([value for value, selected in zip(values, windows.activity == task, strict=True) if selected])
                    for task in np.unique(windows.activity)}
        results[name] = {**pooled, "participant_mean_f1": float(np.mean([person["f1"] for person in people])),
                         "by_activity": activity}
    return {"participants": len(np.unique(windows.participants)), "scheduled_windows": windows.scheduled,
            "paired_windows": len(windows.pulse), "methods": results}


def evaluate(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source, reference_count = load_move(args.move_dir)
    s_input = prepare(source)
    s_targets = reference_targets(source.reference)
    delay = round(ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE / 1000)
    source_predictions = {METHODS[0]: [_pulse_peaks(window, PPG_RATE)[0] - delay for window in source.pulse]}
    models, fold_sizes = {}, []
    for trained, held in GroupKFold(n_splits=5).split(s_input, groups=source.participants):
        fold_sizes.append({"training_participants": len(np.unique(source.participants[trained])),
                           "held_participants": len(np.unique(source.participants[held])), "held_windows": len(held)})
    for channels, method in enumerate(METHODS[1:], 1):
        values = [None] * len(s_input)
        for trained, held in GroupKFold(n_splits=5).split(s_input, groups=source.participants):
            model = train(s_input[trained, :channels], s_targets[trained], source.participants[trained], args.device)
            peaks = predict(model, s_input[held, :channels], args.device)
            for row, estimated in zip(held, peaks, strict=True):
                values[row] = estimated
        source_predictions[method] = values
        models[method] = train(s_input[:, :channels], s_targets, source.participants, args.device)
    validation = load_shimmer(args.shimmer_dir)
    v_input = prepare(validation)
    validation_predictions = {METHODS[0]: [_pulse_peaks(window, PPG_RATE)[0] - delay for window in validation.pulse]}
    validation_predictions.update({method: predict(models[method], v_input[:, :channels], args.device)
                                   for channels, method in enumerate(METHODS[1:], 1)})
    return {"study": "ECG R-peak timing from wrist signals", "seed": SEED,
            "epochs": EPOCHS, "edge_seconds": EDGE / PPG_RATE, "fixed_ppg_delay_ms": ARCHIVE_ASSUMED_DELAY_MS,
            "probability_threshold": PROBABILITY_CUTOFF, "beat_match_tolerance_ms": 100,
            "source_reference_eligible_windows": reference_count, "source_folds": fold_sizes,
            "scientisst_move_grouped": summarize(source, source_predictions),
            "shimmer_frozen_transfer": summarize(validation, validation_predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move-dir", type=Path, required=True)
    parser.add_argument("--shimmer-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
