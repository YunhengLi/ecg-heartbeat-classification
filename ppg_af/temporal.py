"""A compact offline temporal CNN for supervised PPG-to-ECG reconstruction."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DILATIONS = (1, 2, 4, 8, 16)
CONTEXT_SAMPLES = 2 + 2 * sum(DILATIONS)
QRS_RADIUS_SAMPLES = 4


class TemporalECG(nn.Module):
    """Predict normalized ECG amplitude and R-peak logits on the input grid."""

    def __init__(self, channels: int = 12):
        super().__init__()
        self.stem = nn.Conv1d(2, channels, kernel_size=5, padding=2)
        self.layers = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=5, dilation=dilation, padding=2 * dilation)
            for dilation in DILATIONS
        )
        self.head = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, inputs):
        features = F.silu(self.stem(inputs))
        for layer in self.layers:
            features = features + 0.5 * F.silu(layer(features))
        outputs = self.head(features)
        return outputs[:, 0], outputs[:, 1]


def _ppg_tensor(windows):
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("PPG windows must have shape (windows, samples)")
    observed = np.isfinite(values)
    signal = np.zeros_like(values)
    for row in np.flatnonzero(observed.any(axis=1)):
        valid = observed[row]
        samples = values[row, valid]
        samples = samples.astype(np.float64) - float(np.median(samples))
        scale = np.percentile(np.abs(samples), 95)
        if scale == 0:
            scale = np.max(np.abs(samples))
        if scale > 0:
            signal[row, valid] = np.clip(samples / scale, -8, 8)
        else:
            observed[row] = False
    return torch.from_numpy(np.stack((signal, observed.astype(np.float32)), axis=1))


def _select_indices(indices, size, name):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty vector of window indices")
    if np.any((indices < 0) | (indices >= size)) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must contain valid, distinct window indices")
    return indices


def _dataset(ppg, ecg, r_peaks, indices, qrs_weight):
    inputs = _ppg_tensor(ppg[indices])
    target = np.asarray(ecg[indices], dtype=np.float32)
    peaks = np.asarray(r_peaks[indices], dtype=np.float32)
    valid = np.isfinite(target) & inputs[:, 1].numpy().any(axis=1, keepdims=True)
    valid[:, :CONTEXT_SAMPLES] = False
    valid[:, -CONTEXT_SAMPLES:] = False
    peak_valid = valid & np.isfinite(peaks)
    if not valid.any():
        raise ValueError("The selected split has no observed PPG and ECG training targets")
    if np.any((peaks[peak_valid] < 0) | (peaks[peak_valid] > 1)):
        raise ValueError("R-peak targets must be probabilities in [0, 1]")
    waveform_weights = valid.astype(np.float32)
    if qrs_weight > 0:
        peak_centers = torch.from_numpy(((peaks == 1.0) & np.isfinite(target)).astype(np.float32))
        nearby = F.max_pool1d(
            peak_centers[:, None], kernel_size=2 * QRS_RADIUS_SAMPLES + 1,
            stride=1, padding=QRS_RADIUS_SAMPLES,
        )[:, 0].numpy()
        waveform_weights *= 1.0 + qrs_weight * nearby
    return (
        inputs,
        torch.from_numpy(np.where(valid, target, 0)),
        torch.from_numpy(waveform_weights),
        torch.from_numpy(np.where(peak_valid, peaks, 0)),
        torch.from_numpy(peak_valid.astype(np.float32)),
    )


def _batch_losses(model, batch, pos_weight):
    inputs, ecg, waveform_weights, peaks, peak_valid = batch
    waveform, logits = model(inputs)
    waveform_error = F.smooth_l1_loss(waveform, ecg, beta=0.2, reduction="none")
    waveform_sum = (waveform_error * waveform_weights).sum()
    if pos_weight is None:
        return waveform_sum, None
    peak_error = F.binary_cross_entropy_with_logits(logits, peaks, pos_weight=pos_weight, reduction="none")
    return waveform_sum, (peak_error * peak_valid).sum()


def fit_tcn(
    ppg, ecg, r_peaks, train_indices, validation_indices, *, peak_weight, seed,
    epochs=12, batch_size=16, patience=4, channels=12, threads=2, qrs_weight=0.0,
):
    """Train and restore the epoch with lowest validation waveform loss.

    Arrays have shape (windows, samples). ECG targets are already normalized.
    Missing values may be NaN. A caller supplies disjoint subject-level splits.
    Convolutions use up to 64 samples of future context per prediction.
    PPG normalization uses the full window, so this implementation is offline.
    A nonzero qrs_weight adds that weight to valid waveform samples within four
    samples of each annotated R peak during training and validation.
    """
    ppg, ecg, r_peaks = (np.asarray(array) for array in (ppg, ecg, r_peaks))
    if ppg.ndim != 2 or ppg.shape != ecg.shape or ppg.shape != r_peaks.shape:
        raise ValueError("PPG, ECG and R-peak targets must share shape (windows, samples)")
    if ppg.shape[1] <= 2 * CONTEXT_SAMPLES:
        raise ValueError(f"Windows must have more than {2 * CONTEXT_SAMPLES} samples")
    if min(epochs, batch_size, patience, channels, threads) < 1:
        raise ValueError("Epochs, batch size, patience, channels and threads must be positive")
    if not np.isfinite(peak_weight) or peak_weight < 0:
        raise ValueError("Peak loss weight must be finite and nonnegative")
    if not np.isfinite(qrs_weight) or qrs_weight < 0:
        raise ValueError("QRS waveform weight must be finite and nonnegative")
    train_indices = _select_indices(train_indices, len(ppg), "Training indices")
    validation_indices = _select_indices(validation_indices, len(ppg), "Validation indices")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Training and validation indices must be disjoint")

    train = _dataset(ppg, ecg, r_peaks, train_indices, qrs_weight)
    validation = _dataset(ppg, ecg, r_peaks, validation_indices, qrs_weight)
    training_samples = float(train[2].sum())
    training_peak_samples = float(train[4].sum())
    validation_samples = float(validation[2].sum())
    validation_peak_samples = float(validation[4].sum())
    pos_weight = None
    if peak_weight > 0:
        positive_count = float(train[3].sum())
        if positive_count <= 0 or positive_count >= training_peak_samples:
            raise ValueError("Peak-supervised training needs both peak and non-peak examples")
        ratio = (training_peak_samples - positive_count) / positive_count
        pos_weight = torch.tensor(np.clip(ratio, 1.0, 32.0), dtype=torch.float32)

    # Forward and backward on this small network are faster with few CPU threads.
    torch.set_num_threads(threads)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TemporalECG(channels)
    shuffle = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        waveform_total = peak_total = 0.0
        for indices in torch.randperm(len(train_indices), generator=shuffle).split(batch_size):
            batch = tuple(tensor[indices] for tensor in train)
            waveform_sum, peak_sum = _batch_losses(model, batch, pos_weight)
            loss = waveform_sum / batch[2].sum().clamp_min(1)
            if peak_sum is not None:
                loss = loss + peak_weight * peak_sum / batch[4].sum().clamp_min(1)
                peak_total += float(peak_sum.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            waveform_total += float(waveform_sum.detach())

        model.eval()
        validation_total = validation_peak_total = 0.0
        with torch.inference_mode():
            for start in range(0, len(validation_indices), batch_size):
                batch = tuple(tensor[start : start + batch_size] for tensor in validation)
                waveform_sum, peak_sum = _batch_losses(model, batch, pos_weight)
                validation_total += float(waveform_sum)
                if peak_sum is not None:
                    validation_peak_total += float(peak_sum)

        validation_loss = validation_total / validation_samples
        if not np.isfinite(validation_loss):
            raise RuntimeError("TCN training produced a nonfinite validation loss")
        history.append({
            "epoch": epoch,
            "train_waveform_loss": waveform_total / training_samples,
            "train_peak_loss": peak_total / training_peak_samples if pos_weight is not None else None,
            "validation_waveform_loss": validation_loss,
            "validation_peak_loss": (validation_peak_total / validation_peak_samples
                                     if pos_weight is not None and validation_peak_samples else None),
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    for row in history:
        row["selected"] = row["epoch"] == best_epoch
    return model.eval(), history


def predict_tcn(model, ppg_windows, *, batch_size=16):
    """Predict ECG amplitudes and R-peak logits using PPG alone."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    inputs = _ppg_tensor(ppg_windows)
    result = np.full((2, len(inputs), inputs.shape[-1]), np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            waveform, peaks = model(batch)
            result[:, start : start + len(batch)] = torch.stack((waveform, peaks)).numpy()
    result[:, ~inputs[:, 1].numpy().any(axis=1)] = np.nan
    return result[0], result[1]
