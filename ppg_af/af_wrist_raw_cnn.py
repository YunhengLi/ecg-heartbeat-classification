"""Compare a compact raw PPG AF model with paired-data template training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .af_wrist_beat_cnn import standardize
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_training_expansion import _fit_expanded


MODEL_SEED = 2026
EPOCHS = 18
BATCH_SIZE = 32
WIDTH = 16


class AFSignalCNN(nn.Module):
    """Learn optical pulse shape and slower variation in the same window."""

    def __init__(self):
        super().__init__()
        self.pulse = nn.Conv1d(1, WIDTH, 41, padding=20, stride=4)
        self.interval = nn.Conv1d(WIDTH, WIDTH, 11, padding=5, stride=2)
        self.rhythm = nn.Conv1d(WIDTH, WIDTH, 9, dilation=8, padding=32)
        self.readout = nn.Linear(WIDTH * 2, 1)

    def forward(self, values):
        pulse = F.silu(self.pulse(values))
        intervals = F.silu(self.interval(pulse))
        rhythm = intervals + F.silu(self.rhythm(intervals))
        central = rhythm[:, :, 13:-13]
        spread = (central.var(dim=2, unbiased=False) + 1e-5).sqrt()
        return self.readout(torch.cat((central.mean(dim=2), spread), dim=1)).squeeze(1)


def fit_and_predict(raw, labels, inputs, device):
    torch.manual_seed(MODEL_SEED)
    random = np.random.default_rng(MODEL_SEED)
    model = AFSignalCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    pulse = torch.as_tensor(raw[:, None], dtype=torch.float32, device=device)
    outcomes = torch.as_tensor(labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(len(labels) / (2 * np.bincount(labels)[labels]), dtype=torch.float32, device=device)
    for _ in range(EPOCHS):
        for batch in np.array_split(random.permutation(len(raw)), int(np.ceil(len(raw) / BATCH_SIZE))):
            forecast = model(pulse[batch])
            loss = F.binary_cross_entropy_with_logits(forecast, outcomes[batch], weight=weights[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        matrices = np.array_split(inputs, int(np.ceil(len(inputs) / BATCH_SIZE)))
        scores = [torch.sigmoid(model(torch.as_tensor(matrix[:, None], dtype=torch.float32, device=device))).cpu().numpy()
                  for matrix in matrices]
    return np.concatenate(scores)


def evaluate(args):
    grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
    previous = json.loads(args.waveform_results.read_bytes())
    people, pulses, _, labels, _ = read_pairs(args.waveform_pairs, previous, lock_digest, set(grid.subjects))
    with np.load(args.waveform_pairs, allow_pickle=False) as verified:
        supplement = verified["origin"] == "supplemental"
    if np.count_nonzero(supplement) != 109 or np.bincount(labels[supplement]).tolist() != [66, 43]:
        raise ValueError("Use the same verified 109 paired training additions for both model families")
    added_people, added_labels, added_pulse = people[supplement], labels[supplement], pulses[supplement]
    pulse = stage_cores(grid, paths)
    shared = _shared_features(grid.table)
    raw = np.full(pulse.shape, np.nan, dtype=np.float32)
    raw[shared] = standardize(pulse[shared], 8.0)
    added_raw = standardize(added_pulse, 8.0).astype(np.float32)
    model_features = quality["quality_interactions"]
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    added_features = _extract(added_pulse, template)["quality_interactions"]
    if not np.isfinite(added_features).all():
        raise ValueError("All 109 additions must be scoreable in both model families")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fold_calls = 0

    def model(train, labels, holds):
        nonlocal fold_calls
        fold_calls += 1
        return fit_and_predict(train, labels, holds, args.device)

    comparisons = {}
    for seed in SEEDS:
        baseline = fit_template(grid, model_features, lock, DEFAULT_C, seed=seed)
        replay = _replay(grid, baseline)
        if seed == SEEDS[0] and replay[0] != original:
            raise ValueError("Reproduce the original quality template before comparing model families")
        control = _fit_expanded(grid, model_features, added_features, added_people, added_labels, lock, seed)
        candidate = _fit_expanded(grid, raw, added_raw, added_people, added_labels, lock, seed, fold_predictor=model)
        control_route, candidate_route = (value["routes"][ROUTE] for value in (control, candidate))
        fields = ("fit_af_windows", "fit_non_af_windows", "inner_splits", "held_out_finite_scores")
        if (not np.array_equal(control["fold_assignment"], candidate["fold_assignment"])
                or not np.array_equal(np.isfinite(control_route["scores"]), np.isfinite(candidate_route["scores"]))
                or any(a[key] != b[key] for a, b in zip(control_route["folds"], candidate_route["folds"], strict=True)
                       for key in fields)):
            raise ValueError("The template and waveform classifiers need the same fitting and scoring support")
        template_replay = _replay(grid, control)
        waveform_replay = _replay(grid, candidate)
        comparisons[str(seed)] = {"original_quality_template": replay[0], "expanded_quality_template": template_replay[0],
                                  "expanded_raw_ppg_cnn": _compare_with_pre_onset(template_replay, waveform_replay, lock)}
        print(f"{seed}: template {template_replay[0]['episodes_detected']}, {template_replay[0]['wrist_false_alerts']}; "
              f"CNN {waveform_replay[0]['episodes_detected']}, {waveform_replay[0]['wrist_false_alerts']}", flush=True)
    return {"scope": "Train raw optical AF classifier and template classifier on the same wrist windows",
            "added_windows": len(added_labels), "cnn_training_runs": fold_calls, "cnn_epochs": EPOCHS,
            "cnn_parameter_count": sum(parameter.numel() for parameter in AFSignalCNN().parameters()), "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
