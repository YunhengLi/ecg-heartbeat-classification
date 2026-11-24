"""Evaluate a fixed MIMIC PPG CNN on the original wrist development folds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from torch import nn
from torch.nn import functional as F

from . import af_wrist_mimic_transfer as transfer
from .af_wrist_clock_study import load_grid
from .af_wrist_cv import _within_rhythm_participant_weights
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_template_context import _archive_window
from .af_wrist_template_independent import _archives
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS


SEED = 2026
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CPU_THREADS = 2
POLICIES = {"80_percent_target": False, "matched_template_sensitivity": True}


class PulseCNN(nn.Module):
    """Classify the fixed 30-second pulse window with global temporal pooling."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 15, stride=5, padding=7), nn.SiLU(),
            nn.Conv1d(8, 12, 9, stride=2, padding=4), nn.SiLU(),
            nn.Conv1d(12, 16, 9, stride=2, padding=4), nn.SiLU(),
            nn.Conv1d(16, 16, 7, dilation=2, padding=6), nn.SiLU(),
        )
        self.classifier = nn.Linear(32, 1)

    def embedding(self, pulse):
        signal = self.features(pulse)
        return torch.cat((signal.mean(dim=-1), signal.var(dim=-1, unbiased=False)), dim=1)

    def forward(self, pulse):
        return self.classifier(self.embedding(pulse)).squeeze(-1)


def _normalize(windows):
    centered = windows - np.median(windows, axis=1, keepdims=True)
    scale = np.percentile(np.abs(centered), 95, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, np.max(np.abs(centered), axis=1, keepdims=True))
    if not np.isfinite(windows).all() or not np.all(scale > 0):
        raise ValueError("Every allowed CNN input needs a finite, nonflat 30-second pulse window")
    return torch.from_numpy(np.clip(centered / scale, -8, 8).astype(np.float32))[:, None, :]


def _fit_cnn(raw, labels, people):
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    pulse = _normalize(raw)
    target = torch.from_numpy(labels.astype(np.float32))
    weights = torch.from_numpy(_within_rhythm_participant_weights(labels, people).astype(np.float32))
    model = PulseCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for _ in range(EPOCHS):
        total = 0.0
        for rows in torch.randperm(len(target), generator=shuffle).split(BATCH_SIZE):
            loss = F.binary_cross_entropy_with_logits(model(pulse[rows]), target[rows], reduction="none")
            weighted = (loss * weights[rows]).sum()
            optimizer.zero_grad(set_to_none=True)
            (weighted / len(rows)).backward()
            optimizer.step()
            total += float(weighted.detach())
        losses.append(total / float(weights.sum()))
    if not np.isfinite(losses).all():
        raise ValueError("Fixed CNN training produced nonfinite source loss")
    return model.eval().requires_grad_(False), losses


def _wrist_input(grid, archives, version):
    shared = _shared_features(grid.table)
    windows = np.full((len(shared), WINDOW_SECONDS * PPG_RATE), np.nan)
    offset = 0
    parent_raw, parent_features, stride_raw, stride_features = archives
    for finer, raw, features in ((False, parent_raw, parent_features), (True, stride_raw, stride_features)):
        for person in np.unique(grid.subjects):
            name = f"{person}.npz"
            core, _, rows = _archive_window(grid, raw / name, features / name, person, offset,
                                            finer=finer, version=version)
            block = windows[rows]
            block[shared[rows]] = core[shared[rows]]
            offset = rows.stop
    if offset != len(shared) or not np.array_equal(np.isfinite(windows).all(axis=1), shared):
        raise ValueError("The CNN waveform rows must match the original shared grid")
    return windows, shared


def _frozen_scores(grid, windows, shared, logistic, model):
    probabilities = {name: np.full(len(shared), np.nan) for name in ("source_logistic", "source_CNN")}
    probabilities["source_logistic"][shared] = logistic.predict_proba(grid.table[transfer.DIRECT][shared])[:, 1]
    with torch.inference_mode():
        inputs = _normalize(windows[shared])
        probabilities["source_CNN"][shared] = torch.cat([
            torch.sigmoid(model(batch)) for batch in inputs.split(BATCH_SIZE)
        ]).numpy()
    if any(not np.array_equal(np.isfinite(scores), shared) for scores in probabilities.values()):
        raise ValueError("Frozen source scoring changed original wrist eligibility")
    return probabilities


def _compare_scores(grid, lock, source_report, probabilities):
    template = fit_template(grid, grid.table[ROUTE], lock, DEFAULT_C, seed=SEED)
    assignments = template["fold_assignment"]
    template_route = template["routes"][ROUTE]
    _, cutoffs, _ = fold_arrays(template, template_route)
    baseline = _replay(grid, template)
    if baseline[0] != source_report["wrist_template_baseline"]:
        raise ValueError("The original wrist template outcome did not reproduce")
    replays, statuses = {}, {}
    for name, scores in probabilities.items():
        points = [transfer._calibrate(grid, scores, set(grid.subjects[assignments != fold["fold"]]),
                                      lock, fold["operating_point"]) for fold in template_route["folds"]]
        replays[name], statuses[name] = {}, {}
        for label, matched in POLICIES.items():
            replay, status = transfer._replay_candidate(grid, scores, assignments, points, cutoffs, matched=matched)
            replays[name][label], statuses[name][label] = replay, status
            if name == "source_logistic":
                recorded = source_report[f"frozen_mimic_direct_{label}"]
                if replay[0] != recorded["summary"] or status != recorded["training_fold_calibration_status_counts"]:
                    raise ValueError("The frozen source logistic comparison did not reproduce")
            print(f"{name}, {label}: {replay[0]['episodes_detected']}/{replay[0]['complete_episodes']}, "
                  f"{replay[0]['wrist_false_alerts']} wrist false starts", flush=True)

    comparison = {}
    for label in POLICIES:
        before, after = (replays[name][label] for name in ("source_logistic", "source_CNN"))
        comparison[label] = {"source_logistic": before[0], "source_CNN": after[0],
                             "source_CNN_vs_source_logistic": transfer._comparison(before, after, lock),
                             "source_CNN_vs_wrist_template": transfer._comparison(baseline, after, lock),
                             "fold_calibration_status": {name: choices[label] for name, choices in statuses.items()}}
    return baseline[0], comparison


def evaluate(args):
    source_report = json.loads(args.transfer_results.read_bytes())
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    clock = json.loads(args.clock_results.read_bytes())
    table, people, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    development = fit | calibration
    digest = sha256(lock_bytes).hexdigest()
    version = table[FEATURE_VERSION_KEY]
    archives = (args.development_raw, args.development_features, args.stride_raw, args.stride_features)
    if (set(people) != development or fit & calibration or development & reserved
            or any(set(_archives(path)) != development for path in archives)
            or version != lock[FEATURE_VERSION_KEY] or clock["original_lock_sha256"] != digest
            or source_report["wrist_lock_sha256"] != digest):
        raise ValueError("Use the original locked development participants, archives and reports")
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    matrix, labels, groups, inventory, source_raw = transfer._load_source(args.source, with_waveforms=True)
    if (inventory != source_report["source_class_inventory"] or len(labels) != 1342 or len(np.unique(groups)) != 35
            or source_raw.shape != (len(labels), WINDOW_SECONDS * PPG_RATE)
            or source_report["source_doi"] != transfer.SOURCE_DOI
            or source_report["source_md5"] != {name: digest for name, digest, _, _ in transfer.SOURCES}):
        raise ValueError("The CNN must reuse the published checksum-verified source windows")
    logistic = transfer._fit_source(matrix, labels, groups)
    model, losses = _fit_cnn(source_raw, labels, groups)
    print(f"Froze the CNN after {EPOCHS} source-only epochs on {len(labels)} windows", flush=True)

    windows, shared = _wrist_input(grid, archives, version)
    probabilities = _frozen_scores(grid, windows, shared, logistic, model)
    baseline, comparison = _compare_scores(grid, lock, source_report, probabilities)
    evaluated = np.concatenate([block.rows for block in grid.timelines])
    history = np.concatenate(list(grid.history_rows.values()))
    return {
        "scope": "Scoped MIMIC source-only CNN and frozen wrist development comparison",
        "source_doi": transfer.SOURCE_DOI, "source_md5": source_report["source_md5"], "source_inventory": inventory,
        "source_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_recipe": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                          "seed": SEED, "cpu_threads": CPU_THREADS, "pytorch": str(torch.__version__),
                          "pooling": "temporal mean plus variance",
                          "loss_weighting": "Equal total weight by rhythm, equal participant weight within rhythm",
                          "source_training_loss_by_epoch": losses},
        "dependency_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "wrist_shared_rows": int(shared.sum()), "wrist_shared_evaluation_rows": int(shared[evaluated].sum()),
        "wrist_shared_history_rows": int(shared[history].sum()), "wrist_independent_predictions": 0,
        "wrist_lock_sha256": digest, "wrist_template_baseline": baseline, "comparisons": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("source", "development-raw", "development-features", "stride-raw", "stride-features", "lock",
             "clock-results", "transfer-results", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new aggregate output path")
    try:
        _write_json(args.output, evaluate(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"MIMIC source CNN study failed: {error}\n")


if __name__ == "__main__":
    main()
