"""Test synchronized ECG representation guidance for a PPG-only AF predictor."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from . import af_wrist_mimic_transfer as transfer
from .af_wrist_cv import _within_rhythm_participant_weights
from .af_wrist_failure_study import fold_arrays
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_mimic_cnn import BATCH_SIZE, CPU_THREADS, EPOCHS, LEARNING_RATE, POLICIES, SEED, PulseCNN, _normalize
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template


TRANSFER_SHA256 = "8055797166a25a735234fe672bd8d3216de0492e8918ae5a89eb46c6716c7e03"
ALIGNMENT_WEIGHT = 0.1
ARMS = ("PPG_only", "dual_sensor_classification", "synchronized_ECG_alignment", "shifted_ECG_alignment")


def shifted_partners(people):
    """Pair another disjoint window from the same person and source rhythm."""
    partners = np.empty(len(people), dtype=int)
    for person in np.unique(people):
        rows = np.flatnonzero(people == person)
        if len(rows) < 2:
            raise ValueError("The timing control requires two disjoint paired windows per source person")
        partners[rows] = np.roll(rows, max(1, len(rows) // 2))
    return partners


def fit(pulse, ecg, labels, people, arm, device):
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    model = PulseCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(_within_rhythm_participant_weights(labels, people), dtype=torch.float32, device=device)
    partners = shifted_partners(people) if arm == "shifted_ECG_alignment" else np.arange(len(labels))
    histories = []
    for _ in range(EPOCHS):
        total = 0.0
        for rows in torch.randperm(len(labels), generator=shuffle).split(BATCH_SIZE):
            optical = model.embedding(pulse[rows])
            predicted = model.classifier(optical).squeeze(-1)
            ppg_loss = F.binary_cross_entropy_with_logits(predicted, target[rows], reduction="none")
            loss = ppg_loss
            if arm != "PPG_only":
                electrical = model.embedding(ecg[partners[rows.numpy()]])
                observed = model.classifier(electrical).squeeze(-1)
                ecg_loss = F.binary_cross_entropy_with_logits(observed, target[rows], reduction="none")
                loss = 0.5 * (ppg_loss + ecg_loss)
                if arm.endswith("alignment"):
                    alignment = 1.0 - F.cosine_similarity(optical, electrical.detach(), dim=-1)
                    loss = loss + ALIGNMENT_WEIGHT * alignment
            weighted = (weights[rows] * loss).sum()
            optimizer.zero_grad(set_to_none=True)
            (weighted / len(rows)).backward()
            optimizer.step()
            total += float(weighted.detach().cpu())
        histories.append(total / float(weights.sum().cpu()))
    if not np.isfinite(histories).all():
        raise ValueError("The fixed source experiment must produce finite training losses")
    return model.eval().requires_grad_(False), histories


def probabilities(model, windows):
    with torch.inference_mode():
        return torch.cat([torch.sigmoid(model(batch)) for batch in windows.split(BATCH_SIZE)]).cpu().numpy()


def evaluate(args):
    source = _read_pinned(args.transfer_results, TRANSFER_SHA256, public=True)
    if sha256(args.lock.read_bytes()).hexdigest() != source["wrist_lock_sha256"]:
        raise ValueError("Use the original target wrist clock and participant folds")
    grid, _, paths, lock, _, _ = load_motion_development(args)
    _, labels, people, inventory, (raw_ppg, raw_ecg) = transfer._load_source(args.source, with_waveforms=True, with_reference=True)
    if inventory != source["source_class_inventory"] or len(labels) != 1342 or len(np.unique(people)) != 35:
        raise ValueError("The paired source must reproduce the original 35-person ICU input inventory")
    valid = np.isfinite(raw_ecg).all(axis=1) & (np.ptp(raw_ecg, axis=1) > 0)
    counts = np.bincount(people[valid], minlength=35)
    eligible = valid & (counts[people] >= 2)
    pulse = _normalize(raw_ppg[eligible]).to(args.device)
    ecg = _normalize(raw_ecg[eligible]).to(args.device)
    labels, people = labels[eligible], people[eligible]
    if len(np.unique(labels)) != 2:
        raise ValueError("The paired source subset must represent both reference rhythms")
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    device = args.device
    shared = _shared_features(grid.table)
    wrist = _normalize(stage_cores(grid, paths)[shared]).to(device)
    stable = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    predictions, histories = {}, {}
    for arm in ARMS:
        model, histories[arm] = fit(pulse, ecg, labels, people, arm, device)
        all_scores = np.full(len(shared), np.nan)
        all_scores[shared] = probabilities(model, wrist)
        predictions[arm] = all_scores
        print(f"Trained and froze {arm} on {len(labels)} paired source windows", flush=True)
    output = {}
    for seed in SEEDS:
        template = fit_template(grid, grid.table[ROUTE], lock, DEFAULT_C, seed=seed)
        route = template["routes"][ROUTE]
        _, cutoffs, _ = fold_arrays(template, route)
        assignments = template["fold_assignment"]
        baseline = _replay(grid, template)
        if seed == SEED and baseline[0] != source["wrist_template_baseline"]:
            raise ValueError("The published primary wrist reference must reproduce")
        results = {}
        for name, scores in predictions.items():
            points = [transfer._calibrate(grid, scores, set(grid.subjects[assignments != fold["fold"]]), lock,
                                          fold["operating_point"]) for fold in route["folds"]]
            metrics = {}
            for label, matched in POLICIES.items():
                replay, status = transfer._replay_candidate(grid, scores, assignments, points, cutoffs, matched=matched)
                metrics[label] = {**transfer._comparison(baseline, replay, lock), "training_calibration_status": status}
            results[name] = {"wrist_stable_auc": float(roc_auc_score(grid.table["stable_label"][stable], scores[stable])),
                             "comparisons": metrics}
        output[str(seed)] = {"wrist_template": baseline[0], "paired_pretraining": results}
    return {"scope": "Controlled SiamAF-inspired source-only pairing with wrist PPG-only inference",
            "source_doi": transfer.SOURCE_DOI, "source_participants": len(np.unique(people)), "eligible_pairs": len(labels),
            "source_windows_before_pair_filter": len(valid), "source_windows_with_valid_ECG": int(valid.sum()),
            "source_people_excluded_for_fewer_than_two_pairs": int(np.sum(counts < 2)),
            "source_pair_windows_by_rhythm": {"AF": int(np.sum(labels == 1)), "non_AF": int(np.sum(labels == 0))},
            "wrist_common_inputs": int(shared.sum()), "epochs": EPOCHS, "training_seed": SEED,
            "alignment_loss_weight": ALIGNMENT_WEIGHT, "device": device, "torch_version": str(torch.__version__),
            "source_losses": histories, "comparisons": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "transfer-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
