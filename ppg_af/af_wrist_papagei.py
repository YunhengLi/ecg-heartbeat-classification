"""Train group-held-out AF heads on frozen PaPaGei-S and random PPG features."""

import argparse
from hashlib import sha1, sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.signal import cheby2, filtfilt, resample_poly
from sklearn.metrics import roc_auc_score

from .af_wrist_confirmation_spacing import DEVELOPMENT_CONFIRMATION_SHA
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support


WEIGHTS_SHA256 = "79d68671e51bdc951fee853e5a8241099e38148ff26ed888b843849332e0b988"
SOURCE_BLOB = "574d169f2dd097aa6186372b71e8f64ea34f644e"
CONFIG = dict(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
              n_block=18, n_classes=512, n_experts=3)
INPUT_SECONDS, MODEL_RATE = 10, 125
INIT_SEED, BATCH = 2026, 64


def load_encoders(source, weights, device):
    import torch

    source_bytes = source.read_bytes()
    header = f"blob {len(source_bytes)}\0".encode()
    if sha1(header + source_bytes).hexdigest() != SOURCE_BLOB or sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Use the verified public PaPaGei-S source file and published tensor checkpoint")
    spec = spec_from_file_location("papagei_public_resnet", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(INIT_SEED)
    reference = module.ResNet1DMoE(**CONFIG)
    random = module.ResNet1DMoE(**CONFIG)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    reference.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
    return {name: model.to(device).eval() for name, model in (("papagei_s", reference), ("random_frozen_encoder", random))}


def preprocess(cores):
    """Apply the paper's filtering and per-segment normalization on trailing data."""
    if cores.ndim != 2 or cores.shape[1] != WINDOW_SECONDS * PPG_RATE or not np.isfinite(cores).all():
        raise ValueError("PaPaGei requires complete 30-second causal wrist inputs")
    b, a = cheby2(4, 20, (0.5, 12), "bandpass", fs=PPG_RATE)
    kernel = np.full(round(0.05 * PPG_RATE), 1 / round(0.05 * PPG_RATE))
    smooth = filtfilt(kernel, 1, filtfilt(b, a, cores), axis=-1)
    parts = smooth.reshape(-1, INPUT_SECONDS * PPG_RATE)
    scale = parts.std(axis=-1, keepdims=True)
    normalized = np.divide(parts - parts.mean(axis=-1, keepdims=True), scale, where=scale > 0, out=np.zeros_like(parts))
    resampled = resample_poly(normalized, MODEL_RATE, PPG_RATE, axis=-1)
    return np.ascontiguousarray(resampled[:, None], dtype=np.float32), int((scale == 0).sum())


def embed(model, segments, device):
    import torch

    with torch.inference_mode():
        vectors = [model(torch.from_numpy(batch).to(device))[0].cpu().numpy()
                   for batch in (segments[start:start + BATCH] for start in range(0, len(segments), BATCH))]
    values = np.concatenate(vectors)
    width = CONFIG["n_classes"]
    if values.shape != (len(segments), width) or not np.isfinite(values).all():
        raise ValueError("A frozen encoder must return one finite vector per ten-second segment")
    return values.astype(float).reshape(-1, WINDOW_SECONDS // INPUT_SECONDS, width).mean(axis=1)


def ranking(fitted, grid, shared):
    stable = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    scores = fitted["routes"][ROUTE]["scores"][stable]
    labels, people = grid.table["stable_label"][stable], grid.subjects[stable]
    within = []
    for person in np.unique(people):
        selected = people == person
        if len(np.unique(labels[selected])) > 1:
            within.append(float(roc_auc_score(labels[selected], scores[selected])))
    return {"stable_windows": len(labels), "pooled_stable_auc": float(roc_auc_score(labels, scores)),
            "within_participant_stable_auc": float(np.mean(within)) if within else None,
            "participants_with_both_stable_rhythms": len(within)}


def evaluate(args):
    import torch

    grid, quality, paths, lock, _, _ = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)[shared]
    segments, constant = preprocess(cores)
    models = load_encoders(args.architecture, args.weights, args.device)
    matrices = {}
    for name, model in models.items():
        embeddings = embed(model, segments, args.device)
        full = np.full((len(shared), embeddings.shape[1]), np.nan)
        full[shared] = embeddings
        matrices[name] = full
        matrices[f"template_plus_{name}"] = np.column_stack((quality["baseline"], full))
        print(f"Encoded {len(embeddings)} identical causal windows with {name}", flush=True)
    prior = _read_pinned(args.development_confirmation_results, DEVELOPMENT_CONFIRMATION_SHA, public=True)["comparisons"]
    results = {}
    for seed in SEEDS:
        plain = fit_template(grid, quality["baseline"], lock, DEFAULT_C, seed=seed)
        baseline = _replay(grid, plain)
        if baseline[0] != prior[str(seed)]["plain_template"]["original"]:
            raise ValueError("The original 30-second wrist template must reproduce")
        outcomes = {"original_template": {"summary": baseline[0], "ranking": ranking(plain, grid, shared)}}
        for name, matrix in matrices.items():
            fitted = fit_template(grid, matrix, lock, DEFAULT_C, seed=seed)
            verify_fit_support(plain, fitted)
            candidate = _replay(grid, fitted)
            outcomes[name] = {**_compare_with_pre_onset(baseline, candidate, lock), "ranking": ranking(fitted, grid, shared),
                              "calibration_status": _calibration_status(fitted)}
            print(f"{seed}, {name}: {candidate[0]['episodes_detected']} detected, {candidate[0]['wrist_false_alerts']} false starts", flush=True)
        results[str(seed)] = outcomes
    return {"scope": "PaPaGei-S frozen PPG representation on the locked wrist AF development folds",
            "weights_sha256": WEIGHTS_SHA256, "architecture_git_blob": SOURCE_BLOB, "torch_version": str(torch.__version__),
            "device": args.device, "random_encoder_seed": INIT_SEED, "encoder_parameters": sum(p.numel() for p in models["papagei_s"].parameters()),
            "scoreable_rows": int(shared.sum()), "input_seconds": INPUT_SECONDS, "input_rate_hz": MODEL_RATE,
            "ten_second_segments": len(segments), "constant_ten_second_segments": constant,
            "aggregation": "mean of three disjoint trailing ten-second embeddings", "comparisons": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "development-confirmation-results", "architecture", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
