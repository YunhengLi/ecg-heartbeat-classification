"""Evaluate released RDDM weights on external PTT or author-prepared test arrays."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import neurokit2 as nk
import numpy as np
import torch
from scipy.signal import resample_poly

from .rddm_author import AUTHOR_DATASETS

AUTHOR_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
AUTHOR_RATE = 128
AUTHOR_WINDOW = 4
DIFFUSION_STEPS = 10
RANDOM_SEED = 31
WEIGHT_NAMES = ("rddm_main_network.pth", "rddm_condition_encoder_1.pth", "rddm_condition_encoder_2.pth")


def load_author_model(upstream, weights, device):
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise ValueError(f"Expected the authors' commit {AUTHOR_COMMIT}, received {commit}")

    sys.path.insert(0, str(upstream.resolve()))
    from diffusion import RDDM
    from model import ConditionNet, DiffusionUNetCrossAttention

    # Match the upstream loader. Only checkpoint placement and deserialization change.
    diffusion = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device),
        region_model=DiffusionUNetCrossAttention(512, 1, device),
        betas=(1e-4, 0.2),
        n_T=DIFFUSION_STEPS,
    )
    models = (diffusion, ConditionNet(), ConditionNet())
    for module, name in zip(models, WEIGHT_NAMES, strict=True):
        module.load_state_dict(torch.load(weights / name, map_location="cpu", weights_only=True), strict=True)
        module.to(device).eval().requires_grad_(False)
    return models


def author_scaled(windows, input_rate):
    upsampled = resample_poly(windows, AUTHOR_RATE, input_rate, axis=-1, padtype="line")
    chunks = upsampled.reshape(-1, AUTHOR_RATE * AUTHOR_WINDOW).astype(np.float32)
    scaled = chunks - chunks.min(axis=1, keepdims=True)
    span = scaled.max(axis=1, keepdims=True)
    return np.divide(2 * scaled, span, out=np.ones_like(scaled), where=span > 0) - 1


def predict_author_chunks(chunks, models, device):
    dpm, first, second = models
    pulse = torch.from_numpy(np.ascontiguousarray(chunks[:, None], dtype=np.float32)).to(device)
    with torch.inference_mode():
        predicted = dpm(cond1=first(pulse), cond2=second(pulse), mode="sample",
                        window_size=AUTHOR_RATE * AUTHOR_WINDOW)
    return predicted.cpu().numpy()[:, 0]


def generate_author_ecg(ppg, input_rate, models, batch_size, device):
    started = perf_counter()
    torch.manual_seed(RANDOM_SEED)
    scaled = author_scaled(ppg, input_rate)
    chunks = np.stack([nk.ppg_clean(chunk, sampling_rate=AUTHOR_RATE) for chunk in scaled])
    outputs = []
    for start in range(0, len(chunks), batch_size):
        outputs.append(predict_author_chunks(chunks[start:start + batch_size], models, device))
        completed = min(start + batch_size, len(chunks))
        if start == 0 or completed == len(chunks) or completed % 24 == 0:
            print(f"Author model: {completed}/{len(chunks)} four-second chunks", flush=True)
    author_estimate = np.concatenate(outputs)
    estimate = author_estimate.reshape(len(ppg), -1)
    estimate = resample_poly(estimate, input_rate, AUTHOR_RATE, axis=-1, padtype="line")
    if estimate.shape != ppg.shape or not np.isfinite(estimate).all():
        raise RuntimeError("RDDM did not produce finite output on the original time grid")
    return estimate, author_estimate, perf_counter() - started


def select_windows(windows, max_subjects):
    subjects = sorted(np.unique(windows.subjects), key=lambda value: int(value[1:]))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    selected = []
    for subject in subjects:
        for activity in sorted(np.unique(windows.activities)):
            candidates = np.flatnonzero((windows.subjects == subject) & (windows.activities == activity))
            if len(candidates):
                selected.append(candidates[len(candidates) // 2])
    return np.array(sorted(selected), dtype=int)


def existing_forecasts(windows, selected, checkpoints):
    from .temporal import TemporalECG, predict_tcn
    from .template import fit_template_from_stats
    from .train_tcn import SAMPLE_RATE, WINDOW_SECONDS, collect_baselines, make_folds
    from .waveform import fit_waveform_from_stats

    stats = collect_baselines(windows)
    methods = ["ridge_ptt", "pulse_template_ptt"]
    if checkpoints is not None:
        methods.append("qrs_cnn_ptt")
    outputs = {name: np.full(windows.ppg[selected].shape, np.nan) for name in methods}
    for fold, (train, validation, test) in enumerate(make_folds(windows.subjects, 5, 2026), start=1):
        local = np.flatnonzero(np.isin(selected, test))
        if not len(local):
            continue
        subjects = np.unique(windows.subjects[train])
        ridge = fit_waveform_from_stats(stats[subject][0] for subject in subjects)
        template = fit_template_from_stats(stats[subject][1] for subject in subjects)
        ppg = windows.ppg[selected[local]]
        outputs["ridge_ptt"][local] = np.stack([ridge.predict(pulse) for pulse in ppg])
        outputs["pulse_template_ptt"][local] = np.stack([template.predict(pulse) for pulse in ppg])
        if checkpoints is not None:
            state = torch.load(checkpoints / f"tcn_qrs_weighted-fold-{fold}.pt",
                               map_location="cpu", weights_only=True)
            expected_split = {name: np.unique(windows.subjects[indices]).tolist()
                              for name, indices in (("train_subjects", train), ("validation_subjects", validation),
                                                    ("test_subjects", test))}
            if (state["fold"] != fold or state["sample_rate_hz"] != SAMPLE_RATE
                    or state["window_seconds"] != WINDOW_SECONDS or state["method"] != "tcn_qrs_weighted"
                    or state.get("split") != expected_split
                    or state.get("activities") != np.unique(windows.activities).tolist()):
                raise ValueError(f"CNN checkpoint for PTT fold {fold} does not match the documented cohort and split")
            cnn = TemporalECG(channels=state["channels"])
            cnn.load_state_dict(state["weights"])
            outputs["qrs_cnn_ptt"][local], _ = predict_tcn(cnn, ppg)
    return outputs


def score(windows, selected, forecasts, rate):
    from .tcn_metrics import score_group

    def report_group(indices):
        native = selected[indices]
        metrics = score_group(windows.ecg[native], windows.r_peaks[native], windows.ppg[native],
                              {name: waveforms[indices] for name, waveforms in forecasts.items()},
                              {}, rate)
        for model in metrics["models"].values():
            # The upstream RDDM and existing PTT models use different amplitude units.
            model.pop("relative_mae")
        metrics.pop("zero_waveform_relative_mae")
        metrics.pop("auxiliary_peak_heads")
        return {"subjects": len(np.unique(windows.subjects[native])), **metrics}

    activities = windows.activities[selected]
    return {
        "overall": report_group(np.arange(len(selected))),
        "by_activity": {str(activity): report_group(np.flatnonzero(activities == activity))
                        for activity in np.unique(activities)},
    }


def run_ptt(args, models):
    from .train_tcn import SAMPLE_RATE, prepare_windows

    windows = prepare_windows(args.ptt_data, ("sit", "walk", "run"))
    selected = select_windows(windows, args.max_subjects)
    print(f"Selected {len(selected)} midpoint windows", flush=True)
    output, author_estimate, seconds = generate_author_ecg(windows.ppg[selected], SAMPLE_RATE, models,
                                                             args.batch_size, args.device)
    author_target = np.stack([
        nk.ecg_clean(chunk, sampling_rate=AUTHOR_RATE, method="pantompkins1985")
        for chunk in author_scaled(windows.ecg[selected], SAMPLE_RATE)
    ])
    forecasts = {"rddm_author_zero_shot": output, **existing_forecasts(windows, selected, args.tcn_checkpoints)}
    return {
        "scope": "Author-checkpoint inference on external PTT PPG",
        "dataset": "PhysioNet Pulse Transit Time PPG v1.1.0",
        "selection": "One midpoint 12-second window per subject and activity",
        "scoring_sample_rate_hz": SAMPLE_RATE,
        "authors_filtered_target_rmse_on_ptt": float(np.sqrt(np.mean((author_estimate - author_target)**2))),
        "author_inference_seconds": round(seconds, 3),
        **score(windows, selected, forecasts, SAMPLE_RATE),
    }


def run_author(args, models):
    from .rddm_author import evaluate_author_split

    torch.manual_seed(RANDOM_SEED)
    return evaluate_author_split(
        args.author_data, args.dataset,
        lambda chunks: predict_author_chunks(chunks, models, args.device),
        args.batch_size, seed=RANDOM_SEED,
    )


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned author Git checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Directory with the three author checkpoint files")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ptt-data", type=Path, help="Directory with external PhysioNet PTT WFDB records")
    source.add_argument("--author-data", type=Path, help="Directory with author-prepared test split arrays")
    parser.add_argument("--dataset", choices=AUTHOR_DATASETS)
    parser.add_argument("--tcn-checkpoints", type=Path, help="Optional matching PTT CNN fold checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int, help="Limit PTT subjects in numerical ID order for a smoke run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or (args.max_subjects is not None and args.max_subjects < 1):
        parser.error("Batch size, thread count and any subject limit must be positive")
    if args.author_data is not None and args.dataset is None:
        parser.error("--author-data requires --dataset")
    if args.ptt_data is not None and args.dataset is not None:
        parser.error("--dataset applies only to --author-data")
    if args.ptt_data is None and (args.tcn_checkpoints is not None or args.max_subjects is not None):
        parser.error("--tcn-checkpoints and --max-subjects apply only to --ptt-data")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch installation")

    torch.set_num_threads(args.threads)
    try:
        if args.author_data is not None:
            folder = args.author_data / args.dataset
            for name in ("ecg_test_4sec.npy", "ppg_test_4sec.npy"):
                if not (folder / name).is_file():
                    raise FileNotFoundError(f"Missing author test array: {folder / name}")
        hashes = {name: sha256_file(args.weights / name) for name in WEIGHT_NAMES}
        models = load_author_model(args.upstream, args.weights, args.device)
        print("Loaded all three author state dictionaries strictly", flush=True)
        results = run_ptt(args, models) if args.ptt_data is not None else run_author(args, models)
        report = {
            "author_commit": AUTHOR_COMMIT,
            "checkpoint_sha256": hashes,
            "device": args.device,
            "seed": RANDOM_SEED,
            "diffusion_steps": DIFFUSION_STEPS,
            "author_model_sampling_hz": AUTHOR_RATE,
            "author_window_seconds": AUTHOR_WINDOW,
            "batch_size": args.batch_size,
            "versions": {"numpy": np.__version__, "neurokit2": nk.__version__, "torch": str(torch.__version__)},
            **results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"RDDM evaluation failed: {error}\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
