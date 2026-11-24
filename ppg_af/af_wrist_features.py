"""Extract three comparable AF feature sets from staged AFPPGECG windows."""

import argparse
from pathlib import Path

import numpy as np

from .features import ECG_FEATURE_NAMES, PPG_FEATURE_NAMES, ecg_features, ppg_features
from .pulse_timing import correct_timing


ROUTES = ("direct_ppg", "pulse_template", "rddm")
PTT_SUBJECTS = 22
WINDOW_SECONDS = 30
PPG_RATE = 100
_INTERVAL_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")
_COHORT_FIELDS = ("subject_id", "start_s", "run_id", "run_kind", "stable_label", "reference_valid",
                  "af_intervals_s", "sync_method")


def _store_if_supported(matrix, index, features):
    if np.isfinite(features).all() and features[_INTERVAL_SUPPORT] > 0:
        matrix[index] = features


def extract_features(ppg_with_context, template, delay, models, *, device, batch_size):
    """Use one 30-second input for every route and abstain without usable intervals."""
    from .rddm import generate_author_ecg

    pulses = np.asarray(ppg_with_context, dtype=float)
    if pulses.ndim != 2 or pulses.shape[1] != (WINDOW_SECONDS + 2) * PPG_RATE:
        raise ValueError("Each staged PPG row must be 32 seconds at 100 Hz")
    core = pulses[:, PPG_RATE:-PPG_RATE]
    available = np.isfinite(core).all(axis=1)
    usable = available & (np.ptp(core, axis=1) > 0)
    rows = np.flatnonzero(usable)
    matrices = {
        route: np.full((len(core), len(PPG_FEATURE_NAMES if route == "direct_ppg" else ECG_FEATURE_NAMES)), np.nan)
        for route in ROUTES
    }
    quality = np.zeros(len(core))
    if not len(rows):
        return matrices, available, quality

    # RDDM requires 4-second multiples. Left padding uses samples already seen
    # in the window, so a decision at its end does not use future measurements.
    pad = 2 * PPG_RATE
    model_input = np.pad(core[rows], ((0, 0), (pad, 0)), mode="reflect")
    estimates, _, _ = generate_author_ecg(model_input, PPG_RATE, models, batch_size, device)
    for index, pulse, forecast in zip(rows, model_input, estimates, strict=True):
        direct, quality[index] = ppg_features(core[index], PPG_RATE)
        corrected, _ = correct_timing(pulse, forecast, delay, PPG_RATE)
        for route, features in (
            ("direct_ppg", direct),
            ("pulse_template", ecg_features(template.predict(pulse)[pad:], PPG_RATE)),
            ("rddm", ecg_features(corrected[pad:], PPG_RATE)),
        ):
            _store_if_supported(matrices[route], index, features)
    return matrices, available, quality


def _archive_features(source, destination, template, delay, models, *, device, batch_size):
    with np.load(source, allow_pickle=False) as archive:
        if int(archive["ppg_fs"]) != PPG_RATE:
            raise ValueError(f"Expected 100 Hz wrist PPG in {source.name}")
        metadata = {name: archive[name] for name in _COHORT_FIELDS}
        matrices, available, quality = extract_features(
            archive["ppg"], template, delay, models, device=device, batch_size=batch_size,
        )
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **metadata, **matrices, ppg_available=available, quality=quality)
    temporary.replace(destination)
    print(f"Cached {len(quality)} windows from {source.stem}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "output", "ptt-dir", "upstream", "weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and thread count must be positive")

    sources = sorted(path for path in args.cohort.glob("*.npz") if not path.name.endswith(".partial.npz"))
    if not sources:
        parser.error("The cohort directory contains no complete subject archives")
    pending = [path for path in sources if args.overwrite or not (args.output / path.name).exists()
               or (args.output / path.name).stat().st_mtime_ns < path.stat().st_mtime_ns]
    if not pending:
        print("Current features already exist for all staged subjects")
        return

    import torch

    from .af_wrist_smoke import fit_ptt_calibration
    from .rddm import load_author_model

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("The installed PyTorch cannot access CUDA")
    delay, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    models = load_author_model(args.upstream, args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in pending:
        _archive_features(path, args.output / path.name, template, delay, models,
                          device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
