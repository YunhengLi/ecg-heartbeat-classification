"""Train and compare PPG-to-ECG models on PhysioNet PTT PPG v1.1.0."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly

from .ptt_data import iter_ptt_records
from .tcn_metrics import score_group
from .template import fit_template_from_stats, template_stats
from .temporal import fit_tcn, predict_tcn
from .waveform import fit_waveform_from_stats, normalize_ecg, waveform_stats


SAMPLE_RATE = 125
NATIVE_RATE = 500
WINDOW_SECONDS = 12
VAL_FRACTION = 0.22
SCORE_TRIM_SECONDS = 1.0


@dataclass(frozen=True)
class Windows:
    ppg: np.ndarray
    ecg: np.ndarray
    r_peaks: np.ndarray
    subjects: np.ndarray
    activities: np.ndarray


def resample(samples):
    """Downsample on the original time grid, keeping the filter clear of gaps."""
    factor = NATIVE_RATE // SAMPLE_RATE
    values = np.asarray(samples, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.full((len(values) + factor - 1) // factor, np.nan)
    if valid.all():
        return resample_poly(values, 1, factor, padtype="line")

    support = np.flatnonzero(valid)
    filled = np.interp(np.arange(len(values)), support, values[support])
    downsampled = resample_poly(filled, 1, factor, padtype="line")
    gap = maximum_filter1d(~valid, size=20 * factor + 1, mode="nearest")
    downsampled[gap[::factor]] = np.nan
    return downsampled


def prepare_windows(data_dir, activities):
    inputs, targets, markers, subjects, contexts = [], [], [], [], []
    width = SAMPLE_RATE * WINDOW_SECONDS
    for record in iter_ptt_records(data_dir, activities=activities):
        if record.fs != NATIVE_RATE:
            raise ValueError(f"Expected native 500 Hz for {record.subject_id} {record.activity}")
        pulse, voltage = resample(record.ppg), resample(record.ecg)
        if len(pulse) != len(voltage):
            raise ValueError(f"Unequal waveform lengths for {record.subject_id} {record.activity}")
        peaks = np.unique(np.rint(record.r_peaks * SAMPLE_RATE / record.fs).astype(int))
        for start in range(0, len(pulse) - width + 1, width):
            stop = start + width
            window_marks = peaks[(peaks >= start) & (peaks < stop)] - start
            r_labels = np.zeros(width, dtype=np.float32)
            r_labels[window_marks] = 1
            target = normalize_ecg(voltage[start:stop])
            r_labels[~np.isfinite(target)] = np.nan
            inputs.append(pulse[start:stop])
            targets.append(target)
            markers.append(r_labels)
            subjects.append(record.subject_id)
            contexts.append(record.activity)

    if not inputs:
        raise ValueError("No complete 12-second windows were found")
    return Windows(np.stack(inputs), np.stack(targets), np.stack(markers),
                   np.asarray(subjects), np.asarray(contexts))


def make_folds(subjects, folds, seed):
    """Assign subjects, including every activity, to a single test fold."""
    all_subjects = np.unique(subjects)
    if not 2 <= folds <= len(all_subjects) or len(all_subjects) < 4:
        raise ValueError("Need at least four subjects and a valid outer fold count")
    rng = np.random.default_rng(seed)
    for test_subjects in np.array_split(rng.permutation(all_subjects), folds):
        outer = rng.permutation(np.setdiff1d(all_subjects, test_subjects))
        count = max(1, round(len(outer) * VAL_FRACTION))
        validation, training = outer[:count], outer[count:]
        if not len(training):
            raise ValueError("The inner validation split leaves no training subjects")
        yield tuple(np.flatnonzero(np.isin(subjects, ids)) for ids in (training, validation, test_subjects))


def collect_baselines(windows):
    result = {}
    for subject in np.unique(windows.subjects):
        indices = np.flatnonzero(windows.subjects == subject)
        pairs = [(windows.ppg[index], windows.ecg[index]) for index in indices]
        result[subject] = (
            waveform_stats(pairs, fs=SAMPLE_RATE),
            template_stats(pairs, fs=SAMPLE_RATE),
        )
    return result


def save_checkpoint(directory, fold, name, model, args, split):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "channels": args.channels,
                "sample_rate_hz": SAMPLE_RATE, "window_seconds": WINDOW_SECONDS,
                "method": name, "fold": fold, "split": split,
                "activities": sorted(args.activities)}, directory / f"{name}-fold-{fold}.pt")


def run(windows, args):
    baselines = collect_baselines(windows)
    variants = {
        "tcn_waveform": (0.0, 0.0),
        "tcn_waveform_and_peaks": (args.peak_weight, 0.0),
    }
    if args.qrs_weight:
        variants["tcn_qrs_weighted"] = (args.peak_weight, args.qrs_weight)
    names = ("ridge", "pulse_template", *variants)
    waveforms = {name: np.full(windows.ppg.shape, np.nan, dtype=np.float32) for name in names}
    peak_logits = {name: np.full(windows.ppg.shape, np.nan, dtype=np.float32)
                   for name, (weight, _) in variants.items() if weight}
    all_forecasts = np.zeros(len(windows.subjects), dtype=int)
    fold_reports = []

    for fold, (train, validation, test) in enumerate(make_folds(windows.subjects, args.folds, args.seed), 1):
        print(f"Fold {fold}/{args.folds}: {len(train)} train, {len(validation)} validation, {len(test)} test windows",
              flush=True)
        ids = np.unique(windows.subjects[train])
        ridge = fit_waveform_from_stats(baselines[subject][0] for subject in ids)
        template = fit_template_from_stats(baselines[subject][1] for subject in ids)
        waveforms["ridge"][test] = np.stack([ridge.predict(ppg) for ppg in windows.ppg[test]])
        waveforms["pulse_template"][test] = np.stack([template.predict(ppg) for ppg in windows.ppg[test]])
        details = {
            "fold": fold,
            "train_subjects": ids.tolist(),
            "validation_subjects": np.unique(windows.subjects[validation]).tolist(),
            "test_subjects": np.unique(windows.subjects[test]).tolist(),
            "template_delay_ms": template.delay_samples * 1000 / SAMPLE_RATE,
            "training": {},
        }
        for name, (peak_weight, qrs_weight) in variants.items():
            model, history = fit_tcn(
                windows.ppg, windows.ecg, windows.r_peaks, train, validation,
                peak_weight=peak_weight, qrs_weight=qrs_weight, seed=args.seed + fold, epochs=args.epochs,
                batch_size=args.batch_size, channels=args.channels, threads=args.threads,
            )
            waveform, logits = predict_tcn(model, windows.ppg[test], batch_size=args.batch_size)
            waveforms[name][test] = waveform
            if peak_weight:
                peak_logits[name][test] = logits
            details["training"][name] = history
            save_checkpoint(args.checkpoint_dir, fold, name, model, args,
                            {key: details[key] for key in ("train_subjects", "validation_subjects", "test_subjects")})
            print(f"  {name}: {len(history)} training epochs completed", flush=True)

        all_forecasts[test] += 1
        fold_reports.append(details)

    if not np.all(all_forecasts == 1):
        raise RuntimeError("Each window must have exactly one held-out forecast")

    def evaluate(mask):
        return {
            "subjects": len(np.unique(windows.subjects[mask])),
            **score_group(windows.ecg[mask], windows.r_peaks[mask], windows.ppg[mask],
                          {name: values[mask] for name, values in waveforms.items()},
                          {name: logits[mask] for name, logits in peak_logits.items()},
                          SAMPLE_RATE, SCORE_TRIM_SECONDS),
        }

    report = {
        "dataset": "PhysioNet Pulse Transit Time PPG v1.1.0",
        "input_channel": "pleth_3",
        "target_channel": "ecg",
        "native_sample_rate_hz": NATIVE_RATE,
        "model_sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "score_trim_seconds_at_each_edge": SCORE_TRIM_SECONDS,
        "peak_tolerance_ms": 100,
        "settings": {"seed": args.seed, "folds": args.folds, "epochs_max": args.epochs,
                     "batch_size": args.batch_size, "channels": args.channels,
                     "peak_weight": args.peak_weight, "qrs_weight": args.qrs_weight,
                     "threads": args.threads},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "overall": evaluate(np.ones(len(windows.subjects), dtype=bool)),
        "by_activity": {activity: evaluate(windows.activities == activity)
                        for activity in np.unique(windows.activities)},
        "fold_details": fold_reports,
    }
    return report, waveforms


def plot_examples(windows, waveforms, output):
    import matplotlib.pyplot as plt

    activities = np.unique(windows.activities)
    figure, axes = plt.subplots(2, len(activities), squeeze=False, figsize=(4 * len(activities) + 1, 5),
                               sharey="row", layout="constrained")
    time = np.arange(windows.ppg.shape[1]) / SAMPLE_RATE
    segment = (time >= 2.0) & (time < 7.0)
    for column, activity in enumerate(activities):
        index = int(np.flatnonzero(windows.activities == activity)[0])
        pulse_axis, ecg_axis = axes[:, column]
        pulse = windows.ppg[index] - np.nanmedian(windows.ppg[index])
        scale = np.nanpercentile(np.abs(pulse), 95)
        pulse_axis.plot(time[segment], pulse[segment] / max(scale, 1e-12), color="#287b79", linewidth=1.1)
        pulse_axis.set_title(str(activity).title())
        ecg_axis.plot(time[segment], windows.ecg[index, segment], color="#252d38", label="Measured ECG")
        plotted = [("ridge", "#d18c2c", "Ridge"),
                   ("tcn_waveform_and_peaks", "#8272a5", "CNN, waveform + peaks")]
        if "tcn_qrs_weighted" in waveforms:
            plotted.append(("tcn_qrs_weighted", "#1a7583", "CNN, QRS-weighted"))
        for name, color, label in plotted:
            ecg_axis.plot(time[segment], waveforms[name][index, segment], color=color, linewidth=1.05, label=label)
        ecg_axis.set_xlabel("Time in window (s)")
        for axis in (pulse_axis, ecg_axis):
            axis.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Relative PPG amplitude")
    axes[1, 0].set_ylabel("Relative ECG amplitude")
    handles, labels = axes[1, -1].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=8, loc="outside lower center", ncol=len(labels))
    figure.suptitle("PPG-to-ECG estimates on held-out recordings\nMehrgardt et al., PhysioNet PTT PPG v1.1.0 · ODbL 1.0")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--activities", nargs="+", choices=("sit", "walk", "run"), default=("sit", "walk", "run"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--peak-weight", type=float, default=0.05)
    parser.add_argument("--qrs-weight", type=float, default=0.0,
                        help="Add a CNN that upweights waveform errors near annotated R peaks")
    args = parser.parse_args()
    if not np.isfinite(args.peak_weight) or args.peak_weight <= 0:
        parser.error("--peak-weight must be finite and positive")
    if not np.isfinite(args.qrs_weight) or args.qrs_weight < 0:
        parser.error("--qrs-weight must be finite and nonnegative")
    windows = prepare_windows(args.data_dir, args.activities)
    print(f"Prepared {len(windows.ppg)} windows from {len(np.unique(windows.subjects))} subjects", flush=True)
    report, waveforms = run(windows, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.plot:
        plot_examples(windows, waveforms, args.plot)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
