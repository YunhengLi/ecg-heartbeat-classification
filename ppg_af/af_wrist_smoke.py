"""Run the AF feature routes on two aligned wrist windows without training."""

import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_audit import AuditSlice, audit_slice
from .afppgecg import PPG_FS, load_subject
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features
from .ptt_data import iter_ptt_records
from .pulse_timing import correct_timing, fit_pulse_delay, pulse_delay_histogram
from .template import fit_template_from_stats, template_stats


def fit_ptt_calibration(data_dir, rate, subject_count=3):
    """Fit a beat template and pulse timing delay on an external PTT cohort."""
    histograms, stats = [], []
    for record in islice(iter_ptt_records(data_dir, ("sit",)), subject_count):
        stride = int(record.fs / rate)
        pulse = resample_poly(record.ppg, 1, stride)
        voltage = resample_poly(record.ecg, 1, stride)
        marks = np.zeros(len(pulse), dtype=bool)
        peaks = np.rint(record.r_peaks / stride).astype(int)
        marks[peaks[peaks < len(marks)]] = True
        width = 12 * rate
        stop = width * min(5, len(pulse) // width)
        pulses, targets, labels = [array[:stop].reshape(-1, width) for array in (pulse, voltage, marks)]
        histograms.append(pulse_delay_histogram(pulses, labels, rate))
        stats.append(template_stats(zip(pulses, targets, strict=True), fs=rate))
    if len(stats) != subject_count:
        raise ValueError(f"Calibration requires {subject_count} local sitting PTT recordings")
    return fit_pulse_delay(histograms, rate), fit_template_from_stats(stats)


def smoke_routes(windows, ptt_dir, upstream, weights):
    import torch

    from .rddm import generate_author_ecg, load_author_model

    torch.set_num_threads(6)
    rate = int(PPG_FS)
    pulse_rows = np.stack([window.ppg for window in windows])
    if pulse_rows.shape != (2, 32 * rate) or not np.isfinite(pulse_rows).all():
        raise ValueError("Provide two fully observed, 32-second wrist PPG clips")

    delay, template = fit_ptt_calibration(ptt_dir, rate)
    models = load_author_model(upstream, weights, "cpu")
    estimates, chunks, runtime = generate_author_ecg(pulse_rows, rate, models, 4, "cpu")
    core = slice(rate, -rate)
    routes = {name: [] for name in ("native_ecg", "direct_ppg", "pulse_template", "rddm_timing_corrected")}
    for window, forecast in zip(windows, estimates, strict=True):
        rate_ecg = int(window.ecg_fs)
        reference = window.ecg[rate_ecg:-rate_ecg]
        corrected, _ = correct_timing(window.ppg, forecast, delay, rate)
        routes["native_ecg"].append(ecg_features(reference, rate_ecg))
        routes["direct_ppg"].append(ppg_features(window.ppg[core], rate)[0])
        routes["pulse_template"].append(ecg_features(template.predict(window.ppg)[core], rate))
        routes["rddm_timing_corrected"].append(ecg_features(corrected[core], rate))

    summary = {}
    valid_fraction_column = ECG_FEATURE_NAMES.index("valid_interval_fraction")
    for name, rows in routes.items():
        values = np.stack(rows)
        summary[name] = {
            "feature_columns": values.shape[1],
            "windows_with_all_finite_features": int(np.all(np.isfinite(values), axis=1).sum()),
            "windows_with_usable_intervals": int(np.sum(values[:, valid_fraction_column] > 0)),
        }
    return {
        "core_windows": len(windows),
        "rddm_four_second_chunks": len(chunks),
        "rddm_generation_seconds_cpu": round(runtime, 2),
        "routes": summary,
    }


def _parse_window(text):
    try:
        segment, offset = text.split(":")
        index, seconds = int(segment), float(offset)
        if index >= 0 and np.isfinite(seconds) and seconds >= 0:
            return index, seconds
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use SEGMENT:OFFSET_SECONDS with nonnegative values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ecg", "ppg", "ptt-dir", "upstream", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--window", type=_parse_window, nargs=2, required=True)
    args = parser.parse_args()

    with load_subject(args.ecg, args.ppg) as subject:
        windows, reference_labels = [], []
        for index, offset in args.window:
            windows.append(subject.read_segment(index, offset, offset + 32))
            row = audit_slice(subject, AuditSlice(index, offset + 1, 30))["windows"][0]
            reference_labels.append(row["reference"])
    if sorted(reference_labels) != ["af", "non_af"]:
        parser.error("Choose one stable AF window and one stable non-AF window")

    result = {
        "subjects": 1,
        "reference_windows_by_rhythm": {"af": 1, "non_af": 1},
        **smoke_routes(windows, args.ptt_dir, args.upstream, args.weights),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
