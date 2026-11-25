"""Compare the published MSPTDfast v2 pulse detector on the locked wrist clock."""

import argparse
import json
from pathlib import Path

import numpy as np

from .af_wrist_false_start_audit import native_irregularity
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import _compare, load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_pulse_consensus import VALID, interval_diagnostics, pulse_consensus
from .af_wrist_quality_study import quality_interaction_features
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .features import ECG_FEATURE_NAMES, _bandpass, _interval_features, _plateaus, _prepare


VERSION = "0.3.0"


def msptd_features(signal, detect):
    prepared = _prepare(signal, PPG_RATE, minimum_fs=4.0)
    if prepared is None:
        return np.zeros(len(ECG_FEATURE_NAMES))
    optical, present = prepared
    filtered = _bandpass(optical, PPG_RATE, 0.4, 8.0)
    peaks = np.unique(detect(filtered, sampling_rate=PPG_RATE, method="msptdfastv2")["PPG_Peaks"])
    if np.any((peaks < 0) | (peaks >= len(filtered))):
        raise ValueError("The published pulse peak implementation returned positions outside the input window")
    return _interval_features(peaks.astype(int), PPG_RATE, present & ~_plateaus(optical, present, PPG_RATE))


def evaluate(args):
    import neurokit2 as nk

    if nk.__version__ != VERSION:
        raise ValueError("Use NeuroKit2 0.3.0 to reproduce the pinned MSPTDfast v2 comparison")
    grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)
    old = quality["baseline"]
    raw = old.copy()
    raw[shared] = grid.table["direct_ppg"][shared, :len(ECG_FEATURE_NAMES)]
    msptd = old.copy()
    values = np.stack([msptd_features(pulse, nk.ppg_findpeaks) for pulse in cores[shared]])
    supported = values[:, VALID] > 0
    indices = np.flatnonzero(shared)
    msptd[indices[supported]] = values[supported]
    q = quality["quality"][:, -1]
    matrices = {"quality_interactions": quality["quality_interactions"],
                "raw_intervals_with_quality": quality_interaction_features(raw, q),
                "msptd_intervals_with_quality": quality_interaction_features(msptd, q)}

    waveform_results = json.loads(args.waveform_results.read_bytes())
    _, pulses, _, labels, reference = read_pairs(args.waveform_pairs, waveform_results, lock_digest, set(grid.subjects))
    reference_score = np.asarray([native_irregularity(markers / PPG_RATE, WINDOW_SECONDS) for markers in reference])
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    benchmark = {"template": _extract(pulses, template)["baseline"],
                 "broad_band_peaks": np.stack([pulse_consensus(pulse)[0] for pulse in pulses]),
                 "msptdfast_v2": np.stack([msptd_features(pulse, nk.ppg_findpeaks) for pulse in pulses])}
    return {"scope": "Compare published MSPTDfast v2 timing and AF alerts on native wrist inputs",
            "neurokit2_version": VERSION, "original_scoreable_windows": len(indices), "msptd_interval_windows": int(supported.sum()),
            "template_fallback_windows": int((~supported).sum()),
            "paired_stable_diagnostics": interval_diagnostics(reference_score, labels, benchmark),
            "alert_comparisons": _compare(grid, matrices, lock, original)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
