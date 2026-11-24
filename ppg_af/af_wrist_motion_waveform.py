"""Measure how phase-preserving motion filtering changes template ECG beat timing."""

import argparse
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_acceleration_stage import SIDECAR_FOLDERS
from .af_wrist_exercise import HEART_RATE_BAND_HZ, motion_weighted_spectrum
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_motion_quality import ACCELERATION_RATE, verified_motion_rows
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_waveform_study import _metrics, ARCHIVE_ASSUMED_DELAY_MS
from .waveform import normalize_ecg


REFLECT_SECONDS = 2
HIGH_MOTION_POWER_FRACTION = 0.50


def motion_filtered_ppg(pulse, acceleration):
    """Reduce motion-coherent spectral amplitudes using samples inside the window."""
    motion = resample_poly(acceleration, PPG_RATE, ACCELERATION_RATE, axis=0, padtype="line")
    frequencies, power, weighted = motion_weighted_spectrum(pulse, motion)
    band = (frequencies >= HEART_RATE_BAND_HZ[0]) & (frequencies <= HEART_RATE_BAND_HZ[1])
    if not np.any(power[band] > 0):
        return pulse.copy(), 1.0
    ratio = np.divide(weighted, power, out=np.ones_like(power), where=band & (power > 0))
    attenuation = np.sqrt(np.clip(ratio, 0, 1))
    pad = REFLECT_SECONDS * PPG_RATE
    center = np.median(pulse)
    padded = np.pad(pulse - center, (pad, pad), mode="reflect")
    fft_frequencies = np.fft.rfftfreq(len(padded), d=1 / PPG_RATE)
    filtered = np.fft.irfft(np.fft.rfft(padded) * np.interp(fft_frequencies, frequencies, attenuation), n=len(padded))
    return filtered[pad:-pad] + center, float(weighted[band].sum() / power[band].sum())


def _verified_motion(args, participants, lock_digest):
    windows = {}
    for participant in participants:
        name = f"{participant}.npz"
        raw_path = args.development_raw / name
        _, rows, cores, acceleration, valid = verified_motion_rows(
            raw_path, args.development_features / name, args.acceleration_sidecars / SIDECAR_FOLDERS[0] / name,
            lock_digest, stride=False,
        )
        with np.load(raw_path, allow_pickle=False) as raw:
            starts = raw["start_s"][rows][valid]
        for start, pulse, motion in zip(starts, cores[valid], acceleration[valid], strict=True):
            key = (participant, int(start))
            if key in windows:
                raise ValueError("Every original PPG and motion pair must have one native start")
            windows[key] = (pulse, motion)
    return windows


def evaluate(args):
    receipt = json.loads(args.waveform_results.read_bytes())
    with args.waveform_pairs.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != receipt["private_paired_manifest_sha256"]:
            raise ValueError("Use the original verified paired waveform archive")
    with np.load(args.waveform_pairs, allow_pickle=False) as archive:
        paired = {key: archive[key] for key in archive.files if key != "ecg_500hz"}
    lock_digest = receipt["original_protocol_sha256"]
    if lock_digest != str(paired["source_protocol_sha256"].item()):
        raise ValueError("The paired ECG must use the original protocol")
    original = paired["origin"] == "original"
    if original.sum() != 114:
        raise ValueError("Expected the 114 originally verified stable ECG pairs")
    people = paired["subjects"]
    windows = _verified_motion(args, np.unique(people[original]), lock_digest)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    synchronized = replace(template, delay_samples=ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE // 1000)
    rows = []
    for index in np.flatnonzero(original):
        key = people[index], int(paired["start_s"][index])
        if key not in windows:
            continue
        pulse, motion = windows[key]
        if not np.array_equal(pulse, paired["ppg_core"][index]) or len(pulse) != WINDOW_SECONDS * PPG_RATE:
            raise ValueError("The motion sidecar must match the exact paired ECG and PPG core")
        method = paired["sync_method"][index]
        model = synchronized if method == "beat" else template
        filtered, residual_power = motion_filtered_ppg(pulse, motion)
        offsets = paired["native_qrs_offsets"][index:index + 2]
        peaks = paired["native_qrs_float_indices"][slice(*offsets)]
        rows.append((index, residual_power, normalize_ecg(paired["ecg_ppg_clock"][index]), peaks,
                     model.predict(pulse), model.predict(filtered)))
    if not rows:
        raise ValueError("No stable paired ECG windows had verified acceleration")

    def summarize(group):
        indices = [row[0] for row in group]
        target, predictions, filtered = (np.stack([row[field] for row in group]) for field in (2, 4, 5))
        return {"participants": len(np.unique(people[indices])),
                **_metrics(target, paired["ppg_core"][indices], [row[3] for row in group],
                           {"original_ppg_template": predictions, "motion_filtered_ppg_template": filtered})}

    return {"scope": "Development-only template ECG reconstruction with fixed motion frequency weights",
            "original_verified_windows": int(original.sum()), "windows_with_verified_acceleration": len(rows),
            "coherence_weight_band_hz": HEART_RATE_BAND_HZ, "reflection_seconds": REFLECT_SECONDS,
            "beat_synchronized_template_delay_ms": ARCHIVE_ASSUMED_DELAY_MS,
            "coarse_synchronized_template_delay_ms": template.delay_samples * 1000 / PPG_RATE,
            "large_motion_fraction_threshold": HIGH_MOTION_POWER_FRACTION, "all": summarize(rows),
            "by_ppg_motion_weight": {name: summarize(group) for name, group in (
                ("at_least_half_power_removed", [row for row in rows if row[1] <= HIGH_MOTION_POWER_FRACTION]),
                ("less_than_half_power_removed", [row for row in rows if row[1] > HIGH_MOTION_POWER_FRACTION]),
            ) if group}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("development-raw", "development-features", "acceleration-sidecars", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    try:
        _write_json(args.output, evaluate(args))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist motion waveform comparison failed: {error}\n")


if __name__ == "__main__":
    main()
