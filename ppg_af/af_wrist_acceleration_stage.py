"""Pair native accelerometer samples with already-locked wrist PPG windows."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np

from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids
from .afppgecg import (
    ACCEL_FS, PPG_FS, DEFAULT_MAX_REFERENCE_RR_SECONDS, WristSubject, _read_vector, _sample_bounds, _vector_length,
)


SIDECAR_FOLDERS = ("cohort-seeded-development", "cohort-seeded-development-stride10")


def scoreable_rows(raw_path, feature_path, *, stride):
    with (np.load(raw_path, allow_pickle=False) as raw, np.load(feature_path, allow_pickle=False) as feature):
        if not np.array_equal(raw["start_s"], feature["start_s"]):
            raise ValueError("Feature and native PPG input clocks must match")
        shared = _shared_features(feature)
        if stride:
            shared &= raw["input_status"] == "available"
        rows = np.flatnonzero(shared)
        start_key = "input_start_s" if stride else "start_s"
        rate = int(PPG_FS)
        return rows, raw[start_key][rows], raw["sync_method"][rows], raw["ppg"][rows, rate:-rate]


def verified_wrist_bounds(subject, start, method, expected_ppg):
    """Locate the native segment and sample bounds of an exact locked PPG core."""
    stop = start + WINDOW_SECONDS
    for segment in subject.segments:
        if segment.sync_method != method or segment.start_time > start or segment.stop_time < stop:
            continue
        source = subject._ppg_data[segment.index]
        first, last = _sample_bounds(start, stop, segment.start_time, PPG_FS, source.size)
        if np.array_equal(_read_vector(source, slice(first, last)), expected_ppg):
            return segment, first, last
    raise ValueError("The locked window must reproduce exactly on the native wrist PPG clock")


class NativeAcceleration:
    """Open only acceleration cells needed by the fixed PPG window clock."""

    def __init__(self, ecg, wrist, participant):
        self.wrist = wrist
        self.subject = WristSubject(ecg, wrist, participant, DEFAULT_MAX_REFERENCE_RR_SECONDS, False)
        self.axis_references = [_read_vector(wrist[f"Accelerometer_{axis}"]) for axis in "XYZ"]
        if any(len(track) != len(self.subject.segments) for track in self.axis_references):
            raise ValueError("The paired accelerometer and PPG segment counts differ")
        self.axes_by_segment = {}

    def sources_for(self, start, method, expected_ppg):
        """Verify the exact PPG core before locating the corresponding acceleration."""
        stop = start + WINDOW_SECONDS
        segment, first, _ = verified_wrist_bounds(self.subject, start, method, expected_ppg)
        if segment.index not in self.axes_by_segment:
            self.axes_by_segment[segment.index] = tuple(self.wrist[track[segment.index]] for track in self.axis_references)
        axes = self.axes_by_segment[segment.index]
        counts = {_vector_length(axis) for axis in axes}
        if len(counts) != 1 or abs(axes[0].size / ACCEL_FS - segment.duration_seconds) > 1 / ACCEL_FS + 1e-9:
            return None
        lo, hi = _sample_bounds(start, stop, segment.start_time, ACCEL_FS, axes[0].size)
        if hi - lo != WINDOW_SECONDS * ACCEL_FS:
            return None
        return axes, lo, hi, lo / ACCEL_FS - first / PPG_FS


def read_acceleration(axes, first, last):
    return np.column_stack([_read_vector(axis, slice(first, last)) for axis in axes])


def save_sidecar(target, rows, cores, motion, phase, lock_digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, rows=rows, accel=motion, accel_minus_ppg_s=phase,
                        ppg_fs=PPG_FS, accel_fs=ACCEL_FS, staged_ppg_core_sha256=sha256(cores.tobytes()).hexdigest(),
                        original_lock_sha256=lock_digest)
    temporary.replace(target)


def stage_person(ecg_input, wrist_input, participant, archives, output, lock_digest):
    with ExitStack() as stack:
        ecg = stack.enter_context(h5py.File(ecg_input, "r"))
        wrist = stack.enter_context(h5py.File(wrist_input, "r"))
        native = NativeAcceleration(ecg, wrist, participant)
        for folder, (rows, starts, methods, cores) in zip(SIDECAR_FOLDERS, archives, strict=True):
            motion = np.full((len(rows), int(WINDOW_SECONDS * ACCEL_FS), 3), np.nan)
            phase = np.full(len(rows), np.nan)
            for index, (start, method, core) in enumerate(zip(starts, methods, cores, strict=True)):
                clip = native.sources_for(float(start), method, core)
                if clip is None:
                    continue
                axes, first, last, offset = clip
                values = read_acceleration(axes, first, last)
                if np.isfinite(values).all():
                    motion[index], phase[index] = values, offset
            save_sidecar(output / folder / f"{participant}.npz", rows, cores, motion, phase, lock_digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development-raw", "development-features", "stride-raw", "stride-features", "source-dir", "lock", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw_lock = args.lock.read_bytes()
        lock = json.loads(raw_lock)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        participants = fit | calibration
        groups = ((args.development_raw, args.development_features), (args.stride_raw, args.stride_features))
        if (fit & calibration or participants & reserved or (len(participants), len(reserved)) != (10, 25)
                or any({path.stem for path in folder.glob("*.npz")} != participants for group in groups for folder in group)):
            raise ValueError("Stage motion for the original locked wrist development cohort")
        for participant in sorted(participants):
            name = f"{participant}.npz"
            archives = [scoreable_rows(raw / name, features / name, stride=bool(index))
                        for index, (raw, features) in enumerate(groups)]
            stage_person(args.source_dir / f"{participant}_ECG.mat", args.source_dir / f"{participant}_PPG.mat",
                         participant, archives, args.output, sha256(raw_lock).hexdigest())
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Native wrist acceleration staging failed: {error}\n")


if __name__ == "__main__":
    main()
