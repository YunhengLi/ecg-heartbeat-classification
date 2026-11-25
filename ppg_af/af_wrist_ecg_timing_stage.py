"""Stage native ECG-annotated interval features on an existing wrist decision clock."""

import argparse
from hashlib import file_digest, sha256
from pathlib import Path

import h5py
import numpy as np

from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import DEFAULT_MAX_REFERENCE_RR_SECONDS, ECG_FS, _check_nominal_rates, _read_vector, _vector_length
from .features import ECG_FEATURE_NAMES, _interval_features


DURATIONS = (STRIDE_SECONDS, WINDOW_SECONDS)
ROW_SCHEDULE = ("subject_id", "start_s", "run_id", "run_kind", "reference_valid")


def _read_qrs(ecg):
    _check_nominal_rates(ecg, (ECG_FS,))
    indices = _read_vector(ecg["QRSindex"]).astype(float)
    rr = _read_vector(ecg["rr"]).astype(float)
    samples = _vector_length(ecg["ECG"])
    peaks = indices - 1
    if (not np.isfinite(peaks).all() or np.any(peaks != np.floor(peaks)) or np.any(np.diff(peaks) <= 0)
            or np.any((peaks < 0) | (peaks >= samples)) or rr.shape != (max(0, len(peaks) - 1),)):
        raise ValueError("The native ECG must have ordered 1-based QRS samples and one RR value per interval")
    bad = ~np.isfinite(rr) | (rr <= 0) | (np.diff(peaks) / ECG_FS > DEFAULT_MAX_REFERENCE_RR_SECONDS)
    return peaks.astype(np.int64), bad, samples


def qrs_features(ends, peaks, bad, samples, seconds):
    size = int(ECG_FS * seconds)
    allowable = ECG_FS * DEFAULT_MAX_REFERENCE_RR_SECONDS
    values = np.full((len(ends), len(ECG_FEATURE_NAMES)), np.nan)
    present = np.ones(size, dtype=bool)
    for row, stop in enumerate(ends):
        start = stop - size
        if start < 0 or stop > samples:
            continue
        left, right = np.searchsorted(peaks, (start, stop))
        if right - left < 3 or peaks[left] - start > allowable or stop - peaks[right - 1] > allowable or bad[left:right - 1].any():
            continue
        values[row] = _interval_features(peaks[left:right] - start, ECG_FS, present)
    return values


def archive_clock(raw_path, person, *, stride):
    with np.load(raw_path, allow_pickle=False) as raw:
        if raw["subject_id"].item() != person:
            raise ValueError("The ECG reference and wrist archive must refer to the same participant")
        ends = raw["start_s"] + (STRIDE_SECONDS if stride else WINDOW_SECONDS)
        if stride and (raw["decision_stride_seconds"].item() != STRIDE_SECONDS
                       or not np.array_equal(ends, raw["decision_end_s"])):
            raise ValueError("The ECG input must end exactly on the sealed wrist decision clock")
        if not np.isfinite(ends).all() or not np.allclose(np.rint(ends * ECG_FS), ends * ECG_FS, rtol=0, atol=1e-6):
            raise ValueError("Every decision end must fall on a native ECG sample")
        row_key = sha256()
        for name in ROW_SCHEDULE:
            row_key.update(raw[name].tobytes())
    return np.rint(ends * ECG_FS).astype(np.int64), row_key.hexdigest()


def stage_source(ecg_source, person, archives, lock_sha):
    """Read one native ECG once and stage only the feature rows needed for this comparison."""
    with h5py.File(ecg_source, "r") as ecg:
        peaks, bad, samples = _read_qrs(ecg)
    for raw_path, target, stride in archives:
        ends, row_key = archive_clock(raw_path, person, stride=stride)
        matrices = {f"features_{seconds}s": qrs_features(ends, peaks, bad, samples, seconds) for seconds in DURATIONS}
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".partial.npz")
        np.savez_compressed(temp, subject_id=person, decision_end_samples=ends, wrist_row_sha256=row_key,
                            protocol_sha256=lock_sha, **matrices)
        temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-ecg-dir", "parent-raw", "stride-raw", "lock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with args.lock.open("rb") as stream:
        lock_sha = file_digest(stream, "sha256").hexdigest()
    parent = {p.stem: p for p in args.parent_raw.glob("*.npz") if not p.name.endswith(".partial.npz")}
    finer = {p.stem: p for p in args.stride_raw.glob("*.npz") if not p.name.endswith(".partial.npz")}
    if not parent or set(parent) != set(finer):
        raise ValueError("Stage exactly the same participants at both wrist decision cadences")
    for person, path in sorted(parent.items()):
        bundles = [(archive, args.output / name / f"{person}.npz", stride)
                   for archive, name, stride in ((path, "parent", False), (finer[person], "stride", True))]
        stage_source(args.native_ecg_dir / f"{person}_ECG.mat", person, bundles, lock_sha)


if __name__ == "__main__":
    main()
