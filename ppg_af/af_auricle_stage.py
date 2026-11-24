"""Stage ECG, finger PPG and ear PPG from a fixed PhysioNet sample."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import time

import numpy as np
import requests
import wfdb

from .af_wrist_features import WINDOW_SECONDS


SOURCE = "https://physionet.org/files/auricle/1.0.0"
RELEASE_SHA = "b801903cf00152830cbde3cf20f5a62a748ff15cec8cf41f4ecb56d3a1ead04e"
RECORDS_SHA = "86227a23692bce2a34fbd119a7e1f09d25bcbe51fa5eccefb7bc210a89613fb4"
RATE = 1000
COHORTS = ("HC", "HFC")
CASES_PER_COHORT = 10
STARTS_SECONDS = (90, 330, 930)
CHANNELS = ("ECG", "PPG_Finger", "PPG_Ear")


def _get_pinned(session, path, digest):
    with session.get(f"{SOURCE}/{path}", timeout=20) as response:
        response.raise_for_status()
        value = response.content
    if sha256(value).hexdigest() != digest:
        raise ValueError("PhysioNet source metadata does not match the pinned release checksum")
    return value


def plan_records(records):
    patients = defaultdict(list)
    for path in records:
        parts = PurePosixPath(path).parts
        if len(parts) != 2 or parts[0] != parts[1]:
            raise ValueError("Unexpected release waveform record layout")
        cohort, participant, _ = parts[1].split("_")
        patients[cohort, participant].append(f"data/{path}")
    expected = Counter(cohort for cohort, _ in patients)
    if expected != {"HC": 42, "HFC": 28}:
        raise ValueError("The original release contains 42 healthy and 28 heart-failure participants")

    def rank(value):
        return sha256(f"auricle-optical-clock:{RELEASE_SHA}:{value}".encode()).digest()

    return {cohort: [min(patients[key], key=rank) for key in sorted((key for key in patients if key[0] == cohort), key=rank)
                    [:CASES_PER_COHORT]] for cohort in COHORTS}


def _windows(session, stem, header):
    if (header.fs != RATE or len(set(header.file_name)) != 1 or set(header.fmt) != {"32"}
            or any(value not in (None, 0) for value in (*header.byte_offset, *header.skew))
            or any(value != 1 for value in header.samps_per_frame) or not set(CHANNELS) <= set(header.sig_name)):
        raise ValueError("Auricle clock or WFDB interleaving differs from the release representation")
    columns = [header.sig_name.index(name) for name in CHANNELS]
    gains = np.asarray(header.adc_gain)[columns]
    baselines = np.asarray(header.baseline)[columns]
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("Expected positive WFDB calibration gains for all requested channels")
    bytes_per_sample = 4 * header.n_sig
    total_bytes = bytes_per_sample * header.sig_len
    windows = np.full((len(STARTS_SECONDS), len(CHANNELS), WINDOW_SECONDS * RATE), np.nan, dtype=np.float32)
    downloaded = 0
    for index, start_seconds in enumerate(STARTS_SECONDS):
        first = start_seconds * RATE
        stop = first + WINDOW_SECONDS * RATE
        if stop > header.sig_len:
            continue
        lo, hi = bytes_per_sample * first, bytes_per_sample * stop - 1
        requested = {"Range": f"bytes={lo}-{hi}", "Accept-Encoding": "identity"}
        with session.get(f"{SOURCE}/{stem}.dat", headers=requested, timeout=30, stream=True) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{total_bytes}":
                raise OSError(f"PhysioNet did not return the exact waveform range (HTTP {response.status_code})")
            payload = response.raw.read(hi - lo + 2)
        if len(payload) != hi - lo + 1:
            raise OSError("PhysioNet waveform range has a different length from the released header")
        digital = np.frombuffer(payload, dtype="<i4").reshape(-1, header.n_sig)[:, columns]
        analog = (digital.astype(float) - baselines) / gains
        analog[digital == np.iinfo(np.int32).min] = np.nan
        windows[index] = analog.T
        downloaded += len(payload)
        time.sleep(0.30)
    return windows, downloaded


def stage(cache):
    cache.mkdir(parents=True, exist_ok=True)
    headers = cache / "headers"
    headers.mkdir(exist_ok=True)
    total = 0
    with requests.Session() as session:
        checksums = dict((name, digest) for digest, name in (
            line.split(maxsplit=1) for line in _get_pinned(session, "SHA256SUMS.txt", RELEASE_SHA).decode().splitlines()
        ))
        files = _get_pinned(session, "RECORDS", RECORDS_SHA).decode().splitlines()
        plan = plan_records(files)
        for cohort in COHORTS:
            for index, stem in enumerate(plan[cohort]):
                target = cache / f"{cohort}-{index:02}.npz"
                source_digest = sha256(stem.encode()).hexdigest()
                if target.exists():
                    with np.load(target, allow_pickle=False) as previous:
                        if previous["record_digest"].item() != source_digest or previous["release_sha"].item() != RELEASE_SHA:
                            raise ValueError("The stored signal belongs to a different Auricle collection plan")
                    continue
                filename = PurePosixPath(stem).name
                header_path = headers / f"{filename}.hea"
                header_path.write_bytes(_get_pinned(session, f"{stem}.hea", checksums[f"{stem}.hea"]))
                windows, transferred = _windows(session, stem, wfdb.rdheader(str(header_path.with_suffix(""))))
                partial = target.with_suffix(".partial.npz")
                np.savez_compressed(partial, waveforms=windows, cohort=cohort, record_digest=source_digest,
                                    starts_seconds=STARTS_SECONDS, release_sha=RELEASE_SHA)
                partial.replace(target)
                total += transferred
                print(f"Auricle paired waveforms staged for {cohort} {index + 1}/{len(plan[cohort])}", flush=True)
    print({"cohorts": {key: len(value) for key, value in plan.items()}, "downloaded_waveform_bytes": total})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    stage(args.cache_dir)
