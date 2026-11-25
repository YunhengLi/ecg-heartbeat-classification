"""Stage fixed paired waveform ranges around open physician AF boundaries."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import sleep

import numpy as np
import requests
import wfdb

from .af_wrist_holdout import _write_json


RADIUS_SECONDS = 120
DATABASE = "mimic3wdb-matched/1.0"
LEAD_ORDER = ("II", "I", "V", "III", "AVR", "V1", "V2", "MCL", "AVL", "AVF", "V3", "V4", "V5", "V6", "MLII")


def call_wfdb(function, *args, **options):
    for attempt in range(3):
        try:
            return function(*args, **options)
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as error:
            status = error.response.status_code if error.response is not None else None
            if status not in (None, 429, 500, 502, 503, 504) or attempt == 2:
                raise
            header = error.response.headers.get("Retry-After", "") if error.response is not None else ""
            delay = int(header) if header.isdecimal() else 3 * (attempt + 1)
            if delay > 30:
                raise
            sleep(delay)


def make_plan(metadata):
    candidates = [row for row in metadata if row.get("possible_ECG_and_PPG") and row["annotation_valid"]]
    if len(candidates) != 8 or len({row["participant"] for row in candidates}) != len(candidates):
        raise ValueError("The waveform pilot expects the exact eight verified layout and annotation candidates")
    rows = []
    for row in candidates:
        rate = row["rate"]
        reviewed = min(row["declared_seconds"], row["header_seconds"])
        for event, key in (("onset", "annotation_start_seconds"), ("offset", "annotation_end_seconds")):
            center = round(row[key] * rate)
            radius = round(RADIUS_SECONDS * rate)
            first, last = center - radius, center + radius
            room = first >= 0 and last <= reviewed * rate
            segments = [name for name, start, stop in row["segment_clock"] if first < stop and last > start]
            rows.append({"participant": row["participant"], "record": row["record"], "boundary": event,
                         "sampling_rate_hz": rate, "first_sample": first, "last_sample": last,
                         "present_in_reviewed_clock": room, "segments": segments})
    return rows


def declared_channels(segments, folder):
    if not segments or "~" in segments:
        return None
    native = [call_wfdb(wfdb.rdheader, name, pn_dir=folder) for name in segments]
    common = set.intersection(*(set(item.sig_name) for item in native))
    names = {name.upper(): name for name in common}
    ecg = next((names[name] for name in LEAD_ORDER if name in names), None)
    return (ecg, names["PLETH"]) if ecg is not None and "PLETH" in names else None


def stage(row, marker, output, digest):
    if not row["present_in_reviewed_clock"]:
        return "outside_reviewed_clock"
    person = row["participant"]
    folder = f"{DATABASE}/{person[:3]}/{person}"
    if row["segments"]:
        channels = declared_channels(row["segments"], folder)
    else:
        channels = declared_channels([row["record"]], folder)
    if channels is None:
        return "paired_channels_not_continuous"
    start, end = (row[key] for key in ("first_sample", "last_sample"))
    record = call_wfdb(wfdb.rdrecord, row["record"], pn_dir=folder, sampfrom=start, sampto=end,
                       channel_names=list(channels), physical=True, return_res=32)
    if (record.fs != row["sampling_rate_hz"] or record.p_signal.shape != (end - start, 2)
            or set(record.sig_name) != set(channels)):
        raise ValueError("The aligned native waveform data must match the prespecified interval and channels")
    indices = [record.sig_name.index(name) for name in channels]
    ecg, ppg = record.p_signal[:, indices].T
    target = output / f"{marker:03d}.npz"
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, ecg=ecg, ppg=ppg, sampling_rate_hz=record.fs, boundary=row["boundary"],
                        private_record_digest=sha256(row["record"].encode()).hexdigest(),
                        expected_window_seconds=2 * RADIUS_SECONDS, plan_sha256=digest)
    temp.replace(target)
    return "both_waves_staged"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("private-clock", "private-plan", "signal-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = args.private_clock.read_bytes()
    plan = make_plan(json.loads(source))
    path = args.private_plan
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != plan_bytes:
        raise ValueError("The plan must remain sealed before examining the signal windows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plan_bytes)
    digest = sha256(plan_bytes).hexdigest()
    destination = args.signal_cache
    destination.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for index, row in enumerate(plan):
        state = stage(row, index, destination, digest)
        counts[state] += 1
        print(f"Checked waveform boundary {index + 1}/{len(plan)}: {state}", flush=True)
    summary = {"planned_boundaries": len(plan), "scheduled_seconds_each": RADIUS_SECONDS * 2,
               "plan_sha256": digest, "status": dict(counts)}
    _write_json(args.output, summary)


if __name__ == "__main__":
    main()
