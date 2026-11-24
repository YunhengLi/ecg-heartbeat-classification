"""Stage checksummed ECG beat and rhythm annotations from SHDB-AF v1.0.1."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from email.utils import parsedate_to_datetime
from hashlib import sha256
import io
import json
from pathlib import Path
import time

import requests


SOURCE = "https://physionet.org/files/shdb-af/1.0.1"
MANIFEST_SHA = "a101475f41df8300493b9cba963499334fc5aa118c0b4c01503e7bf7621ac3c9"
GROUP_MANIFEST = "annotation-groups-private.json"
EXTENSIONS = ("atr", "qrs", "hea")


def fetch(session, key, expected):
    for attempt in range(4):
        with session.get(f"{SOURCE}/{key}", timeout=25) as reply:
            if reply.status_code == 429:
                delay = reply.headers.get("Retry-After")
                if not delay:
                    seconds = 2 + attempt
                elif delay.isdecimal():
                    seconds = int(delay)
                else:
                    seconds = max(0, parsedate_to_datetime(delay).timestamp() - time.time())
                if seconds > 30:
                    raise OSError("Annotation retrieval deferred by the provider")
                time.sleep(seconds)
                continue
            reply.raise_for_status()
            data = reply.content
        if sha256(data).hexdigest() != expected:
            raise ValueError("A published source checksum did not match the retrieved annotation")
        return data
    raise OSError("The annotation server continues to rate-limit requests")


def record_groups(metadata):
    rows = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
    all_members = {row["Data_ID"].zfill(3): row["Subject_ID"] for row in rows}
    labeled = {row["Data_ID"].zfill(3): row["Subject_ID"] for row in rows if row["Annotated"] == "True"}
    if (len(rows), len(all_members), len(set(all_members.values())), len(labeled), len(set(labeled.values()))) != (128, 128, 122, 98, 93):
        raise ValueError("The group map must match the original 98 annotated recordings from 93 people")
    return labeled


def stage_record(directory, record, hashes):
    transferred = 0
    with requests.Session() as session:
        for ext in EXTENSIONS:
            name = f"{record}.{ext}"
            path = directory / name
            if path.exists() and sha256(path.read_bytes()).hexdigest() == hashes[name]:
                continue
            payload = fetch(session, name, hashes[name])
            draft = path.with_suffix(f".{ext}.partial")
            draft.write_bytes(payload)
            draft.replace(path)
            transferred += len(payload)
    return transferred


def stage(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        manifest = fetch(session, "SHA256SUMS.txt", MANIFEST_SHA)
        checksums = {key: digest for digest, key in (row.split(maxsplit=1) for row in manifest.decode().splitlines())}
        groups = record_groups(fetch(session, "AdditionalData.csv", checksums["AdditionalData.csv"]))
    if set(groups) != {key.split(".")[0] for key in checksums if key.endswith(".atr")}:
        raise ValueError("All source rhythm annotation files must map to exactly the published annotated roster")
    (directory / "SHA256SUMS.txt").write_bytes(manifest)
    total = 0
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(stage_record, directory, record, checksums) for record in groups]
        for finished, future in enumerate(as_completed(futures), 1):
            total += future.result()
            if finished % 10 == 0 or finished == len(groups):
                print(f"Verified SHDB-AF annotations for {finished}/{len(groups)} recordings", flush=True)
    receipt = {"release_sha256": MANIFEST_SHA, "recordings": groups}
    temp = directory / f"{GROUP_MANIFEST}.partial"
    temp.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(directory / GROUP_MANIFEST)
    print({"annotated_recordings": len(groups), "participants": len(set(groups.values())), "downloaded_bytes": total})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    stage(args.cache_dir)
