"""Stage a fixed four-center paired waveform sample from the public release."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from email.utils import parsedate_to_datetime
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
from zipfile import ZipFile

import numpy as np
import requests


from .af_periwave_benchmark import PATIENTS_PER_CENTER, REVISION, SOURCE_RATE
from .af_wrist_features import WINDOW_SECONDS


REPO = "https://huggingface.co/datasets/DX-anes/PeriWaveSync-ABG"
SIZES = {"A": 3270989497, "B": 2435517886, "C": 2644258765, "D": 3476977156}
BLOCK_BYTES = 2 * 1024 * 1024
STARTS_SECONDS = (30, 90, 150, 210, 270)
MEMBER = re.compile(r"([A-D])-([0-9]+)-([0-9]+)\.txt\Z")


class SparseArchive(io.RawIOBase):
    """Read only validated byte ranges of a pinned public archive."""

    def __init__(self, center, session, cache):
        self.size = SIZES[center]
        self.uri = f"{REPO}/resolve/{REVISION}/{center}.zip"
        self.session = session
        self.cache = cache / "ranges" / REVISION / center
        self.cache.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Unsupported file seek origin")
        origin = (0, self.position, self.size)[whence]
        if origin + offset < 0:
            raise OSError("Negative file position")
        self.position = origin + offset
        return self.position

    def _block(self, number):
        start = number * BLOCK_BYTES
        end = min(start + BLOCK_BYTES, self.size) - 1
        count = end + 1 - start
        cached = self.cache / f"{number}.bin"
        if cached.exists() and cached.stat().st_size == count:
            return cached.read_bytes()
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        acceptable = f"bytes {start}-{end}/{self.size}"
        for attempt in range(4):
            with self.session.get(self.uri, headers=headers, timeout=30, stream=True) as reply:
                if reply.status_code == requests.codes.too_many_requests:
                    header = reply.headers.get("Retry-After")
                    if not header:
                        delay = 5 + 2 * attempt
                    elif header.isdecimal():
                        delay = int(header)
                    else:
                        delay = max(0, parsedate_to_datetime(header).timestamp() - time.time())
                    if delay > 30:
                        raise OSError("Public range retrieval deferred by the provider. Resume from the verified cache later.")
                    time.sleep(delay)
                    continue
                if reply.status_code != requests.codes.partial_content or reply.headers.get("Content-Range") != acceptable:
                    raise ValueError(f"Public archive server did not return the exact requested range: HTTP {reply.status_code}")
                payload = reply.raw.read(count + 1)
                if len(payload) != count:
                    raise ValueError("Received a truncated or oversized public byte range")
                temp = cached.with_suffix(".part")
                temp.write_bytes(payload)
                temp.replace(cached)
                self.downloaded += len(payload)
                return payload
        raise OSError("Public range requests continue to be rate limited")

    def read(self, count=-1):
        remaining = max(0, self.size - self.position)
        count = remaining if count < 0 else min(count, remaining)
        if count > 16 * BLOCK_BYTES:
            raise ValueError("Refusing a full-archive read in the sparse reader")
        output = bytearray()
        while len(output) < count:
            number, offset = divmod(self.position, BLOCK_BYTES)
            block = self._block(number)
            take = min(count - len(output), len(block) - offset)
            output.extend(block[offset:offset + take])
            self.position += take
        return bytes(output)


def choose_members(archive, center):
    patients = defaultdict(list)
    for info in archive.infolist():
        match = MEMBER.fullmatch(PurePosixPath(info.filename).name)
        if match and match[1] == center:
            patients[match[2]].append(info.filename)
    if len(patients) < PATIENTS_PER_CENTER:
        raise ValueError("The center ZIP has too few eligible patients")

    def rank(value):
        return sha256(f"paired-release-proof:{REVISION}:{center}:{value}".encode()).digest()

    selected = [(subject, min(patients[subject], key=rank)) for subject in sorted(patients, key=rank)[:PATIENTS_PER_CENTER]]
    return len(patients), selected


def read_waveforms(source):
    if not source.readline().startswith(b"Signal start time: "):
        raise ValueError("Missing the published relative waveform clock fields")
    output = []
    for label in (b"ECG1 data: ", b"PPG1 data: ", b"PPG2 data: "):
        line = source.readline().rstrip()
        if not line.startswith(label + b"[") or not line.endswith(b"]"):
            raise ValueError("Expected the three native waveform lists in the release order")
        encoded = line[len(label) + 1:-1]
        values = np.fromstring(encoded, sep=",", dtype=np.float32)
        if len(values) != encoded.count(b",") + 1:
            raise ValueError("The source waveform list contains unsupported fields")
        output.append(values)
    if source.read(1).strip() or len({len(row) for row in output}) != 1:
        raise ValueError("Published channels are not equal-length synchronous waveforms")
    return np.stack(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Private directory for staged data and source byte ranges")
    parser.add_argument("--zip-dir", type=Path, help="Use the four complete local archives instead of public byte ranges")
    args = parser.parse_args()
    receipt_path = args.cache_dir / "plan-private.json"
    inventory, member_names, downloaded = {}, {}, 0
    cache = args.cache_dir / "staged"
    cache.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for center in SIZES:
            with ExitStack() as stack:
                if args.zip_dir is None:
                    source = stack.enter_context(SparseArchive(center, session, args.cache_dir))
                else:
                    source = args.zip_dir / f"{center}.zip"
                    if source.stat().st_size != SIZES[center]:
                        raise ValueError("Local archive size does not match the pinned release")
                archive = stack.enter_context(ZipFile(source))
                inventory[center], selected = choose_members(archive, center)
                member_names[center] = selected
                for index, (_, member) in enumerate(selected):
                    target = cache / f"{center}-{index:02}.npz"
                    member_digest = sha256(member.encode()).hexdigest()
                    if target.exists():
                        with np.load(target, allow_pickle=False) as previous:
                            if previous["release_revision"].item() != REVISION or previous["member_digest"].item() != member_digest:
                                raise ValueError("Existing waveforms were staged from another member or release")
                        continue
                    with archive.open(member) as stream:
                        raw = read_waveforms(stream)
                    windows = [raw[:, start * SOURCE_RATE:(start + WINDOW_SECONDS) * SOURCE_RATE]
                               for start in STARTS_SECONDS if (start + WINDOW_SECONDS) * SOURCE_RATE <= raw.shape[-1]]
                    temp = target.with_suffix(".part.npz")
                    np.savez_compressed(temp, waveforms=np.asarray(windows).reshape(-1, 3, WINDOW_SECONDS * SOURCE_RATE),
                                        release_revision=REVISION, center=center,
                                        member_digest=member_digest)
                    temp.replace(target)
                    print(f"Paired waveforms staged in center {center}: {index + 1}/{len(selected)}", flush=True)
                if isinstance(source, SparseArchive):
                    downloaded += source.downloaded
    report = {"release_revision": REVISION, "participants_by_center": inventory, "selected_per_center": PATIENTS_PER_CENTER,
              "scheduled_start_seconds": STARTS_SECONDS, "window_seconds": WINDOW_SECONDS, "retrieved_bytes": downloaded}
    receipt_path.write_text(json.dumps({"public_receipt": report, "members": member_names}, indent=2) + "\n", encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
