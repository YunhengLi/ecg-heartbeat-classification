"""Join the published ICU AF workbook to the original PhysioNet waveform headers."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, time, timedelta
from functools import partial
from hashlib import md5
import json
from pathlib import Path
import re
from time import sleep
from urllib.error import HTTPError
from urllib.request import urlopen

from openpyxl import load_workbook
from openpyxl.utils.datetime import to_excel

from .af_wrist_holdout import _write_json


WAVEFORMS = "https://physionet.org/files/mimic3wdb-matched/1.0"
CHECKSUM = "2a3de0609ab806eae37a4a534fa806a2"
RECORD = re.compile(r"(p\d{6})-\d{4}(?:-\d{2}){4}\Z")
ECG_LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V", "V1", "V2", "V3", "V4", "V5", "V6", "MLII", "MCL"}


def workbook_seconds(value, epoch):
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (time, datetime)):
        return to_excel(value, epoch) * 86400
    if isinstance(value, str) and re.fullmatch(r"\d+:[0-5]\d:[0-5]\d(?:\.\d+)?", value):
        hours, minutes, seconds = value.split(":")
        return 3600 * int(hours) + 60 * int(minutes) + float(seconds)
    raise ValueError(f"Unexpected annotation time type {type(value).__name__}")


def annotation_records(input_file):
    if md5(input_file.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("Use the original Figshare annotation workbook")
    book = load_workbook(input_file, read_only=True, data_only=True)
    rows = []
    rejected_identifiers = 0
    for status, code, *values in book.active.values:
        if not isinstance(code, int):
            continue
        for offset in range(0, len(values), 4):
            name, total, onset, end = values[offset:offset + 4]
            if name is None:
                if any(value is not None for value in (total, onset, end)):
                    raise ValueError("A waveform duration or annotation has no named reference")
                continue
            name = name.strip()
            match = RECORD.fullmatch(name)
            if match is None or status not in (0, 1):
                raise ValueError("Use valid coded waveforms and patient labels")
            if match[1] != f"p{code:06}":
                rejected_identifiers += 1
                continue
            rows.append((match[1], name, status, *(workbook_seconds(value, book.epoch) for value in (total, onset, end))))
    book.close()
    if len({row[0] for row in rows}) != 45 or len({row[1] for row in rows}) != len(rows):
        raise ValueError("The release must retain its full set of grouped waveforms")
    return rows, rejected_identifiers


def load_header(person, name, cache_dir):
    cache = cache_dir / person
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.hea"
    if path.exists():
        payload = path.read_bytes()
    else:
        for attempt in range(4):
            try:
                with urlopen(f"{WAVEFORMS}/{person[:3]}/{person}/{name}.hea", timeout=25) as reply:
                    payload = reply.read()
                break
            except HTTPError as error:
                if error.code == 404:
                    return None
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                retry = error.headers.get("Retry-After", "")
                delay = int(retry) if retry.isdecimal() else 3 * (attempt + 1)
                if delay > 30:
                    raise
                sleep(delay)
        path.write_bytes(payload)
    lines = [line.strip() for line in payload.decode("ascii").splitlines() if line.strip() and not line.startswith("#")]
    if not lines or lines[0].split()[0].split("/")[0] != name:
        raise ValueError("The returned WFDB header must match the requested public record")
    return lines


def probe(row, cache_dir):
    person, name, status, listed, onset, end = row
    header = load_header(person, name, cache_dir)
    result = {"participant": person, "record": name, "patient_af_status": status,
              "declared_seconds": listed, "annotation_start_seconds": onset, "annotation_end_seconds": end,
              "public_header_available": header is not None}
    if not header:
        return result
    title = header[0].split()
    rate = float(title[2].split("/")[0])
    length = int(title[3])
    channels = int(title[1])
    count = int(title[0].split("/")[1]) if "/" in title[0] else 0
    result.update(rate=rate, header_seconds=length / rate)
    segments = []
    layout = header
    if count:
        offset = 0
        for line in header[1:1 + count]:
            short, frames = line.split()[:2]
            if not int(frames):
                layout = load_header(person, short, cache_dir)
            segments.append((short, offset, offset + int(frames)))
            offset += int(frames)
        if offset != length or layout is None:
            result.update(segment_clock_consistent=offset == length, layout_header_available=layout is not None,
                          possible_ECG_and_PPG=False, annotation_valid=False)
            return result
    names = {line.split(maxsplit=8)[-1].upper() for line in layout[1:channels + 1]}
    result.update(segment_clock_consistent=True, layout_header_available=True, segment_clock=segments,
                  possible_ECG_and_PPG="PLETH" in names and bool(names & ECG_LEADS),
                  annotation_valid=(onset is not None and end is not None and 0 < onset < end < length / rate))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "header-cache", "private-clock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source, rejected_identifiers = annotation_records(args.annotations)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(partial(probe, cache_dir=args.header_cache), source))
    found = [r for r in rows if r["public_header_available"]]
    paired = [r for r in found if r["possible_ECG_and_PPG"]]
    complete = [r for r in paired if r["annotation_valid"]]
    status = Counter({row[0]: row[2] for row in source}.values())
    result = {"annotation_doi": "10.6084/m9.figshare.12149091.v1", "waveform_release": "mimic3wdb-matched/1.0",
              "subjects_in_workbook": len({row[0] for row in source}), "subjects_by_AF_status": dict(status),
              "named_waveforms": len(source) + rejected_identifiers, "waveforms_rejected_identifier_mismatch": rejected_identifiers,
              "public_header_available": len(found), "layout_with_ECG_and_pleth": len(paired),
              "inconsistent_segment_clocks": sum(not r["segment_clock_consistent"] for r in found),
              "unavailable_layout_headers": sum(not r["layout_header_available"] for r in found),
              "paired_layout_and_complete_AF_annotation": len(complete),
              "participants_with_paired_layout_and_complete_annotation": len({r['participant'] for r in complete}),
              "complete_annotations_all_available_headers": sum(r["annotation_valid"] for r in found),
              "declared_and_header_duration_within_one_second": sum(r["declared_seconds"] is not None and
                                                                     abs(r["declared_seconds"] - r["header_seconds"]) <= 1 for r in found)}
    _write_json(args.output, result)
    _write_json(args.private_clock, rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
