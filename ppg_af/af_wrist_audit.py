"""Audit selected paired AFPPGECG windows without training a classifier."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .wrist_audit import LABEL_MARGIN_SECONDS, WINDOW_SECONDS, audit_windows


@dataclass(frozen=True)
class AuditSlice:
    segment: int
    offset_s: float
    duration_s: int


def _parse_slice(value):
    try:
        index, offset, duration = value.split(":")
        request = AuditSlice(int(index), float(offset), int(duration))
        if (request.segment < 0 or not np.isfinite(request.offset_s) or request.offset_s < 0
                or request.duration_s <= 0 or request.duration_s % WINDOW_SECONDS):
            raise ValueError
        return request
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use SEGMENT:OFFSET_SECONDS:DURATION_SECONDS with duration a positive multiple of 30"
        ) from exc


def audit_slice(subject, request):
    segments = {segment.index: segment for segment in subject.segments}
    if request.segment not in segments:
        raise ValueError(f"Unknown wrist PPG segment: {request.segment}")
    segment = segments[request.segment]
    stop_offset = request.offset_s + request.duration_s
    if stop_offset > segment.duration_seconds:
        raise ValueError(f"The requested interval extends past segment {request.segment}")
    start = segment.start_time + request.offset_s
    stop = start + request.duration_s
    if start < 0 or stop > subject.ecg_duration_seconds + 1e-8:
        raise ValueError("The requested audit interval extends past the ECG recording")

    source = subject.read_segment(request.segment, request.offset_s, stop_offset)
    reference = subject.reference_rhythm_intervals(
        max(0, start - LABEL_MARGIN_SECONDS),
        min(subject.ecg_duration_seconds, stop + LABEL_MARGIN_SECONDS),
    )
    results = audit_windows(
        start_s=start, ppg=source.ppg, ppg_fs=source.ppg_fs, ecg=source.ecg, ecg_fs=source.ecg_fs,
        af_intervals_s=reference[reference[:, 2] == 1, :2], reference_intervals_s=reference[:, :2],
        acceleration=source.accel, acceleration_fs=source.accel_fs,
    )
    if len(results) != request.duration_s // WINDOW_SECONDS:
        raise ValueError("The native-rate clips did not cover every requested audit window")
    rows = [row.to_dict() for row in results]
    return {
        "segment": request.segment,
        "sync_method": segment.sync_method,
        "window_reference_counts": dict(sorted(Counter(row.reference for row in results).items())),
        "windows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg", type=Path, required=True, help="Local v6 AFPPGECG ECG MAT file")
    parser.add_argument("--ppg", type=Path, required=True, help="Local v6 AFPPGECG wrist PPG MAT file")
    parser.add_argument("--slice", action="append", type=_parse_slice, required=True,
                        help="SEGMENT:OFFSET_SECONDS:DURATION_SECONDS, repeatable")
    parser.add_argument("--output", type=Path, required=True, help="Local JSON report path")
    args = parser.parse_args()

    from .afppgecg import load_subject

    try:
        with load_subject(args.ecg, args.ppg) as subject:
            audits = [audit_slice(subject, request) for request in args.slice]
        report = {
            "dataset": "AFPPGECG v6 (10.5281/zenodo.23921320)",
            "scope": "Selected paired window data audit; no screening performance estimate",
            "window_seconds": WINDOW_SECONDS,
            "label_margin_seconds": LABEL_MARGIN_SECONDS,
            "audits": audits,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, IndexError) as exc:
        parser.exit(1, f"AFPPGECG audit failed: {exc}\n")
    print(f"Audited {sum(len(audit['windows']) for audit in audits)} windows")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
