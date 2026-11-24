"""Build a measured ECG interval benchmark on the annotated SHDB-AF clock."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import wfdb

from .af_events import _coverage, _union
from .af_shdb_stage import EXTENSIONS, GROUP_MANIFEST, MANIFEST_SHA
from .af_wrist_holdout import _write_json
from .features import ECG_FEATURE_NAMES


RATE = 200
STRIDE = 10
HORIZONS = (10, 30)
MAX_GAP = 4.0
RHYTHM = {"(AFIB": 1, "(N": 0, "(AFL": 0, "(AT": 0, "(PAT": 0, "(NOD": 0, "(AB": -1}


def inventory(directory):
    manifest = (directory / "SHA256SUMS.txt").read_bytes()
    groups = json.loads((directory / GROUP_MANIFEST).read_bytes())
    records = groups["recordings"]
    if (sha256(manifest).hexdigest() != MANIFEST_SHA or groups["release_sha256"] != MANIFEST_SHA
            or len(records) != 98 or len(set(records.values())) != 93):
        raise ValueError("Use the checksummed SHDB-AF release and published subject group mapping")
    hashes = {name: digest for digest, name in (line.split(maxsplit=1) for line in manifest.decode().splitlines())}
    if set(records) != {name.split(".")[0] for name in hashes if name.endswith(".atr")}:
        raise ValueError("The staged group map has changed the published annotated records")
    return records, hashes


def annotations(directory, record, hashes):
    for extension in EXTENSIONS:
        name = f"{record}.{extension}"
        if sha256((directory / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("A staged Holter annotation did not match its released checksum")
    base = str(directory / record)
    clock = wfdb.rdheader(base)
    beats = wfdb.rdann(base, "qrs")
    episodes = wfdb.rdann(base, "atr")
    notes = np.asarray(episodes.aux_note)
    selected = notes != ""
    labels = [RHYTHM[note] for note in notes[selected]]
    starts = episodes.sample[selected]
    if (clock.fs != RATE or episodes.fs != RATE or beats.fs != RATE or set(beats.symbol) != {"N"}
            or not len(starts) or np.any(np.diff(starts) < 0) or starts[0] < 0 or starts[-1] >= clock.sig_len
            or np.any(np.diff(beats.sample) < 0) or beats.sample[0] < 0 or beats.sample[-1] >= clock.sig_len):
        raise ValueError("ECG reference annotations must use the released 200 Hz Holter clock")
    # A subsequent mark at the same sample overrides the earlier state.
    last_marker = np.r_[starts[1:] > starts[:-1], True]
    starts = starts[last_marker]
    labels = np.asarray(labels)[last_marker]
    spans = np.column_stack((starts, np.r_[starts[1:], clock.sig_len], labels))
    peaks = np.unique(beats.sample)
    counts = Counter(notes[selected])
    return clock.sig_len, peaks, spans, dict(counts), len(beats.sample) - len(peaks)


def reference_grid(spans, length):
    ends = np.arange(max(HORIZONS), length // RATE + 1, STRIDE) * RATE
    decision_start = ends - STRIDE * RATE
    known = decision_start >= spans[0, 0]
    for left, right, _ in spans[spans[:, 2] < 0]:
        known &= (right <= decision_start) | (left >= ends)
    onset = ends - max(HORIZONS) * RATE
    membership = np.searchsorted(spans[:, 0], onset, side="right") - 1
    stable = (membership >= 0) & (ends <= spans[membership, 1])
    labels = np.where(stable, spans[membership, 2], -1)
    af = _union(spans[spans[:, 2] == 1, :2])
    if len(af) and af[0, 0] == spans[0, 0]:
        af[0, 0] = -1
    if len(af) and af[-1, 1] == length:
        af[-1, 1] += 1
    return ends, known, labels, af / RATE


def interval_matrix(peaks, ends, seconds):
    start = ends - seconds * RATE
    left, right = np.searchsorted(peaks, np.column_stack((start, ends))).T
    counts = right - left
    intervals = np.diff(peaks) / RATE
    offsets = np.arange(max(2, int(counts.max(initial=0)) - 1))
    positions = left[:, None] + offsets
    present = offsets[None, :] < (counts - 1)[:, None]
    rows = intervals[np.minimum(positions, len(intervals) - 1)]
    ordered = np.sort(np.where(present, rows, np.inf), axis=1)
    valid = (counts >= 3) & (peaks[np.minimum(left, len(peaks) - 1)] - start <= MAX_GAP * RATE)
    valid &= ends - peaks[np.maximum(right - 1, 0)] <= MAX_GAP * RATE
    valid &= np.max(np.where(present, rows, 0), axis=1) <= MAX_GAP
    result = np.full((len(ends), len(ECG_FEATURE_NAMES)), np.nan, dtype=np.float32)
    if not np.any(valid):
        return result
    count = counts[valid] - 1
    fractions = (count[:, None] - 1) * np.array([0.25, 0.5, 0.75])
    lower, upper = np.floor(fractions).astype(int), np.ceil(fractions).astype(int)
    compact = ordered[valid]
    bottom = np.take_along_axis(compact, lower, axis=1)
    top = np.take_along_axis(compact, upper, axis=1)
    p25, median, p75 = (bottom + (top - bottom) * (fractions - lower)).T
    values, mask = rows[valid], present[valid]
    means = np.sum(np.where(mask, values, 0), axis=1) / count
    variance = np.sum(np.where(mask, np.square(values - means[:, None]), 0), axis=1) / count
    differences = np.diff(values, axis=1)
    diff_valid = mask[:, 1:]
    rmssd = np.sqrt(np.sum(np.where(diff_valid, np.square(differences), 0), axis=1) / (count - 1)) / median
    large = np.sum(diff_valid & (np.abs(differences) > 0.05), axis=1) / (count - 1)
    result[valid] = np.column_stack((counts[valid] / seconds, median, (p75 - p25) / median, np.sqrt(variance) / means,
                                    rmssd, large, np.ones(len(count))))
    return result


def stage_features(directory, feature_dir):
    roster, hashes = inventory(directory)
    feature_dir.mkdir(parents=True, exist_ok=True)
    rhythm_marks, totals = Counter(), Counter()
    for index, (record, person) in enumerate(roster.items(), 1):
        length, peaks, spans, marks, duplicates = annotations(directory, record, hashes)
        ends, known, labels, af = reference_grid(spans, length)
        matrices = np.stack([interval_matrix(peaks, ends, seconds) for seconds in HORIZONS])
        common = np.isfinite(matrices).all(axis=(0, 2))
        boundaries = np.flatnonzero(np.diff(known, prepend=False, append=False))
        edges = np.r_[ends[0] - STRIDE * RATE, ends] / RATE
        complete, _ = _coverage(af, edges[boundaries.reshape(-1, 2)])
        totals.update(unique_qrs_peaks=len(peaks), duplicate_qrs_samples=duplicates, scheduled_windows=len(ends),
                      known_reference_windows=int(known.sum()), scored_common_windows=int(common.sum()),
                      known_reference_scored_windows=int(np.sum(known & common)), complete_af_episodes=int(complete.sum()),
                      complete_af_episodes_below_30_seconds=int(np.sum(complete & ((af[:, 1] - af[:, 0]) < 30))),
                      stable_af_fit_candidates=int(np.sum((labels == 1) & common)),
                      stable_non_af_fit_candidates=int(np.sum((labels == 0) & common)))
        rhythm_marks.update(marks)
        target = feature_dir / f"{record}.npz"
        draft = target.with_suffix(".partial.npz")
        np.savez_compressed(draft, features=matrices, reference_valid=known, stable_label=labels, ends_s=ends / RATE,
                            af_intervals_s=af, group=person, release_sha=MANIFEST_SHA)
        draft.replace(target)
        if index % 10 == 0 or index == len(roster):
            print(f"Prepared ECG interval features for {index}/{len(roster)} Holters", flush=True)
    return {"scope": "Annotated ECG beat and rhythm preflight for the SHDB-AF continuous reference",
            "release_sha256": MANIFEST_SHA, "annotated_recordings": len(roster), "participants": len(set(roster.values())),
            "native_annotation_rate_hz": RATE, "decision_stride_seconds": STRIDE, "feature_horizons_seconds": HORIZONS,
            "rhythm_annotation_mark_counts": dict(rhythm_marks), "clinical_record_totals": dict(totals)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("annotations-dir", "feature-dir", "output"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, stage_features(args.annotations_dir, args.feature_dir))
