"""Audit the public VitalDB arrhythmia annotations and paired waveform clocks."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
from math import floor, log10
from pathlib import Path

import numpy as np

from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import ecg_peaks


ANNOTATIONS = "https://physionet.org/content/vitaldb-arrhythmia/1.0.0/"
TRACK_API = "https://api.vitaldb.net/"
RELEASE_SHA256 = "d896fcc1420723d66649d0cf172c446e0185cc0913774326a66955882165d56a"
TRACKS = ("SNUADC/ECG_II", "SNUADC/PLETH")
RATE = 500
AF_OR_FLUTTER = "AFIB/AFL"
ECTOPY = {"Patterned Atrial Ectopy", "Patterned Ventricular Ectopy"}
LABELED_BEATS = {"N", "S", "V", "P"}
MAX_REFERENCE_GAP_S = 2.0
PEAK_TOLERANCE_S = 0.15
PILOT_GROUPS = ("af_flutter_and_sinus", "af_flutter_without_sinus", "ectopy_and_sinus_without_af_flutter")
CASES_PER_GROUP = 2


def read_release(directory):
    checksums = (directory / "SHA256SUMS.txt").read_bytes()
    if sha256(checksums).hexdigest() != RELEASE_SHA256:
        raise ValueError("Use the published VitalDB arrhythmia 1.0.0 checksum list")
    for line in checksums.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Annotation file does not match the release: {name}")
    with (directory / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    roster = {int(row["case_id"]): row["subjectid"] for row in rows}
    if len(rows) != 482 or len(roster) != len(rows) or not all(roster.values()):
        raise ValueError("Each annotation case must have a known participant grouping value")
    return roster


def read_tracks(path):
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    open_path = gzip.open if compressed else open
    tracks = defaultdict(dict)
    with open_path(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["tname"]
            if name in TRACKS:
                case = int(row["caseid"])
                if name in tracks[case]:
                    raise ValueError("Track list contains duplicate ECG or plethysmogram entries")
                tracks[case][name] = row["tid"]
    return tracks


def stable_reference(times, labels, usable):
    """Use elapsed time bracketed by adjacent clean beats with the same label."""
    spans = []
    gaps = np.diff(times)
    good_edge = usable[:-1] & usable[1:] & (labels[:-1] == labels[1:]) & (gaps > 0) & (gaps <= MAX_REFERENCE_GAP_S)
    left = np.flatnonzero(good_edge & ~np.r_[False, good_edge[:-1]])
    right = np.flatnonzero(good_edge & ~np.r_[good_edge[1:], False]) + 1
    for start, stop in zip(left, right, strict=True):
        # Adjacent valid edges can share a beat only if they have the same label.
        spans.append((float(times[start]), float(times[stop]), str(labels[start])))
    return spans


def read_annotations(directory, case):
    path = directory / "Annotation_Files" / f"Annotation_file_{case}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["time_second"]) for row in rows])
    labels = np.array([row["rhythm_label"] for row in rows])
    bad = np.array([row["bad_signal_quality"] for row in rows])
    if (len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) < 0)
            or not np.isin(bad, ("True", "False")).all()):
        raise ValueError("Annotation times and quality flags must be valid on the original case clock")
    beat = np.array([row["beat_type"] in LABELED_BEATS for row in rows])
    marker = np.array([bool(row.get("bad_signal_quality_label")) for row in rows])
    clean_beats = beat & ~marker & (bad == "False")
    usable = clean_beats & ~np.isin(labels, ("", "Noise", "Unclassifiable"))
    return times, labels, clean_beats, stable_reference(times, labels, usable)


def decision_windows(spans):
    """Return scheduled window ends fully supported by one clean rhythm span."""
    for start, stop, label in spans:
        first = int(np.ceil((start + WINDOW_SECONDS) / STRIDE_SECONDS)) * STRIDE_SECONDS
        for end in range(first, int(np.floor(stop)) + 1, STRIDE_SECONDS):
            yield end, label


def plan_cases(directory, track_path):
    roster = read_release(directory)
    tracks = read_tracks(track_path)
    paired = {case for case in roster if set(tracks[case]) == set(TRACKS)}
    per_label_cases, windows, labels, groups = defaultdict(set), Counter(), set(), defaultdict(list)
    annotated_af_cases, af_flutter_patients = set(), set()
    for case, participant in roster.items():
        _, rhythms, _, spans = read_annotations(directory, case)
        if AF_OR_FLUTTER in rhythms:
            annotated_af_cases.add(case)
        if case not in paired:
            continue
        labels.update(rhythms)
        counts = Counter(label for _, label in decision_windows(spans))
        windows.update(counts)
        for label in counts:
            per_label_cases[label].add(case)
        if AF_OR_FLUTTER in rhythms:
            af_flutter_patients.add(participant)
        if AF_OR_FLUTTER in counts and "N" in counts:
            groups[PILOT_GROUPS[0]].append(case)
        elif AF_OR_FLUTTER in counts:
            groups[PILOT_GROUPS[1]].append(case)
        elif AF_OR_FLUTTER not in rhythms and "N" in counts and ECTOPY & counts.keys():
            groups[PILOT_GROUPS[2]].append(case)

    seen_participants = set()
    pilot = []
    for group in PILOT_GROUPS:
        selection = sorted(groups[group], key=lambda case: sha256(f"wrist-af-vitaldb-2026:{case}".encode()).digest())
        selected = []
        for case in selection:
            participant = roster[case]
            if participant in seen_participants:
                continue
            seen_participants.add(participant)
            selected.append({"case_id": case, "stratum": group, "tracks": tracks[case]})
            if len(selected) == CASES_PER_GROUP:
                break
        if len(selected) != CASES_PER_GROUP:
            raise ValueError(f"The input-only pilot cannot fill stratum {group}")
        pilot.extend(selected)
    support = {"annotated_cases": len(roster), "annotated_participants": len(set(roster.values())),
               "annotated_af_or_flutter_cases": len(annotated_af_cases),
               "paired_catalog_cases": len(paired), "paired_catalog_participants": len({roster[case] for case in paired}),
               "paired_catalog_af_or_flutter_cases": len(annotated_af_cases & paired),
               "paired_catalog_af_or_flutter_participants": len(af_flutter_patients),
               "source_af_label": AF_OR_FLUTTER, "rhythm_labels_in_paired_catalog": sorted(labels - {""}),
               "candidate_clean_windows_by_label": dict(windows),
               "cases_with_candidate_clean_windows_by_label": {label: len(cases) for label, cases in per_label_cases.items()}}
    return {"annotation_release_sha256": RELEASE_SHA256, "track_listing_sha256": sha256(track_path.read_bytes()).hexdigest(),
            "support": support, "pilot": pilot}


def load_track(path, name):
    """Decode the legacy public 500 Hz track CSV, preserving empty rows."""
    documented_edges = []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Time", name]:
            raise ValueError("Waveform CSV must name the planned track")

        def rows():
            for i, (stamp, value) in enumerate(reader):
                if stamp:
                    documented_edges.append((i, float(stamp)))
                yield float(value) if value else np.nan

        values = np.fromiter(rows(), dtype=np.float32)
    if (len(documented_edges) != 3 or documented_edges[:2] != [(0, 0.0), (1, 1 / RATE)]
            or documented_edges[-1][0] != len(values) - 1):
        raise ValueError("Waveform CSV does not have the documented full-case 500 Hz clock")
    last_time = (len(values) - 1) / RATE
    # The endpoint is printed to six significant digits. Row indices retain the sample clock.
    rounding = 0.5 * 10 ** (floor(log10(last_time)) - 5)
    if abs(documented_edges[-1][1] - last_time) > rounding + 1e-8:
        raise ValueError("Waveform endpoint is inconsistent with the documented sample count")
    return values


def load_native_case(path):
    """Keep the EVENT track so the annotation and waveform have the same origin."""
    import vitaldb

    recording = vitaldb.VitalFile(str(path), ["EVENT", *TRACKS])
    origin = recording.find_track("EVENT")
    tracks = [recording.find_track(name) for name in TRACKS]
    if not origin or not origin.recs or origin.recs[0]["dt"] != recording.dtstart:
        raise ValueError("The EVENT track must anchor the original full-record clock")
    if any(not track or not track.recs or track.srate != RATE for track in tracks):
        raise ValueError("The native file must contain both original 500 Hz waveform tracks")
    first = [track.recs[0]["dt"] - recording.dtstart for track in tracks]
    if first[0] != first[1] or first[0] < 0:
        raise ValueError("The native ECG and PPG do not start together on the original case clock")
    return tuple(recording.get_track_samples(name, 1 / RATE) for name in TRACKS), first[0]


def _waveform_support(waveforms, times, clean, spans):
    if len(waveforms[0]) != len(waveforms[1]) or times[-1] * RATE >= len(waveforms[0]):
        raise ValueError("Both tracks and the annotation endpoints must share the same full case clock")
    paired = Counter()
    planned_windows = list(decision_windows(spans))
    reference_count = matched = 0
    for end, label in planned_windows:
        start = end - WINDOW_SECONDS
        ecg_clip, pulse_clip = (signal[start * RATE:end * RATE] for signal in waveforms)
        if np.isfinite(ecg_clip).all() and np.isfinite(pulse_clip).all() and np.ptp(ecg_clip) > 0 and np.ptp(pulse_clip) > 0:
            paired[label] += 1
        if end % WINDOW_SECONDS or not np.isfinite(ecg_clip).all():
            continue
        reference = times[clean & (times >= start + 1) & (times < end - 1)]
        detected = start + ecg_peaks(ecg_clip, RATE) / RATE
        if len(detected):
            indices = np.searchsorted(detected, reference)
            before = detected[np.maximum(0, indices - 1)]
            after = detected[np.minimum(len(detected) - 1, indices)]
            matched += int(np.sum(np.minimum(np.abs(before - reference), np.abs(after - reference)) <= PEAK_TOLERANCE_S))
        reference_count += len(reference)
    if reference_count == 0:
        raise ValueError("No clean reference beats fell in a nonoverlapping clock-check window")
    return {"annotation_stable_windows": len(planned_windows), "finite_paired_windows": sum(paired.values()),
            "finite_paired_windows_by_label": dict(paired), "reference_peaks_checked": reference_count,
            "reference_peaks_with_nearby_waveform_peak": matched, "matched_peak_fraction": matched / reference_count}


def verify_case(directory, waveform_cache, native_cache, planned):
    case = planned["case_id"]
    times, _, clean, spans = read_annotations(directory, case)
    legacy = tuple(load_track(waveform_cache / f"{case}-{name.rsplit('/', 1)[-1]}.csv.gz", name) for name in TRACKS)
    if any(not np.isfinite(track).any() for track in legacy):
        raise ValueError("The legacy waveform tracks must contain observed samples")
    legacy_aggregate = _waveform_support(legacy, times, clean, spans)
    legacy_first = int(np.argmax(np.isfinite(legacy[0]))) / RATE
    del legacy
    native, native_first = load_native_case(native_cache / f"{case}.vital")
    return {"native": _waveform_support(native, times, clean, spans), "legacy_csv": legacy_aggregate,
            "native_minus_legacy_first_ecg_seconds": native_first - legacy_first}


def _aggregate(rows):
    peak_fractions = [row["matched_peak_fraction"] for row in rows]
    by_label = Counter()
    for row in rows:
        by_label.update(row["finite_paired_windows_by_label"])
    return {"annotation_stable_windows": sum(row["annotation_stable_windows"] for row in rows),
            "finite_paired_windows": sum(row["finite_paired_windows"] for row in rows),
            "finite_paired_windows_by_label": dict(by_label),
            "reference_peaks_checked": sum(row["reference_peaks_checked"] for row in rows),
            "reference_peaks_with_nearby_waveform_peak": sum(row["reference_peaks_with_nearby_waveform_peak"] for row in rows),
            "case_peak_agreement_min": min(peak_fractions), "case_peak_agreement_median": float(np.median(peak_fractions))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--track-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--private-plan", action="store_true", help="Write the local acquisition plan with source keys")
    choices.add_argument("--waveform-cache", type=Path, help="Folder with the legacy per-track CSV files")
    parser.add_argument("--native-cache", type=Path, help="Folder with the original .vital case files")
    args = parser.parse_args()
    if args.waveform_cache and not args.native_cache:
        parser.error("Supply both the legacy CSV and native waveform caches for the clock comparison")
    try:
        plan = plan_cases(args.annotations_dir, args.track_list)
        if args.private_plan:
            _write_json(args.output, plan)
        else:
            rows = [verify_case(args.annotations_dir, args.waveform_cache, args.native_cache, item) for item in plan["pilot"]]
            offsets = [row["native_minus_legacy_first_ecg_seconds"] for row in rows]
            pilot = {"participants": len(rows), "cases": len(rows), "case_counts_by_stratum": dict(Counter(
                        row["stratum"] for row in plan["pilot"])),
                     "native": _aggregate([row["native"] for row in rows]),
                     "legacy_csv": _aggregate([row["legacy_csv"] for row in rows]),
                     "native_minus_legacy_first_ecg_seconds": {"minimum": min(offsets), "maximum": max(offsets),
                                                               "median": float(np.median(offsets))}}
            _write_json(args.output, {"source": ANNOTATIONS, "track_api": TRACK_API,
                                      "annotation_release_sha256": plan["annotation_release_sha256"],
                                      "track_listing_sha256": plan["track_listing_sha256"], "source_waveform_rate_hz": RATE,
                                      "reference_maximum_beat_gap_seconds": MAX_REFERENCE_GAP_S,
                                      "peak_agreement_tolerance_seconds": PEAK_TOLERANCE_S, "window_seconds": WINDOW_SECONDS,
                                      "decision_stride_seconds": STRIDE_SECONDS, "source_support": plan["support"],
                                      "paired_waveform_pilot": pilot})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"VitalDB preflight failed: {error}\n")


if __name__ == "__main__":
    main()
