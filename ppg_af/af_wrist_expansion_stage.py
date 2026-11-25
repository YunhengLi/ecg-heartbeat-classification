"""Stage disjoint stable AF fitting windows from local v6 sources and caches."""

import argparse
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path

import numpy as np

from .af_wrist_cohort import CONTEXT, WINDOW, core_label, in_wrist_segment
from .af_wrist_data_preflight import _inventory
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_template_independent import _archives
from .af_wrist_train import load_features
from .afppgecg import ECG_FS, PPG_FS, _read_vector, _sample_bounds, load_subject


SEED = 20261124
MAX_PER_RHYTHM = 8
BIN_SECONDS = 300
BLOCK_BYTES = 1 << 20


class CacheMiss(OSError):
    """The local archive lacks a required source byte block."""


class CachedMAT(io.RawIOBase):
    """Read HDF5 through length-verified local one-MiB blocks."""

    def __init__(self, directory, size):
        super().__init__()
        self.directory, self.size, self.position = directory, size, 0
        self.block = lru_cache(maxsize=128)(self._block)

    def _block(self, number):
        path = self.directory / f"{BLOCK_BYTES}-{number}.bin"
        expected = min(BLOCK_BYTES, self.size - number * BLOCK_BYTES)
        try:
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise CacheMiss("A required source byte block is absent") from error
        if len(data) != expected:
            raise CacheMiss("A required source byte block is absent or incomplete")
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origins or origins[whence] + offset < 0:
            raise OSError("Invalid MAT file byte seek")
        self.position = origins[whence] + offset
        return self.position

    def read(self, count=-1):
        stop = self.size if count < 0 else min(self.size, self.position + count)
        chunks = []
        while self.position < stop:
            number, begin = divmod(self.position, BLOCK_BYTES)
            length = min(stop - self.position, BLOCK_BYTES - begin)
            chunks.append(self.block(number)[begin:begin + length])
            self.position += length
        return b"".join(chunks)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)

    def close(self):
        self.block.cache_clear()
        super().close()


def _open_source(stack, entry, source_dir, cache_dir):
    path = source_dir / entry["key"]
    if path.is_file():
        with path.open("rb") as stream:
            actual = "md5:" + file_digest(stream, "md5").hexdigest()
        if path.stat().st_size != entry["size"] or actual != entry["checksum"]:
            raise ValueError("A complete local MAT file disagrees with the source inventory")
        return path
    return stack.enter_context(CachedMAT(cache_dir / entry["key"], entry["size"]))


def _disjoint(starts, reserved):
    width = WINDOW + 2 * CONTEXT
    return (np.searchsorted(reserved, starts + width, side="left")
            == np.searchsorted(reserved, starts - width, side="right"))


def _read_pair(subject, start):
    end = start + WINDOW
    segment = in_wrist_segment(subject, start - CONTEXT, end + CONTEXT)
    if segment is None:
        return None
    data = subject._ppg_data[segment.index]
    left, right = _sample_bounds(start - CONTEXT, end + CONTEXT, segment.start_time, PPG_FS, data.size)
    try:
        pulse = _read_vector(data, slice(left, right))
        flank = int(CONTEXT * PPG_FS)
        if len(pulse) != int((WINDOW + 2 * CONTEXT) * PPG_FS):
            raise ValueError("Unexpected PPG clock for a paired source window")
        if not np.isfinite(pulse).all() or np.ptp(pulse[flank:-flank]) <= 0:
            return None
        ecg = _read_vector(subject._voltage, slice(int(ECG_FS * start), int(ECG_FS * end)))
    except CacheMiss:
        return None
    if len(ecg) != int(WINDOW * ECG_FS):
        raise ValueError("Unexpected ECG clock for a paired source window")
    return (pulse, ecg, segment.sync_method) if np.isfinite(ecg).all() else None


def _stable_candidates(subject, reference, reserved):
    possible = {label: set() for label in (0, 1)}
    for segment in subject.segments:
        low, high = max(0.0, segment.start_time), min(subject.ecg_duration_seconds, segment.stop_time)
        first = np.searchsorted(reference[:, 1], low, side="right")
        last = np.searchsorted(reference[:, 0], high, side="left")
        for start, stop, label in reference[first:last]:
            lower = int(np.ceil((max(start, low) + CONTEXT) / WINDOW))
            upper = int(np.floor((min(stop, high) - WINDOW - CONTEXT) / WINDOW))
            possible[int(label)].update(range(lower * WINDOW, (upper + 1) * WINDOW, WINDOW))
    verified = {}
    for label, candidates in possible.items():
        starts = np.fromiter(sorted(candidates), dtype=int)
        usable = (int(start) for start in starts[_disjoint(starts, reserved)])
        verified[str(label)] = [start for start in usable if _read_pair(subject, start) is not None]
    return verified


def _select_windows(verified, reserved, subject):
    selected = []
    for label in (1, 0):
        eligible = np.asarray(verified[str(label)])
        if eligible.ndim != 1 or (eligible.size and not np.issubdtype(eligible.dtype, np.integer)):
            raise ValueError("Candidate starts must be an integer vector")
        eligible = np.sort(eligible.astype(int, copy=False))
        if (len(np.unique(eligible)) != len(eligible) or np.any(eligible % WINDOW)
                or not _disjoint(eligible, reserved).all()):
            raise ValueError("An eligibility plan repeats, shifts, or overlaps a reserved fitting interval")
        bins = eligible // BIN_SECONDS
        available = np.unique(bins)
        if not len(available):
            continue
        seed = int.from_bytes(sha256(f"{SEED}:{subject}:{label}".encode()).digest()[:8], "little")
        random = np.random.default_rng(seed)
        for stratum in np.array_split(available, min(MAX_PER_RHYTHM, len(available))):
            used = np.array(sorted(start for start, _ in selected), dtype=int)
            for bin_id in random.permutation(stratum):
                candidates = eligible[bins == bin_id]
                candidates = candidates[_disjoint(candidates, used)]
                if len(candidates):
                    selected.append((int(random.choice(candidates)), label))
                    break
    return sorted(selected)


def _reserved(raw_path, preroll_path, subject):
    with (np.load(raw_path, allow_pickle=False) as raw, np.load(preroll_path, allow_pickle=False) as preroll):
        if raw["subject_id"].item() != subject or preroll["subject_id"].item() != subject:
            raise ValueError("Reserved archive identity mismatch")
        return np.unique(np.r_[raw["start_s"], preroll["start_s"]])


def _stage(args, roster, entries, plan):
    selected, candidates = [], {}
    for person in sorted(roster):
        name = f"{person}.npz"
        reserved = _reserved(args.development_raw / name, args.development_preroll / name, person)
        with ExitStack() as stack:
            sources = [_open_source(stack, entries[f"{person}_{signal}.mat"], args.source_dir, args.cache_dir)
                       for signal in ("ECG", "PPG")]
            subject = stack.enter_context(load_subject(*sources, subject_id=person, include_acceleration=False))
            reference = subject.reference_rhythm_intervals()
            verified = _stable_candidates(subject, reference, reserved) if plan is None else plan[person]
            candidates[person] = verified
            for start, rhythm in _select_windows(verified, reserved, person):
                end = start + WINDOW
                pair = _read_pair(subject, start)
                if pair is None or core_label(reference, start, end, pair[1]) != (rhythm, True):
                    raise ValueError("A selected candidate lacks its paired waveform or stable ECG reference")
                selected.append((person, start, end, rhythm, pair[2], pair[0]))
    return selected, candidates


def _rhythm_support(features, feature_people, subjects, labels):
    fit = _shared_features(features) & (features["run_kind"] != "enriched_episode")
    stats = {}
    for label, rhythm in ((0, "non_AF"), (1, "AF")):
        original = Counter(feature_people[fit & (features["stable_label"] == label)])
        selected = Counter(subjects[labels == label])
        combined = original + selected
        stats[rhythm] = {"original_windows": original.total(), "original_participants": len(original),
                         "selected_windows": selected.total(), "selected_participants": len(selected),
                         "new_label_donors": len(selected.keys() - original.keys()),
                         "original_top_two_share": sum(n for _, n in original.most_common(2)) / original.total(),
                         "expanded_top_two_share_if_all_score": sum(n for _, n in combined.most_common(2)) / combined.total()}
    return stats


def _retain_prefix(selected, path, lock_hash):
    rows = {(row[0], row[1]): row for row in selected}
    prefix = []
    with np.load(path, allow_pickle=False) as previous:
        if previous["source_protocol_sha256"].item() != lock_hash:
            raise ValueError("A previous bundle must belong to the same development protocol")
        fields = ("subjects", "start_s", "decision_end_s", "stable_label", "sync_method", "ppg_with_context")
        for old in zip(*(previous[key] for key in fields), strict=True):
            row = rows.pop(old[:2], None)
            if row is None or old[:5] != row[:5] or not np.array_equal(old[5], row[5]):
                raise ValueError("The new selection must contain each previous bundle row unchanged")
            prefix.append(row)
    return [*prefix, *rows.values()]


def stage(args):
    manifests = [path.read_bytes() for path in (args.source_inventory, args.lock)]
    inventory, lock = map(json.loads, manifests)
    inventory_hash, lock_hash = (sha256(blob).hexdigest() for blob in manifests)
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    roster = fit | calibration
    features, feature_people, _ = load_features(args.development_features)
    if (fit & calibration or roster & reserved or set(feature_people) != roster
            or any(set(_archives(path)) != roster for path in (args.development_raw, args.development_preroll))
            or features[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or (len(fit), len(calibration), len(reserved)) != (5, 5, 25)):
        raise ValueError("Use the original locked development roster")
    _inventory(inventory, roster | reserved)
    entries = {item["key"]: item for item in inventory["files"]}
    plan, plan_hash = None, None
    if args.eligibility_plan is not None:
        raw_plan = args.eligibility_plan.read_bytes()
        plan_hash = sha256(raw_plan).hexdigest()
        recorded = json.loads(raw_plan)
        plan = {row["subject_id"]: row["stable_starts_by_class"] for row in recorded["development"]}
        if (set(plan) != roster or len(recorded["development"]) != len(roster)
                or recorded["source_protocol_sha256"] != lock_hash):
            raise ValueError("The eligibility plan must match the locked development roster")

    selected, candidates = _stage(args, roster, entries, plan)
    prefix_hash = None
    if args.retain_prefix is not None:
        selected = _retain_prefix(selected, args.retain_prefix, lock_hash)
        with args.retain_prefix.open("rb") as stream:
            prefix_hash = file_digest(stream, "sha256").hexdigest()
    subjects, starts, ends, labels, methods, ppg = map(np.asarray, zip(*selected, strict=True))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with args.bundle.open("xb") as stream:
        np.savez_compressed(stream, subjects=subjects, start_s=starts, decision_end_s=ends, stable_label=labels.astype(np.int8),
                            sync_method=methods, ppg_with_context=ppg, ppg_fs=int(PPG_FS), ecg_reference_fs=int(ECG_FS),
                            feature_window_seconds=WINDOW, ppg_flank_seconds=CONTEXT, source_protocol_sha256=lock_hash)
    with args.bundle.open("rb") as stream:
        bundle_hash = file_digest(stream, "sha256").hexdigest()
    _write_json(args.report, {
        "scope": "Verified stable rhythm fitting expansion in disjoint cached development intervals",
        "source_release_doi": inventory["doi"], "source_inventory_sha256": inventory_hash,
        "source_protocol_sha256": lock_hash, "eligibility_plan_sha256": plan_hash,
        "extra_bundle_sha256": bundle_hash, "retained_prefix_sha256": prefix_hash, "network_requests": 0,
        "development_participants": len(roster),
        "participants_with_extra_windows": len(set(subjects)), "selected_windows": len(starts),
        "selection_seed": SEED, "maximum_windows_per_participant_rhythm": MAX_PER_RHYTHM,
        "maximum_windows_per_participant_rhythm_bin": 1, "bin_seconds": BIN_SECONDS,
        "ppg_sample_rate_hz": int(PPG_FS), "feature_window_seconds": WINDOW, "ppg_flank_seconds": CONTEXT,
        "decision_end_equals_start_plus_window": True, "finite_padded_nonflat_core_waveforms": len(ppg),
        "stable_labels_require_full_ecg_and_one_second_same_rhythm_flanks": True,
        "no_existing_staging_or_preroll_waveform_overlap": True, "sync_method_counts": dict(Counter(methods)),
        "verified_stable_candidate_windows": {rhythm: sum(len(person[str(label)]) for person in candidates.values())
                                              for label, rhythm in ((0, "non_AF"), (1, "AF"))},
        "by_stable_rhythm": _rhythm_support(features, feature_people, subjects, labels),
    })
    return len(subjects), len(set(subjects))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = ("source-inventory", "source-dir", "cache-dir", "development-raw", "development-features",
                "development-preroll", "lock", "bundle", "report")
    for argument in required:
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--eligibility-plan", type=Path, help="Optional private stable-candidate snapshot")
    parser.add_argument("--retain-prefix", type=Path, help="Optional previous private bundle to preserve unchanged")
    args = parser.parse_args()
    if args.bundle.resolve() == args.report.resolve() or args.bundle.exists() or args.report.exists():
        parser.error("Choose new bundle and receipt paths")
    try:
        count, people = stage(args)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF expansion staging failed: {error}\n")
    print(f"Staged {count} verified stable fitting windows from {people} development participants")


if __name__ == "__main__":
    main()
