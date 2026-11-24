"""Choose extra wrist monitoring using only source clocks and excluded time ranges."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_features import WINDOW_SECONDS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_monitor_extension import BLOCK_S, CONTEXT_S, EARLIER_S
from .afppgecg import load_subject


BLOCKS_PER_PERSON = 3


def exclusion_intervals(person, paths, supplement):
    spans = []
    for path in paths:
        with np.load(path / f"{person}.npz", allow_pickle=False) as raw:
            spans.extend((float(start) - CONTEXT_S, float(start) + WINDOW_SECONDS + CONTEXT_S) for start in raw["start_s"])
    spans.extend((float(start) - CONTEXT_S, float(start) + WINDOW_SECONDS + CONTEXT_S) for start in supplement.get(person, ()))
    return np.asarray(spans).reshape(-1, 2)


def choose_monitor_blocks(segments, ecg_seconds, excluded, seed, *, blocks_per_person=BLOCKS_PER_PERSON):
    """Draw uniform starts with disjoint preceding inputs and monitoring time."""
    index_by_start = {}
    for segment in segments:
        earliest = max(0, segment.start_time) + EARLIER_S + CONTEXT_S
        latest = min(segment.stop_time, ecg_seconds) - BLOCK_S - CONTEXT_S
        first = int(np.ceil(earliest / WINDOW_SECONDS)) * WINDOW_SECONDS
        last = int(np.floor(latest / WINDOW_SECONDS)) * WINDOW_SECONDS
        index_by_start.update((start, segment.index) for start in range(first, last + 1, WINDOW_SECONDS))
    starts = np.asarray(sorted(index_by_start), dtype=int)
    support = np.column_stack((starts - EARLIER_S - CONTEXT_S, starts + BLOCK_S + CONTEXT_S))
    overlapping = (support[:, 0, None] < excluded[None, :, 1]) & (support[:, 1, None] > excluded[None, :, 0])
    eligible = ~overlapping.any(axis=1)
    random = np.random.default_rng(seed)
    picks = []
    for _ in range(blocks_per_person):
        available = np.flatnonzero(eligible)
        if not len(available):
            break
        choice = int(random.choice(available))
        start = int(starts[choice])
        picks.append({"input_support_s": support[choice].tolist(), "evaluation_s": [start, start + BLOCK_S],
                      "native_wrist_segment": index_by_start[start]})
        eligible &= (support[:, 0] >= support[choice, 1]) | (support[:, 1] <= support[choice, 0])
    return sorted(picks, key=lambda row: row["evaluation_s"][0])


def per_person_seed(locked_protocol, person):
    return int.from_bytes(sha256(f"blind-wear-extension:{locked_protocol}:{person}".encode()).digest()[:8], "little")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "development-raw", "development-preroll", "stable-supplement", "lock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        digest = sha256(lock_bytes).hexdigest()
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        people = fit | calibration
        if fit & calibration or people & reserved or (len(people), len(reserved)) != (10, 25):
            raise ValueError("Use the original development roster")
        supplement = defaultdict(list)
        with np.load(args.stable_supplement, allow_pickle=False) as source:
            for person, start in zip(source["subjects"], source["start_s"], strict=True):
                supplement[person].append(start)
        if not supplement.keys() <= people:
            raise ValueError("Only original development participants may supply excluded training inputs")
        schedule = {}
        for person in sorted(people):
            excluded = exclusion_intervals(person, (args.development_raw, args.development_preroll), supplement)
            with load_subject(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat",
                              subject_id=person, include_acceleration=False) as subject:
                schedule[person] = choose_monitor_blocks(subject.segments, subject.ecg_duration_seconds, excluded,
                                                         per_person_seed(digest, person))
        count = sum(len(blocks) for blocks in schedule.values())
        receipt = {"source_protocol_sha256": digest, "block_seconds": BLOCK_S, "predecessor_seconds": EARLIER_S,
                   "scheduled_blocks": count, "scheduled_seconds": count * BLOCK_S,
                   "participants_with_planned_blocks": sum(bool(blocks) for blocks in schedule.values())}
        _write_json(args.output, {"plan_receipt": receipt, "private_schedule": schedule})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist monitoring plan failed: {error}\n")


if __name__ == "__main__":
    main()
