"""Trace historical carried-alert origins into sealed and cached native preroll."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np

from .af_wrist_confirmation_miss_audit import CONFIRMATION_RESULTS_SHA, _audit, _outcome
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, ROUTES, WINDOW_SECONDS, extract_features
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_independent_miss_audit import _timeline
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_template_study import ROUTE
from .af_wrist_vote_study import _IMMEDIATE, _held_out_replay
from .afppgecg import ECG_FS, PPG_FS, _read_vector, _sample_bounds, load_subject


NATIVE_LOOKBACK_SECONDS = 180
BLOCK_BYTES = 1048576
RECORD_ID = 23921320
CORE = slice(PPG_RATE, -PPG_RATE)


class CacheMiss(OSError):
    """The native data are not present in the local byte cache."""


class CachedMat(io.RawIOBase):
    """Expose size-checked local range files as a seekable HDF5 input."""

    def __init__(self, blocks, size):
        self.blocks = blocks
        self.size = size
        self.position = 0
        self.block = lru_cache(maxsize=32)(self._read_block)

    def _read_block(self, number):
        path = self.blocks / f"{BLOCK_BYTES}-{number}.bin"
        expected = min(BLOCK_BYTES, self.size - number * BLOCK_BYTES)
        if not path.is_file() or path.stat().st_size != expected:
            raise CacheMiss("A required native byte range is absent or incomplete")
        data = path.read_bytes()
        if len(data) != expected:
            raise CacheMiss("A native byte range changed while being read")
        return data

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        position = origin + offset
        if position < 0:
            raise OSError("Cannot seek before the start of a MAT file")
        self.position = position
        return position

    def readinto(self, buffer):
        maximum = min(self.size, self.position + len(buffer))
        count = 0
        while self.position < maximum:
            number, offset = divmod(self.position, BLOCK_BYTES)
            width = min(maximum - self.position, BLOCK_BYTES - offset)
            buffer[count:count + width] = self.block(number)[offset:offset + width]
            self.position += width
            count += width
        return count


def _origin(decisions, endpoints):
    """Return the start of a positive tail only when its predecessor is a known reset."""
    last_reset = len(decisions) - 1
    while last_reset >= 0 and decisions[last_reset] == 1:
        last_reset -= 1
    if 0 <= last_reset < len(decisions) - 1 and decisions[last_reset] in (-1, 0):
        return float(endpoints[last_reset + 1]), decisions[last_reset]
    return None


def _history_states(scores, quality, known, cutoff, confirmed):
    previous = np.arange(len(scores)) - 1
    filtered = confirm_scores(scores, previous, quality < QUALITY_CUTOFF) if confirmed else scores
    result = []
    for row, value in enumerate(filtered):
        if not known[row]:
            result.append(None)
        elif confirmed and quality[row] < QUALITY_CUTOFF and np.isfinite(scores[row]) and (
            row == 0 or not known[row - 1]
        ):
            # Unknown earlier raw score can affect a positive, but cannot raise a negative.
            result.append(0 if scores[row] < cutoff else None)
        else:
            result.append(int(value >= cutoff) if np.isfinite(value) else -1)
    return result


def _cases_by_block(grid, scores, cutoffs, evidence, expected):
    cases, _, report = _audit(grid, scores, cutoffs, evidence, expected)
    _, runs, _ = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.subject_id].append(run)
    traces = {person: _timeline(items)[-1] for person, items in grouped.items()}
    by_end = {(run.subject_id, run.start_s + grid.seconds * len(run.decisions)): run for run in runs}
    origins_by_subject = defaultdict(Counter)
    for case in cases:
        if _outcome(case) != "carried_alert_with_positive_in_episode":
            continue
        person = case["subject_id"]
        trace = traces[person]
        start, origin = next((start, trace[start, row][1]) for start, row in case["decision_refs"]
                             if (start, row) in trace)
        if origin is not None:
            continue
        previous = by_end.get((person, start))
        while previous is not None and previous.decisions[-1] == 1:
            start = previous.start_s
            previous = by_end.get((person, start))
        origins_by_subject[person][start] += 1
    unknown = report["miss_attribution"]["carried_alerts"]["observed_origins"][
        "already_active_at_start_of_replayed_segment"
    ]
    if sum(count for blocks in origins_by_subject.values() for count in blocks.values()) != unknown:
        raise ValueError("The prefix cases do not reconcile with the frozen carried-alert audit")
    return origins_by_subject, report["frozen_replay"]


def _native_history(subject, start, lookback):
    left = start - lookback
    segment = next((item for item in subject.segments if item.start_time <= left and item.stop_time >= start), None)
    if segment is None or left < 0:
        return None, "wear_gap_or_record_boundary"
    green = subject._ppg_data[segment.index]
    lo, hi = _sample_bounds(left, start, segment.start_time, PPG_FS, green.size)
    try:
        ppg = _read_vector(green, slice(lo, hi))
        ecg = _read_vector(subject._voltage, slice(round(left * ECG_FS), round(start * ECG_FS)))
    except CacheMiss:
        return None, "missing_native_bytes"
    if (len(ppg) != lookback * PPG_FS or len(ecg) != lookback * ECG_FS
            or not np.isfinite(ppg).all() or not np.isfinite(ecg).all() or np.ptp(ecg) == 0):
        return None, "unusable_native_pair"
    reference = subject.reference_rhythm_intervals(left, start)
    if not np.isclose(np.sum(reference[:, 1] - reference[:, 0]), start - left, rtol=0, atol=1e-6):
        return None, "incomplete_ecg_annotation"
    return np.asarray(ppg, dtype=float), "complete_native_pair"


def _record_origin(summary, count, origin, reference, subject, stride, source):
    endpoint, reset = origin
    candidate = reference[(reference[:, 0] < endpoint) & (endpoint <= reference[:, 1])]
    phase = "unobserved" if len(candidate) != 1 else ("AF" if candidate[0, 2] == 1 else "non_AF")
    summary[source]["origins"] += count
    summary[source]["after_negative" if reset == 0 else "after_unavailable"] += count
    summary[source][f"cached_ECG_rhythm_{phase}"] += count
    try:
        ecg = _read_vector(subject._voltage, slice(round((endpoint - stride) * ECG_FS), round(endpoint * ECG_FS)))
    except CacheMiss:
        return
    observed = subject.reference_rhythm_intervals(endpoint - stride, endpoint)
    if (len(ecg) == stride * ECG_FS and np.isfinite(ecg).all() and np.ptp(ecg) > 0
            and np.isclose(np.sum(observed[:, 1] - observed[:, 0]), stride, rtol=0, atol=1e-6)):
        summary[source]["native_ECG_and_annotations_cover_preceding_decision_cell"] += count


def evaluate(args):
    import torch

    from .af_wrist_smoke import fit_ptt_calibration
    from .rddm import load_author_model

    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    published = _read_pinned(args.confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)[
        "historical_episodes_and_monitoring"
    ]
    grid, model, available, _, quality, _ = historical_inputs(args, protocol, model_lock)
    optical_quality = quality["quality"][:, -1]
    raw_dir = args.prior_inputs / "stride-raw"
    evidence = load_input_evidence(grid, raw_dir, args.prior_inputs / "stride-features")
    raw_scores = _predict(grid.table[ROUTE], available, model["model"])
    cutoff = model["cutoff"]
    cutoffs = np.full(len(raw_scores), cutoff)
    predecessor, active = preceding_rows(grid)
    filtered = confirm_scores(raw_scores, predecessor, active & (optical_quality < QUALITY_CUTOFF))
    original, original_replay = _cases_by_block(grid, raw_scores, cutoffs, evidence, published["baseline"])
    candidate, candidate_replay = _cases_by_block(
        grid, filtered, cutoffs, evidence, published["conditional_confirmation"]["summary"],
    )

    release = json.loads(args.native_metadata.read_bytes())
    if release["id"] != RECORD_ID:
        raise ValueError("Expected the AFPPGECG v6 native manifest")
    files = {item["key"]: item["size"] for item in release["files"]}
    torch.set_num_threads(args.threads)
    delay, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    models = load_author_model(args.upstream, args.weights, args.device)

    arms = {"baseline": original, "quality_confirmation": candidate}
    reports = {name: defaultdict(Counter) for name in arms}
    support = Counter()
    attempted_support = defaultdict(Counter)
    for person in sorted(original.keys() | candidate.keys()):
        per_arm = {name: cases.get(person, {}) for name, cases in arms.items()}
        starts = sorted(set().union(*(cases.keys() for cases in per_arm.values())))
        with ExitStack() as stack, np.load(raw_dir / f"{person}.npz", allow_pickle=False) as raw:
            paths = [f"{person}_{signal}.mat" for signal in ("ECG", "PPG")]
            streams = [stack.enter_context(CachedMat(args.native_cache / path, files[path])) for path in paths]
            subject = stack.enter_context(load_subject(*streams, subject_id=person, include_acceleration=False))
            reference = subject.reference_rhythm_intervals()
            for start in starts:
                history = grid.history_rows[person, start]
                known = ~grid.unknown_source[history]
                rows = np.flatnonzero((raw["run_start_s"] == start) & ~raw["evaluation"])
                endpoints = raw["decision_end_s"][rows]
                expected = np.arange(start - 60, start + grid.seconds, grid.seconds)
                if not np.array_equal(endpoints, expected) or raw["reference_valid"][rows].any():
                    raise ValueError("Expected sealed state-only history on the published clock")
                scores, q = raw_scores[history], optical_quality[history]
                unresolved = {}
                for name, cases in per_arm.items():
                    if start not in cases:
                        continue
                    count = cases[start]
                    states = _history_states(scores, q, known, cutoff, name == "quality_confirmation")
                    origin = _origin(states, endpoints)
                    if origin is None:
                        unresolved[name] = count
                    else:
                        _record_origin(reports[name], count, origin, reference, subject, grid.seconds, "sealed_preroll")
                if not unresolved:
                    continue
                for lookback in dict.fromkeys((args.native_lookback_seconds, NATIVE_LOOKBACK_SECONDS)):
                    ppg, support_status = _native_history(subject, start, lookback)
                    attempted_support[lookback][support_status] += 1
                    if ppg is not None:
                        break
                support[support_status] += 1
                if ppg is None:
                    for name, count in unresolved.items():
                        reports[name]["unresolved"][support_status] += count
                    continue
                left = start - lookback
                windows = np.lib.stride_tricks.sliding_window_view(ppg, WINDOW_SECONDS * PPG_RATE)[::grid.seconds * PPG_RATE]
                native_ends = np.arange(left + WINDOW_SECONDS, start + grid.seconds, grid.seconds)
                overlap = np.searchsorted(native_ends, endpoints[0])
                if not np.array_equal(windows[overlap:][known], raw["ppg"][rows, CORE][known], equal_nan=True):
                    for name, count in unresolved.items():
                        reports[name]["unresolved"]["native_overlap_does_not_match_sealed_ppg"] += count
                    continue
                inputs = np.pad(windows[:overlap], ((0, 0), (PPG_RATE, PPG_RATE)), constant_values=np.nan)
                matrices, _, new_quality = extract_features(
                    inputs, template, delay, models, device=args.device, batch_size=args.batch_size,
                )
                common = np.logical_and.reduce([np.isfinite(matrices[route]).all(axis=1) for route in ROUTES])
                new_scores = _predict(matrices[ROUTE], common, model["model"])
                extended_scores, extended_quality = np.r_[new_scores, scores], np.r_[new_quality, q]
                extended_known = np.r_[np.ones(len(new_scores), dtype=bool), known]
                for name, count in unresolved.items():
                    states = _history_states(extended_scores, extended_quality, extended_known, cutoff,
                                             name == "quality_confirmation")
                    origin = _origin(states, native_ends)
                    if origin is None:
                        reports[name]["unresolved"]["no_definite_start_in_bounded_native_history"] += count
                    else:
                        _record_origin(reports[name], count, origin, reference, subject, grid.seconds, "native_extension")

    for name, subjects in arms.items():
        report = reports[name]
        cases = sum(count for blocks in subjects.values() for count in blocks.values())
        resolved = report["sealed_preroll"]["origins"] + report["native_extension"]["origins"]
        if resolved + sum(report["unresolved"].values()) != cases:
            raise ValueError("Every previously unknown carried origin needs a disposition")
        report["counts"] = {"previously_unknown": cases, "resolved": resolved, "still_unknown": cases - resolved}
    result = {
        "scope": "Retrospective origin audit on the previously scored AFPPGECG v6 roster",
        "decision_stride_seconds": grid.seconds,
        "sealed_ppg_lookback_seconds": 90,
        "maximum_native_lookback_seconds": args.native_lookback_seconds,
        "native_extension_support_by_block": support,
        "scoring_reference_exposure_added_seconds": 0,
        "cached_rhythm_definition": "ECG annotation immediately before the preroll alert start",
        "native_ECG_support_definition": "Finite nonflat native ECG and contiguous annotations over the preceding decision cell, for attribution only",
        "frozen_replay": {"baseline": original_replay, "quality_confirmation": candidate_replay},
        "origin_attribution": reports,
    }
    if args.native_lookback_seconds != NATIVE_LOOKBACK_SECONDS:
        result["native_extension_support_by_attempt_seconds"] = dict(attempted_support)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report",
                 "historical-results", "confirmation-results", "native-metadata", "native-cache",
                 "ptt-dir", "upstream", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--native-lookback-seconds", type=int, choices=(180, 300, 600), default=NATIVE_LOOKBACK_SECONDS,
                        help="Maximum native preroll; use 180 seconds when the longer local pair is incomplete")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
