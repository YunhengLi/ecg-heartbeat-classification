"""Stage lower-quality non-AF windows in the original development training bins."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from .af_wrist_cohort import CONTEXT, WINDOW, core_label
from .af_wrist_data_preflight import _inventory
from .af_wrist_expansion_stage import BIN_SECONDS, _open_source, _read_pair, _reserved
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_train import load_features
from .af_wrist_training_expansion import _check_alignment
from .afppgecg import ECG_FS, load_subject
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features


_FIELDS = ("subjects", "start_s", "decision_end_s", "stable_label", "sync_method", "ppg_with_context")
_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")
_WIDTH = WINDOW + 2 * CONTEXT


def _digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _native_quality(padded, template):
    pulse = padded[PPG_RATE:-PPG_RATE]
    if not np.isfinite(padded).all() or np.ptp(pulse) <= 0:
        return None
    pad = 2 * PPG_RATE
    forecast = template.predict(np.pad(pulse, (pad, 0), mode="reflect"))[pad:]
    intervals = ecg_features(forecast, PPG_RATE)
    if not np.isfinite(intervals).all() or intervals[_SUPPORT] <= 0:
        return None
    return ppg_features(pulse, PPG_RATE)[1]


def _occupied(person, args, fixed_af):
    name = f"{person}.npz"
    original = np.r_[_reserved(args.development_raw / name, args.development_preroll / name, person), fixed_af]
    with np.load(args.stride_raw / name, allow_pickle=False) as stride:
        if stride["subject_id"].item() != person:
            raise ValueError("The finer native archive must match the candidate participant")
        return (np.r_[original, stride["input_start_s"]] - CONTEXT,
                np.r_[original + WINDOW, stride["decision_end_s"]] + CONTEXT)


def _overlaps(start, occupied):
    first, last = occupied
    return np.any((start - CONTEXT < last) & (first < start + WINDOW + CONTEXT))


def _optimal_nonoverlap(options):
    """Minimize total quality with one row per slot and nonoverlapping padded time.

    Each state holds the best history ending at its final candidate. At a fixed
    end candidate, further feasibility depends only on that candidate's time.
    Compare full chronological start sequences to break exact objective ties.
    """
    states = [((0.0, ()), [])]
    for _, choices in options:
        following = []
        for entry in choices:
            start, quality, _, _ = entry
            compatible = [(key, path) for key, path in states if not path or start >= path[-1][0] + _WIDTH]
            if compatible:
                (total, starts), path = min(compatible, key=lambda item: item[0])
                following.append(((total + quality, (*starts, start)), [*path, entry]))
        if not following:
            raise ValueError("One disjoint candidate per original non-AF bin is required")
        states = following
    chosen = min(states, key=lambda item: item[0])[1]
    for index, ((_, choices), winner) in enumerate(zip(options, chosen, strict=True)):
        compatible = (entry for entry in choices if all(abs(entry[0] - row[0]) >= _WIDTH
                                                        for position, row in enumerate(chosen) if position != index))
        if winner[1] != min(entry[1] for entry in compatible):
            raise ValueError("A selected row has a lower-quality compatible candidate in the same bin")
    return chosen


def _alternatives(subject, reference, candidates, occupied, original_quality, template, counts):
    for start in sorted(candidates):
        counts["same_bin_alternatives"] += 1
        if _overlaps(start, occupied):
            counts["excluded_fixed_overlap"] += 1
            continue
        pair = _read_pair(subject, start)
        if pair is None or core_label(reference, start, start + WINDOW, pair[1]) != (0, True):
            counts["excluded_unverified_stable_pair"] += 1
            continue
        quality = _native_quality(pair[0], template)
        if quality is None:
            counts["excluded_template_support"] += 1
        elif quality < original_quality:
            counts["strictly_lower_eligible_alternatives"] += 1
            yield start, quality, pair[2], pair[0]


def _replace_rows(args, before, planned, entries, template, original_quality):
    subjects, starts, _, labels, methods, pulses = (before[key] for key in _FIELDS)
    after = {key: value.copy() for key, value in before.items()}
    after_quality = original_quality.copy()
    counts = Counter()
    for person in np.unique(subjects):
        rows = np.flatnonzero((subjects == person) & (labels == 0))
        if not len(rows):
            continue
        rows = rows[np.argsort(starts[rows])]
        occupied = _occupied(person, args, starts[(subjects == person) & (labels == 1)])
        candidates_by_bin = {}
        for start in planned[person]["0"]:
            if start % WINDOW:
                raise ValueError("Verified candidate plans must retain the native sampling grid")
            candidates_by_bin.setdefault(start // BIN_SECONDS, []).append(start)
        options = []
        with ExitStack() as stack:
            sources = [_open_source(stack, entries[f"{person}_{signal}.mat"], args.source_dir, args.cache_dir)
                       for signal in ("ECG", "PPG")]
            subject = stack.enter_context(load_subject(*sources, subject_id=person, include_acceleration=False))
            reference = subject.reference_rhythm_intervals()
            for row in rows:
                start = starts[row]
                if _overlaps(start, occupied):
                    raise ValueError("The original non-AF row conflicts with a fixed input interval")
                candidates = candidates_by_bin.get(start // BIN_SECONDS, [])
                if start not in candidates or len(candidates) != len(set(candidates)):
                    raise ValueError("The original non-AF row must come from a unique verified bin candidate")
                fallback = (start, original_quality[row], methods[row], pulses[row])
                alternatives = _alternatives(subject, reference, set(candidates) - {start}, occupied,
                                             original_quality[row], template, counts)
                options.append((row, [fallback, *alternatives]))

        for (row, _), (start, quality, method, padded) in zip(options, _optimal_nonoverlap(options), strict=True):
            if start != starts[row]:
                counts["swapped_non_AF_windows"] += 1
                counts["swaps_with_changed_sync_method"] += int(method != methods[row])
                after["start_s"][row], after["decision_end_s"][row] = start, start + WINDOW
                after["sync_method"][row], after["ppg_with_context"][row] = method, padded
                after_quality[row] = quality
    return after, after_quality, counts


def _quality_summary(values):
    return {"windows": len(values), "mean": float(np.mean(values)),
            "min_q25_median_q75_max": np.quantile(values, [0, .25, .5, .75, 1]).tolist()}


def stage(args):
    existing = json.loads(args.original_receipt.read_bytes())
    original_digest, plan_digest, inventory_digest, lock_digest = map(
        _digest, (args.original_bundle, args.eligibility_plan, args.source_inventory, args.lock),
    )
    if (original_digest != existing["extra_bundle_sha256"] or plan_digest != existing["eligibility_plan_sha256"]
            or inventory_digest != existing["source_inventory_sha256"] or lock_digest != existing["source_protocol_sha256"]):
        raise ValueError("Use the original 107-row bundle, stable candidate plan, and source lock")
    lock, plan, inventory = (json.loads(path.read_bytes()) for path in (args.lock, args.eligibility_plan, args.source_inventory))
    fit, calibration, independent = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    roster = fit | calibration
    features, feature_people, _ = load_features(args.development_features)
    planned = {row["subject_id"]: row["stable_starts_by_class"] for row in plan["development"]}
    if ((len(fit), len(calibration), len(independent)) != (5, 5, 25) or fit & calibration or roster & independent
            or set(feature_people) != roster or features[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or set(planned) != roster or len(plan["development"]) != len(roster)
            or plan["source_protocol_sha256"] != lock_digest):
        raise ValueError("Only the original locked development participants can supply replacements")
    _inventory(inventory, roster | independent)
    entries = {item["key"]: item for item in inventory["files"]}

    with np.load(args.original_bundle, allow_pickle=False) as archive:
        before = {key: archive[key] for key in archive.files}
    subjects, starts, ends, labels, methods, pulses = (before[key] for key in _FIELDS)
    if (before["source_protocol_sha256"].item() != lock_digest
            or tuple(before[key].item() for key in ("ppg_fs", "ecg_reference_fs", "feature_window_seconds", "ppg_flank_seconds"))
               != (PPG_RATE, ECG_FS, WINDOW, CONTEXT)
            or not set(subjects) <= roster or Counter(map(int, labels)) != {0: 66, 1: 41}
            or pulses.shape != (len(labels), int(_WIDTH * PPG_RATE))):
        raise ValueError("Use the pinned rhythm counts and original 32-second PPG shape")
    _check_alignment(subjects, labels, starts, ends, args.development_raw, args.stride_raw)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    original_quality = np.asarray([_native_quality(row, template) for row in pulses], dtype=float)
    if not np.isfinite(original_quality).all():
        raise ValueError("Every original row must have supported template features")
    after, after_quality, counts = _replace_rows(args, before, planned, entries, template, original_quality)

    _check_alignment(subjects, labels, after["start_s"], after["decision_end_s"], args.development_raw, args.stride_raw)
    non_af, af = labels == 0, labels == 1
    if (not np.array_equal(starts // BIN_SECONDS, after["start_s"] // BIN_SECONDS)
            or any(not np.array_equal(before[key][af], after[key][af]) for key in _FIELDS)
            or np.any(after_quality > original_quality)
            or np.count_nonzero(after_quality < original_quality) != counts["swapped_non_AF_windows"]):
        raise ValueError("Replacements must lower only non-AF quality and preserve each bin, AF row and participant")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with args.bundle.open("xb") as stream:
        np.savez_compressed(stream, **after)
    swaps = starts != after["start_s"]
    report = {
        "scope": "Local paired development data ablation with same-participant same-bin non-AF replacement",
        "source_release_doi": inventory["doi"], "source_protocol_sha256": lock_digest,
        "original_bundle_sha256": original_digest, "eligibility_plan_sha256": plan_digest,
        "replaced_bundle_sha256": _digest(args.bundle), "network_requests": 0,
        "development_participants": len(roster), "participants_in_bundle": len(set(subjects)),
        "AF_windows_preserved": int(af.sum()), "non_AF_windows_preserved_by_slot": int(non_af.sum()),
        "non_AF_participants": len(set(subjects[non_af])), "participants_with_swaps": len(set(subjects[swaps])),
        "quality_selection_counts": dict(sorted(counts.items())),
        "both_augmented_bundles_have_template_feature_support": True,
        "selection_method": "One row per original non-AF bin, minimizing summed quality under exact padded interval disjointness",
        "quality": {"AF_unchanged": _quality_summary(original_quality[af]),
                    "original_non_AF": _quality_summary(original_quality[non_af]),
                    "replacement_non_AF": _quality_summary(after_quality[non_af]),
                    "original_non_AF_sync_counts": dict(sorted(Counter(methods[non_af]).items())),
                    "replacement_non_AF_sync_counts": dict(sorted(Counter(after["sync_method"][non_af]).items()))},
    }
    _write_json(args.report, report)
    return len(labels), counts["swapped_non_AF_windows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("source-inventory", "source-dir", "cache-dir", "development-raw", "development-features",
             "development-preroll", "stride-raw", "lock", "ptt-dir", "original-bundle", "original-receipt",
             "eligibility-plan", "bundle", "report")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle.resolve() == args.report.resolve() or args.bundle.exists() or args.report.exists():
        parser.error("Choose new private bundle and aggregate receipt paths")
    try:
        count, swapped = stage(args)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Low-quality wrist AF staging failed: {error}\n")
    print(f"Preserved {count} training slots and replaced {swapped} non-AF windows", flush=True)


if __name__ == "__main__":
    main()
