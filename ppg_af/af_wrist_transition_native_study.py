"""Test one new native ECG-labelled onset donor on the original wrist folds."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_cohort import CONTEXT, in_wrist_segment
from .af_wrist_cv import train_oof
from .af_wrist_data_preflight import _inventory
from .af_wrist_expansion_stage import _disjoint, _open_source, _read_pair
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import build_quality_matrices, quality_interaction_features, _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template
from .af_wrist_train import load_features
from .af_wrist_transition_examples import EARLY_AF_SECONDS, PRE_ONSET_SECONDS, select_transition_examples
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE
from .afppgecg import ECG_FS, load_subject
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features


SEED = 2026
FAMILIES = ("baseline", "quality_interactions")


def _within_rhythm(reference, start, stop, label):
    index = np.searchsorted(reference[:, 1], start, side="right")
    return bool(index < len(reference) and reference[index, 0] <= start
                and reference[index, 1] >= stop and reference[index, 2] == label)


def _native_clock(subject, starts, ends, onset, offset, methods):
    pre_start, early_start = starts
    pre_end, early_end = ends
    reference = subject.reference_rhythm_intervals()
    covered = np.maximum(0, np.minimum(reference[:, 1], early_end) - np.maximum(reference[:, 0], early_start)).sum()
    segment = in_wrist_segment(subject, pre_start - CONTEXT, early_end + CONTEXT)
    if (np.any(starts % STRIDE_SECONDS) or not _within_rhythm(reference, onset, offset, 1)
            or not onset - PRE_ONSET_SECONDS <= pre_end <= onset
            or not _within_rhythm(reference, pre_start, pre_end, 0)
            or not onset + STRIDE_SECONDS <= early_end <= min(onset + EARLY_AF_SECONDS, offset)
            or not _within_rhythm(reference, early_end - STRIDE_SECONDS, early_end, 1)
            or abs(covered - WINDOW_SECONDS) >= 1e-7
            or segment is None or not np.all(methods == segment.sync_method)):
        raise ValueError("The native pair must retain its ECG-verified onset labels and clock")


def _protected_starts(args, person, lock_hash, protected_hash):
    name = f"{person}.npz"
    if sha256(args.protected_supplement.read_bytes()).hexdigest() != protected_hash:
        raise ValueError("The protected stable window bundle must match the staging receipt")
    with (np.load(args.development_raw / name, allow_pickle=False) as raw,
          np.load(args.stride_raw / name, allow_pickle=False) as stride,
          np.load(args.development_preroll / name, allow_pickle=False) as preroll,
          np.load(args.protected_supplement, allow_pickle=False) as protected):
        if (any(item["subject_id"].item() != person for item in (raw, stride, preroll))
                or len(protected["subjects"]) != 109 or protected["source_protocol_sha256"].item() != lock_hash
                or not np.array_equal(protected["decision_end_s"], protected["start_s"] + WINDOW_SECONDS)
                or not np.array_equal(stride["decision_end_s"], stride["input_start_s"] + WINDOW_SECONDS)):
            raise ValueError("The protected source clocks must align with the locked development archives")
        return np.unique(np.r_[raw["start_s"], stride["input_start_s"], preroll["start_s"],
                               protected["start_s"][protected["subjects"] == person]])


def _new_features(args, receipt, grid, labels, lock_hash, entries):
    digest = sha256(args.pair.read_bytes()).hexdigest()
    if digest != receipt["bundle_sha256"] or receipt["source_protocol_sha256"] != lock_hash:
        raise ValueError("The native transition pair must match the verified staging receipt")
    with np.load(args.pair, allow_pickle=False) as source:
        people, phases, extra_labels, padded = (source[key] for key in (
            "subjects", "phase", "transition_label", "ppg_with_context"))
        starts, ends, ecg, methods = (source[key] for key in (
            "start_s", "decision_end_s", "ecg_reference_waveform", "sync_method"))
        onset, offset = (source[key].item() for key in ("episode_onset_s", "episode_offset_s"))
        if (source["source_protocol_sha256"].item() != lock_hash
                or source["protected_109_sha256"].item() != receipt["supplement_sha256"]
                or tuple(phases) != ("pre_AF_non_AF", "early_AF") or tuple(extra_labels) != (0, 1)
                or source["ppg_fs"].item() != PPG_RATE or source["feature_window_seconds"].item() != WINDOW_SECONDS
                or source["ecg_reference_fs"].item() != ECG_FS
                or source["ppg_flank_seconds"].item() != 1 or source["decision_stride_seconds"].item() != STRIDE_SECONDS
                or starts.shape != (2,) or not np.isfinite(starts).all() or starts[0] >= starts[1]
                or not np.array_equal(ends, starts + WINDOW_SECONDS)):
            raise ValueError("Use the staged causal onset windows and pinned ECG labels")
    if (people.shape != (2,) or padded.shape != (2, (WINDOW_SECONDS + 2) * PPG_RATE)
            or ecg.shape != (2, int(ECG_FS) * WINDOW_SECONDS) or methods.shape != (2,)
            or len(set(people)) != 1 or not set(people) <= set(grid.subjects)
            or set(people) & set(grid.subjects[labels >= 0])):
        raise ValueError("The native pair must add exactly one new development transition donor")
    person = str(people[0])
    if not _disjoint(starts, _protected_starts(args, person, lock_hash, receipt["supplement_sha256"])).all():
        raise ValueError("Native pair time must be disjoint from all original and protected 109 waveform inputs")
    with ExitStack() as stack:
        inputs = [_open_source(stack, entries[f"{person}_{signal}.mat"], args.source_dir, args.cache_dir)
                  for signal in ("ECG", "PPG")]
        subject = stack.enter_context(load_subject(*inputs, subject_id=person, include_acceleration=False))
        _native_clock(subject, starts, ends, onset, offset, methods)
        for n, start in enumerate(starts):
            pair = _read_pair(subject, int(start))
            if (pair is None or not np.array_equal(pair[0], padded[n])
                    or not np.array_equal(pair[1], ecg[n]) or pair[2] != methods[n]):
                raise ValueError("Native ECG and PPG arrays must reproduce the two staged inputs exactly")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    pad = 2 * PPG_RATE
    pulse = padded[:, PPG_RATE:-PPG_RATE]
    if not np.isfinite(pulse).all() or np.any(np.ptp(pulse, axis=1) <= 0):
        raise ValueError("Require finite, non-flat native PPG input cores")
    extracted = np.stack([ecg_features(template.predict(np.pad(row, (pad, 0), mode="reflect"))[pad:], PPG_RATE)
                          for row in pulse])
    if not np.isfinite(extracted).all() or np.any(extracted[:, ECG_FEATURE_NAMES.index("valid_interval_fraction")] <= 0):
        raise ValueError("Both native pair windows must support the original template features")
    quality = np.asarray([ppg_features(row, PPG_RATE)[1] for row in pulse])
    return people, extra_labels, {"baseline": extracted, "quality_interactions": quality_interaction_features(extracted, quality)}, digest


def _fit_both(grid, matrix, labels, extra_people, extra_labels, extra_matrix, lock):
    n = len(grid.subjects)
    shared = _shared_features(grid.table)
    original_fit = shared & (grid.table["run_kind"] != "enriched_episode")
    if not np.isfinite(matrix[shared]).all() or not np.isfinite(extra_matrix).all():
        raise ValueError("Both old and new rows must retain their expected feature support")

    def calibrate(scores, rows):
        original_rows = rows < n
        probabilities = np.full(n, np.nan)
        probabilities[rows[original_rows]] = scores[original_rows]
        people = set(grid.subjects[rows[original_rows]])
        evaluate, pool = scoring_policy(grid, probabilities, people, _IMMEDIATE)
        return select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                       max_thresholds=max(3, len(pool) + 1))

    fitted = train_oof({ROUTE: np.vstack((np.where(shared[:, None], matrix, np.nan), extra_matrix))},
                       np.r_[grid.table["stable_label"], -1, -1], np.r_[grid.subjects, extra_people],
                       n_splits=5, inner_splits=3, seed=SEED, regularization=DEFAULT_C, calibration_selector=calibrate,
                       train_eligible=np.r_[original_fit, False, False], supplemental_fit_labels=np.r_[labels, extra_labels])
    assignment = fitted["fold_assignment"]
    route = fitted["routes"][ROUTE]
    for key in ("scores", "decisions"):
        route[key] = route[key][:n]
    for fold in route["folds"]:
        held_out = assignment[:n] == fold["fold"]
        fold["held_out_finite_scores"] = int(np.isfinite(route["scores"][held_out]).sum())
        if set(extra_people[assignment[n:] != fold["fold"]]) & set(grid.subjects[held_out]):
            raise ValueError("The native pair entered its own outer training fold")
    fitted["fold_assignment"] = assignment[:n]
    return fitted


def evaluate(grid, matrix, labels, new, lock):
    people, new_labels, extra, digest = new
    variants = {
        "stable_baseline": fit_template(grid, matrix, lock, DEFAULT_C, seed=SEED),
        "original_26": fit_template(grid, matrix, lock, DEFAULT_C, seed=SEED, supplemental_fit_labels=labels),
        "original_26_plus_new_pair": _fit_both(grid, matrix, labels, people, new_labels, extra, lock),
    }
    original = variants["stable_baseline"]
    previous = variants["original_26"]
    candidate = variants["original_26_plus_new_pair"]
    original_route = original["routes"][ROUTE]
    for fit in (previous, candidate):
        route = fit["routes"][ROUTE]
        if (not np.array_equal(original["fold_assignment"], fit["fold_assignment"])
                or not np.array_equal(np.isfinite(original_route["scores"]), np.isfinite(route["scores"]))
                or not np.array_equal(original_route["decisions"] >= 0, route["decisions"] >= 0)):
            raise ValueError("Every model must retain original outer membership, scoring and calibration support")
    old_folds, new_folds = (fit["routes"][ROUTE]["folds"] for fit in (previous, candidate))
    training_pairs = []
    for original_fold, old_fold, new_fold in zip(original_route["folds"], old_folds, new_folds, strict=True):
        held = original["fold_assignment"] == old_fold["fold"]
        extra = int(not set(people) & set(grid.subjects[held]))
        if (original_fold["inner_splits"] != old_fold["inner_splits"] or old_fold["inner_splits"] != new_fold["inner_splits"]
                or any(new_fold[f"fit_{rhythm}_windows"] - old_fold[f"fit_{rhythm}_windows"] != extra
                       for rhythm in ("af", "non_af"))):
            raise ValueError("The new pair affected the wrong participant-fold fitting rows")
        training_pairs.append(extra)
    if sorted(training_pairs) != [0, 1, 1, 1, 1]:
        raise ValueError("The native pair must be excluded in its donor's held-out fold")
    replay = {name: _replay(grid, model) for name, model in variants.items()}
    return {
        "extra_bundle_sha256": digest, "outer_folds": original["n_splits"],
        "new_pairs_added_by_outer_fit": training_pairs,
        "variants": {name: {"summary": score[0], "calibration_status": _calibration_status(variants[name])}
                     for name, score in replay.items()},
        "original_26_vs_stable_baseline": _compare_with_pre_onset(replay["stable_baseline"], replay["original_26"], lock),
        "all_28_vs_stable_baseline": _compare_with_pre_onset(replay["stable_baseline"], replay["original_26_plus_new_pair"], lock),
        "new_pair_vs_original_26": _compare_with_pre_onset(replay["original_26"], replay["original_26_plus_new_pair"], lock),
    }


def _run(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    source_bytes = args.source_manifest.read_bytes()
    source = json.loads(source_bytes)
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    table, members, timelines = load_features(args.development_features)
    fit, calibration, test = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & test or set(members) != fit | calibration
            or (len(fit), len(calibration), len(test)) != (5, 5, 25)
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or receipt["source_inventory_sha256"] != sha256(source_bytes).hexdigest()):
        raise ValueError("Use only the locked development participants")
    _inventory(source, fit | calibration | test)
    entries = {item["key"]: item for item in source["files"]}
    grid = load_grid(table, members, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    labels, support = select_transition_examples(grid, timelines, args.development_raw)
    if (np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)) != (10, 16):
        raise ValueError("Use the exact 26 prior onset examples")
    people, extra_labels, extras, digest = _new_features(args, receipt, grid, labels, sha256(lock_bytes).hexdigest(), entries)
    matrices = build_quality_matrices(grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features)
    reports = {}
    for family in FAMILIES:
        result = evaluate(grid, matrices[family], labels, (people, extra_labels, extras[family], digest), lock)
        expected = {"stable_baseline": (12, 6) if family == "baseline" else (14, 4)}
        if family == "baseline":
            expected["original_26"] = (11, 6)
        for name, counts in expected.items():
            summary = result["variants"][name]["summary"]
            if (summary["episodes_detected"], summary["wrist_false_alerts"]) != counts:
                raise ValueError("The previously reported template model results did not reproduce")
        reports[family] = {"scope": "One new development transition donor on fixed original subject folds",
                           "family": family, "seed": SEED, "logistic_C": DEFAULT_C,
                           "old_transition_support": support, "new_transition_donors": 1,
                           "stable_107_or_109_training_rows_used": 0, "independent_predictions": 0, **result}
        for name, entry in result["variants"].items():
            summary = entry["summary"]
            print(f"{family} {name}: {summary['episodes_detected']}/{summary['complete_episodes']}, "
                  f"{summary['wrist_false_alerts']} false starts", flush=True)
    return {"scope": "Fixed seed-2026 test of one additional development onset donor",
            "source_release_doi": source["doi"], "source_inventory_sha256": sha256(source_bytes).hexdigest(),
            "transition_staging_receipt_sha256": sha256(receipt_bytes).hexdigest(),
            "native_pair_sha256": digest, "protected_109_sha256": receipt["supplement_sha256"],
            "local_source_arrays_and_ecg_timing_reverified": True, "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = ("source-manifest", "source-dir", "cache-dir", "development-raw", "development-features",
                "development-preroll", "stride-raw", "stride-features", "lock", "pair", "receipt",
                "protected-supplement", "ptt-dir", "output")
    for argument in required:
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new aggregate output")
    try:
        _write_json(args.output, _run(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Native transition study failed: {error}\n")


if __name__ == "__main__":
    main()
