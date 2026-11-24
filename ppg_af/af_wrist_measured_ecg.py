"""Compare measured ECG and template ECG on the same wrist development clock."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

from .afppgecg import ECG_FS, _read_vector, _sample_bounds
from .af_wrist_clock_study import load_grid, predecessor_decisions, scoring_runs
from .af_wrist_expansion_stage import CacheMiss, _open_source
from .af_wrist_failure_events import audit_events
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_sequence_study import _comparison
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE
from .features import ECG_FEATURE_NAMES, ecg_features


_DOI = "10.5281/zenodo.23921320"
_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def _archives(path, person, *, finer):
    with np.load(path / f"{person}.npz", allow_pickle=False) as raw:
        if str(raw["subject_id"].item()) != person:
            raise ValueError("The measured ECG and staged archive must describe the same participant")
        start = raw["start_s"]
        end = start + (STRIDE_SECONDS if finer else WINDOW_SECONDS)
        if finer and (not np.array_equal(end, raw["decision_end_s"])
                      or not np.array_equal(end - WINDOW_SECONDS, raw["input_start_s"])):
            raise ValueError("The measured ECG and wrist routes must use the same trailing inputs")
        return end, raw["reference_valid"], raw["run_kind"]


def measured_features(grid, source, source_dir, cache_dir, parent, stride):
    shared = _shared_features(grid.table)
    output = np.full_like(grid.table[ROUTE], np.nan)
    people = np.unique(grid.subjects)
    status = Counter()
    original_rows = sum(len(_archives(parent, person, finer=False)[0]) for person in people)
    parent_index, fine_index = 0, original_rows
    for person in people:
        first = _archives(parent, person, finer=False)
        second = _archives(stride, person, finer=True)
        key = f"{person}_ECG.mat"
        with ExitStack() as stack:
            stream = _open_source(stack, source[key], source_dir, cache_dir)
            record = stack.enter_context(h5py.File(stream, "r"))
            voltage = record["ECG"]
            for offset, (end, reference, kind) in ((parent_index, first), (fine_index, second)):
                rows = slice(offset, offset + len(end))
                if (not np.all(grid.subjects[rows] == person)
                        or not np.array_equal(grid.table["reference_valid"][rows], reference)
                        or not np.array_equal(grid.table["run_kind"][rows], kind)):
                    raise ValueError("ECG feature rows do not align with the shared wrist clock")
                for relative in np.flatnonzero(shared[rows]):
                    lo, hi = _sample_bounds(end[relative] - WINDOW_SECONDS, end[relative], 0.0, ECG_FS, voltage.size)
                    if hi - lo != WINDOW_SECONDS * ECG_FS:
                        status["outside_ecg_record"] += 1
                        continue
                    try:
                        waveform = _read_vector(voltage, slice(lo, hi))
                    except CacheMiss:
                        status["uncached_measured_ecg"] += 1
                        continue
                    if not np.isfinite(waveform).all() or np.ptp(waveform) == 0:
                        status["unusable_measured_ecg"] += 1
                        continue
                    values = ecg_features(waveform, ECG_FS)
                    if not np.isfinite(values).all() or values[_SUPPORT] <= 0:
                        status["no_supported_measured_intervals"] += 1
                        continue
                    output[offset + relative] = values
                    status["scoreable_measured_ecg"] += 1
        parent_index += len(first[0])
        fine_index += len(second[0])
    if parent_index != original_rows or fine_index != len(grid.subjects) or sum(status.values()) != int(shared.sum()):
        raise ValueError("Measured ECG support did not reconcile to the original decision rows")
    return output, dict(sorted(status.items()))


def evaluate_measured_ecg(grid, measured, status, lock):
    original = grid.table[ROUTE]
    shared = _shared_features(grid.table)
    common = shared & np.isfinite(measured).all(axis=1)
    template = np.where(common[:, None], original, np.nan)
    comparison_grid = replace(grid, table={**grid.table, ROUTE: template})
    baseline = fit_template(comparison_grid, template, lock, DEFAULT_C)
    candidate = fit_template(comparison_grid, measured, lock, DEFAULT_C)
    verify_fit_support(baseline, candidate)
    scored_template = _replay(comparison_grid, baseline)
    scored_ecg = _replay(comparison_grid, candidate)
    states = {"all_evaluation": np.concatenate([block.rows for block in grid.timelines]),
              "preceding_history": np.concatenate(list(grid.history_rows.values())),
              "original_fitting": np.flatnonzero((grid.table["stable_label"] >= 0)
                                                 & (grid.table["run_kind"] != "enriched_episode"))}
    diagnostics = {}
    for name, fitted, replay in (("template", baseline, scored_template), ("measured_ecg", candidate, scored_ecg)):
        scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
        stable = states["original_fitting"]
        stable = stable[np.isfinite(scores[stable])]
        labels = grid.table["stable_label"][stable]
        fixed = np.where(np.isfinite(cutoffs), 0.5, np.nan)
        decision = states["all_evaluation"]
        present = decision[np.isfinite(scores[decision]) & np.isfinite(cutoffs[decision])]
        known = np.isfinite(scores) & np.isfinite(cutoffs)
        binary = np.where(known, scores >= cutoffs, np.nan)
        predecessor = predecessor_decisions(comparison_grid, np.where(known, binary, -1).astype(np.int8))
        _, episode_runs = scoring_runs(comparison_grid, binary, set(grid.subjects))
        event_audit = audit_events((run.decisions(0.5) for run in episode_runs), replay[1], confirmation_windows=1,
                                   prior_decisions=predecessor, feature_window_s=WINDOW_SECONDS)
        cases = event_audit["complete_episodes"]
        if len(cases) != replay[0]["complete_episodes"] or sum(case["detected"] for case in cases) != replay[0]["episodes_detected"]:
            raise ValueError("The measured ECG error trace must reconcile with event scoring")
        diagnostics[name] = {
            "pooled_held_out_stable_window_ROC_AUC": float(roc_auc_score(labels, scores[stable])),
            "stable_AF_windows": int(np.count_nonzero(labels == 1)),
            "stable_non_AF_windows": int(np.count_nonzero(labels == 0)),
            "positive_evaluation_decisions_at_fold_cutoff": int(np.sum(scores[present] >= cutoffs[present])),
            "positive_evaluation_decisions_at_fixed_cutoff": int(np.sum(scores[present] >= fixed[present])),
            "fixed_0_5_replay": _held_out_replay(comparison_grid, scores, fixed, _IMMEDIATE)[0],
            "missed_episode_mechanisms": dict(sorted(Counter(case["primary"] for case in cases if not case["detected"]).items())),
            "already_active_misses_with_positive_in_episode": sum(
                case["primary"] == "already_active_at_onset" and case["positive_count"] > 0 for case in cases
            ),
        }
    return {
        "scope": "Measured ECG waveform reference control on a common wrist development grid",
        "outer_folds": baseline["n_splits"], "development_participants": len(np.unique(grid.subjects)),
        "native_ecg_hz": int(ECG_FS), "feature_window_seconds": WINDOW_SECONDS,
        "decision_stride_seconds": grid.seconds, "logistic_C": DEFAULT_C,
        "ecg_status_on_original_shared_rows": status,
        "feature_support": {name: {"original_shared_rows": int(shared[rows].sum()),
                                  "common_ecg_and_template_rows": int(common[rows].sum())} for name, rows in states.items()},
        "template_on_common_footprint": {**scored_template[0], "inner_calibration": _calibration_status(baseline)},
        "measured_ecg_on_common_footprint": {**_comparison(scored_template, scored_ecg, lock),
                                             "inner_calibration": _calibration_status(candidate)},
        "discrimination_and_cutoff_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args_names = ("source-inventory", "source-dir", "cache-dir", "development-raw", "development-features",
                  "stride-raw", "stride-features", "lock", "clock-results", "output")
    for name in args_names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    try:
        lock_raw = args.lock.read_bytes()
        lock = json.loads(lock_raw)
        clock = json.loads(args.clock_results.read_bytes())
        source = json.loads(args.source_inventory.read_bytes())
        table, people, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in
                                      ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(people) != (fit | calibration)
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != sha256(lock_raw).hexdigest() or source["doi"] != _DOI):
            raise ValueError("Use the pinned v6 inventory, original feature version and development cohort")
        grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        measured, status = measured_features(grid, {entry["key"]: entry for entry in source["files"]},
                                             args.source_dir, args.cache_dir, args.development_raw, args.stride_raw)
        result = evaluate_measured_ecg(grid, measured, status, lock)
        _write_json(args.output, result)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Measured ECG control failed: {error}\n")
    baseline, candidate = result["template_on_common_footprint"], result["measured_ecg_on_common_footprint"]["summary"]
    print(f"Measured ECG {candidate['episodes_detected']}/{candidate['complete_episodes']}, "
          f"{candidate['wrist_false_alerts']} wrist false starts; matched template {baseline['episodes_detected']}/"
          f"{baseline['complete_episodes']}, {baseline['wrist_false_alerts']} false starts")


if __name__ == "__main__":
    main()
