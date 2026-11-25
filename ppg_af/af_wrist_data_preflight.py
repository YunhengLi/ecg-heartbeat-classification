"""Verify locked wrist AF data and evaluation denominators without model inference."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_runs
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, ROUTES, WINDOW_SECONDS, read_feature_version
from .af_wrist_holdout import MONITOR_KIND, _shared_features, _subject_ids, _write_json
from .af_wrist_operating import score_decisions
from .af_wrist_stride import STRIDE, stage_subject
from .af_wrist_template_coverage import _baseline, _subject_support, _sum_support
from .af_wrist_template_independent import _archives, _match_coverage, load_independent_inputs
from .af_wrist_train import load_features


_SOURCE_DOI = "10.5281/zenodo.23921320"
_SCOPES = ("parent-raw", "parent-features", "preroll", "stride-raw", "stride-features")
_INPUT_STATES = ("available", "unavailable", "unknown")
_PARENT_FIELDS = ("subject_id", "start_s", "run_id", "run_kind", "reference_valid", "stable_label", "af_intervals_s")
_STRIDE_FIELDS = (*_PARENT_FIELDS, "decision_end_s", "input_start_s", "run_start_s", "evaluation", "ppg_available",
                  "input_status", "ppg_fs", "feature_window_seconds", "decision_stride_seconds")


def _inventory(manifest, assigned):
    if manifest["doi"] != _SOURCE_DOI:
        raise ValueError("Use the pinned AFPPGECG v6 source inventory")
    counts = {name: set() for name in ("ECG", "PPG")}
    for item in manifest["files"]:
        match = re.fullmatch(r"(\d+)_(ECG|PPG)\.mat", item["key"])
        if not match:
            continue
        participant, signal = match.groups()
        if (participant in counts[signal] or item["size"] <= 0
                or not re.fullmatch(r"md5:[0-9a-f]{32}", item["checksum"])):
            raise ValueError("The source inventory contains an invalid waveform entry")
        counts[signal].add(participant)
    paired = counts["ECG"] & counts["PPG"]
    if paired != assigned or counts["PPG"] != paired or len(counts["ECG"]) != 39:
        raise ValueError("Every paired v6 participant must belong to exactly one locked cohort")
    return {"doi": _SOURCE_DOI, "ecg_recordings": len(counts["ECG"]), "ppg_recordings": len(counts["PPG"]),
            "paired_recordings": len(paired), "ecg_only_recordings": len(counts["ECG"] - paired),
            "paired_recordings_assigned_to_locked_cohorts": len(assigned)}


def _cohort_archives(directories, expected):
    archives = {scope: _archives(directories[scope]) for scope in _SCOPES}
    for scope, paths in archives.items():
        if set(paths) != expected:
            raise ValueError(f"The {scope} archives must exactly match the locked participant roster")
    return archives


def _verify_seal(archives, seal, lock_hash, coverage):
    hashes = {}
    for scope, paths in archives.items():
        digest = sha256()
        for path in sorted(paths.values(), key=lambda item: item.name):
            with path.open("rb") as stream:
                content = file_digest(stream, "sha256").hexdigest()
            digest.update(f"{path.name} {content}\n".encode())
        hashes[scope] = digest.hexdigest()
    if (seal["sha256_model_lock"] != lock_hash or seal["classifier_predictions_generated"] != 0
            or seal["scope_sha256_by_sorted_filename_and_content_digest"] != hashes):
        raise ValueError("The independent archives must match the model-free preparation seal")
    _match_coverage(seal["ten_second_grid"], coverage["ten_second_grid"])
    return hashes


def _verify_fields(expected, observed, fields, error):
    if any(not np.array_equal(expected[key], observed[key]) for key in fields):
        raise ValueError(error)


def _verify_parent_recordings(parent, features):
    starts, reference, labels = (parent[key] for key in ("start_s", "reference_valid", "stable_label"))
    ecg_rate, pulse = parent["ecg_fs"].item(), parent["ppg"]
    ecg = parent["ecg"]
    if (ecg_rate != 500 or ecg.shape != (len(starts), WINDOW_SECONDS * ecg_rate)
            or pulse.shape != (len(starts), (WINDOW_SECONDS + 2) * PPG_RATE)
            or reference.shape != starts.shape or reference.dtype != bool or labels.shape != starts.shape
            or not np.isin(labels, (-1, 0, 1)).all() or np.any((labels >= 0) & ~reference)
            or not np.isfinite(ecg[reference]).all()):
        raise ValueError("Parent waveform, ECG reference, and stable label rows must align")
    if not np.array_equal(np.isfinite(pulse[:, PPG_RATE:-PPG_RATE]).all(1), features["ppg_available"]):
        raise ValueError("Parent feature PPG availability must agree with its recorded waveform")


def _verify_archives(archives, version, *, verify_staging):
    parents, finer, kinds = [], [], Counter()
    for member, parent_path in archives["parent-raw"].items():
        parents.append(_subject_support(member, _baseline(parent_path), WINDOW_SECONDS))
        with (np.load(parent_path, allow_pickle=False) as parent,
              np.load(archives["parent-features"][member], allow_pickle=False) as parent_features,
              np.load(archives["preroll"][member], allow_pickle=False) as history,
              np.load(archives["stride-raw"][member], allow_pickle=False) as raw,
              np.load(archives["stride-features"][member], allow_pickle=False) as features):
            _verify_fields(parent, parent_features, _PARENT_FIELDS, "Parent raw and feature scopes or ECG labels differ")
            _verify_fields(parent, raw, ("subject_id", "af_intervals_s"), "The finer grid changed ECG annotations")
            _verify_fields(parent, history, ("subject_id", "af_intervals_s"), "Native history changed ECG annotations")
            _verify_fields(raw, features, _STRIDE_FIELDS, "Finer raw and feature clocks, scopes, or ECG labels differ")
            _verify_parent_recordings(parent, parent_features)
            if any(read_feature_version(item) != version for item in (parent_features, features)):
                raise ValueError("The parent and finer feature versions must match the protocol")
            if verify_staging:
                staged = stage_subject(parent_path, archives["preroll"][member])
                _verify_fields(staged, raw, _STRIDE_FIELDS, "Development staging does not reproduce the archived grid")
                if not np.array_equal(staged["ppg"], raw["ppg"], equal_nan=True):
                    raise ValueError("Development staging does not reproduce the archived PPG windows")
            states, end = raw["input_status"], raw["decision_end_s"]
            if (not np.isin(states, _INPUT_STATES).all() or raw["decision_stride_seconds"].item() != STRIDE
                    or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                    or not np.array_equal(end - STRIDE, raw["start_s"])
                    or not np.array_equal(end - WINDOW_SECONDS, raw["input_start_s"])):
                raise ValueError("Use the locked causal PPG input and decision clocks")
            available = (states == "available") & raw["ppg_available"]
            for route in ROUTES:
                matrix = features[route]
                if matrix.ndim != 2 or len(matrix) != len(available) or not matrix.shape[1] or np.isinf(matrix).any():
                    raise ValueError("Feature matrices must align on the finer decision grid")
                if np.any(np.isfinite(matrix).all(1) & ~available):
                    raise ValueError("A complete feature row requires a usable native PPG input")
            finer.append(_subject_support(member, raw, STRIDE))
            for kind in np.unique(raw["run_kind"]):
                kinds[str(kind)] += int(len(np.unique(raw["run_id"][raw["run_kind"] == kind])))
    original, staged = _sum_support(parents, WINDOW_SECONDS), _sum_support(finer, STRIDE)
    invariant = ("complete_reference_episodes", "partial_reference_episodes", "reference_observed_seconds",
                 "complete_reference_episodes_in_wrist_monitoring", "monitoring_observed_seconds",
                 "monitoring_observed_non_af_seconds", "monitoring_blocks", "evaluation_blocks")
    _match_coverage({key: staged[key] for key in invariant}, original)
    return original, staged, dict(kinds)


def _summarize_grid(grid, people, support):
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    monitoring = np.concatenate([block.rows for block in grid.timelines if block.kind == MONITOR_KIND])
    history = np.concatenate(list(grid.history_rows.values()))
    appended = np.r_[evaluation, history]
    if (not np.array_equal(np.sort(appended), np.arange(len(grid.subjects) - len(appended), len(grid.subjects)))
            or grid.table["reference_valid"][history].any() or len(grid.history_rows) != len(grid.timelines)
            or len(grid.timelines) != support["evaluation_blocks"]):
        raise ValueError("Evaluation and unscored history must partition every finer-grid row exactly once")
    shared = _shared_features(grid.table)
    rows = {"all_evaluation": evaluation, "wrist_monitoring": monitoring, "predecessor_history": history}
    scopes = {}
    for scope, selected in rows.items():
        scopes[scope] = {
            "decision_cells": len(selected),
            "current_ppg_present_cells": int(grid.table["ppg_available"][selected].sum()),
            "shared_feature_cells": int(shared[selected].sum()),
            "finite_feature_cells_by_route": {name: int(np.isfinite(grid.table[name][selected]).all(1).sum())
                                             for name in ROUTES},
        }
    # Constant values encode feature opportunity only. This exercise runs no classifier.
    wrist, all_episodes = scoring_runs(grid, np.where(shared, 1.0, np.nan), people)
    probe = score_decisions((run.decisions(0.5) for run in wrist),
                            episode_runs=(run.decisions(0.5) for run in all_episodes))
    outcome, burden = probe["episode_metrics"], probe["burden_metrics"]
    _match_coverage({
        "complete_reference_episodes": outcome["complete_reference_episodes"],
        "partial_reference_episodes": outcome["partial_reference_episodes"],
        "reference_observed_seconds": outcome["ecg_reference_seconds"],
        "monitoring_observed_seconds": burden["ecg_reference_seconds"],
        "monitoring_observed_non_af_seconds": burden["ecg_non_af_seconds"],
    }, support)
    supported = outcome["complete_episodes_with_positive_decision"]
    if not supported or supported > support["complete_reference_episodes_with_usable_raw_input_endpoint"]:
        raise ValueError("Feature-supported episodes must have usable native PPG")
    return {
        "scopes": scopes,
        "complete_reference_episodes_with_shared_feature_endpoint": supported,
        "monitoring_shared_feature_seconds": burden["analyzable_seconds"],
        "monitoring_shared_feature_non_af_seconds": burden["analyzable_non_af_seconds"],
        "participants_with_shared_feature_endpoint": len(set(grid.subjects[evaluation[shared[evaluation]]])),
    }


def _read(path):
    data = path.read_bytes()
    return json.loads(data), sha256(data).hexdigest()


def preflight(args):
    protocol, protocol_hash = _read(args.protocol_lock)
    lock, lock_hash = _read(args.independent_lock)
    clock, clock_hash = _read(args.clock_results)
    coverage, coverage_hash = _read(args.coverage_report)
    manifest, manifest_hash = _read(args.source_manifest)
    seal, seal_hash = _read(args.independent_seal)
    fit, calibration, independent = (_subject_ids(protocol[role])
                                     for role in ("fit_subjects", "calibration_subjects", "test_roster"))
    development = fit | calibration
    if ((len(fit), len(calibration), len(independent)) != (5, 5, 25) or fit & calibration
            or development & independent or lock["source_protocol_sha256"] != protocol_hash
            or clock["original_lock_sha256"] != protocol_hash):
        raise ValueError("The protocol must match the original five fit, five calibration, and 25 independent participants")
    if (protocol["window_seconds"] != WINDOW_SECONDS or protocol["monitor_kind"] != MONITOR_KIND
            or clock["development_participants"] != len(development)
            or clock["warm_10s"]["stride_seconds"] != STRIDE or clock["warm_10s"]["feature_window_seconds"] != WINDOW_SECONDS
            or coverage["decoded_roster_members"] != len(independent) or seal["participants"] != len(independent)):
        raise ValueError("The original scope, feature window, decision stride, and coverage reports must agree")

    inventory = _inventory(manifest, development | independent)
    version = protocol[FEATURE_VERSION_KEY]
    dev_dir = dict(zip(_SCOPES, (args.development_raw, args.development_features, args.development_preroll,
                                  args.development_stride_raw, args.development_stride_features), strict=True))
    independent_dir = {name: args.independent_prepared / name for name in _SCOPES}
    dev_archives = _cohort_archives(dev_dir, development)
    independent_archives = _cohort_archives(independent_dir, independent)
    fingerprints = _verify_seal(independent_archives, seal, lock_hash, coverage)

    dev_original, dev_fine, dev_kinds = _verify_archives(dev_archives, version, verify_staging=True)
    table, people, timelines = load_features(args.development_features)
    if set(people) != development:
        raise ValueError("Development features must only contain the locked development participants")
    grid = load_grid(table, people, timelines, args.development_stride_raw, args.development_stride_features, seconds=STRIDE)
    dev_features = _summarize_grid(grid, development, dev_fine)
    eligible_fit = _shared_features(table) & (table["run_kind"] != "enriched_episode")
    dev_features["original_feature_supported_stable_fit_cells"] = {
        "af": int((eligible_fit & (table["stable_label"] == 1)).sum()),
        "non_af": int((eligible_fit & (table["stable_label"] == 0)).sum()),
    }
    baseline = clock["warm_10s"]["nested_development_cv"]["1"]["pulse_template"]
    _match_coverage({"complete_episodes": dev_fine["complete_reference_episodes"],
                     "wrist_ecg_observed_seconds": dev_fine["monitoring_observed_seconds"],
                     "wrist_non_af_seconds": dev_fine["monitoring_observed_non_af_seconds"]}, baseline)

    ind_original, ind_fine, ind_kinds = _verify_archives(independent_archives, version, verify_staging=False)
    _match_coverage(ind_original, coverage["original_30_second_grid"])
    _match_coverage(ind_fine, coverage["ten_second_grid"])
    if ind_kinds != seal["decoded_runs_by_kind"]:
        raise ValueError("Independent evaluation scopes must match the preparation seal")
    grid, _, available, _, roster, _, _ = load_independent_inputs(
        lock, protocol, independent_dir["parent-features"], independent_dir["stride-raw"],
        independent_dir["stride-features"], coverage,
    )
    if set(grid.subjects) != roster or roster != independent:
        raise ValueError("All locked independent participants must be present on the verified grid")
    ind_features = _summarize_grid(grid, independent, ind_fine)
    shared = {seal_key: ind_features["scopes"][scope]["shared_feature_cells"] for seal_key, scope in
              (("evaluation", "all_evaluation"), ("history", "predecessor_history"))}
    if shared != seal["shared_feature_cells"] or sum(shared.values()) != int(available.sum()):
        raise ValueError("Independent feature eligibility must match the preparation seal and reference verifier")

    return {
        "scope": "Data and reference preflight for the locked wrist AF experiment",
        "classifier_predictions_generated": 0,
        "feature_window_seconds": WINDOW_SECONDS,
        "decision_stride_seconds": STRIDE,
        "monitor_kind": MONITOR_KIND,
        FEATURE_VERSION_KEY: version,
        "source_inventory": inventory,
        "sha256": {"source_inventory": manifest_hash, "protocol_lock": protocol_hash, "independent_lock": lock_hash,
                   "clock_results": clock_hash, "independent_coverage_report": coverage_hash, "independent_seal": seal_hash},
        "development": {"participants": len(development), "fit_participants": len(fit),
                        "calibration_participants": len(calibration), "evaluation_blocks_by_kind": dev_kinds,
                        "original_30_second_grid": dev_original, "ten_second_grid": dev_fine, **dev_features},
        "independent": {"participants": len(independent), "evaluation_blocks_by_kind": ind_kinds,
                        "sealed_archive_sha256_by_scope": fingerprints,
                        "original_30_second_grid": ind_original, "ten_second_grid": ind_fine, **ind_features},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-manifest", "protocol-lock", "independent-lock", "clock-results", "coverage-report",
                 "development-raw", "development-features", "development-preroll", "development-stride-raw",
                 "development-stride-features", "independent-prepared", "independent-seal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Use a new output path to preserve the existing preflight report")
        report = preflight(args)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF data preflight failed: {error}\n")
    print(f"Verified {report['development']['participants']} development and "
          f"{report['independent']['participants']} independent participants without classifier inference")


if __name__ == "__main__":
    main()
