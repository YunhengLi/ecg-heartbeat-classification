"""Freeze two wrist quality variants on development and replay them retrospectively."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, load_grid, scoring_policy
from .af_wrist_data_preflight import _SCOPES, _cohort_archives, _summarize_grid, _verify_archives, _verify_seal
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_operating import _NO_ALERT_THRESHOLD, select_scored_threshold
from .af_wrist_quality_study import build_quality_matrices
from .af_wrist_template_independent import ROUTE, _match_coverage, load_independent_inputs
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare, _held_out_replay


VARIANTS = ("quality", "quality_interactions")
_C = 1.0
_LOCK_VERSION = 1
_PREFLIGHT_SHA256 = "184d05d85d0e3c8b7546add7ce78f4f356ec7b90ccd4ad0109684604982bf25e"
_HISTORICAL_RESULTS_SHA256 = "0af4f089923cc446f207ba9cc459432faabcf28821571cb11e0ad1d5757ef122"


def _read_pinned(path, digest, *, public=False):
    data = path.read_bytes()
    if public:
        data = data.replace(b"\r\n", b"\n")
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    return json.loads(data)


def _quality_matrices(grid, directories):
    return build_quality_matrices(
        grid, *(directories[key] for key in ("parent-raw", "parent-features", "stride-raw", "stride-features")),
    )


def freeze_variants(protocol, protocol_digest, preflight, directories):
    """Fit and calibrate the two exact variants without loading retrospective inputs."""
    fit, calibration, reserved = (_subject_ids(protocol[key])
                                  for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    people = fit | calibration
    if (fit & calibration or people & reserved or (len(fit), len(calibration), len(reserved)) != (5, 5, 25)
            or protocol["window_seconds"] != WINDOW_SECONDS or protocol["target_sensitivity"] != 0.8):
        raise ValueError("Use the original fixed development fit and calibration groups")
    version = protocol[FEATURE_VERSION_KEY]
    archives = _cohort_archives(directories, people)
    original, fine, kinds = _verify_archives(archives, version, verify_staging=True)
    expected = preflight["development"]
    if (original, fine, kinds) != (expected["original_30_second_grid"], expected["ten_second_grid"],
                                   expected["evaluation_blocks_by_kind"]):
        raise ValueError("Development reference coverage must match the original data preflight")

    table, subjects, timelines = load_features(directories["parent-features"])
    if set(subjects) != people or table[FEATURE_VERSION_KEY] != version:
        raise ValueError("Use only the original development features")
    grid = load_grid(table, subjects, timelines, directories["stride-raw"], directories["stride-features"],
                     seconds=STRIDE_SECONDS)
    if _summarize_grid(grid, people, fine)["scopes"] != expected["scopes"]:
        raise ValueError("Development feature support must match the original data preflight")
    matrices = _quality_matrices(grid, directories)
    shared = _shared_features(grid.table)
    labels = grid.table["stable_label"]
    fit_rows = (np.isin(grid.subjects, sorted(fit)) & shared & (labels >= 0)
                & (grid.table["run_kind"] != "enriched_episode"))
    fit_counts = {name: int(np.sum(fit_rows & (labels == label))) for name, label in (("AF", 1), ("non_AF", 0))}
    if fit_counts != {"AF": 21, "non_AF": 22}:
        raise ValueError("The fixed classifier must use the original 43 stable fitting cells")
    calibration_rows = np.isin(grid.subjects, sorted(calibration)) & shared

    models = {}
    for name in VARIANTS:
        matrix = matrices[name]
        model = _fit_model(matrix[fit_rows], labels[fit_rows], regularization=_C)
        scores = _predict(matrix, calibration_rows, model)
        evaluate, pool = scoring_policy(grid, scores, calibration, _IMMEDIATE)
        choice = select_scored_threshold(pool, evaluate, target_sensitivity=protocol["target_sensitivity"],
                                         max_thresholds=max(3, len(pool) + 1))
        if choice["threshold"] is None:
            raise ValueError("Each quality variant needs a development-calibrated cutoff")
        models[name] = {
            "model": model, "cutoff": choice["threshold"],
            "calibration": {"status": choice["status"], "threshold_candidates": choice["threshold_candidates"],
                            **_summarize(choice["validation"])},
        }
    return {
        "scope": "Private development-only freeze for retrospective wrist quality replay",
        "lock_version": _LOCK_VERSION,
        "source_protocol_sha256": protocol_digest,
        FEATURE_VERSION_KEY: version,
        "logistic_C": _C,
        "fit_windows": fit_counts,
        "models": models,
    }


def replay_variants(protocol, baseline_lock, frozen, coverage, published, preflight, directories, seal):
    """Reconcile the historical baseline, then replay both locked quality alternatives."""
    if (frozen["lock_version"] != _LOCK_VERSION or set(frozen["models"]) != set(VARIANTS)
            or frozen["source_protocol_sha256"] != baseline_lock["source_protocol_sha256"]
            or frozen[FEATURE_VERSION_KEY] != baseline_lock[FEATURE_VERSION_KEY]
            or frozen["logistic_C"] != _C or frozen["logistic_C"] != published["logistic_C"]
            or not all(0 <= option["cutoff"] <= _NO_ALERT_THRESHOLD for option in frozen["models"].values())):
        raise ValueError("Use the two development-frozen C=1 quality variants")
    roster = _subject_ids(protocol["test_roster"])
    archives = _cohort_archives(directories, roster)
    _verify_seal(archives, seal, published["model_lock_sha256"], coverage)
    grid, model, available, support, _, fit_count, calibration_count = load_independent_inputs(
        baseline_lock, protocol, directories["parent-features"], directories["stride-raw"],
        directories["stride-features"], coverage,
    )
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    historical = published["independent"]
    shared_counts = {"evaluation": int(available[evaluation].sum()),
                     "history": int(available.sum() - available[evaluation].sum())}
    if (set(grid.subjects) != roster or len(roster) != published["evaluated_independent_participants"]
            or shared_counts != seal["shared_feature_cells"]
            or shared_counts != {"evaluation": historical["scored_evaluation_cells"],
                                 "history": historical["scored_predecessor_cells"]}):
        raise ValueError("Retrospective replay must use the exact published roster and score support")

    def replay(matrix, option):
        scores = _predict(matrix, available, option["model"])
        if not np.array_equal(np.isfinite(scores), available):
            raise ValueError("Every replay must preserve the original score eligibility")
        return _held_out_replay(grid, scores, np.full(len(scores), option["cutoff"]), _IMMEDIATE)

    baseline = replay(grid.table[ROUTE], model)
    _match_coverage(baseline[0], historical)
    matrices = _quality_matrices(grid, directories)
    variants = {}
    for name in VARIANTS:
        option = frozen["models"][name]
        comparison = _compare(baseline, replay(matrices[name], option), protocol)
        paired = comparison["paired_changes"]
        metrics = comparison["summary"]
        if (paired["episodes"]["detected_by_both"] + paired["episodes"]["gained"] != metrics["episodes_detected"]
                or paired["episodes"]["detected_by_both"] + paired["episodes"]["lost"] != baseline[0]["episodes_detected"]
                or paired["false_starts"]["candidate"] != metrics["wrist_false_alerts"]
                or paired["false_starts"]["baseline"] != baseline[0]["wrist_false_alerts"]):
            raise ValueError("Paired changes must reconcile with the replayed event totals")
        variants[name] = comparison
    return {
        "scope": "Retrospective replay on the previously scored and audited wrist cohort",
        "fit_participants": fit_count, "calibration_participants": calibration_count,
        "retrospective_participants": len(roster),
        "logistic_C": _C, FEATURE_VERSION_KEY: frozen[FEATURE_VERSION_KEY],
        "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": grid.seconds,
        "alert_policy": baseline_lock["alert_policy"],
        "target_episode_sensitivity": protocol["target_sensitivity"],
        "sensitivity_tolerance": protocol["sensitivity_tolerance"],
        "fit_windows": frozen["fit_windows"],
        "development_calibration": {"baseline": published["calibration"],
                                    **{name: frozen["models"][name]["calibration"] for name in VARIANTS}},
        "common_retrospective_support": {
            "scored_evaluation_cells": shared_counts["evaluation"],
            "scored_predecessor_cells": shared_counts["history"],
            "complete_reference_episodes": support["complete_reference_episodes"],
            "partial_reference_episodes": support["partial_reference_episodes"],
            "all_evaluation_ecg_observed_seconds": support["reference_observed_seconds"],
            "wrist_ecg_observed_seconds": support["monitoring_observed_seconds"],
            "wrist_non_af_seconds": support["monitoring_observed_non_af_seconds"],
            "complete_episodes_with_shared_feature_endpoint": historical["complete_episodes_with_shared_feature_endpoint"],
            "participants_with_complete_episodes": (
                published["participant_cluster_uncertainty"]["participants_with_complete_episodes"]
            ),
            "maximum_complete_episodes_from_one_participant": (
                published["participant_cluster_uncertainty"]["maximum_complete_episodes_from_one_participant"]
            ),
        },
        "historical_baseline_reproduced": True,
        "historical_baseline": baseline[0],
        "retrospective_variants": variants,
        "source_receipts_sha256": {**{name: preflight["sha256"][name] for name in
                                      ("protocol_lock", "independent_lock", "independent_coverage_report", "independent_seal")},
                                   "historical_results": _HISTORICAL_RESULTS_SHA256, "data_preflight": _PREFLIGHT_SHA256},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Create a private model lock from development only")
    replay = commands.add_parser("replay", help="Replay that private lock on the previously evaluated wrist cohort")
    for command in (freeze, replay):
        for name in ("protocol-lock", "data-preflight", "output"):
            command.add_argument(f"--{name}", type=Path, required=True)
    for scope in _SCOPES:
        freeze.add_argument(f"--development-{scope}", type=Path, required=True)
    for name in ("variant-lock", "historical-model-lock", "historical-results", "coverage-report", "sealed-inputs"):
        replay.add_argument(f"--{name}", type=Path, required=True)
    replay.add_argument("--variant-lock-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Use a new output path")
        preflight = _read_pinned(args.data_preflight, _PREFLIGHT_SHA256, public=True)
        pinned = preflight["sha256"]
        protocol = _read_pinned(args.protocol_lock, pinned["protocol_lock"])
        if args.command == "freeze":
            directories = {scope: getattr(args, f"development_{scope.replace('-', '_')}") for scope in _SCOPES}
            lock = freeze_variants(protocol, pinned["protocol_lock"], preflight, directories)
            _write_json(args.output, lock)
            print(f"Wrote private development lock SHA-256 {sha256(args.output.read_bytes()).hexdigest()}")
        else:
            frozen = _read_pinned(args.variant_lock, args.variant_lock_sha256)
            baseline = _read_pinned(args.historical_model_lock, pinned["independent_lock"])
            history = _read_pinned(args.historical_results, _HISTORICAL_RESULTS_SHA256, public=True)
            coverage = _read_pinned(args.coverage_report, pinned["independent_coverage_report"], public=True)
            seal = _read_pinned(args.sealed_inputs / "sealed-input-proof.json", pinned["independent_seal"])
            directories = {scope: args.sealed_inputs / scope for scope in _SCOPES}
            report = replay_variants(protocol, baseline, frozen, coverage, history, preflight, directories, seal)
            report["development_variant_lock_sha256"] = args.variant_lock_sha256
            _write_json(args.output, report)
            print(f"Replayed the historical baseline and two frozen variants on {report['retrospective_participants']} participants")
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Retrospective wrist quality {args.command} failed: {error}\n")


if __name__ == "__main__":
    main()
