"""Compare true ECG-annotated interval timing with the frozen wrist template."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, scoring_policy
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_ecg_timing_stage import DURATIONS, archive_clock
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_monitor_extension import _cluster_uncertainty
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_short_retrospective import by_duration
from .af_wrist_template_independent import _locked_model, _match_coverage
from .af_wrist_template_study import DEFAULT_C, ROUTE
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE, _select_matched
from .features import ECG_FEATURE_NAMES


def load_features_for_clock(grid, directories, sidecars, seal):
    columns = {width: [] for width in DURATIONS}
    order = []
    for raw_dir, kind in zip(directories, ("parent", "stride"), strict=True):
        for person in np.unique(grid.subjects):
            name = f"{person}.npz"
            expected, row_digest = archive_clock(raw_dir / name, person, stride=kind == "stride")
            with np.load(sidecars / kind / name, allow_pickle=False) as file:
                if (file["subject_id"].item() != person or file["protocol_sha256"].item() != seal
                        or file["wrist_row_sha256"].item() != row_digest or not np.array_equal(file["decision_end_samples"], expected)):
                    raise ValueError("ECG interval rows must match the exact sealed wrist clock")
                for width in DURATIONS:
                    values = file[f"features_{width}s"]
                    if values.shape != (len(expected), len(ECG_FEATURE_NAMES)):
                        raise ValueError("ECG timing columns must match the original template interval names")
                    columns[width].append(values)
            order.extend([person] * len(expected))
    if not np.array_equal(order, grid.subjects):
        raise ValueError("Native ECG features must use the identical participant row order")
    return {width: np.concatenate(sets) for width, sets in columns.items()}


def available_at_both(matrices):
    return np.logical_and.reduce([np.isfinite(values).all(axis=1) for values in matrices.values()])


def freeze(args, protocol, model_lock):
    grid, _, paths, _, digest, _ = load_motion_development(args)
    ecg = load_features_for_clock(grid, tuple(pair[0] for pair in paths), args.ecg_timing / "development", digest)
    keep = _shared_features(grid.table) & available_at_both(ecg)
    fitting = _subject_ids(model_lock["fit_subjects"])
    calibration = _subject_ids(model_lock["calibration_subjects"])
    labels = grid.table["stable_label"]
    train = (keep & np.isin(grid.subjects, sorted(fitting)) & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode"))
    cal_rows = keep & np.isin(grid.subjects, sorted(calibration))
    original, _, _, _ = _locked_model(model_lock, protocol)
    original_scores = _predict(grid.table[ROUTE], cal_rows, original["model"])
    validation, _ = scoring_policy(grid, original_scores, calibration, _IMMEDIATE)
    reference = {"threshold": original["cutoff"], "validation": validation(original["cutoff"])}
    options, reports = {}, {}
    for width, matrix in ecg.items():
        model = _fit_model(matrix[train], labels[train], regularization=DEFAULT_C)
        scores = _predict(matrix, cal_rows, model)
        validate, pool = scoring_policy(grid, scores, calibration, _IMMEDIATE)
        candidates = {}

        def record(cutoff):
            metrics = validate(cutoff)
            candidates[cutoff] = metrics
            return metrics

        target = select_scored_threshold(pool, record, target_sensitivity=protocol["target_sensitivity"],
                                         max_thresholds=max(3, len(pool) + 1))
        matched = _select_matched(candidates, reference, 0.0)
        if target["threshold"] is None or matched["threshold"] is None:
            raise ValueError("Native ECG thresholds must use only valid development calibration support")
        choices = {"target": target, "matched": matched}
        options[width] = {"model": model, "thresholds": {name: item["threshold"] for name, item in choices.items()}}
        reports[str(width)] = {name: {"status": item["status"], "summary": _summarize(item["validation"])}
                               for name, item in choices.items()}
    return options, {"fitting_people": len(set(grid.subjects[train])), "calibration_people": len(set(grid.subjects[cal_rows])),
                     "fitting_windows_by_rhythm": {"AF": int((train & (labels == 1)).sum()), "non_AF": int((train & (labels == 0)).sum())},
                     "common_calibration_scoreable_rows": int(cal_rows.sum()), "wrist_reference": _summarize(reference["validation"]),
                     "native_ecg": reports}


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    models, receipt = freeze(args, protocol, model_lock)
    grid, original, ppg, roster, _, published = historical_inputs(args, protocol, model_lock)
    directories = (args.prior_inputs / "parent-raw", args.prior_inputs / "stride-raw")
    ecg = load_features_for_clock(grid, directories, args.ecg_timing / "historical", PROTOCOL_SHA)
    native = available_at_both(ecg)
    common = native & ppg
    original_scores = _predict(grid.table[ROUTE], ppg, original["model"])
    original_cutoff = np.full(len(ppg), original["cutoff"])
    full = _held_out_replay(grid, original_scores, original_cutoff, _IMMEDIATE)
    _match_coverage(full[0], published["independent"])
    baseline = _held_out_replay(grid, np.where(common, original_scores, np.nan), original_cutoff, _IMMEDIATE)
    ecg_grid = replace(grid, table={**grid.table, "ppg_available": native}, unknown_source=np.zeros(len(native), dtype=bool))
    matched = {"original": baseline}
    ecg_full = {}
    for width, matrix in ecg.items():
        for policy, cutoff in models[width]["thresholds"].items():
            name = f"native_ecg_{width}s_{policy}"
            full_scores = _predict(matrix, native, models[width]["model"])
            thresholds = np.full(len(native), cutoff)
            matched[name] = _held_out_replay(grid, np.where(common, full_scores, np.nan), thresholds, _IMMEDIATE)
            ecg_full[name] = _held_out_replay(ecg_grid, full_scores, thresholds, _IMMEDIATE)[0]
    evaluation = np.concatenate([item.rows for item in grid.timelines])
    return {"scope": "True ECG QRS reference and wrist PPG on the same historical decision clock",
            "development_only_model_fitting": receipt, "reference_full_ppg": full[0],
            "historical_input_counts": {"wrist_evaluation_rows": int(ppg[evaluation].sum()),
                                        "native_ecg_evaluation_rows": int(native[evaluation].sum()),
                                        "common_evaluation_rows": int(common[evaluation].sum())},
            "common_support": {name: _compare_with_pre_onset(baseline, replay, protocol) for name, replay in matched.items()},
            "paired_participant_uncertainty": {name: _cluster_uncertainty(baseline, replay, sorted(roster))
                                               for name, replay in matched.items() if name != "original"},
            "native_ecg_available_support": ecg_full, "duration_strata_common_support": by_duration(matched)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "historical-model-lock", "data-preflight", "coverage-report",
                 "historical-results", "ecg-timing", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
