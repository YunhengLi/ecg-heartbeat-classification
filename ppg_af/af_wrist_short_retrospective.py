"""Evaluate a development-frozen ten-second template on historical wrist AF episodes."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize, scoring_policy
from .af_wrist_confirmation_opportunity import DURATION_LABELS
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_subgroups import complete_episodes, select_scope
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_history import apply_negative_rearm
from .af_wrist_local_template_study import stage_cores
from .af_wrist_monitor_extension import _cluster_uncertainty
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_context import SHORT_SECONDS, _pulse_features, short_template_features
from .af_wrist_template_independent import _locked_model, _match_coverage
from .af_wrist_template_study import DEFAULT_C, ROUTE
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE, _select_matched


REARMED_SHORT = "short_matched_quality_and_two_negative_rearm"
SHORT_REARM = partial(apply_negative_rearm, negative_windows=2)


def short_matrix(cores, available, template, shape):
    values = np.full(shape, np.nan)
    for row in np.flatnonzero(available):
        recent = cores[row, -SHORT_SECONDS * PPG_RATE:]
        if np.ptp(recent) > 0:
            features = _pulse_features(template, recent)
            if features[-1] > 0:
                values[row] = features
    return values, np.isfinite(values).all(axis=1)


def freeze_short(args, protocol, model_lock):
    grid, _, paths, development_lock, _, _ = load_motion_development(args)
    if development_lock != protocol:
        raise ValueError("The development and historical protocol must be the same lock")
    fitting, calibration, reserved = (_subject_ids(model_lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fitting & calibration or (fitting | calibration) & reserved or set(grid.subjects) != fitting | calibration):
        raise ValueError("The ten-second model must be frozen without the historical roster")
    pairs = tuple(path for duo in paths for path in duo)
    recent, support, _ = short_template_features(grid, *pairs, args.ptt_dir, protocol[FEATURE_VERSION_KEY])
    shared = _shared_features(grid.table)
    if not np.array_equal(shared, np.isfinite(recent).all(axis=1)):
        raise ValueError("Development 10-second and 30-second fitting and calibration must have identical support")
    labels = grid.table["stable_label"]
    fit_rows = shared & np.isin(grid.subjects, sorted(fitting)) & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    counts = {name: int(np.sum(fit_rows & (labels == label))) for name, label in (("AF", 1), ("non_AF", 0))}
    if counts != {"AF": 21, "non_AF": 22}:
        raise ValueError("Use exactly the original 43 stable fitting examples")

    fitted = _fit_model(recent[fit_rows], labels[fit_rows], regularization=DEFAULT_C)
    calibration_rows = shared & np.isin(grid.subjects, sorted(calibration))
    scores = _predict(recent, calibration_rows, fitted)
    evaluate, pool = scoring_policy(grid, scores, calibration, _IMMEDIATE)
    evaluated = {}

    def record(threshold):
        summary = evaluate(threshold)
        evaluated[threshold] = summary
        return summary

    choice = select_scored_threshold(pool, record, target_sensitivity=protocol["target_sensitivity"], max_thresholds=max(3, len(pool) + 1))
    if choice["threshold"] is None:
        raise ValueError("The ten-second model must have a valid development-only threshold")
    original, _, _, _ = _locked_model(model_lock, protocol)
    original_scores = _predict(grid.table[ROUTE], calibration_rows, original["model"])
    compare, _ = scoring_policy(grid, original_scores, calibration, _IMMEDIATE)
    baseline = {"threshold": original["cutoff"], "validation": compare(original["cutoff"])}
    matched = _select_matched(evaluated, baseline, 0.0)
    if matched["threshold"] is None:
        raise ValueError("The matched sensitivity comparison requires a training-side threshold")
    cutoffs = {"short": choice["threshold"], "short_matched": matched["threshold"]}
    return fitted, cutoffs, {"fitting_examples": counts, "short_input_support": support,
                             "fitting_participants": len(fitting), "calibration_participants": len(calibration),
                             "calibration_status": choice["status"], "calibration_summary": _summarize(choice["validation"]),
                             "match_calibration_status": matched["status"], "matched_calibration_summary": _summarize(matched["validation"]),
                             "reference_calibration_summary": _summarize(baseline["validation"])}


def by_duration(replays):
    original, people, _ = complete_episodes(replays["original"][1])
    latencies = {}
    for name, (_, runs, _) in replays.items():
        intervals, roster, latency = complete_episodes(runs)
        if not np.array_equal(intervals, original) or not np.array_equal(roster, people):
            raise ValueError("Historical AF durations and participant groupings must be identical")
        latencies[name] = latency
    duration = original[:, 1] - original[:, 0]
    bins = np.searchsorted((10, 30, 60), duration, side="left")
    return {label: select_scope(people[bins == index], {name: values[bins == index] for name, values in latencies.items()})
            for index, label in enumerate(DURATION_LABELS)}


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    new_model, new_cutoffs, fitting = freeze_short(args, protocol, model_lock)
    grid, original, available, roster, quality, published = historical_inputs(args, protocol, model_lock)
    paths = tuple((args.prior_inputs / f"{key}-raw", args.prior_inputs / f"{key}-features") for key in ("parent", "stride"))
    cores = stage_cores(grid, paths)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    recent, common = short_matrix(cores, available, template, grid.table[ROUTE].shape)
    original_cutoff = np.full(len(grid.subjects), original["cutoff"])
    original_scores = _predict(grid.table[ROUTE], available, original["model"])
    full = _held_out_replay(grid, original_scores, original_cutoff, _IMMEDIATE)
    _match_coverage(full[0], published["independent"])
    older, newer = np.where(common, original_scores, np.nan), _predict(recent, common, new_model)
    prior, active = preceding_rows(grid)
    low = active & (quality["quality"][:, -1] < QUALITY_CUTOFF)
    arms = {"original": (older, original_cutoff),
            "original_with_quality_confirmation": (confirm_scores(older, prior, low), original_cutoff)}
    short_confirmed = confirm_scores(newer, prior, low)
    for name, value in new_cutoffs.items():
        cutoff = np.full(len(newer), value)
        arms[name] = newer, cutoff
        arms[f"{name}_with_quality_confirmation"] = short_confirmed, cutoff
    replays = {name: _held_out_replay(grid, scores, threshold, _IMMEDIATE) for name, (scores, threshold) in arms.items()}
    previous_short = replays["short_matched_with_quality_confirmation"]
    replays[REARMED_SHORT] = _held_out_replay(grid, *arms["short_matched_with_quality_confirmation"], SHORT_REARM)
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    support = {"matched_evaluation_rows": int(common[evaluation].sum()),
               "matched_history_rows": int(common.sum() - common[evaluation].sum()),
               "lost_evaluation_rows": int((available[evaluation] & ~common[evaluation]).sum()),
               "lost_history_rows": int((available & ~common).sum() - (available[evaluation] & ~common[evaluation]).sum())}
    matched = replays["original"]
    return {"scope": "Historical ten-second versus 30-second PPG template with shared wrist score eligibility",
            "short_input_seconds": SHORT_SECONDS, "frozen_development_only_fit": fitting,
            "full_original": full[0], "historical_signal_support": support,
            "comparisons": {name: _compare_with_pre_onset(matched, other, protocol) for name, other in replays.items()},
            "short_vs_original_uncertainty": _cluster_uncertainty(matched, replays["short"], sorted(roster)),
            "short_matched_vs_original_uncertainty": _cluster_uncertainty(matched, replays["short_matched"], sorted(roster)),
            "short_vs_original_with_confirmation_uncertainty": _cluster_uncertainty(
                replays["original_with_quality_confirmation"], replays["short_with_quality_confirmation"], sorted(roster)),
            "rearmed_vs_immediate_short": _compare_with_pre_onset(previous_short, replays[REARMED_SHORT], protocol),
            "rearmed_vs_immediate_short_uncertainty": _cluster_uncertainty(previous_short, replays[REARMED_SHORT], sorted(roster)),
            "duration_strata": by_duration(replays)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "historical-model-lock", "data-preflight", "coverage-report",
                 "historical-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
