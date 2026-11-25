"""Test a short-window alert trigger on the locked wrist development participants."""

import argparse
from pathlib import Path

import numpy as np

from .af_state import state_durations, state_summary
from .af_wrist_confirmation_spacing import DEVELOPMENT_CONFIRMATION_SHA
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_context import SHORT_SECONDS, short_template_features
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, fit_template, verify_fit_support
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


SHORT_MARGIN = 0.10


def short_trigger(confirmed, long_cutoff, short, short_cutoff, quality, predecessor):
    """Allow a new positive without changing the original model's scoreable rows."""
    available = (np.isfinite(confirmed) & (long_cutoff >= 0) & (long_cutoff <= 1) & np.isfinite(short)
                 & np.isfinite(short_cutoff) & (quality >= QUALITY_CUTOFF) & (predecessor >= 0))
    eligible = available & (confirmed < long_cutoff) & (short >= short_cutoff + SHORT_MARGIN)
    rows = np.flatnonzero(eligible)
    previous = predecessor[rows]
    rows = rows[np.isfinite(short[previous]) & (short[previous] < short_cutoff[rows])]
    scored = confirmed.copy()
    scored[rows] = long_cutoff[rows]
    return scored, rows


def evaluate(args):
    grid, quality, paths, protocol, _, _ = load_motion_development(args)
    previous = _read_pinned(args.confirmation_results, DEVELOPMENT_CONFIRMATION_SHA, public=True)["comparisons"]
    short, support, _ = short_template_features(grid, *(path for pair in paths for path in pair),
                                                 args.ptt_dir, protocol[FEATURE_VERSION_KEY])
    if not np.array_equal(np.isfinite(short).all(axis=1), _shared_features(grid.table)):
        raise ValueError("Both feature lengths must use the original development support")
    predecessor, active = preceding_rows(grid)
    optical_quality = quality["quality"][:, -1]
    needs_confirmation = active & (optical_quality < QUALITY_CUTOFF)
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    comparisons = {}
    for seed in SEEDS:
        long_fit = fit_template(grid, quality["baseline"], protocol, DEFAULT_C, seed=seed)
        short_fit = fit_template(grid, short, protocol, DEFAULT_C, seed=seed)
        verify_fit_support(long_fit, short_fit)
        long_scores, long_cutoff, _ = fold_arrays(long_fit, long_fit["routes"][ROUTE])
        short_scores, short_cutoff, _ = fold_arrays(short_fit, short_fit["routes"][ROUTE])
        baseline_scores = confirm_scores(long_scores, predecessor, needs_confirmation)
        baseline = _held_out_replay(grid, baseline_scores, long_cutoff, _IMMEDIATE)
        confirmed_prior = previous[str(seed)]["plain_template"]["lower_quality_only"]["fixed_cutoffs"]["summary"]
        if baseline[0] != confirmed_prior:
            raise ValueError("The published quality-confirmed development baseline must reproduce exactly")
        scores, added = short_trigger(baseline_scores, long_cutoff, short_scores, short_cutoff, optical_quality, predecessor)
        challenger = _held_out_replay(grid, scores, long_cutoff, _IMMEDIATE)
        if not np.array_equal(np.isfinite(baseline_scores), np.isfinite(scores)):
            raise ValueError("The short trigger changed the long route's scoreable decisions")
        coverage = {name: state_summary(state_durations(replay[2])) for name, replay in
                    (("original_confirmation", baseline), ("plus_short_trigger", challenger))}
        if coverage["original_confirmation"]["scoreable_seconds"] != coverage["plus_short_trigger"]["scoreable_seconds"]:
            raise ValueError("Both policies must retain the same monitorable wear time")
        comparison = _compare_with_pre_onset(baseline, challenger, protocol)
        comparisons[str(seed)] = {"original_confirmation": baseline[0], "plus_short_trigger": comparison,
                                  "short_calibration_status": _calibration_status(short_fit),
                                  "added_positive_evaluation_decisions": int(np.isin(added, evaluation).sum()),
                                  "added_positive_history_decisions": int((~np.isin(added, evaluation)).sum()),
                                  "monitorable_wear": {name: {
                                      "seconds": row["scoreable_seconds"], "observed_seconds": row["ecg_observed_seconds"],
                                      "observed_coverage": row["scoreable_coverage"],
                                  } for name, row in coverage.items()},
                                  "challenger_reaches_target": challenger[0]["episode_sensitivity"] >= protocol["target_sensitivity"],
                                  "challenger_does_not_add_false_starts": challenger[0]["wrist_false_alerts"] <= baseline[0]["wrist_false_alerts"]}
        print(f"{seed}: {baseline[0]['episodes_detected']} -> {challenger[0]['episodes_detected']} episodes, "
              f"{baseline[0]['wrist_false_alerts']} -> {challenger[0]['wrist_false_alerts']} false starts", flush=True)
    return {"scope": "Two locked nested participant splits on the existing development participants",
            "outer_folds": 5, "inner_folds": 3, "short_seconds": SHORT_SECONDS,
            "minimum_quality": QUALITY_CUTOFF, "fixed_short_probability_margin": SHORT_MARGIN,
            "target_episode_sensitivity": protocol["target_sensitivity"], "short_input_support": support,
            "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "ptt-dir", "confirmation-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
