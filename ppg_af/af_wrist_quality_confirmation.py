"""Test confirmation only when the current optical input has lower quality."""

import argparse
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import scoring_policy
from .af_wrist_failure_study import fold_arrays
from .af_wrist_holdout import _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


QUALITY_CUTOFF = 0.40


def preceding_rows(grid, *, steps=1):
    if steps < 1:
        raise ValueError("Confirmation must use an earlier decision")
    preceding = np.full(len(grid.subjects), -1, dtype=int)
    active = np.zeros(len(grid.subjects), dtype=bool)
    for block in grid.timelines:
        history = grid.history_rows[block.subject, block.start]
        chain = np.r_[history, block.rows]
        if (len(np.unique(chain)) != len(chain) or not np.all(grid.subjects[chain] == block.subject)
                or active[chain].any()):
            raise ValueError("Confirmation split must preserve unique participant-local causal histories")
        preceding[chain[steps:]] = chain[:-steps]
        active[chain] = True
    return preceding, active


def confirm_scores(scores, preceding, needs_confirmation):
    """Take the lesser of the current and prior raw score at confirmed decisions."""
    result = scores.copy()
    requested = needs_confirmation & np.isfinite(scores)
    selected = requested & (preceding >= 0)
    result[requested & ~selected] = np.nan
    result[selected] = np.minimum(scores[selected], scores[preceding[selected]])
    return result


def evaluate(args):
    grid, quality, _, lock, _, original = load_motion_development(args)
    predecessor, active = preceding_rows(grid)
    newest_quality = quality["quality"][:, -1]
    low = active & (newest_quality < QUALITY_CUTOFF)
    needs_confirmation = {"lower_quality_only": low, "all_inputs": active}
    matrices = {"quality_interactions": quality["quality_interactions"], "plain_template": quality["baseline"]}
    comparisons = {}
    for seed in SEEDS:
        comparisons[str(seed)] = {}
        for name, matrix in matrices.items():
            fitted = fit_template(grid, matrix, lock, DEFAULT_C, seed=seed)
            baseline = _replay(grid, fitted)
            if seed == SEEDS[0] and name == "quality_interactions" and baseline[0] != original:
                raise ValueError("The fixed original quality baseline must reproduce")
            scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
            studies = {"original": baseline[0]}
            for rule, confirm in needs_confirmation.items():
                changed = confirm_scores(scores, predecessor, confirm)
                fixed = _held_out_replay(grid, changed, cutoffs, _IMMEDIATE)

                def calibrate(inner_scores, rows):
                    full = np.full(len(grid.subjects), np.nan)
                    full[rows] = inner_scores
                    filtered = confirm_scores(full, predecessor, confirm)
                    score, pool = scoring_policy(grid, filtered, set(grid.subjects[rows]), _IMMEDIATE)
                    return select_scored_threshold(pool, score, target_sensitivity=lock["target_sensitivity"],
                                                   max_thresholds=max(3, len(pool) + 1))

                tuned = fit_template(grid, matrix, lock, DEFAULT_C, seed=seed, calibration_selector=calibrate)
                reused, new_cutoffs, _ = fold_arrays(tuned, tuned["routes"][ROUTE])
                if not np.array_equal(scores, reused, equal_nan=True):
                    raise ValueError("Quality confirmation changes a decision rule, so all native probabilities must remain fixed")
                recalibrated = _held_out_replay(grid, changed, new_cutoffs, _IMMEDIATE)
                studies[rule] = {"fixed_cutoffs": _compare_with_pre_onset(baseline, fixed, lock),
                                 "nested_calibration": _compare_with_pre_onset(baseline, recalibrated, lock),
                                 "calibration_status": _calibration_status(tuned),
                                 "unavailable_after_confirmation": int(np.sum(np.isfinite(scores) & ~np.isfinite(changed)))}
                print(f"{seed}, {name}, {rule}: fixed {fixed[0]['episodes_detected']}/{fixed[0]['wrist_false_alerts']}; "
                      f"recalibrated {recalibrated[0]['episodes_detected']}/{recalibrated[0]['wrist_false_alerts']}", flush=True)
            comparisons[str(seed)][name] = studies
    return {"scope": "Conditional consecutive-score confirmation on lower-quality wrist inputs",
            "quality_boundary": QUALITY_CUTOFF, "decision_stride_seconds": grid.seconds,
            "lower_quality_rows_in_replay": int(low.sum()), "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
