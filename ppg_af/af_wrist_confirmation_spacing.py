"""Compare adjacent and nonoverlapping quality-conditioned wrist confirmations."""

import argparse
from pathlib import Path

import numpy as np

from .af_events import _event_at_decision, _grid, _intervals, _union
from .af_state import state_durations, state_summary
from .af_wrist_confirmation_retrospective import historical_replay, replay_on_schedule
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import _cluster_uncertainty
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_vote_compare import _definite_starts, compare_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE


CONFIRMATION_RESULTS_SHA = "4b2e3e5eb90b5e49690b668b517be7d70aff02bffea020ece626c020ee7e7388"
DEVELOPMENT_CONFIRMATION_SHA = "3d00b74e24a6e2744754ecbd3aa2b6985ed425a3f0d0c032f4edb10f8d79b268"
LAGS = {"adjacent_10_seconds": 1, "nonoverlapping_30_seconds": WINDOW_SECONDS // STRIDE_SECONDS}


def development(args):
    grid, quality, _, protocol, _, _ = load_motion_development(args)
    preceding = {name: preceding_rows(grid, steps=steps) for name, steps in LAGS.items()}
    published = args.development_confirmation_results
    prior = _read_pinned(published, DEVELOPMENT_CONFIRMATION_SHA, public=True)
    output = {}
    for seed in SEEDS:
        fitted = fit_template(grid, quality["baseline"], protocol, DEFAULT_C, seed=seed)
        baseline = _replay(grid, fitted)
        scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
        candidates = {}
        for name, (previous, active) in preceding.items():
            changed = confirm_scores(scores, previous, active & (quality["quality"][:, -1] < QUALITY_CUTOFF))
            candidate = _held_out_replay(grid, changed, cutoffs, _IMMEDIATE)
            candidates[name] = _compare_with_pre_onset(baseline, candidate, protocol)
        original = prior["comparisons"][str(seed)]["plain_template"]
        if baseline[0] != original["original"] or candidates["adjacent_10_seconds"] != original["lower_quality_only"]["fixed_cutoffs"]:
            raise ValueError("Use the published original and fixed-adjacent development replay")
        output[str(seed)] = {"original": baseline[0], **candidates}
    return output


def false_start_fragmentation(original, transformed):
    extra_starts = fragmented_runs = 0
    for left, right in zip(original, transformed, strict=True):
        edges, decisions, reference, _ = _grid(left)
        right_positive = right.decisions == 1
        if np.any(right_positive & (decisions != 1)):
            raise ValueError("Confirmation cannot make a raw negative positive")
        episodes = _union(_intervals(left.af_intervals_s))
        non_af = _event_at_decision(edges[1:], episodes) < 0
        candidate_false = reference & non_af & _definite_starts(right, right.decisions)
        original_group = np.cumsum(np.r_[True, decisions[:-1] != 1])
        _, counts = np.unique(original_group[candidate_false], return_counts=True)
        extra_starts += int(np.sum(counts - 1))
        fragmented_runs += int(np.sum(counts > 1))
    return {"original_positive_streaks_with_multiple_candidate_false_starts": fragmented_runs,
            "candidate_false_starts_after_first_in_same_original_streak": extra_starts}


def half_hour(args, protocol, model_lock):
    earlier, _, earlier_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, earlier, earlier_digest)
    prior = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    if digest != prior["plan_sha256"] or set(schedule) != roster:
        raise ValueError("Use the sealed half-hour plan and out-of-fit roster")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    output = {}
    for name, steps in LAGS.items():
        baseline, candidate = replay_on_schedule(schedule, digest, args.monitoring_cache, template, model,
                                                 block_seconds=BLOCK_SECONDS, confirmation_steps=steps)
        if baseline[0] != prior["original"] or (steps == 1 and candidate[0] != prior["conditional_confirmation"]):
            raise ValueError("The earlier sealed half-hour replays must reproduce")
        paired = compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2])
        output[name] = {"summary": candidate[0], "paired_changes_from_original": paired,
                        "uncertainty_from_original": _cluster_uncertainty(baseline, candidate, sorted(roster),
                                                                         allow_score_coverage_change=True),
                        "false_start_fragmentation": false_start_fragmentation(baseline[1], candidate[1]),
                        "scoreable_decisions": sum(int((run.decisions >= 0).sum()) for run in candidate[1]),
                        "forward_state": state_summary(state_durations(candidate[1]))}
    return {"plan_sha256": digest, "original": prior["original"], **output}


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    prior = _read_pinned(args.confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)
    historical = {name: historical_replay(args, protocol, model_lock, confirmation_steps=steps)
                  for name, steps in LAGS.items()}
    if historical["adjacent_10_seconds"] != prior["historical_episodes_and_monitoring"]:
        raise ValueError("The earlier fixed-adjacent historical replay must reproduce")
    return {"scope": "Previously scored wrist cohorts with identical frozen PPG scores and cutoffs",
            "quality_boundary": QUALITY_CUTOFF, "confirmation_delay_seconds": {key: value * STRIDE_SECONDS for key, value in LAGS.items()},
            "development": development(args), "historical": historical, "half_hour": half_hour(args, protocol, model_lock)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "development-confirmation-results", "prior-inputs", "data-preflight", "historical-model-lock",
                 "coverage-report", "historical-results", "confirmation-results", "prior-monitor-plan", "plan",
                 "monitoring-cache", "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
