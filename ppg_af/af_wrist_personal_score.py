"""Test causal unlabeled wearer normalization of frozen AF scores."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

from .af_events import AFRun
from .af_state import state_durations, state_summary
from .af_state_probability import logits
from .af_wrist_clock_study import _summarize
from .af_wrist_exercise import BASELINE_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import MONITOR_KIND, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import _archive_features, _cluster_uncertainty
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_operating import score_decisions
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_compare import compare_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE


BASELINE_SECONDS = 300
BASELINE_ROWS = BASELINE_SECONDS // STRIDE_SECONDS
MIN_BASELINE_ROWS = 20
SHRINKAGE = 0.5
ROUTES = ("original", "fixed_wearer_center", "rolling_wearer_center")


def center(scores):
    valid = scores[np.isfinite(scores)]
    return float(np.median(logits(valid))) if len(valid) >= MIN_BASELINE_ROWS else None


def adapt(scores, source_center):
    """Return shifts for exactly the decisions after a common five-minute warm-up."""
    if len(scores) <= BASELINE_ROWS:
        raise ValueError("The evaluation block must extend beyond the wearer's baseline")
    first = center(scores[:BASELINE_ROWS])
    past = [center(scores[row - BASELINE_ROWS:row]) for row in range(BASELINE_ROWS, len(scores))]
    offsets = {}
    for key, centers in (("fixed_wearer_center", [first] * len(past)), ("rolling_wearer_center", past)):
        offsets[key] = np.array([0.0 if value is None else SHRINKAGE * (value - source_center) for value in centers])
    return offsets, {"fixed_baseline_available": first is not None, "rolling_baseline_rows": sum(value is not None for value in past)}


def replay(start, valid, episodes, availability, scores, cutoff, person, *, begin=0):
    preroll = len(scores) - len(valid)
    raw = np.where(np.isfinite(scores), scores >= cutoff, -1).astype(np.int8)
    evaluation = AFRun(float(start + begin * STRIDE_SECONDS), STRIDE_SECONDS, episodes, raw[preroll + begin:],
                       valid[begin:], availability[preroll + begin:], person)
    (scored,), _ = _IMMEDIATE((evaluation,), prior_decisions=lambda _: raw[:preroll + begin])
    return scored


def replay_totals(runs):
    return _summarize({**score_decisions(runs), "monitor_blocks_with_unknown_cutoff_state": 0,
                       "episode_blocks_with_unknown_cutoff_state": 0})


def study(args):
    grid, quality, _, protocol, _, _ = load_motion_development(args)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    fitting = _subject_ids(model_lock["fit_subjects"])
    wear = (_shared_features(grid.table) & (grid.table["run_kind"] == MONITOR_KIND)
            & np.isin(grid.subjects, sorted(fitting)))
    source_scores = _predict(quality["baseline"], wear, model["model"])
    fitting_with_wear = np.unique(grid.subjects[wear])
    source_medians = [np.median(logits(source_scores[wear & (grid.subjects == person)])) for person in fitting_with_wear]
    if not source_medians or not np.isfinite(source_medians).all() or fitting & roster:
        raise ValueError("The reference center must use only the frozen model's fitting wearers")
    source_center = float(np.median(source_medians))

    earlier, _, earlier_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, earlier, earlier_digest)
    previous = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    if set(schedule) != roster or digest != previous["plan_sha256"]:
        raise ValueError("Wearer normalization needs the same original half-hour roster and clock")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    all_original, runs = [], {route: [] for route in ROUTES}
    fixed_offsets = []
    fixed_support = rolling_support = rolling_total = 0
    for person, blocks in schedule.items():
        if not blocks:
            continue
        starts, reference, episodes, features, available = _archive_features(
            args.monitoring_cache / f"{person}.npz", person, blocks, digest, template, families=("baseline",),
            block_seconds=BLOCK_SECONDS,
        )
        for start, valid, values, availability in zip(starts, reference, features["baseline"], available, strict=True):
            scores = _predict(values, np.isfinite(values).all(axis=1), model["model"])
            all_original.append(replay(start, valid, episodes, availability, scores, model["cutoff"], person))
            evaluation_scores = scores[-len(valid):]
            offsets, support = adapt(evaluation_scores, source_center)
            fixed_support += support["fixed_baseline_available"]
            rolling_support += support["rolling_baseline_rows"]
            rolling_total += len(offsets["rolling_wearer_center"])
            if support["fixed_baseline_available"]:
                fixed_offsets.append(offsets["fixed_wearer_center"][0])
            runs["original"].append(replay(start, valid, episodes, availability, scores, model["cutoff"], person, begin=BASELINE_ROWS))
            for name, shift in offsets.items():
                updated = scores.copy()
                recent_scores = updated[-len(shift):]
                selected = np.isfinite(recent_scores) & (shift != 0)
                recent_scores[selected] = expit(logits(recent_scores[selected])[:, 0] - shift[selected])
                runs[name].append(replay(start, valid, episodes, availability, updated, model["cutoff"], person, begin=BASELINE_ROWS))
    if replay_totals(all_original) != previous["original"]:
        raise ValueError("The original half-hour replay must reproduce before excluding the common warm-up")

    output = {name: {"summary": replay_totals(replayed), "forward_state": state_summary(state_durations(replayed))}
              for name, replayed in runs.items()}
    baseline = (output["original"]["summary"], runs["original"], runs["original"])
    for name in ROUTES[1:]:
        candidate = (output[name]["summary"], runs[name], runs[name])
        output[name]["paired_changes"] = compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2])
        output[name]["paired_uncertainty"] = _cluster_uncertainty(baseline, candidate, sorted(roster))
    return {"scope": "Label-blind causal AF log-odds centering after a common half-hour block warm-up",
            "warmup_seconds": BASELINE_SECONDS, "minimum_scoreable_baseline_decisions": MIN_BASELINE_ROWS,
            "shrinkage_weight": SHRINKAGE, "fitting_wear_participants": len(fitting_with_wear),
            "model_fitting_participants": len(fitting),
            "fitting_wear_rows": int(wear.sum()), "fitting_wear_median_log_odds_center": source_center,
            "held_out_blocks_with_fixed_center": fixed_support, "held_out_rows_with_rolling_center": rolling_support,
            "held_out_post_warmup_rows": rolling_total,
            "fixed_offset_log_odds_quantiles_10_50_90": np.quantile(fixed_offsets, (.1, .5, .9)).tolist() if fixed_offsets else [],
            "plan_sha256": digest, "results": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "historical-model-lock", "prior-monitor-plan", "plan",
                 "monitoring-cache", "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, study(args))
