"""Evaluate frozen wrist AF confirmation on fixed episode and wear schedules."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .af_events import _grid, evaluate_af
from .af_state import state_durations
from .af_wrist_confirmation_retrospective import (
    historical_replays, RECENT_RESULTS_SHA, replay_on_schedule,
)
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_quality_confirmation import QUALITY_CUTOFF
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, _locked_model
from .af_wrist_vote_compare import compare_alert_runs


def _participant_wear(runs, people):
    by_person = defaultdict(list)
    for run in runs:
        by_person[run.subject_id].append(run)
    if set(by_person) - set(people):
        raise ValueError("Wear runs include a participant outside the locked roster")
    durations = state_durations(runs)
    rows = []
    for person in people:
        person_runs = by_person[person]
        if not person_runs:
            rows.append((0, 0, 0, 0))
            continue
        scheduled = sum(len(run.decisions) * run.window_s for run in person_runs)
        observed, _, scoreable, *_ = durations[("participant", person)]
        events = evaluate_af(person_runs)
        if not np.isclose(observed, events["ecg_reference_seconds"]):
            raise ValueError("Wear and alert calculations must share the same ECG exposure")
        rows.append((scheduled, observed, scoreable, events["false_alert_starts"]))
    return np.asarray(rows)


def _paired_rate(numerator, denominator, draws, *, scale=1):
    sampled_denominator = denominator[draws].sum(axis=1)
    valid = sampled_denominator > 0
    if not valid.any():
        return None
    rates = scale * numerator[draws[valid]].sum(axis=1) / sampled_denominator[valid]
    return {"point": float(scale * numerator.sum() / denominator.sum()),
            "percentile_95": np.percentile(rates, [2.5, 97.5]).tolist(),
            "valid_draws": int(valid.sum())}


def _check_shared_support(original, candidate):
    for left, right in zip(original, candidate, strict=True):
        edges, _, ecg, ppg = _grid(left)
        other_edges, _, other_ecg, other_ppg = _grid(right)
        if (left.subject_id != right.subject_id or not np.array_equal(edges, other_edges)
                or not np.array_equal(ecg, other_ecg) or not np.array_equal(ppg, other_ppg)
                or not np.array_equal(left.af_intervals_s, right.af_intervals_s)):
            raise ValueError("Both policies must use identical reference and native PPG support")


def summarize_wear(original, candidate, people, draws):
    _check_shared_support(original, candidate)
    before, after = (_participant_wear(runs, people) for runs in (original, candidate))
    scheduled, observed, _, _ = before.T
    if not observed.sum():
        raise ValueError("The wear comparison needs ECG-observed time")

    result = {"scheduled_wrist_seconds": float(scheduled.sum()),
              "ecg_observed_wrist_seconds": float(observed.sum()),
              "participants_with_ecg_exposure": int(np.count_nonzero(observed))}
    for name, rows in (("original", before), ("confirmation", after)):
        _, _, scoreable, false = rows.T
        result[name] = {"false_starts": int(false.sum()),
                        "false_starts_per_observed_hour": float(3600 * false.sum() / observed.sum()),
                        "monitorable_seconds": float(scoreable.sum()),
                        "coverage_of_observed_time": _paired_rate(scoreable, observed, draws),
                        "coverage_of_scheduled_time": float(scoreable.sum() / scheduled.sum())}
    _, _, coverage_change, false_change = (after - before).T
    result["paired_changes"] = {
        "false_starts_per_observed_hour": _paired_rate(false_change, observed, draws, scale=3600),
        "coverage_of_observed_time": _paired_rate(coverage_change, observed, draws),
    }
    return result


def _episode_counts(runs, people):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.subject_id].append(run)
    if set(grouped) - set(people):
        raise ValueError("Episode runs include a participant outside the locked roster")
    rows = []
    for person in people:
        if not grouped[person]:
            rows.append((0, 0))
            continue
        metrics = evaluate_af(grouped[person])
        rows.append((metrics["detected_complete_episodes"], metrics["complete_reference_episodes"]))
    return np.asarray(rows)


def summarize_episodes(original, candidate, people, draws):
    _check_shared_support(original, candidate)
    baseline, total = _episode_counts(original, people).T
    confirmed, candidate_total = _episode_counts(candidate, people).T
    if not np.array_equal(total, candidate_total) or not total.sum():
        raise ValueError("Episode policies must share a nonempty complete-reference denominator")
    has_episode = total > 0
    result = {"complete_ecg_episodes": int(total.sum()), "participants_with_complete_episodes": int(has_episode.sum()),
              "largest_participant_episode_count": int(total.max()),
              "paired_sensitivity_change": _paired_rate(confirmed - baseline, total, draws),
              "paired_episode_changes": compare_alert_runs(original, candidate, (), ())["episodes"]}
    for name, detected in (("original", baseline), ("confirmation", confirmed)):
        result[name] = {"detected": int(detected.sum()), "sensitivity": float(detected.sum() / total.sum()),
                        "participant_average_sensitivity": float(np.mean(detected[has_episode] / total[has_episode]))}
    return result


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    historical, confirmed, people, _ = historical_replays(args, protocol, model_lock)
    earlier, _, earlier_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    long, _, long_digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, earlier, earlier_digest)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    if set(earlier) != roster or set(long) != roster:
        raise ValueError("Both wear schedules must cover the frozen evaluation roster")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    short_replays = replay_on_schedule(earlier, earlier_digest, args.prior_monitoring_cache, template, model)
    long_replays = replay_on_schedule(long, long_digest, args.monitoring_cache, template, model, block_seconds=BLOCK_SECONDS)
    short_prior = _read_pinned(args.prior_monitoring_results, RECENT_RESULTS_SHA, public=True)
    long_prior = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    if (earlier_digest != short_prior["plan_sha256"] or short_replays[0][0] != short_prior["baseline"]
            or long_digest != long_prior["plan_sha256"] or long_replays[0][0] != long_prior["original"]
            or long_replays[1][0] != long_prior["conditional_confirmation"]):
        raise ValueError("Published wear comparisons must reproduce before pooling")

    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(people), size=(BOOTSTRAP_REPETITIONS, len(people)))
    episodes = summarize_episodes(historical[1], confirmed[1], people, draws)
    wear_scopes = {"historical": (historical[2], confirmed[2]),
                   "two_five_minute_blocks": tuple(run[2] for run in short_replays),
                   "half_hour_block": tuple(run[2] for run in long_replays),
                   "pooled_additional_wear": tuple(short[2] + long_run[2]
                                                  for short, long_run in zip(short_replays, long_replays, strict=True))}
    wear = {name: summarize_wear(before, after, people, draws) for name, (before, after) in wear_scopes.items()}
    pooled = wear["pooled_additional_wear"]
    candidate_coverage_low = pooled["confirmation"]["coverage_of_observed_time"]["percentile_95"][0]
    coverage_change_low = pooled["paired_changes"]["coverage_of_observed_time"]["percentile_95"][0]
    false_start_change_high = pooled["paired_changes"]["false_starts_per_observed_hour"]["percentile_95"][1]
    lowest_sensitivity = min(episodes[name]["sensitivity"] for name in ("original", "confirmation"))
    sensitivity_low, sensitivity_high = episodes["paired_sensitivity_change"]["percentile_95"]
    tolerance = protocol["sensitivity_tolerance"]
    limits = {"episode_sensitivity_target": protocol["target_sensitivity"], "max_sensitivity_difference": tolerance,
              "min_observed_coverage": args.min_coverage, "max_observed_coverage_loss": args.max_coverage_loss}
    checks = {"both_reach_sensitivity_target": lowest_sensitivity >= limits["episode_sensitivity_target"],
              "paired_sensitivity_interval_inside_tolerance": sensitivity_low >= -tolerance and sensitivity_high <= tolerance,
              "pooled_false_start_interval_favors_confirmation": false_start_change_high < 0,
              "pooled_confirmation_coverage_meets_minimum": candidate_coverage_low >= limits["min_observed_coverage"],
              "pooled_coverage_loss_within_limit": coverage_change_low >= -limits["max_observed_coverage_loss"]}
    return {"scope": "Retrospective fixed-policy comparison on the same previously scored participants",
            "participants": len(people), "quality_boundary": QUALITY_CUTOFF, "model_lock_sha256": BASELINE_SHA,
            "resamples": len(draws), "seed": BOOTSTRAP_SEED, "limits": limits, "historical_complete_episodes": episodes,
            "wear": wear, "retrospective_checks": checks, "retrospective_gate_pass": all(checks.values()),
            "independent_validation_available": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report", "historical-results",
                 "prior-monitor-plan", "prior-monitoring-cache", "prior-monitoring-results", "plan", "monitoring-cache",
                 "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, required=True)
    parser.add_argument("--max-coverage-loss", type=float, required=True)
    args = parser.parse_args()
    if not all(0 <= value <= 1 for value in (args.min_coverage, args.max_coverage_loss)):
        parser.error("Coverage limits must be in [0, 1]")
    try:
        _write_json(args.output, evaluate(args))
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Frozen confirmation evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
