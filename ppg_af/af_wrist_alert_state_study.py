"""Compare fixed alert entry and release rules on the original wrist cohort."""

import argparse
from pathlib import Path

import numpy as np

from .af_events import AFRun
from .af_state import state_durations, state_summary
from .af_wrist_alert_state import replay_alert_state, replay_score_block
from .af_wrist_confirmation_evaluation import summarize_episodes, summarize_wear
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_spacing import CONFIRMATION_RESULTS_SHA, DEVELOPMENT_CONFIRMATION_SHA
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import BLOCK_S, _archive_features
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, _locked_model
from .af_wrist_template_study import DEFAULT_C, ROUTE, fit_template
from .af_wrist_vote_study import STRIDE_SECONDS


REFERENCE = "quality_confirmation"
POLICIES = {REFERENCE: (0.0, 0.0), "entry": (0.10, 0.0), "release": (0.0, 0.10), "entry_and_release": (0.10, 0.10)}


def _confirmed(scores, quality, previous, active):
    return confirm_scores(scores, previous, active & (quality < QUALITY_CUTOFF))


def _replay_grid(grid, scores, cutoffs, quality):
    filtered = _confirmed(scores, quality, *preceding_rows(grid))
    return {name: replay_alert_state(grid, filtered, cutoffs, entry_margin=entry, release_margin=release)
            for name, (entry, release) in POLICIES.items()}


def _time_metrics(runs):
    summary = state_summary(state_durations(runs))
    return {key: summary[key] for key in ("ecg_observed_seconds", "ecg_af_seconds", "scoreable_seconds",
                                          "scoreable_coverage", "non_af_seconds_with_positive_state",
                                          "non_af_time_false_positive_rate", "af_time_sensitivity")}


def development(args, protocol):
    grid, quality, _, development_lock, _, _ = load_motion_development(args)
    if development_lock != protocol:
        raise ValueError("Development and historical replay must use the same source protocol lock")
    pinned = _read_pinned(args.development_confirmation_results, DEVELOPMENT_CONFIRMATION_SHA, public=True)
    results = {}
    for seed in SEEDS:
        fitted = fit_template(grid, quality["baseline"], protocol, DEFAULT_C, seed=seed)
        scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
        replays = _replay_grid(grid, scores, cutoffs, quality["quality"][:, -1])
        baseline = replays[REFERENCE]
        expected = pinned["comparisons"][str(seed)]["plain_template"]["lower_quality_only"]["fixed_cutoffs"]["summary"]
        if baseline[0] != expected:
            raise ValueError("The published development confirmation replay must reproduce")
        results[str(seed)] = {name: {"comparison": _compare_with_pre_onset(baseline, replay, protocol),
                                     "wear_state_time": _time_metrics(replay[2])} for name, replay in replays.items()}
        print(f"Development {seed}: " + ", ".join(f"{name} {replay[0]['episodes_detected']}/{replay[0]['wrist_false_alerts']}"
                                                  for name, replay in replays.items()), flush=True)
    return results


def _wear_replays(schedule, digest, cache, template, model, *, block_seconds):
    result = {name: [] for name in POLICIES}
    cutoff = model["cutoff"]
    for person, blocks in schedule.items():
        if not blocks:
            continue
        starts, references, episodes, features, availability = _archive_features(
            cache / f"{person}.npz", person, blocks, digest, template,
            families=("baseline", "quality"), block_seconds=block_seconds,
        )
        rows = zip(starts, references, features["baseline"], features["quality"][:, :, -1], availability, strict=True)
        for start, reference, matrix, quality, ppg in rows:
            probabilities = _predict(matrix, np.isfinite(matrix).all(axis=1), model["model"])
            positions = np.arange(len(probabilities))
            scores = _confirmed(probabilities, quality, positions - 1, np.ones(len(positions), dtype=bool))
            history = len(scores) - len(reference)
            cells = scores[history:]
            decisions = np.where(np.isfinite(cells), cells >= cutoff, -1).astype(np.int8)
            run = AFRun(float(start), STRIDE_SECONDS, episodes, decisions, reference, ppg[history:], person)
            for name, (entry, release) in POLICIES.items():
                scored, _ = replay_score_block(run, cells, scores[:history], cutoff, entry_margin=entry, release_margin=release)
                result[name].append(scored)
    return result


def _present_wear(original, candidate, people, draws):
    paired = summarize_wear(original, candidate, people, draws)
    if paired["original"]["monitorable_seconds"] != paired["confirmation"]["monitorable_seconds"]:
        raise ValueError("The candidate changed monitorable time despite unchanged score support")
    return {"ecg_observed_seconds": paired["ecg_observed_wrist_seconds"],
            **paired["confirmation"], "paired_changes_from_quality_confirmation": paired["paired_changes"],
            "rhythm_state_time": _time_metrics(candidate)}


def retrospective(args, protocol):
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    pinned_historical = _read_pinned(args.historical_confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)
    grid, model, available, roster, quality, _ = historical_inputs(args, protocol, model_lock)
    probabilities = _predict(grid.table[ROUTE], available, model["model"])
    historical = _replay_grid(grid, probabilities, np.full(len(probabilities), model["cutoff"]), quality["quality"][:, -1])
    baseline = historical[REFERENCE]
    if baseline[0] != pinned_historical["historical_episodes_and_monitoring"]["conditional_confirmation"]["summary"]:
        raise ValueError("The historical quality confirmation replay must reproduce")
    short, _, short_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    long, _, long_digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, short, short_digest)
    model, frozen_roster, _, _ = _locked_model(model_lock, protocol)
    if set(short) != roster or set(long) != roster or frozen_roster != roster:
        raise ValueError("Both wear schedules must keep the original out-of-fit roster")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    wear_short = _wear_replays(short, short_digest, args.prior_monitoring_cache, template, model, block_seconds=BLOCK_S)
    wear_long = _wear_replays(long, long_digest, args.monitoring_cache, template, model, block_seconds=BLOCK_SECONDS)
    prior_long = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    paired_wear = {name: wear_short[name] + wear_long[name] for name in POLICIES}
    people = sorted(roster)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(people), size=(BOOTSTRAP_REPETITIONS, len(people)))

    monitor = {name: _present_wear(paired_wear[REFERENCE], runs, people, draws) for name, runs in paired_wear.items()}
    old_short = pinned_historical["additional_monitoring"]["conditional_confirmation"]
    old_long = prior_long["conditional_confirmation"]
    for runs, published, expected_seconds in ((wear_short[REFERENCE], old_short, 14490),
                                               (wear_long[REFERENCE], old_long, 44640)):
        observed = _present_wear(runs, runs, people, draws)
        if (observed["false_starts"] != published["wrist_false_alerts"]
                or observed["ecg_observed_seconds"] != published["wrist_ecg_observed_seconds"]
                or observed["ecg_observed_seconds"] != expected_seconds):
            raise ValueError("Published scheduled wear results must reproduce before comparing policies")
    if prior_long["plan_sha256"] != long_digest:
        raise ValueError("The half-hour plan does not match the published results")
    if (monitor[REFERENCE]["false_starts"], monitor[REFERENCE]["monitorable_seconds"]) != (210, 58100):
        raise ValueError("The published joint wear evaluation must reproduce")

    results = {}
    for name, replay in historical.items():
        episodes = summarize_episodes(baseline[1], replay[1], people, draws)
        comparison = _compare_with_pre_onset(baseline, replay, protocol)
        tolerance = protocol["sensitivity_tolerance"]
        low, high = episodes["paired_sensitivity_change"]["percentile_95"]
        results[name] = {"historical_episodes": {**episodes["confirmation"],
                                               "complete": episodes["complete_ecg_episodes"],
                                               "participants_with_complete": episodes["participants_with_complete_episodes"],
                                               "paired_sensitivity_change": episodes["paired_sensitivity_change"],
                                               "paired_detections": episodes["paired_episode_changes"],
                                               "similarity_interval_within_tolerance": low >= -tolerance and high <= tolerance,
                                               "reaches_development_sensitivity_target": episodes["confirmation"]["sensitivity"]
                                               >= protocol["target_sensitivity"]},
                         "historical_pre_onset": comparison["pre_onset_comparison"],
                         "historical_wear": replay[0], "pooled_additional_wear": monitor[name]}
        print(f"Historical {name}: {replay[0]['episodes_detected']} complete detections, "
              f"{monitor[name]['false_starts']} pooled wear false starts", flush=True)
    return {"participants": len(people), "bootstrap_resamples": len(draws), "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("lock", "development-raw", "development-features", "stride-raw", "stride-features", "clock-results",
                "quality-results", "development-confirmation-results", "historical-confirmation-results", "prior-inputs",
                "data-preflight", "historical-model-lock", "coverage-report", "historical-results", "prior-monitor-plan",
                "prior-monitoring-cache", "plan", "monitoring-cache", "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    report = {"scope": "Fixed-policy development and retrospective replay of the original wrist cohort",
              "source_model_lock_sha256": BASELINE_SHA, "quality_boundary": QUALITY_CUTOFF,
              "policy_margins": {name: {"entry": entry, "release": release} for name, (entry, release) in POLICIES.items()},
              "episode_sensitivity_target": protocol["target_sensitivity"],
              "episode_sensitivity_tolerance": protocol["sensitivity_tolerance"],
              "development": development(args, protocol), "retrospective": retrospective(args, protocol)}
    _write_json(args.output, report)


if __name__ == "__main__":
    main()
