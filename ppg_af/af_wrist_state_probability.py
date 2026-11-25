"""Compare independent log-odds calibration for wrist AF-time probabilities."""

import argparse
from pathlib import Path

import numpy as np

from .af_state_probability import fit_logit_map, logits, summarize_probabilities, time_targets
from .af_wrist_clock_study import scoring_runs
from .af_wrist_exercise import BASELINE_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import BLOCK_S, _archive_features, read_plan
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_operating import AFScoredRun
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_study import STRIDE_SECONDS


HALF_HOUR_RESULTS_SHA = "ea8e3364ee3078b978858ea79b23916c8d02572c7337533b7a0d42a59c0d3ca7"


def monitor_probabilities(schedule, digest, cache, template, model, block_seconds):
    for person, blocks in schedule.items():
        if not blocks:
            continue
        starts, reference, episodes, matrices, available = _archive_features(
            cache / f"{person}.npz", person, blocks, digest, template, families=("baseline",), block_seconds=block_seconds,
        )
        for start, valid, features, present in zip(starts, reference, matrices["baseline"], available, strict=True):
            stride_rows = len(valid)
            values = features[-stride_rows:]
            scores = _predict(values, np.isfinite(values).all(axis=1), model)
            yield AFScoredRun(float(start), STRIDE_SECONDS, episodes, scores, valid, present[-stride_rows:], person)


def support(truth, people):
    return {"scoreable_seconds": len(truth) * STRIDE_SECONDS, "AF_seconds": float(truth.sum() * STRIDE_SECONDS),
            "participants": len(np.unique(people)), "participants_with_AF_seconds": len(np.unique(people[truth > 0]))}


def study(args):
    grid, inputs, _, protocol, _, _ = load_motion_development(args)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    frozen, held_people, _, _ = _locked_model(model_lock, protocol)
    calibration_people = _subject_ids(model_lock["calibration_subjects"])
    eligible = _shared_features(grid.table) & np.isin(grid.subjects, sorted(calibration_people))
    original_scores = _predict(inputs["baseline"], eligible, frozen["model"])
    original_monitoring, _ = scoring_runs(grid, original_scores, calibration_people)
    earlier, _, earlier_digest = read_plan(args.extension_plan, args.lock)
    extension = {person: earlier[person] for person in sorted(calibration_people)}
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    additional = tuple(monitor_probabilities(extension, earlier_digest, args.extension_cache, template, frozen["model"], BLOCK_S))
    calibration = {"original_calibration_wear": original_monitoring,
                   "original_and_additional_wear": (*original_monitoring, *additional)}
    maps, training = {}, {}
    for name, runs in calibration.items():
        scores, truth, participants = time_targets(runs, STRIDE_SECONDS)
        if set(participants) != calibration_people:
            raise ValueError("All probability maps must use the original designated calibration participants")
        training[name] = support(truth, participants)
        maps[name] = fit_logit_map(scores, truth)

    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    recent = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    held = monitor_probabilities(schedule, digest, args.monitoring_cache, template, frozen["model"], BLOCK_SECONDS)
    scores, truth, participants = time_targets(held, STRIDE_SECONDS)
    test_support = support(truth, participants)
    expected = recent["forward_state_duration"]["original"]
    if (digest != recent["plan_sha256"] or set(participants) != held_people or held_people & calibration_people
            or test_support["scoreable_seconds"] != expected["scoreable_seconds"]
            or not np.isclose(test_support["AF_seconds"], expected["af_seconds_with_state"], atol=1e-8, rtol=0)):
        raise ValueError("Use the same held-out reference and probability coverage as the sealed half-hour replay")
    probabilities = {"original": scores, **{name: model.predict_proba(logits(scores))[:, 1] for name, model in maps.items()}}
    for name, data in training.items():
        probabilities[f"{name}_prevalence_only"] = np.full(len(scores), data["AF_seconds"] / data["scoreable_seconds"])
    results = {name: {**summarize_probabilities(estimate, truth, participants),
                      "mean_probability_during_AF": float(np.average(estimate, weights=truth)),
                      "mean_probability_during_non_AF": float(np.average(estimate, weights=1 - truth))}
               for name, estimate in probabilities.items()}
    return {"scope": "Forward AF time estimates from the frozen wrist model on newly scored half-hour times",
            "plan_sha256": digest, "pinned_model_sha256": BASELINE_SHA, "calibration_support": training,
            "held_out_support": test_support, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "historical-model-lock", "ptt-dir", "extension-plan", "extension-cache", "prior-monitor-plan",
                 "prior-inputs", "plan", "monitoring-cache", "half-hour-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, study(args))
