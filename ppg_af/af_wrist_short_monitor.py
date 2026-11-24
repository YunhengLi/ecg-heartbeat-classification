"""Replay the frozen short-template models over representative half-hour wrist blocks."""

import argparse
from pathlib import Path

import numpy as np

from .af_state import state_durations, state_summary
from .af_wrist_clock_study import _summarize
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import _archive_features, _cluster_uncertainty, _person_runs
from .af_wrist_operating import score_decisions
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_short_retrospective import REARMED_SHORT, SHORT_REARM, freeze_short, short_matrix
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_compare import compare_alert_runs


def archive_for_both(path, person, blocks, digest, template):
    starts, valid, episodes, matrices, ppg = _archive_features(
        path, person, blocks, digest, template, families=("baseline", "quality"), block_seconds=BLOCK_SECONDS,
    )
    with np.load(path, allow_pickle=False) as source:
        raw = source["raw_ppg"]
    baseline = matrices["baseline"]
    flat = baseline.reshape(-1, baseline.shape[-1])
    old = np.isfinite(flat).all(axis=1)
    short, both = short_matrix(raw.reshape(len(flat), -1), old, template, flat.shape)
    aligned = np.where(both[:, None], flat, np.nan)
    matrices.update(short=short.reshape(baseline.shape), matched=aligned.reshape(baseline.shape))
    return (starts, valid, episodes, matrices, ppg), int(old.sum() - both.sum())


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    new_model, new_cutoffs, fit_receipt = freeze_short(args, protocol, model_lock)
    original, roster, _, _ = _locked_model(model_lock, protocol)
    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    prior = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    if set(schedule) != roster or digest != prior["plan_sha256"]:
        raise ValueError("Use the same preselected half-hour test exposures")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    configurations = {
        "original_full": ("baseline", original["model"], original["cutoff"], False),
        "original_full_with_quality_confirmation": ("baseline", original["model"], original["cutoff"], True),
        "original": ("matched", original["model"], original["cutoff"], False),
        "original_with_quality_confirmation": ("matched", original["model"], original["cutoff"], True),
    }
    for name, cutoff in new_cutoffs.items():
        configurations[name] = "short", new_model, cutoff, False
        configurations[f"{name}_with_quality_confirmation"] = "short", new_model, cutoff, True
    runs = {name: [] for name in (*configurations, REARMED_SHORT)}
    lost = 0
    for person, blocks in schedule.items():
        if not blocks:
            continue
        archive, removed = archive_for_both(args.monitoring_cache / f"{person}.npz", person, blocks, digest, template)
        lost += removed
        needs = archive[3]["quality"][:, :, -1] < QUALITY_CUTOFF

        def confirmed(scores, index):
            return confirm_scores(scores, np.arange(len(scores)) - 1, needs[index])

        for name, (family, model, cutoff, confirm) in configurations.items():
            runs[name].extend(_person_runs(person, archive, model, cutoff, family, transform=confirmed if confirm else None))
        runs[REARMED_SHORT].extend(_person_runs(person, archive, new_model, new_cutoffs["short_matched"], "short",
                                               transform=confirmed, replay=SHORT_REARM))
    totals = {}
    for name, individual in runs.items():
        summary = _summarize({**score_decisions(individual), "monitor_blocks_with_unknown_cutoff_state": 0,
                              "episode_blocks_with_unknown_cutoff_state": 0})
        totals[name] = (summary, individual, individual)
    if totals["original_full"][0] != prior["original"] or totals["original_full_with_quality_confirmation"][0] != prior["conditional_confirmation"]:
        raise ValueError("The original full-support and adjacent half-hour outcomes must reproduce")
    output = {}
    for name, (summary, individual, _) in totals.items():
        output[name] = {"summary": summary, "forward_state": state_summary(state_durations(individual)),
                        "scoreable_decisions": sum(int(np.sum(run.decisions >= 0)) for run in individual)}
    for name in new_cutoffs:
        for suffix in ("", "_with_quality_confirmation"):
            route = f"{name}{suffix}"
            baseline = totals[f"original{suffix}"]
            candidate = totals[route]
            output[route]["paired_changes"] = compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2])
            output[route]["uncertainty"] = _cluster_uncertainty(baseline, candidate, sorted(roster))
    for reference in ("short_matched_with_quality_confirmation", "original_with_quality_confirmation"):
        baseline, candidate = totals[reference], totals[REARMED_SHORT]
        output[REARMED_SHORT][f"comparison_to_{reference}"] = {
            "paired_changes": compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2]),
            "uncertainty": _cluster_uncertainty(baseline, candidate, sorted(roster)),
        }
    return {"scope": "Same-duration half-hour wrist replay of development-frozen shorter-input models",
            "plan_sha256": digest, "fitting_receipt": fit_receipt, "unmatched_original_input_rows": lost, "results": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "prior-inputs", "historical-model-lock", "prior-monitor-plan", "plan",
                 "monitoring-cache", "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
