"""Recheck conditional confirmation on the previously evaluated 25-person roster."""

import argparse
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize
from .af_wrist_data_preflight import _SCOPES, _cohort_archives, _verify_seal
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _predict, _subject_ids, _write_json
from .af_wrist_monitor_extension import BLOCK_S, _archive_features, _cluster_uncertainty, _person_runs
from .af_wrist_operating import score_decisions
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _HISTORICAL_RESULTS_SHA256, _PREFLIGHT_SHA256, _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset, build_quality_matrices
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_independent import _locked_model, _match_coverage, load_independent_inputs
from .af_wrist_template_study import ROUTE
from .af_wrist_vote_compare import compare_alert_runs
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


RECENT_RESULTS_SHA = "4756a3da959324b0713ff511496ed16130fecc631a6a76cd119de1993ad21057"


def historical_inputs(args, protocol, model_lock):
    receipts = _read_pinned(args.data_preflight, _PREFLIGHT_SHA256, public=True)["sha256"]
    coverage = _read_pinned(args.coverage_report, receipts["independent_coverage_report"], public=True)
    previous = _read_pinned(args.historical_results, _HISTORICAL_RESULTS_SHA256, public=True)
    staged = {key: args.prior_inputs / key for key in _SCOPES}
    seal = _read_pinned(args.prior_inputs / "sealed-input-proof.json", receipts["independent_seal"])
    archives = _cohort_archives(staged, _subject_ids(protocol["test_roster"]))
    _verify_seal(archives, seal, previous["model_lock_sha256"], coverage)
    grid, model, available, _, roster, _, _ = load_independent_inputs(
        model_lock, protocol, staged["parent-features"], staged["stride-raw"], staged["stride-features"], coverage,
    )
    quality = build_quality_matrices(grid, *(staged[key] for key in ("parent-raw", "parent-features", "stride-raw", "stride-features")))
    return grid, model, available, roster, quality, previous


def historical_replay(args, protocol, model_lock, *, confirmation_steps=1):
    grid, model, available, roster, quality, previous = historical_inputs(args, protocol, model_lock)
    predecessor, active = preceding_rows(grid, steps=confirmation_steps)
    scores = _predict(grid.table[ROUTE], available, model["model"])
    cutoffs = np.full(len(scores), model["cutoff"])
    baseline = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    _match_coverage(baseline[0], previous["independent"])
    filtered = confirm_scores(scores, predecessor, active & (quality["quality"][:, -1] < QUALITY_CUTOFF))
    candidate = _held_out_replay(grid, filtered, cutoffs, _IMMEDIATE)
    return {"baseline": baseline[0], "conditional_confirmation": _compare_with_pre_onset(baseline, candidate, protocol),
            "paired_participant_uncertainty": _cluster_uncertainty(baseline, candidate, sorted(roster)),
            "lost_scoreable_decisions": int(np.sum(np.isfinite(scores) & ~np.isfinite(filtered)))}


def replay_on_schedule(schedule, digest, cache, template, model, *, block_seconds=BLOCK_S, confirmation_steps=1):
    baseline_runs, candidate_runs = [], []
    for person, blocks in schedule.items():
        if not blocks:
            continue
        archive = _archive_features(cache / f"{person}.npz", person, blocks, digest, template,
                                    families=("baseline", "quality"), block_seconds=block_seconds)
        quality = archive[3]["quality"][:, :, -1]

        def transform(scores, index):
            previous_row = np.arange(len(scores)) - confirmation_steps
            return confirm_scores(scores, previous_row, quality[index] < QUALITY_CUTOFF)

        baseline_runs.extend(_person_runs(person, archive, model["model"], model["cutoff"], "baseline"))
        candidate_runs.extend(_person_runs(person, archive, model["model"], model["cutoff"], "baseline", transform=transform))

    def replay(runs):
        totals = _summarize({**score_decisions(runs), "monitor_blocks_with_unknown_cutoff_state": 0,
                             "episode_blocks_with_unknown_cutoff_state": 0})
        return totals, runs, runs

    return replay(baseline_runs), replay(candidate_runs)


def additional_replay(args, protocol, model_lock):
    schedule, receipt, digest = verify_schedule(args.plan, args.lock, args.prior_inputs)
    previous = _read_pinned(args.monitoring_results, RECENT_RESULTS_SHA, public=True)
    model, _, _, _ = _locked_model(model_lock, protocol)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    baseline, candidate = replay_on_schedule(schedule, digest, args.monitoring_cache, template, model)
    if digest != previous["plan_sha256"] or baseline[0] != previous["baseline"]:
        raise ValueError("The 250-minute frozen plain baseline must reproduce exactly")
    paired = compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2])
    return {"baseline": baseline[0], "conditional_confirmation": candidate[0], "plan_receipt": receipt,
            "paired_changes": paired, "paired_participant_uncertainty": _cluster_uncertainty(baseline, candidate, list(schedule)),
            "lost_scoreable_decisions": sum(int(np.sum((before.decisions >= 0) & (after.decisions < 0)))
                                             for before, after in zip(baseline[1], candidate[1], strict=True))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report", "historical-results",
                "plan", "monitoring-cache", "monitoring-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    _write_json(args.output, {"scope": "Retrospective causal confirmation on the already reviewed wrist cohort",
                              "quality_boundary": QUALITY_CUTOFF, "fixed_original_model_and_cutoff": True,
                              "historical_episodes_and_monitoring": historical_replay(args, protocol, model_lock),
                              "additional_monitoring": additional_replay(args, protocol, model_lock)})
