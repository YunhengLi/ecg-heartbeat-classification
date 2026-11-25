"""Compare frozen wrist alert rules on newly scheduled continuous half-hour blocks."""

import argparse
from pathlib import Path

import numpy as np

from .af_state import state_durations, state_summary
from .af_wrist_confirmation_retrospective import replay_on_schedule
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_monitor_extension import EARLIER_S, _cluster_uncertainty, read_plan, stage_person
from .af_wrist_monitor_plan import choose_monitor_blocks, per_person_seed
from .af_wrist_quality_confirmation import QUALITY_CUTOFF
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import exclusions, verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_compare import compare_alert_runs
from .afppgecg import load_subject


BLOCK_SECONDS = 30 * 60


def combined_exclusions(prior_inputs, prior_schedule, person):
    newer = np.asarray([block["input_support_s"] for block in prior_schedule[person]]).reshape(-1, 2)
    return np.concatenate((exclusions(prior_inputs, person), newer))


def select_block(source, excluded, person):
    return choose_monitor_blocks(source.segments, source.ecg_duration_seconds, excluded,
                                 per_person_seed(f"holdout-half-hour:{PROTOCOL_SHA}", person),
                                 blocks_per_person=1, block_seconds=BLOCK_SECONDS)


def plan_from_schedule(schedule, previous_digest):
    count = sum(map(len, schedule.values()))
    return {"plan_receipt": {"source_protocol_sha256": PROTOCOL_SHA,
                             "previous_monitor_plan_sha256": previous_digest, "block_seconds": BLOCK_SECONDS,
                             "predecessor_seconds": EARLIER_S, "scheduled_blocks": count,
                             "scheduled_seconds": count * BLOCK_SECONDS,
                             "participants_with_planned_blocks": sum(bool(blocks) for blocks in schedule.values())},
            "private_schedule": schedule}


def verify_long_schedule(path, lock_path, prior_inputs, previous_schedule, previous_digest):
    schedule, receipt, digest = read_plan(path, lock_path, retrospective=True, block_seconds=BLOCK_SECONDS)
    if (receipt != plan_from_schedule(schedule, previous_digest)["plan_receipt"]
            or not receipt["scheduled_blocks"] or any(len(blocks) > 1 for blocks in schedule.values())):
        raise ValueError("Use the sealed, single-block half-hour monitoring plan")
    for person, blocks in schedule.items():
        excluded = combined_exclusions(prior_inputs, previous_schedule, person)
        for block in blocks:
            first, last = block["input_support_s"]
            if np.any((first < excluded[:, 1]) & (last > excluded[:, 0])):
                raise ValueError("New monitoring and its preceding PPG overlap earlier scoring inputs")
    return schedule, receipt, digest


def evaluate(args, protocol, schedule, receipt, digest):
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    if set(schedule) != roster:
        raise ValueError("All new blocks must come from the same out-of-fit evaluation roster")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    baseline, candidate = replay_on_schedule(schedule, digest, args.monitoring_cache, template, model,
                                             block_seconds=BLOCK_SECONDS)
    return {"scope": "Fixed alert comparison on disjoint outcome-blind continuous time from previously scored people",
            "quality_boundary": QUALITY_CUTOFF, "plan_receipt": receipt, "plan_sha256": digest,
            "frozen_model_sha256": BASELINE_SHA, "original": baseline[0], "conditional_confirmation": candidate[0],
            "paired_changes": compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2]),
            "paired_participant_uncertainty": _cluster_uncertainty(baseline, candidate, sorted(schedule),
                                                                   allow_score_coverage_change=True),
            "forward_state_duration": {name: state_summary(state_durations(runs[1]))
                                        for name, runs in (("original", baseline), ("conditional_confirmation", candidate))},
            "scheduled_decisions": sum(len(run.decisions) for run in baseline[1]),
            "scoreable_original": sum(int((run.decisions >= 0).sum()) for run in baseline[1]),
            "scoreable_conditional": sum(int((run.decisions >= 0).sum()) for run in candidate[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "stage", "score"))
    for key in ("plan", "lock", "prior-inputs", "prior-monitor-plan"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("source-dir", "monitoring-cache", "historical-model-lock", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    if args.phase == "plan":
        if args.source_dir is None:
            parser.error("Planning requires the local native ECG and wrist files")
        schedule = {}
        for person in sorted(_subject_ids(protocol["test_roster"])):
            excluded = combined_exclusions(args.prior_inputs, previous, person)
            with load_subject(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat",
                              subject_id=person, include_acceleration=False) as source:
                schedule[person] = select_block(source, excluded, person)
        _write_json(args.plan, plan_from_schedule(schedule, previous_digest))
        return
    schedule, receipt, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    if args.phase == "stage":
        if args.source_dir is None or args.monitoring_cache is None:
            parser.error("Staging requires native files and a private monitoring cache")
        for person, blocks in schedule.items():
            if blocks:
                stage_person(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat", person,
                             blocks, digest, args.monitoring_cache, block_seconds=BLOCK_SECONDS)
        return
    if not all((args.monitoring_cache, args.historical_model_lock, args.ptt_dir, args.output)):
        parser.error("Scoring requires the monitoring cache, original model, PTT calibration and an output path")
    _write_json(args.output, evaluate(args, protocol, schedule, receipt, digest))


if __name__ == "__main__":
    main()
