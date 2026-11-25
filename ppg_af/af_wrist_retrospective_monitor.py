"""Score new monitored times on the historical 25-person wrist roster."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import _summarize
from .af_wrist_direct_fusion import SPECTRAL_ROUTE
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA, VARIANTS_SHA
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_monitor_extension import (
    BLOCK_S, CONTEXT_S, EARLIER_S, _archive_features, _cluster_uncertainty, _person_runs, read_plan, stage_person,
)
from .af_wrist_monitor_plan import choose_monitor_blocks, per_person_seed
from .af_wrist_operating import score_decisions
from .af_wrist_quality_retrospective import VARIANTS, _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_compare import compare_alert_runs
from .afppgecg import load_subject


FAMILIES = ("baseline", *VARIANTS)
BLOCKS_PER_PERSON = 2
SPECTRAL_SHA = "fe639167e77c0c7b6951dc682992b67bf332cf0b6bf945a4f1b295fc6e9b7400"


def exclusions(prior_inputs, person):
    spans = []
    for folder in ("parent-raw", "pending-parent-raw", "preroll", "stride-raw"):
        path = prior_inputs / folder / f"{person}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as raw:
            if str(raw["subject_id"].item()) != person:
                raise ValueError("The previous raw-input archive must match the planned participant")
            starts = raw["input_start_s"] if folder == "stride-raw" else raw["start_s"]
            stops = raw["decision_end_s"] if folder == "stride-raw" else starts + WINDOW_SECONDS
            spans.extend(zip(starts - CONTEXT_S, stops + CONTEXT_S, strict=True))
    if not spans:
        raise ValueError("Every historical held-out participant needs a prior input exclusion clock")
    return np.asarray(spans).reshape(-1, 2)


def make_plan(schedule, protocol_digest):
    count = sum(map(len, schedule.values()))
    return {"plan_receipt": {"scope": "previously scored 25-person roster, disjoint new times",
                            "source_protocol_sha256": protocol_digest, "block_seconds": BLOCK_S,
                            "predecessor_seconds": EARLIER_S, "scheduled_blocks": count,
                            "scheduled_seconds": count * BLOCK_S,
                            "participants_with_planned_blocks": sum(bool(value) for value in schedule.values())},
            "private_schedule": schedule}


def verify_schedule(plan_path, lock_path, prior_inputs):
    schedule, receipt, digest = read_plan(plan_path, lock_path, retrospective=True)
    if receipt != make_plan(schedule, PROTOCOL_SHA)["plan_receipt"] or any(len(blocks) != BLOCKS_PER_PERSON for blocks in schedule.values()):
        raise ValueError("Use the original full two-block plan for the 25-person roster")
    for person, blocks in schedule.items():
        spans = exclusions(prior_inputs, person)
        for block in blocks:
            start, stop = block["input_support_s"]
            if np.any((start < spans[:, 1]) & (stop > spans[:, 0])):
                raise ValueError("New monitoring and its preceding PPG must be disjoint from previously used inputs")
    return schedule, receipt, digest


def freeze_from_receipts(protocol, baseline_path, variants_path):
    baseline = _read_pinned(baseline_path, BASELINE_SHA)
    variants = _read_pinned(variants_path, VARIANTS_SHA)
    if (variants["source_protocol_sha256"] != PROTOCOL_SHA or baseline["source_protocol_sha256"] != PROTOCOL_SHA
            or variants[FEATURE_VERSION_KEY] != baseline[FEATURE_VERSION_KEY] or set(variants["models"]) != set(VARIANTS)):
        raise ValueError("Both model locks must be the original historical development-only fits")
    route, roster, fit, calibration = _locked_model(baseline, protocol)
    return {"baseline": route, **variants["models"]}, roster, fit, calibration


def evaluate(args, protocol, schedule, receipt, digest):
    models, roster, fit, calibration = freeze_from_receipts(protocol, args.historical_model_lock, args.variant_lock)
    if set(schedule) != roster:
        raise ValueError("The supplemental schedule must match the historical held-out model roster")
    hashes = {"template": BASELINE_SHA, "quality_variants": VARIANTS_SHA}
    if args.spectral_model_lock is not None:
        spectral = _read_pinned(args.spectral_model_lock, SPECTRAL_SHA)
        if (spectral["source_protocol_sha256"] != PROTOCOL_SHA or spectral["route"] != SPECTRAL_ROUTE
                or spectral["fit_participants"] != fit or spectral["calibration_participants"] != calibration):
            raise ValueError("Use the development-only spectral comparator locked before supplemental scoring")
        models[SPECTRAL_ROUTE] = spectral["frozen_model"]
        hashes[SPECTRAL_ROUTE] = SPECTRAL_SHA
    families = tuple(models)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    recordings = {person: _archive_features(args.monitoring_cache / f"{person}.npz", person, blocks, digest,
                                            template, families=families) for person, blocks in schedule.items()}
    replays = {}
    for name, route in models.items():
        runs = tuple(run for person, archive in recordings.items()
                     for run in _person_runs(person, archive, route["model"], route["cutoff"], name))
        scored = score_decisions(runs)
        totals = _summarize({**scored, "monitor_blocks_with_unknown_cutoff_state": 0,
                             "episode_blocks_with_unknown_cutoff_state": 0})
        replays[name] = totals, runs, runs
    baseline = replays["baseline"]
    comparisons = {}
    for name in families[1:]:
        candidate = replays[name]
        if any(not np.array_equal(before.decisions >= 0, after.decisions >= 0)
               for before, after in zip(baseline[1], candidate[1], strict=True)):
            raise ValueError("The original frozen classifiers must be scored on identical additional PPG inputs")
        metrics = _compare_with_pre_onset(baseline, candidate, protocol) if baseline[0]["complete_episodes"] else {
            "summary": candidate[0], "sensitivity_change": None, "within_sensitivity_tolerance": None,
            "paired_changes": compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2]),
        }
        comparisons[name] = {**metrics, "paired_participant_uncertainty": _cluster_uncertainty(baseline, candidate, list(schedule))}
    coverage = {"scheduled_decisions": sum(len(run.decisions) for run in baseline[1]),
                "observed_wrist_decisions": sum(int(run.ppg_available.sum()) for run in baseline[1]),
                "scoreable_decisions_per_route": sum(int((run.decisions >= 0).sum()) for run in baseline[1])}
    return {"scope": "New times on the previously compared retrospective 25-person roster", "plan_receipt": receipt,
            "plan_sha256": digest, "fit_participants": fit, "calibration_participants": calibration, "coverage": coverage,
            "baseline": baseline[0], "variants": comparisons,
            "pinned_model_hashes": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "stage", "score"))
    for key in ("plan", "lock", "prior-inputs"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("source-dir", "monitoring-cache", "historical-model-lock", "variant-lock", "spectral-model-lock", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    if args.phase == "plan":
        if args.source_dir is None:
            parser.error("Planning requires the original local MAT sources")
        schedule = {}
        for person in sorted(_subject_ids(protocol["test_roster"])):
            with load_subject(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat",
                              subject_id=person, include_acceleration=False) as source:
                schedule[person] = choose_monitor_blocks(source.segments, source.ecg_duration_seconds,
                                                         exclusions(args.prior_inputs, person),
                                                         per_person_seed(f"holdout:{PROTOCOL_SHA}", person),
                                                         blocks_per_person=BLOCKS_PER_PERSON)
        _write_json(args.plan, make_plan(schedule, PROTOCOL_SHA))
        return
    schedule, receipt, digest = verify_schedule(args.plan, args.lock, args.prior_inputs)
    if args.phase == "stage":
        if args.source_dir is None or args.monitoring_cache is None:
            parser.error("Staging requires local MAT sources and a private monitoring cache directory")
        for person, blocks in schedule.items():
            stage_person(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat", person, blocks,
                         digest, args.monitoring_cache)
        return
    if not all((args.monitoring_cache, args.historical_model_lock, args.variant_lock, args.ptt_dir, args.output)):
        parser.error("Scoring requires both model locks, the monitoring cache, and an output path")
    _write_json(args.output, evaluate(args, protocol, schedule, receipt, digest))


if __name__ == "__main__":
    main()
