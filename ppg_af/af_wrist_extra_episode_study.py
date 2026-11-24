"""Replay one new native development episode with its original held-out fit."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_events import evaluate_af
from .af_wrist_clock_study import load_grid, scoring_runs
from .af_wrist_cohort import CONTEXT, in_wrist_segment
from .af_wrist_data_preflight import _inventory
from .af_wrist_expansion_stage import _open_source
from .af_wrist_failure_events import audit_events
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_operating import AFScoredRun
from .af_wrist_quality_study import build_quality_matrices, quality_interaction_features
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_train import load_features
from .af_wrist_transition_native_study import _protected_starts, _within_rhythm
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE
from .afppgecg import ECG_FS, _read_vector, _sample_bounds, load_subject
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features


PREHISTORY_SECONDS = 90
SEED = 2026
_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def _bounds(start, stop):
    return start - PREHISTORY_SECONDS - CONTEXT, stop + CONTEXT


def _verify_protected(args, plan, person, start, stop, lock_hash, development):
    protected = _protected_starts(args, person, lock_hash, plan["supplement_sha256"])
    if sha256(args.protected_transition.read_bytes()).hexdigest() != plan["bundle_sha256"]:
        raise ValueError("The protected native pair must match the verified episode plan")
    with np.load(args.protected_transition, allow_pickle=False) as pair:
        people, starts, ends = (pair[key] for key in ("subjects", "start_s", "decision_end_s"))
        if (pair["source_protocol_sha256"].item() != lock_hash or people.shape != (2,)
                or not set(people) <= development or not np.array_equal(ends, starts + WINDOW_SECONDS)):
            raise ValueError("The two protected native inputs must match the original protocol")
        protected = np.r_[protected, starts[people == person]]
    low, high = _bounds(start, stop)
    if np.any((low < protected + WINDOW_SECONDS + CONTEXT) & (protected - CONTEXT < high)):
        raise ValueError("New waveform and predecessor time must be disjoint from all protected inputs")


def _read_pulse(args, source, entry, start, stop, episode):
    person = entry["subject_id"]
    low, high = _bounds(start, stop)
    with ExitStack() as stack:
        files = []
        for modality in ("ECG", "PPG"):
            recorded, manifest = entry["files"][modality], source[f"{person}_{modality}.mat"]
            if (recorded["official_checksum"] != manifest["checksum"] or recorded["file_bytes"] != manifest["size"]
                    or recorded["missing_blocks"]):
                raise ValueError("The private plan must certify both local paired source files")
            files.append(_open_source(stack, manifest, args.source_dir, args.cache_dir))
        subject = stack.enter_context(load_subject(*files, subject_id=person, include_acceleration=False))
        intervals = subject.annotated_af_episodes()
        complete = (intervals[:, 0] >= start) & (intervals[:, 1] <= stop)
        reference = subject.reference_rhythm_intervals()
        covered = np.maximum(0, np.minimum(reference[:, 1], stop) - np.maximum(reference[:, 0], start)).sum()
        if (not np.array_equal(intervals[complete], episode[None]) or not _within_rhythm(reference, *episode, 1)
                or abs(covered - (stop - start)) >= 1e-7):
            raise ValueError("The entire extra block and its complete episode need continuous ECG reference")
        segment = in_wrist_segment(subject, low, high)
        if segment is None:
            raise ValueError("A single native wrist segment must cover scoring time and predecessor history")
        wave = subject._ppg_data[segment.index]
        begin, end = _sample_bounds(low, high, segment.start_time, PPG_RATE, wave.size)
        ppg = _read_vector(wave, slice(begin, end))
        voltage = subject._voltage
        begin, end = _sample_bounds(start, stop, 0.0, ECG_FS, voltage.size)
        ecg = _read_vector(voltage, slice(begin, end))
        if (len(ppg) != (high - low) * PPG_RATE or len(ecg) != (stop - start) * ECG_FS
                or not np.isfinite(ppg).all() or not np.isfinite(ecg).all() or np.ptp(ecg) <= 0):
            raise ValueError("The local paired waveforms must be finite and match the original sample clocks")
        intersects = (intervals[:, 0] < stop) & (intervals[:, 1] > start)
        return ppg, intervals[intersects]


def _features(ppg, start, stop, ptt_dir):
    _, model = fit_ptt_calibration(ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    endpoints = np.arange(start - PREHISTORY_SECONDS + WINDOW_SECONDS, stop + STRIDE_SECONDS, STRIDE_SECONDS)
    rhythm = np.full((len(endpoints), len(ECG_FEATURE_NAMES)), np.nan)
    quality = np.full(len(endpoints), np.nan)
    low, _ = _bounds(start, stop)
    pad = 2 * PPG_RATE
    for index, endpoint in enumerate(endpoints):
        offset = int((endpoint - WINDOW_SECONDS - low) * PPG_RATE)
        pulse = ppg[offset:offset + WINDOW_SECONDS * PPG_RATE]
        if pulse.shape != (WINDOW_SECONDS * PPG_RATE,) or np.ptp(pulse) <= 0:
            raise ValueError("Each new decision needs its nonflat native trailing 30-second input")
        forecast = model.predict(np.pad(pulse, (pad, 0), mode="reflect"))[pad:]
        values = ecg_features(forecast, PPG_RATE)
        if np.isfinite(values).all() and values[_SUPPORT] > 0:
            rhythm[index] = values
            quality[index] = ppg_features(pulse, PPG_RATE)[1]
    return endpoints, {"baseline": rhythm, "quality_interactions": quality_interaction_features(rhythm, quality)}


def _replay_extra(grid, lock, matrix, new, person, start, evaluation, annotations, expected):
    fixed = fit_template(grid, matrix, lock, DEFAULT_C, seed=SEED)
    original = _replay(grid, fixed)[0]
    if any(expected[key] != value for key, value in original.items()):
        raise ValueError("The original development comparison must reproduce before replaying the extra case")
    (held_fold,) = np.unique(fixed["fold_assignment"][grid.subjects == person])
    stable = (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    shared = _shared_features(grid.table)
    training = fixed["fold_assignment"] != held_fold
    if person in grid.subjects[training]:
        raise ValueError("The new episode participant must remain outside its fitting and calibration roster")
    fit_rows = training & stable & shared
    model = _fit_model(matrix[fit_rows], grid.table["stable_label"][fit_rows], regularization=DEFAULT_C)
    route = fixed["routes"][ROUTE]
    original_scores = _predict(matrix, shared & ~training, model)
    np.testing.assert_allclose(original_scores, np.where(training, np.nan, route["scores"]), rtol=1e-12, atol=1e-12)

    cutoff = route["folds"][held_fold]["operating_point"]["threshold"]
    prefix = np.ones(int(evaluation.sum()), dtype=bool)
    available = np.isfinite(new).all(axis=1)
    scores = _predict(new, available, model)
    history = np.where(np.isfinite(scores[~evaluation]), scores[~evaluation] >= cutoff, -1).astype(np.int8).tolist()
    run = AFScoredRun(start, STRIDE_SECONDS, annotations, scores[evaluation], prefix, prefix, person)
    raw = (run.decisions(cutoff),)
    replay, _ = _IMMEDIATE(raw, prior_decisions=lambda _: history)
    metrics = evaluate_af(replay)
    (case,) = audit_events(raw, replay, confirmation_windows=1, prior_decisions=lambda _: history)["complete_episodes"]
    if metrics["complete_reference_episodes"] != 1 or metrics["detected_complete_episodes"] != int(case["detected"]):
        raise ValueError("The event trace must reconcile the single extra complete episode")
    return {"detected": metrics["detected_complete_episodes"],
            "in_episode_positive": metrics["complete_episodes_with_positive_decision"],
            "available_decisions": int(np.count_nonzero(available[evaluation])),
            "available_predecessor_decisions": int(np.count_nonzero(available[~evaluation])),
            "in_episode_decisions": case["decision_count"], "alert_active_at_onset": case["active_at_onset"],
            "episode_outcome": case["primary"]}


def evaluate(args):
    plan_bytes, lock_bytes, manifest_bytes = (path.read_bytes() for path in
                                              (args.verified_private_plan, args.lock, args.source_manifest))
    plan, lock, manifest = map(json.loads, (plan_bytes, lock_bytes, manifest_bytes))
    lock_hash, inventory_hash = (sha256(blob).hexdigest() for blob in (lock_bytes, manifest_bytes))
    if (plan["waveforms_fully_verified"] is not True or plan["source_protocol_sha256"] != lock_hash
            or plan["source_inventory_sha256"] != inventory_hash or plan["source_release_doi"] != manifest["doi"]):
        raise ValueError("Use the verified private case plan pinned to the original source and protocol")
    (entry,) = plan["range_plans"]
    person = entry["subject_id"]
    start, stop = (entry[key] for key in ("block_start_s", "block_stop_s"))
    episode = np.array([entry["onset_s"], entry["offset_s"]])
    # Keep the exact containing parent cell and require a decision in the interval.
    if (start % WINDOW_SECONDS or stop - start != WINDOW_SECONDS or not start <= episode[0] < episode[1] <= stop
            or not np.any((np.arange(start + STRIDE_SECONDS, stop + 1, STRIDE_SECONDS) > episode[0])
                          & (np.arange(start + STRIDE_SECONDS, stop + 1, STRIDE_SECONDS) <= episode[1]))):
        raise ValueError("Use the verified single parent cell with an in-episode decision")
    table, people, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    development = fit | calibration
    clock = json.loads(args.clock_results.read_bytes())
    quality = json.loads(args.quality_results.read_bytes())
    if (fit & calibration or development & reserved or set(people) != development or person not in development
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY] or clock["original_lock_sha256"] != lock_hash):
        raise ValueError("The extra episode belongs only to the original development roster")
    _inventory(manifest, development | reserved)
    _verify_protected(args, plan, person, start, stop, lock_hash, development)
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    _, original_blocks = scoring_runs(grid, np.full(len(grid.subjects), np.nan), {person})
    if evaluate_af(run.decisions(0.5) for run in original_blocks)["complete_reference_episodes"] != 0:
        raise ValueError("The extra case must come from a participant with no original complete episode")
    old = build_quality_matrices(grid, args.development_raw, args.development_features,
                                 args.stride_raw, args.stride_features)
    source = {item["key"]: item for item in manifest["files"]}
    ppg, annotations = _read_pulse(args, source, entry, start, stop, episode)
    endpoints, new = _features(ppg, start, stop, args.ptt_dir)
    evaluation = endpoints > start
    original = {"baseline": clock["warm_10s"]["nested_development_cv"]["1"][ROUTE],
                "quality_interactions": quality["fixed_regularization"]["quality_interactions"]["summary"]}
    outcomes = {name: _replay_extra(grid, lock, old[name], new[name], person, start, evaluation, annotations, baseline)
                for name, baseline in original.items()}
    return {"scope": "One additional rhythm-enriched development case scored without threshold retuning",
            "source_release_doi": manifest["doi"], "source_inventory_sha256": inventory_hash,
            "source_protocol_sha256": lock_hash, "verified_private_plan_sha256": sha256(plan_bytes).hexdigest(),
            "protected_stable_bundle_sha256": plan["supplement_sha256"],
            "protected_transition_bundle_sha256": plan["bundle_sha256"], "network_requests": 0,
            "episode_duration_seconds": float(episode[1] - episode[0]), "evaluation_block_seconds": stop - start,
            "native_prehistory_seconds": PREHISTORY_SECONDS, "additional_wrist_false_alert_exposure_seconds": 0,
            "retained_original_development_episodes": original["baseline"]["complete_episodes"],
            "retained_original_wrist_exposure_seconds": original["baseline"]["wrist_ecg_observed_seconds"],
            "new_complete_episode_contributors": 1, "candidate_results": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("verified-private-plan", "source-manifest", "source-dir", "cache-dir", "development-raw",
             "development-features", "development-preroll", "stride-raw", "stride-features", "lock", "ptt-dir",
             "protected-supplement", "protected-transition", "clock-results", "quality-results", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new aggregate output path")
    try:
        _write_json(args.output, evaluate(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Extra native episode replay failed: {error}\n")
    print("Reproduced both original fits and replayed one development episode with no cutoff retuning")


if __name__ == "__main__":
    main()
