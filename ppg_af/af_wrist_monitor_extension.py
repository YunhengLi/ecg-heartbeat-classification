"""Stage and replay extra rhythm-blind wrist monitoring with frozen fold classifiers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_events import AFRun
from .af_wrist_clock_study import _summarize
from .af_wrist_cohort import core_label
from .af_wrist_direct_fusion import SPECTRAL_ROUTE
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _fit_model, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_motion_quality import load_motion_development
from .af_wrist_operating import score_decisions
from .af_wrist_quality_retrospective_uncertainty import _paired_interval, _participant_counts
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_vote_study import _IMMEDIATE, STRIDE_SECONDS
from .af_wrist_vote_compare import compare_alert_runs
from .afppgecg import ECG_FS, load_subject
from .features import ECG_FEATURE_NAMES, PPG_FEATURE_NAMES, ppg_features


BLOCK_S = 300
EARLIER_S = 90
CONTEXT_S = 1
SEED = 2026
FAMILIES = ("baseline", "quality_interactions")
BOOTSTRAP_REPETITIONS = 10_000


def read_plan(path, lock_path, *, retrospective=False, block_seconds=BLOCK_S):
    raw, lock_bytes = path.read_bytes(), lock_path.read_bytes()
    plan, lock = json.loads(raw), json.loads(lock_bytes)
    receipt, schedule = plan["plan_receipt"], plan["private_schedule"]
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    development = fit | calibration
    people = reserved if retrospective else development
    if (fit & calibration or development & reserved or len(development) != 10 or len(reserved) != 25 or set(schedule) != people
            or receipt["source_protocol_sha256"] != sha256(lock_bytes).hexdigest()
            or receipt["block_seconds"] != block_seconds or receipt["predecessor_seconds"] != EARLIER_S):
        raise ValueError("The new monitoring plan must use the original development protocol")
    if sum(len(blocks) for blocks in schedule.values()) != receipt["scheduled_blocks"]:
        raise ValueError("Scheduled monitoring counts must match the frozen plan")
    for blocks in schedule.values():
        previous_end = -1
        for block in blocks:
            start, stop = block["evaluation_s"]
            support = (start - EARLIER_S - CONTEXT_S, stop + CONTEXT_S)
            if (start % WINDOW_SECONDS or stop - start != block_seconds or support[0] < previous_end
                    or tuple(block["input_support_s"]) != support):
                raise ValueError("New monitored time and its preceding signal must be disjoint and fixed")
            previous_end = support[1]
    return schedule, receipt, sha256(raw).hexdigest()


def stage_person(ecg_source, wrist_source, person, blocks, plan_digest, output, *, block_seconds=BLOCK_S):
    with load_subject(ecg_source, wrist_source, subject_id=person, include_acceleration=False) as subject:
        episodes = subject.annotated_af_episodes()
        reference = subject.reference_rhythm_intervals()
        raw_pulse, valid_reference, starts, episode_support = [], [], [], []
        for block in blocks:
            start, stop = block["evaluation_s"]
            if stop - start != block_seconds:
                raise ValueError("Native input duration differs from the frozen monitoring plan")
            outer_start, outer_stop = block["input_support_s"]
            segment = subject.segments[block["native_wrist_segment"]]
            clip = subject.read_segment(segment.index, outer_start - segment.start_time, outer_stop - segment.start_time)
            if (clip.ppg.shape != ((outer_stop - outer_start) * PPG_RATE,)
                    or clip.ecg.shape != ((outer_stop - outer_start) * int(ECG_FS),)):
                raise ValueError("A planned native wrist block cannot be recovered on the frozen source clock")
            ends = np.arange(start - EARLIER_S + WINDOW_SECONDS, stop + 1, STRIDE_SECONDS)
            starts.append(start)
            ppg = [clip.ppg[(end - WINDOW_SECONDS - outer_start) * PPG_RATE:(end - outer_start) * PPG_RATE]
                   for end in ends]
            raw_pulse.append(np.stack(ppg))
            row_ref = []
            for end in ends[ends > start]:
                left, right = end - STRIDE_SECONDS, end
                trace = clip.ecg[(left - outer_start) * int(ECG_FS):(right - outer_start) * int(ECG_FS)]
                _, valid = core_label(reference, left, right, trace)
                row_ref.append(valid)
            valid_reference.append(row_ref)
            episode_support.append(episodes[(episodes[:, 1] > start) & (episodes[:, 0] < stop)])
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{person}.npz"
    tmp = target.with_suffix(".partial.npz")
    per_block_rows = (block_seconds + EARLIER_S - WINDOW_SECONDS) // STRIDE_SECONDS + 1
    np.savez_compressed(tmp, raw_ppg=np.asarray(raw_pulse).reshape(-1, per_block_rows, WINDOW_SECONDS * PPG_RATE),
                        reference_valid=np.asarray(valid_reference, dtype=bool).reshape(-1, block_seconds // STRIDE_SECONDS),
                        start_s=np.asarray(starts, dtype=int), af_intervals_s=np.concatenate(episode_support)
                        if episode_support else np.empty((0, 2)), subject_id=person, plan_sha256=plan_digest)
    tmp.replace(target)


def _archive_features(path, person, scheduled, digest, template, *, families=FAMILIES, block_seconds=BLOCK_S):
    with np.load(path, allow_pickle=False) as raw:
        pulse, valid, starts, episodes = (raw[key] for key in ("raw_ppg", "reference_valid", "start_s", "af_intervals_s"))
        if (raw["subject_id"].item() != person or raw["plan_sha256"].item() != digest
                or starts.tolist() != [row["evaluation_s"][0] for row in scheduled]
                or valid.shape != (len(starts), block_seconds // STRIDE_SECONDS)
                or pulse.shape != (len(starts), (block_seconds + EARLIER_S - WINDOW_SECONDS) // STRIDE_SECONDS + 1,
                                   WINDOW_SECONDS * PPG_RATE)):
            raise ValueError("Staged monitoring must match the original blind plan")
    if not len(starts):
        return starts, valid, episodes, {}, np.empty((0, 0), dtype=bool)
    flat = pulse.reshape(-1, WINDOW_SECONDS * PPG_RATE)
    available = np.isfinite(flat[:, -STRIDE_SECONDS * PPG_RATE:]).all(axis=1).reshape(pulse.shape[:2])
    eligible = np.isfinite(flat).all(axis=1) & (np.ptp(flat, axis=1) > 0)
    learned = _extract(flat[eligible], template)
    if "quality" in families:
        learned["quality"] = learned["quality_interactions"][:, :len(ECG_FEATURE_NAMES) + 1]
    if SPECTRAL_ROUTE in families:
        direct = np.asarray([ppg_features(pulse, PPG_RATE)[0] for pulse in flat[eligible]]).reshape(-1, len(PPG_FEATURE_NAMES))
        learned[SPECTRAL_ROUTE] = np.column_stack((learned["quality_interactions"], direct[:, len(ECG_FEATURE_NAMES):]))
    matrices = {}
    for model, values in learned.items():
        if model not in families:
            continue
        aligned = np.full((len(flat), values.shape[1]), np.nan)
        aligned[eligible] = values
        matrices[model] = aligned.reshape(*pulse.shape[:2], values.shape[1])
    return starts, valid, episodes, matrices, available


def _reuse_fold_models(grid, lock, quality, original):
    shared = _shared_features(grid.table)
    labels = grid.table["stable_label"]
    train_rows = shared & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    classifiers = {}
    for name in FAMILIES:
        matrix = quality[name]
        fitted = fit_template(grid, matrix, lock, DEFAULT_C, seed=SEED)
        if name == "quality_interactions" and _replay(grid, fitted)[0] != original:
            raise ValueError("The original quality comparison must reproduce before supplemental scoring")
        scores, _, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
        assignments = fitted["fold_assignment"]
        members = {}
        for fold in fitted["routes"][ROUTE]["folds"]:
            held = assignments == fold["fold"]
            trained = train_rows & ~held
            model = _fit_model(matrix[trained], labels[trained], regularization=DEFAULT_C)
            expected = _predict(matrix, held & shared, model)
            if not np.allclose(expected[held], scores[held], rtol=1e-12, atol=1e-14, equal_nan=True):
                raise ValueError("Supplemental scoring must reuse each original model without training on new blocks")
            cutoff = fold["operating_point"]["threshold"]
            members.update({person: (model, cutoff) for person in np.unique(grid.subjects[held])})
        classifiers[name] = members
    return classifiers


def _person_runs(person, archive, model, cutoff, family, *, transform=None, replay=_IMMEDIATE):
    starts, validity, episodes, matrices, ppg_available = archive
    if not len(starts):
        return []
    output = []
    for index, (start, valid, values, availability) in enumerate(zip(starts, validity, matrices[family], ppg_available, strict=True)):
        probability = _predict(values, np.isfinite(values).all(axis=1), model)
        if transform is not None:
            probability = transform(probability, index)
        decision = np.where(np.isfinite(probability), probability >= cutoff, -1).astype(np.int8)
        preroll = len(decision) - len(valid)
        run = AFRun(float(start), STRIDE_SECONDS, episodes, decision[preroll:], valid, availability[preroll:], person)
        (scored,), _ = replay((run,), prior_decisions=lambda _: decision[:preroll])
        output.append(scored)
    return output


def _cluster_uncertainty(baseline, candidate, people, *, allow_score_coverage_change=False):
    before, after = (_participant_counts(replay, people) for replay in (baseline, candidate))
    support = [1, 3] if allow_score_coverage_change else [1, 3, 4]
    if not np.array_equal(before[:, support], after[:, support]):
        raise ValueError("Paired supplemental models must use identical reference support")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(people), size=(BOOTSTRAP_REPETITIONS, len(people)))
    _, complete, _, elapsed, _ = before[draws].sum(axis=1).T
    changes = after - before
    detected, _, false, _, _ = changes[draws].sum(axis=1).T
    source = before.sum(axis=0)
    delta = changes.sum(axis=0)
    result = {"resamples": BOOTSTRAP_REPETITIONS, "seed": SEED,
              "participants_with_complete_episodes": int(np.sum(before[:, 1] > 0)),
              "false_starts_per_ecg_observed_wrist_hour_change": {
                  "point": float(3600 * delta[2] / source[3]), **_paired_interval(3600 * false, elapsed, elapsed > 0),
              }}
    if source[1] > 0:
        result["sensitivity_change"] = {"point": float(delta[0] / source[1]), **_paired_interval(detected, complete, complete > 0)}
    return result


def evaluate(args):
    schedule, receipt, digest = read_plan(args.plan, args.lock)
    grid, quality, paths, lock, _, original = load_motion_development(args)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    for person, blocks in schedule.items():
        with np.load(paths[1][0] / f"{person}.npz", allow_pickle=False) as old:
            starts, ends = old["input_start_s"], old["decision_end_s"]
        with np.load(paths[0][0] / f"{person}.npz", allow_pickle=False) as old:
            starts, ends = np.r_[starts, old["start_s"]], np.r_[ends, old["start_s"] + WINDOW_SECONDS]
        if any(np.any((starts < stop) & (ends > start)) for start, stop in (block["input_support_s"] for block in blocks)):
            raise ValueError("The supplemental wrist plan must be disjoint from original fit and scoring inputs")
    records = {person: _archive_features(args.monitoring_cache / f"{person}.npz", person, scheduled, digest, template)
               for person, scheduled in schedule.items()}
    classifier = _reuse_fold_models(grid, lock, quality, original)
    results = {}
    for family in FAMILIES:
        runs = []
        for person, record in records.items():
            model, cutoff = classifier[family][person]
            if cutoff is None:
                raise ValueError("An original fold with planned new monitoring lacks a cutoff")
            runs.extend(_person_runs(person, record, model, cutoff, family))
        scored = score_decisions(runs)
        metrics = _summarize({**scored, "monitor_blocks_with_unknown_cutoff_state": 0,
                              "episode_blocks_with_unknown_cutoff_state": 0})
        results[family] = (metrics, tuple(runs), tuple(runs))
    base, candidate = (results[name] for name in FAMILIES)
    coverage = {"scheduled_decisions": sum(len(run.decisions) for run in base[1]),
                "observed_wrist_decisions": sum(int(run.ppg_available.sum()) for run in base[1]),
                "scoreable_decisions": {family: sum(int((run.decisions >= 0).sum()) for run in replay[1])
                                        for family, replay in results.items()}}
    if base[0]["complete_episodes"]:
        comparison = _compare_with_pre_onset(base, candidate, lock)
    else:
        comparison = {"summary": candidate[0], "sensitivity_change": None, "within_sensitivity_tolerance": None,
                      "paired_changes": compare_alert_runs(base[1], candidate[1], base[2], candidate[2])}
    return {"scope": "Supplemental wrist monitoring on new times from original development participants",
            "plan_receipt": receipt, "plan_sha256": digest, "frozen_source_seed": SEED, "coverage": coverage,
            "baseline": base[0], "quality_interactions": comparison,
            "paired_participant_uncertainty": _cluster_uncertainty(base, candidate, list(schedule))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "lock", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Stage native blocks from local paired MAT files")
    for key in ("monitoring-cache", "development-raw", "development-features", "stride-raw", "stride-features",
                "clock-results", "quality-results", "ptt-dir"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    try:
        if args.source_dir is not None:
            schedule, _, digest = read_plan(args.plan, args.lock)
            for person, blocks in schedule.items():
                stage_person(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat",
                             person, blocks, digest, args.output)
            return
        if not all((args.monitoring_cache, args.development_raw, args.development_features, args.stride_raw,
                    args.stride_features, args.clock_results, args.quality_results, args.ptt_dir)):
            parser.error("Supply all original archives and the staged supplemental monitoring cache")
        _write_json(args.output, evaluate(args))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Supplemental wrist monitoring failed: {error}\n")


if __name__ == "__main__":
    main()
