"""Compare an optical-entropy gate with the original low-quality confirmation gate."""

import argparse
from pathlib import Path

import numpy as np

from .af_state import state_durations, state_summary
from .af_wrist_clock_study import _summarize
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_spacing import CONFIRMATION_RESULTS_SHA, DEVELOPMENT_CONFIRMATION_SHA
from .af_wrist_direct_fusion import SPECTRAL_ROUTE
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import MONITOR_KIND, _predict, _shared_features, _subject_ids, _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import _archive_features, _cluster_uncertainty, _person_runs
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_operating import score_decisions
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_vote_compare import compare_alert_runs
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE
from .features import PPG_FEATURE_NAMES


ENTROPY = PPG_FEATURE_NAMES.index("pulse_band_entropy")


def matched_gate(quality, entropy):
    """Select entropy on fitting wear time to match the existing gate frequency."""
    if not len(entropy) or not np.isfinite(entropy).all() or not np.isfinite(quality).all():
        raise ValueError("The gate needs finite training-side PPG wear inputs")
    requested = int(np.sum(quality < QUALITY_CUTOFF))
    if not requested:
        return None, {"training_scoreable_rows": len(entropy), "quality_gate_rows": 0, "entropy_gate_rows": 0}
    boundary = float(np.partition(entropy, len(entropy) - requested)[len(entropy) - requested])
    return boundary, {"training_scoreable_rows": len(entropy), "quality_gate_rows": requested,
                      "entropy_gate_rows": int(np.sum(entropy >= boundary))}


def entails_confirmation(entropy, boundary):
    return entropy >= boundary if boundary is not None else np.zeros(len(entropy), dtype=bool)


def paired_gate_replays(grid, quality, entropy_gate, scores, cutoffs):
    prior, active = preceding_rows(grid)
    gates = {"quality": active & (quality < QUALITY_CUTOFF), "entropy": active & entropy_gate}
    output = {"original": _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)}
    changes = {}
    for name, gate in gates.items():
        transformed = confirm_scores(scores, prior, gate)
        output[name] = _held_out_replay(grid, transformed, cutoffs, _IMMEDIATE)
        changes[name] = {"scoreable_replay_rows_gated": int(np.sum(gate & np.isfinite(scores))),
                         "scoreable_replay_rows_lost": int(np.sum(np.isfinite(scores) & ~np.isfinite(transformed)))}
    return output, changes


def development(grid, matrices, lock, prior):
    q = matrices["quality"][:, -1]
    entropy = grid.table["direct_ppg"][:, ENTROPY]
    wear = _shared_features(grid.table) & (grid.table["run_kind"] == MONITOR_KIND)
    result = {}
    for seed in SEEDS:
        fitted = fit_template(grid, matrices["baseline"], lock, DEFAULT_C, seed=seed)
        scores, cutoffs, _ = fold_arrays(fitted, fitted["routes"][ROUTE])
        assignments = fitted["fold_assignment"]
        selected = np.zeros(len(scores), dtype=bool)
        training_support = []
        for fold in np.unique(assignments):
            held = assignments == fold
            train = wear & ~held
            boundary, receipt = matched_gate(q[train], entropy[train])
            selected[held] = entails_confirmation(entropy[held], boundary)
            training_support.append({"threshold": boundary, **receipt})
        replays, changes = paired_gate_replays(grid, q, selected, scores, cutoffs)
        original = prior[str(seed)]["plain_template"]
        baseline = replays["original"]
        if baseline[0] != _replay(grid, fitted)[0] or baseline[0] != original["original"]:
            raise ValueError("The plain template AF comparison must reproduce")
        comparisons = {name: _compare_with_pre_onset(baseline, replays[name], lock) for name in ("quality", "entropy")}
        if comparisons["quality"] != original["lower_quality_only"]["fixed_cutoffs"]:
            raise ValueError("The published quality gate replay must reproduce")
        result[str(seed)] = {"original": baseline[0], "development_training_gate_support": training_support,
                             "confirmation_support": changes, **comparisons}
    return result


def historical(args, protocol, model_lock, boundary):
    grid, model, available, roster, quality, _ = historical_inputs(args, protocol, model_lock)
    original = _read_pinned(args.confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)["historical_episodes_and_monitoring"]
    scores = _predict(grid.table[ROUTE], available, model["model"])
    gate = entails_confirmation(grid.table["direct_ppg"][:, ENTROPY], boundary)
    replays, changes = paired_gate_replays(grid, quality["quality"][:, -1], gate, scores, np.full(len(scores), model["cutoff"]))
    baseline = replays["original"]
    comparisons = {name: _compare_with_pre_onset(baseline, replays[name], protocol) for name in ("quality", "entropy")}
    if baseline[0] != original["baseline"] or comparisons["quality"] != original["conditional_confirmation"]:
        raise ValueError("The two frozen historical comparisons must reproduce")
    return {"original": baseline[0], "confirmation_support": changes, **comparisons,
            "uncertainty_from_original": _cluster_uncertainty(baseline, replays["entropy"], sorted(roster)),
            "uncertainty_from_quality": _cluster_uncertainty(replays["quality"], replays["entropy"], sorted(roster))}


def half_hour(args, protocol, model_lock, boundary):
    from .af_wrist_template_independent import _locked_model

    earlier, _, old_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, earlier, old_digest)
    prior = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    if digest != prior["plan_sha256"] or set(schedule) != roster:
        raise ValueError("Use only the original sealed half-hour evaluation plan")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    all_runs = {"original": [], "quality": [], "entropy": []}
    for person, blocks in schedule.items():
        if not blocks:
            continue
        archive = _archive_features(args.monitoring_cache / f"{person}.npz", person, blocks, digest, template,
                                    families=("baseline", "quality", SPECTRAL_ROUTE), block_seconds=BLOCK_SECONDS)
        matrices = archive[3]
        gates = {"quality": matrices["quality"][:, :, -1] < QUALITY_CUTOFF,
                 "entropy": np.stack([entails_confirmation(row, boundary) for row in matrices[SPECTRAL_ROUTE][:, :, -1]])}
        all_runs["original"].extend(_person_runs(person, archive, model["model"], model["cutoff"], "baseline"))
        for name, gate in gates.items():
            def transform(scores, index):
                return confirm_scores(scores, np.arange(len(scores)) - 1, gate[index])

            all_runs[name].extend(_person_runs(person, archive, model["model"], model["cutoff"], "baseline", transform=transform))
    totals = {}
    for name, runs in all_runs.items():
        summary = _summarize({**score_decisions(runs), "monitor_blocks_with_unknown_cutoff_state": 0,
                              "episode_blocks_with_unknown_cutoff_state": 0})
        totals[name] = (summary, runs, runs)
    if totals["original"][0] != prior["original"] or totals["quality"][0] != prior["conditional_confirmation"]:
        raise ValueError("The same two historical half-hour replays must reproduce")
    comparisons = {}
    for name, (summary, runs, _) in totals.items():
        comparisons[name] = {"summary": summary, "forward_state": state_summary(state_durations(runs)),
                             "scoreable_decisions": sum(int((run.decisions >= 0).sum()) for run in runs)}
    for name in ("original", "quality"):
        baseline, candidate = totals[name], totals["entropy"]
        comparisons[f"entropy_vs_{name}"] = {
            "paired_changes": compare_alert_runs(baseline[1], candidate[1], baseline[2], candidate[2]),
            "uncertainty": _cluster_uncertainty(baseline, candidate, sorted(roster), allow_score_coverage_change=True),
        }
    return comparisons


def evaluate(args):
    grid, matrices, _, protocol, _, _ = load_motion_development(args)
    _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    prior = _read_pinned(args.development_confirmation_results, DEVELOPMENT_CONFIRMATION_SHA, public=True)["comparisons"]
    fixed_people = _subject_ids(model_lock["fit_subjects"])
    wear = (_shared_features(grid.table) & (grid.table["run_kind"] == MONITOR_KIND)
            & np.isin(grid.subjects, sorted(fixed_people)))
    boundary, receipt = matched_gate(matrices["quality"][wear, -1], grid.table["direct_ppg"][wear, ENTROPY])
    return {"scope": "Exploratory entropy gate at the original gate activation budget on fitting-only wear",
            "quality_boundary": QUALITY_CUTOFF,
            "historical_entropy_threshold": boundary, "historical_training_gate_support": receipt,
            "development": development(grid, matrices, protocol, prior),
            "historical": historical(args, protocol, model_lock, boundary),
            "half_hour": half_hour(args, protocol, model_lock, boundary)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "development-confirmation-results", "prior-inputs", "data-preflight", "historical-model-lock",
                 "coverage-report", "historical-results", "confirmation-results", "prior-monitor-plan", "plan",
                 "monitoring-cache", "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
