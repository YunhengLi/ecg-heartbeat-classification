"""Describe uncertainty in two fixed participant-held-out AF replays."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_features import FEATURE_VERSION_KEY
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_operating import score_decisions
from .af_wrist_quality_study import build_quality_matrices
from .af_wrist_template_study import DEFAULT_C, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS


BOOTSTRAP_SEED = 20261124
FIT_SEED = 2026
RESAMPLES = 10_000
FAMILIES = ("baseline", "quality_interactions")


def cluster_summaries(people, replay):
    _, episode_runs, wrist_runs = replay
    rows = []
    for participant in people:
        episodes = tuple(run for run in episode_runs if run.subject_id == participant)
        wrist = tuple(run for run in wrist_runs if run.subject_id == participant)
        metrics = score_decisions(wrist, episode_runs=episodes)
        episode = metrics["episode_metrics"] or {}
        burden = metrics["burden_metrics"] or {}
        rows.append((episode.get("complete_reference_episodes", 0), episode.get("detected_complete_episodes", 0),
                     burden.get("ecg_reference_seconds", 0), burden.get("false_alert_starts", 0)))
    return np.asarray(rows)


def _interval(delta, denominator, scale):
    support = denominator > 0
    values = scale * delta[support] / denominator[support]
    return {"finite_replicates": int(support.sum()), "omitted_zero_denominator": int((~support).sum()),
            "percentile_95_interval": list(map(float, np.quantile(values, (0.025, 0.975))))}


def summarize(people, replays):
    baseline, quality = (cluster_summaries(people, replays[name]) for name in FAMILIES)
    if not np.array_equal(baseline[:, [0, 2]], quality[:, [0, 2]]):
        raise ValueError("Participant ECG episode and wrist denominators must match between replays")
    for model, values in zip(FAMILIES, (baseline, quality), strict=True):
        total = values.sum(axis=0)
        prior = replays[model][0]
        keys = ("complete_episodes", "episodes_detected", "wrist_ecg_observed_seconds", "wrist_false_alerts")
        if not np.array_equal(total, [prior[key] for key in keys]):
            raise ValueError("Participant-level events and exposure must reconcile exactly to the published totals")
    if tuple(baseline.sum(axis=0)) != (17, 12, 2790, 6) or tuple(quality.sum(axis=0)) != (17, 14, 2790, 4):
        raise ValueError("Use the fixed original template vs quality-interaction comparison")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(people), size=(RESAMPLES, len(people)))
    base_draws, quality_draws = baseline[draws].sum(axis=1), quality[draws].sum(axis=1)
    detected_delta = quality_draws[:, 1] - base_draws[:, 1]
    false_delta = quality_draws[:, 3] - base_draws[:, 3]
    return {
        "paired_bootstrap_replicates": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED, "clusters_sampled_per_draw": len(people),
        "support": {"complete_episodes": int(baseline[:, 0].sum()),
                    "participants_with_complete_episodes": int(np.sum(baseline[:, 0] > 0)),
                    "largest_participant_episode_count": int(baseline[:, 0].max()),
                    "wrist_ecg_observed_seconds": int(baseline[:, 2].sum()),
                    "participants_with_wrist_ecg_exposure": int(np.sum(baseline[:, 2] > 0))},
        "fixed_replays": {name: replays[name][0] for name in FAMILIES},
        "quality_minus_original_sensitivity_percentage_points": {
            "point_difference": float(100 * (quality[:, 1].sum() - baseline[:, 1].sum()) / baseline[:, 0].sum()),
            **_interval(detected_delta, base_draws[:, 0], 100),
        },
        "quality_minus_original_false_starts_per_observed_wrist_hour": {
            "point_difference": float(3600 * (quality[:, 3].sum() - baseline[:, 3].sum()) / baseline[:, 2].sum()),
            **_interval(false_delta, base_draws[:, 2], 3600),
        },
    }


def _run(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    receipt_bytes = args.quality_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    table, members, timelines = load_features(args.development_features)
    fit, calibration, test = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & test or set(members) != fit | calibration
            or (len(fit), len(calibration), len(test)) != (5, 5, 25)
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]):
        raise ValueError("Use only the original ten development participants")
    grid = load_grid(table, members, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    matrices = build_quality_matrices(grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features)
    fitted = {name: fit_template(grid, matrices[name], lock, DEFAULT_C, seed=FIT_SEED) for name in FAMILIES}
    verify_fit_support(*(fitted[name] for name in FAMILIES))
    replays = {name: _replay(grid, model) for name, model in fitted.items()}
    if any(replays[name][0] != receipt["fixed_regularization"][name]["summary"] for name in FAMILIES):
        raise ValueError("The fixed quality study must reproduce before uncertainty analysis")
    return {"scope": "Descriptive participant-cluster bootstrap of two fixed development replays",
            "original_fit_seed": FIT_SEED, "source_protocol_sha256": sha256(lock_bytes).hexdigest(),
            "quality_receipt_sha256": sha256(receipt_bytes).hexdigest(), "refits_inside_bootstrap": 0,
            "independent_predictions": 0, **summarize(sorted(fit | calibration), replays)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "quality-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new aggregate output")
    try:
        result = _run(args)
        _write_json(args.output, result)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Development quality uncertainty analysis failed: {error}\n")
    print(json.dumps({"support": result["support"], "sensitivity": result["quality_minus_original_sensitivity_percentage_points"],
                      "false_alert_rate": result["quality_minus_original_false_starts_per_observed_wrist_hour"]}), flush=True)


if __name__ == "__main__":
    main()
