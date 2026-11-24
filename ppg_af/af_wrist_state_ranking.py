"""Compare frozen PPG rhythm scores within participants and across the same roster."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .af_state_probability import time_targets
from .af_wrist_clock_study import scoring_runs
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_quality_retrospective_uncertainty import (
    _RETROSPECTIVE_SHA256, _read_retrospective_inputs, _verified_score_inputs,
)
from .af_wrist_template_independent import _match_coverage
from .af_wrist_vote_study import _IMMEDIATE, _held_out_replay


BOOTSTRAPS = 10_000


def duration_auc(scores, truth):
    labels = np.repeat([0, 1], len(scores))
    weights = np.r_[1 - truth, truth]
    return float(roc_auc_score(labels, np.r_[scores, scores], sample_weight=weights))


def leading_summary(methods, clock_seconds):
    _, truth, people = methods["baseline"]
    unique = np.unique(people)
    mixed = [person for person in unique if np.any(truth[people == person] > 0) and np.any(truth[people == person] < 1)]
    participant_masks = [people == person for person in mixed]
    summaries = {}
    aucs = {}
    for name, (estimate, reference, roster) in methods.items():
        if not np.array_equal(reference, truth) or not np.array_equal(roster, people):
            raise ValueError("PPG comparisons need identical time and participant coverage")
        within = np.asarray([duration_auc(estimate[index], truth[index]) for index in participant_masks])
        aucs[name] = within
        summaries[name] = {"pooled_duration_auc": duration_auc(estimate, truth),
                           "participant_macro_duration_auc": float(within.mean()) if len(within) else None,
                           "median_participant_duration_auc": float(np.median(within)) if len(within) else None}
    paired = {}
    if mixed:
        draws = (np.random.default_rng(2026).integers(len(mixed), size=(BOOTSTRAPS, len(mixed))) if len(mixed) > 1 else None)
        for name in sorted(set(methods) - {"baseline"}):
            delta = aucs[name] - aucs["baseline"]
            paired[name] = {"mean_auc_change": float(delta.mean()), "participants_with_improvement": int((delta > 0).sum()),
                            "percentile_95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
                            if draws is not None else None}
    af = float(truth.sum() * clock_seconds)
    return {"participants": len(unique), "participants_with_both_rhythm_states": len(mixed),
            "forward_scoreable_seconds": len(truth) * clock_seconds, "forward_AF_seconds": af,
            "routes": summaries, "paired_participant_differences_from_baseline": paired}


def evaluate(args):
    inputs = _read_retrospective_inputs(args)
    published = inputs[4]
    grid, matrices, models, available, people = _verified_score_inputs(*inputs)
    scopes = {"wear_monitoring": {}, "all_scored_times_including_AF_enriched": {}}
    for name, model in models.items():
        probabilities = _predict(matrices[name], available, model["model"])
        if not np.array_equal(np.isfinite(probabilities), available):
            raise ValueError("Ranking must compare the same finite probability support")
        verified = published["historical_baseline"] if name == "baseline" else published["retrospective_variants"][name]["summary"]
        replay = _held_out_replay(grid, probabilities, np.full(len(probabilities), model["cutoff"]), _IMMEDIATE)
        _match_coverage(replay[0], verified)
        for scope, runs in zip(scopes.values(), scoring_runs(grid, probabilities, set(people)), strict=True):
            scope[name] = time_targets(runs, grid.seconds)
    return {"scope": "Within-person forward-time ranking on the previously scored 25-person wrist roster",
            "retrospective_results_sha256": _RETROSPECTIVE_SHA256, "decision_stride_seconds": grid.seconds,
            "scopes": {name: leading_summary(results, grid.seconds) for name, results in scopes.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("retro-results", "protocol-lock", "data-preflight", "variant-lock", "historical-model-lock",
                 "coverage-report", "sealed-inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
