"""Report frozen wrist alert sensitivity by AF duration and participant contribution."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .af_events import _coverage, _grid, _intervals, _union
from .af_wrist_confirmation_opportunity import DURATION_LABELS
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_spacing import CONFIRMATION_RESULTS_SHA
from .af_wrist_entropy_confirmation import ENTROPY, entails_confirmation
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_template_study import ROUTE
from .af_wrist_vote_compare import _definite_starts, _first_alerts
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


ENTROPY_RESULTS_SHA = "064eeffc5448a6d827d60ebfbd60965e3b21899554ffdcdcf44146deaea7e72d"
BOOTSTRAP = 10_000


def complete_episodes(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.subject_id].append(run)
    labels, people, delays = [], [], []
    for person in sorted(grouped):
        spans, starts, annotations = [], [], []
        for run in grouped[person]:
            edges, decisions, reference, _ = _grid(run)
            spans.append(edges[np.flatnonzero(np.diff(reference, prepend=False, append=False)).reshape(-1, 2)])
            starts.append(edges[1:][reference & _definite_starts(run, decisions)])
            annotations.append(_intervals(run.af_intervals_s))
        merged = _union(np.concatenate(annotations))
        whole, _ = _coverage(merged, _union(np.concatenate(spans)))
        all_first = _first_alerts(np.concatenate(starts), merged)
        labels.extend(merged[whole])
        people.extend([person] * int(whole.sum()))
        delays.extend((all_first - merged[:, 0])[whole])
    return np.asarray(labels), np.asarray(people), np.asarray(delays)


def select_scope(people, delays):
    participants, counts = np.unique(people, return_counts=True)
    if not len(people):
        return {"episodes": 0, "contributors": 0, "compared_routes": {}}
    largest_position = int(np.argmax(counts))
    nonlargest = people != participants[largest_position]
    before = np.isfinite(delays["original"])
    rng = np.random.default_rng(2026)
    draws = rng.integers(len(participants), size=(BOOTSTRAP, len(participants))) if len(participants) >= 3 else None
    baseline_counts = np.array([int(before[people == person].sum()) for person in participants])
    comparisons = {}
    for name, latency in delays.items():
        detected = np.isfinite(latency)
        selected = np.array([int(detected[people == person].sum()) for person in participants])
        shared = detected & before
        shifts = latency[shared] - delays["original"][shared]
        interval = (np.quantile((selected - baseline_counts)[draws].sum(axis=1) / counts[draws].sum(axis=1),
                                (0.025, 0.975)).tolist() if draws is not None else None)
        comparisons[name] = {"detected_episodes": int(detected.sum()), "pooled_sensitivity": float(detected.mean()),
                             "participant_average_sensitivity": float(np.mean(selected / counts)),
                             "largest_contributor_detections": int(selected[largest_position]),
                             "other_contributors_detections": int(np.sum(detected & nonlargest)),
                             "paired_gain_over_original": int(np.sum(detected & ~before)),
                             "paired_loss_from_original": int(np.sum(~detected & before)),
                             "paired_sensitivity_change_interval_95": interval,
                             "shared_detection_median_delay_change_s": float(np.median(shifts)) if len(shifts) else None}
    return {"episodes": len(people), "contributors": len(participants), "largest_contributor_episodes": int(counts.max()),
            "compared_routes": comparisons}


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    ground = _read_pinned(args.confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)["historical_episodes_and_monitoring"]
    new = _read_pinned(args.entropy_results, ENTROPY_RESULTS_SHA, public=True)
    grid, model, available, _, quality, _ = historical_inputs(args, protocol, model_lock)
    scores = _predict(grid.table[ROUTE], available, model["model"])
    cutoff = np.full(len(scores), model["cutoff"])
    adjacent, active = preceding_rows(grid)
    low = active & (quality["quality"][:, -1] < QUALITY_CUTOFF)
    entropy = active & entails_confirmation(grid.table["direct_ppg"][:, ENTROPY], new["historical_entropy_threshold"])
    arms = {"original": scores, "adjacent_quality": confirm_scores(scores, adjacent, low),
            "adjacent_entropy": confirm_scores(scores, adjacent, entropy)}
    expected = {"original": ground["baseline"], "adjacent_quality": ground["conditional_confirmation"]["summary"],
                "adjacent_entropy": new["historical"]["entropy"]["summary"]}
    labels = people = None
    all_delays = {}
    for name, values in arms.items():
        result, runs, _ = _held_out_replay(grid, values, cutoff, _IMMEDIATE)
        if result != expected[name]:
            raise ValueError("Historical AF totals must reproduce before duration stratification")
        intervals, participants, latency = complete_episodes(runs)
        if len(latency) != result["complete_episodes"] or np.isfinite(latency).sum() != result["episodes_detected"]:
            raise ValueError("Episode attribution must match the frozen published counts")
        if labels is None:
            labels, people = intervals, participants
        if not np.array_equal(intervals, labels) or not np.array_equal(participants, people):
            raise ValueError("Every route must score the same complete person-linked AF intervals")
        all_delays[name] = latency
    lengths = labels[:, 1] - labels[:, 0]
    strata = np.searchsorted((10, 30, 60), lengths, side="left")
    output = {"all_complete_episodes": select_scope(people, all_delays)}
    for index, label in enumerate(DURATION_LABELS):
        use = strata == index
        output[label] = select_scope(people[use], {name: latency[use] for name, latency in all_delays.items()})
    return {"scope": "Descriptive AF duration and contributor audit on the same 25 historical wrist participants",
            "bootstrap_repetitions": BOOTSTRAP, "results": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report", "historical-results",
                 "confirmation-results", "entropy-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
