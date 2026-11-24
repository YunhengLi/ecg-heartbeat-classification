"""Estimate paired participant uncertainty for the frozen retrospective wrist replay."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .af_events import evaluate_af
from .af_wrist_data_preflight import _SCOPES, _cohort_archives, _verify_seal
from .af_wrist_holdout import _predict, _subject_ids, _write_json
from .af_wrist_operating import _remaining_non_af_time
from .af_wrist_quality_retrospective import VARIANTS, _PREFLIGHT_SHA256, _quality_matrices, _read_pinned
from .af_wrist_template_independent import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, _match_coverage, load_independent_inputs,
)
from .af_wrist_vote_compare import compare_alert_runs
from .af_wrist_vote_study import _IMMEDIATE, _held_out_replay


_RETROSPECTIVE_SHA256 = "81a6af20f98bceccce288ad4002d0634edaa850576739cd1b7bc29350ffa4cd7"


def _verified_score_inputs(protocol, lock, frozen, coverage, report, directories, seal):
    expected_people = _subject_ids(protocol["test_roster"])
    _verify_seal(_cohort_archives(directories, expected_people), seal,
                 report["source_receipts_sha256"]["independent_lock"], coverage)
    grid, baseline, available, support, roster, _, _ = load_independent_inputs(
        lock, protocol, directories["parent-features"], directories["stride-raw"],
        directories["stride-features"], coverage,
    )
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    scored = int(available[evaluation].sum())
    common = report["common_retrospective_support"]
    if (set(grid.subjects) != roster or len(roster) != report["retrospective_participants"]
            or scored != common["scored_evaluation_cells"]
            or int(available.sum()) - scored != common["scored_predecessor_cells"]
            or support["partial_reference_episodes"] != common["partial_reference_episodes"]):
        raise ValueError("Use the original sealed retrospective roster, score support and reference coverage")

    matrices = _quality_matrices(grid, directories)
    return grid, matrices, {"baseline": baseline, **frozen["models"]}, available, sorted(roster)


def _verified_replays(protocol, lock, frozen, coverage, report, directories, seal):
    grid, matrices, models, available, people = _verified_score_inputs(protocol, lock, frozen, coverage, report, directories, seal)
    expected = {"baseline": report["historical_baseline"],
                **{name: report["retrospective_variants"][name]["summary"] for name in VARIANTS}}
    replays = {}
    for name, option in models.items():
        scores = _predict(matrices[name], available, option["model"])
        if not np.array_equal(np.isfinite(scores), available):
            raise ValueError("The frozen models must retain the original shared score gate")
        replays[name] = _held_out_replay(grid, scores, np.full(len(scores), option["cutoff"]), _IMMEDIATE)
        _match_coverage(replays[name][0], expected[name])
    return replays, people


def _participant_counts(replay, people):
    summary, episodes, wrist = replay
    episode_by_person, wrist_by_person = defaultdict(list), defaultdict(list)
    for run in episodes:
        episode_by_person[run.subject_id].append(run)
    for run in wrist:
        wrist_by_person[run.subject_id].append(run)
    counts = []
    for person in people:
        event_metrics = evaluate_af(episode_by_person[person])
        monitor_runs = wrist_by_person[person]
        monitor_metrics = evaluate_af(monitor_runs)
        counts.append((event_metrics["detected_complete_episodes"], event_metrics["complete_reference_episodes"],
                       monitor_metrics["false_alert_starts"], monitor_metrics["ecg_reference_seconds"],
                       _remaining_non_af_time(monitor_runs, analyzable_only=True)))
    values = np.asarray(counts)
    headline_fields = ("episodes_detected", "complete_episodes", "wrist_false_alerts", "wrist_ecg_observed_seconds")
    if not np.array_equal(values.sum(axis=0)[:4], [summary[key] for key in headline_fields]):
        raise ValueError("Participant contributions must sum to the original headline counts and exposure")
    return values


def _paired_interval(numerator, denominator, valid):
    return {"percentile_95": np.percentile(numerator[valid] / denominator[valid], [2.5, 97.5]).tolist(),
            "valid_paired_draws": int(valid.sum())}


def _episode_changes(before, after, people):
    left, right = (tuple(run for run in replay[1] if run.subject_id in people) for replay in (before, after))
    paired = compare_alert_runs(left, right, (), ())["episodes"]
    return {field: paired[field] for field in ("gained", "lost")}


def summarize_uncertainty(replays, people):
    """Use identical participant draws for both models and preserve paired denominators."""
    counts = {name: _participant_counts(replay, people) for name, replay in replays.items()}
    baseline = counts["baseline"]
    support = [1, 3, 4]
    if any(not np.array_equal(baseline[:, support], rows[:, support]) for rows in counts.values()):
        raise ValueError("Every frozen replay must have the same participant reference support")
    episode_people = baseline[:, 1] > 0
    largest = int(np.argmax(baseline[:, 1]))
    if int(episode_people.sum()) != 16 or int(baseline[largest, 1]) != 157:
        raise ValueError("The original participant episode distribution must be preserved")
    others = episode_people.copy()
    others[largest] = False
    influence = {
        name: {
            "macro_recall_16": float(np.mean(rows[episode_people, 0] / rows[episode_people, 1])),
            "largest_episode_contributor": {"detected": int(rows[largest, 0]), "complete": int(rows[largest, 1])},
            "other_15_participants_with_complete_episodes": {
                "detected": int(rows[others, 0].sum()), "complete": int(rows[others, 1].sum()),
                "macro_recall": float(np.mean(rows[others, 0] / rows[others, 1])),
            },
        }
        for name, rows in counts.items()
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(people), size=(BOOTSTRAP_REPETITIONS, len(people)))
    _, episodes, _, exposure, analyzable_non_af = baseline[draws].sum(axis=1).T
    wrist_valid = (exposure > 0) & (analyzable_non_af > 0)
    largest_people = {people[largest]}
    other_people = set(np.asarray(people)[others])
    paired = {}
    for name in VARIANTS:
        delta = counts[name] - baseline
        detected, _, false_starts, _, _ = delta[draws].sum(axis=1).T
        largest_changes = _episode_changes(replays["baseline"], replays[name], largest_people)
        other_changes = _episode_changes(replays["baseline"], replays[name], other_people)
        largest_net = largest_changes["gained"] - largest_changes["lost"]
        others_net = other_changes["gained"] - other_changes["lost"]
        net = largest_net + others_net
        if largest_net != delta[largest, 0] or net != delta[:, 0].sum():
            raise ValueError("The paired influence strata must sum to the total detection change")
        people_deltas = delta[episode_people, 0]
        paired[name] = {
            "sensitivity_difference": {"point": float(net / baseline[:, 1].sum()),
                                       **_paired_interval(detected, episodes, episodes > 0)},
            "false_starts_per_observed_wrist_hour_difference": {
                "point": float(3600 * delta[:, 2].sum() / baseline[:, 3].sum()),
                **_paired_interval(3600 * false_starts, exposure, wrist_valid),
            },
            "participant_macro_sensitivity_difference": (
                influence[name]["macro_recall_16"] - influence["baseline"]["macro_recall_16"]
            ),
            "participants_with_complete_episodes_by_detection_change": {
                "increased": int(np.sum(people_deltas > 0)),
                "decreased": int(np.sum(people_deltas < 0)),
                "unchanged": int(np.sum(people_deltas == 0)),
            },
            "paired_counts_by_contribution_stratum": {"largest": largest_changes, "others": other_changes},
            "largest_fraction_of_net_detection_gain": largest_net / net if net else None,
            "largest_fraction_of_gross_gains": largest_changes["gained"] / (
                largest_changes["gained"] + other_changes["gained"]
            ),
        }
    return {
        "scope": "Descriptive paired participant-cluster bootstrap of frozen retrospective wrist replays",
        "retrospective_participants": len(people),
        "resamples": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "reconciled_headlines": {name: replay[0] for name, replay in replays.items()},
        "participant_recall": influence,
        "paired_differences_versus_historical": paired,
    }


def _read_retrospective_inputs(args):
    retro = _read_pinned(args.retro_results, _RETROSPECTIVE_SHA256, public=True)
    pins = _read_pinned(args.data_preflight, _PREFLIGHT_SHA256, public=True)["sha256"]
    protocol = _read_pinned(args.protocol_lock, pins["protocol_lock"])
    lock = _read_pinned(args.historical_model_lock, pins["independent_lock"])
    frozen = _read_pinned(args.variant_lock, retro["development_variant_lock_sha256"])
    coverage = _read_pinned(args.coverage_report, pins["independent_coverage_report"], public=True)
    seal = _read_pinned(args.sealed_inputs / "sealed-input-proof.json", pins["independent_seal"])
    directories = {scope: args.sealed_inputs / scope for scope in _SCOPES}
    return protocol, lock, frozen, coverage, retro, directories, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("retro-results", "protocol-lock", "data-preflight", "variant-lock", "historical-model-lock",
                 "coverage-report", "sealed-inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Use a new output path")
        report = summarize_uncertainty(*_verified_replays(*_read_retrospective_inputs(args)))
        report["retrospective_results_sha256"] = _RETROSPECTIVE_SHA256
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Retrospective wrist uncertainty audit failed: {error}\n")
    print(f"Completed {report['resamples']} paired retrospective participant draws")


if __name__ == "__main__":
    main()
