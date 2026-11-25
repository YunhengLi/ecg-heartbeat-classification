"""Measure attainable episode detections and false starts on frozen wrist scores."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .af_events import _TIME_EPSILON_S, _coverage, _event_at_decision, _grid, _intervals, _union
from .af_wrist_clock_study import scoring_runs
from .af_wrist_confirmation_retrospective import historical_inputs
from .af_wrist_confirmation_spacing import CONFIRMATION_RESULTS_SHA
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import BLOCK_S, _archive_features
from .af_wrist_operating import AFScoredRun, _NO_ALERT_THRESHOLD
from .af_wrist_quality_confirmation import QUALITY_CUTOFF, confirm_scores, preceding_rows
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_template_study import ROUTE
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE, STRIDE_SECONDS


CONFIRMED = "quality_confirmation"
ORIGINAL = "original"
BATCH_SIZE = 128


def _eligible_endpoints(runs, prior, *, count_episodes):
    """Extract current and preceding scores, with the original ECG event support."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.subject_id].append(run)
    selected = []
    complete_count, exposure = 0, 0.0
    for person, blocks in grouped.items():
        annotations = _union(np.concatenate([_intervals(run.af_intervals_s) for run in blocks]))
        clock = [(_grid(run.decisions(2.0)), run.scores) for run in sorted(blocks, key=lambda item: item.start_s)]
        spans = [edges[np.flatnonzero(np.diff(ref, prepend=False, append=False)).reshape(-1, 2)]
                 for (edges, _, ref, _), _ in clock]
        whole, _ = _coverage(annotations, _union(np.concatenate(spans)))
        ids = np.full(len(annotations) + 1, -1, dtype=int)
        ids[np.flatnonzero(whole)] = complete_count + np.arange(whole.sum())
        complete_count += int(whole.sum())

        end, previous = -np.inf, None
        for (edges, _, reference, _), scores in clock:
            start = edges[0]
            if start < end - _TIME_EPSILON_S:
                raise ValueError("Scored runs must be disjoint")
            if abs(start - end) > _TIME_EPSILON_S:
                previous = prior[(person, start)]
            # An unknown predecessor could be positive. Exclude its possible starts.
            previous_scores = np.r_[np.inf if previous is None else previous, scores[:-1]]
            previous_scores[np.isnan(previous_scores)] = -np.inf
            event = _event_at_decision(edges[1:], annotations)
            keep = reference & np.isfinite(scores) & (scores > previous_scores)
            keep &= (ids[event] >= 0) if count_episodes else (event < 0)
            selected.append((previous_scores[keep], scores[keep], ids[event[keep]]))
            exposure += float(np.diff(edges)[reference].sum())
            end, previous = edges[-1], scores[-1]
    arrays = tuple(np.concatenate(items) for items in zip(*selected, strict=True))
    return arrays, complete_count, exposure


def cutoff_frontier(episode_runs, episode_history, wear_runs, wear_history, cutoff, sensitivity_target, burden_cap):
    """Enumerate the threshold intervals giving a definite start at each endpoint."""
    (episode_previous, episode_current, event_ids), complete, _ = _eligible_endpoints(
        episode_runs, episode_history, count_episodes=True,
    )
    (wear_previous, wear_current, _), _, observed_seconds = _eligible_endpoints(wear_runs, wear_history, count_episodes=False)
    if complete <= 0 or observed_seconds <= 0:
        raise ValueError("Require complete episodes and ECG-observed wear")
    points = np.r_[episode_previous, episode_current, wear_previous, wear_current]
    thresholds = np.unique(np.r_[0.0, 1.0, _NO_ALERT_THRESHOLD, cutoff, points[np.isfinite(points)]])
    episode_counts, false_counts = np.zeros(len(thresholds), dtype=int), np.zeros(len(thresholds), dtype=int)
    for start in range(0, len(thresholds), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(thresholds))
        values = thresholds[start:stop]
        activated = (episode_previous[:, None] < values) & (values <= episode_current[:, None])
        rows, columns = np.nonzero(activated)
        hits = np.zeros((complete, len(values)), dtype=bool)
        hits[event_ids[rows], columns] = True
        episode_counts[start:stop] = hits.sum(axis=0)
        false_counts[start:stop] = ((wear_previous[:, None] < values) & (values <= wear_current[:, None])).sum(axis=0)

    def _point(index):
        return {"detected_complete_episodes": int(episode_counts[index]), "sensitivity": float(episode_counts[index] / complete),
                "false_starts": int(false_counts[index]),
                "false_starts_per_ecg_observed_hour": float(3600 * false_counts[index] / observed_seconds)}

    def summarize(mask):
        if not np.any(mask):
            return None
        eligible = np.flatnonzero(mask)
        index = min(eligible, key=lambda i: (-episode_counts[i], false_counts[i]))
        return _point(index)

    target = episode_counts / complete >= sensitivity_target
    target_points = np.flatnonzero(target)
    least_burden = min(target_points, key=lambda i: (false_counts[i], -episode_counts[i])) if len(target_points) else None
    order = np.lexsort((-episode_counts, false_counts))
    keep = order[episode_counts[order] > np.r_[-1, np.maximum.accumulate(episode_counts[order][:-1])]]
    report = {"complete_episodes": complete, "ecg_observed_wear_seconds": observed_seconds,
              "distinct_cutoff_candidates": len(thresholds), "episodes_with_any_definite_start_threshold": int(len(np.unique(event_ids))),
              "at_locked_cutoff": _point(np.searchsorted(thresholds, cutoff)),
              "maximum_sensitivity": summarize(np.ones(len(thresholds), dtype=bool)),
              "maximum_sensitivity_within_false_start_cap": summarize(false_counts <= burden_cap),
              "least_burden_at_sensitivity_target": _point(least_burden) if least_burden is not None else None,
              "nondominated_tradeoffs": [_point(index) for index in keep]}
    # Check one oracle threshold against the actual causal scoring pipeline, keeping its value private.
    return report, thresholds[min(range(len(thresholds)), key=lambda i: (-episode_counts[i], false_counts[i]))]


def _scheduled_scores(schedule, digest, cache, template, model, *, block_seconds):
    batches = {name: ([], {}) for name in (ORIGINAL, CONFIRMED)}
    for person, blocks in schedule.items():
        if not blocks:
            continue
        starts, reference, episodes, features, available = _archive_features(
            cache / f"{person}.npz", person, blocks, digest, template,
            families=("baseline", "quality"), block_seconds=block_seconds,
        )
        for start, valid, data, optical, ppg in zip(starts, reference, features["baseline"],
                                                    features["quality"][:, :, -1], available, strict=True):
            scores = _predict(data, np.isfinite(data).all(axis=1), model["model"])
            confirmed = confirm_scores(scores, np.arange(len(scores)) - 1, optical < QUALITY_CUTOFF)
            history_size = len(scores) - len(valid)
            for name, values in ((ORIGINAL, scores), (CONFIRMED, confirmed)):
                runs, history = batches[name]
                runs.append(AFScoredRun(float(start), STRIDE_SECONDS, episodes, values[history_size:], valid, ppg[history_size:], person))
                history[person, float(start)] = values[history_size - 1] if history_size else None
    return batches


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    previous = _read_pinned(args.historical_confirmation_results, CONFIRMATION_RESULTS_SHA, public=True)
    grid, model, available, roster, quality, _ = historical_inputs(args, protocol, model_lock)
    scores = _predict(grid.table[ROUTE], available, model["model"])
    predecessor, active = preceding_rows(grid)
    confirmed = confirm_scores(scores, predecessor, active & (quality["quality"][:, -1] < QUALITY_CUTOFF))
    earlier, _, first_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    later, _, second_digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, earlier, first_digest)
    prior_long = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    if prior_long["plan_sha256"] != second_digest or set(earlier) != roster or set(later) != roster:
        raise ValueError("Use the original disjoint wear plans")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    frozen, _, _, _ = _locked_model(model_lock, protocol)
    earlier_scores = _scheduled_scores(earlier, first_digest, args.prior_monitoring_cache, template, frozen, block_seconds=BLOCK_S)
    later_scores = _scheduled_scores(later, second_digest, args.monitoring_cache, template, frozen, block_seconds=BLOCK_SECONDS)
    cutoff = model["cutoff"]
    limits = {ORIGINAL: (155, 253), CONFIRMED: (151, 210)}
    arms = {}
    for name, values in ((ORIGINAL, scores), (CONFIRMED, confirmed)):
        _, episodes = scoring_runs(grid, values, roster)
        prior = {}
        for key, rows in grid.history_rows.items():
            prior[key] = None if not len(rows) or grid.unknown_source[rows[-1]] else values[rows[-1]]
        earlier_runs, earlier_prior = earlier_scores[name]
        later_runs, later_prior = later_scores[name]
        arm, oracle_cutoff = cutoff_frontier(episodes, prior, earlier_runs + later_runs, earlier_prior | later_prior,
                                             cutoff, protocol["target_sensitivity"], limits[CONFIRMED][1])
        locked = arm["at_locked_cutoff"]
        if ((locked["detected_complete_episodes"], locked["false_starts"]) != limits[name]
                or arm["complete_episodes"] != 260 or arm["ecg_observed_wear_seconds"] != 59130):
            raise ValueError("The original joint evaluation must reproduce exactly")
        replay, _, _ = _held_out_replay(grid, values, np.full(len(values), oracle_cutoff), _IMMEDIATE)
        if replay["episodes_detected"] != arm["maximum_sensitivity"]["detected_complete_episodes"]:
            raise ValueError("The interval sweep must match the standard causal episode scorer")
        original = previous["historical_episodes_and_monitoring"]
        expected = original["baseline"] if name == ORIGINAL else original["conditional_confirmation"]["summary"]
        if expected["episodes_detected"] != locked["detected_complete_episodes"]:
            raise ValueError("The pinned episode counts must match the locked cutoff")
        arms[name] = arm
        print(f"{name}: {arm['maximum_sensitivity']['detected_complete_episodes']} max detections; "
              f"{arm['maximum_sensitivity_within_false_start_cap']['detected_complete_episodes']} within the false-start cap", flush=True)
    return {"scope": "Outcome-aware threshold diagnostic on previously scored participants",
            "new_threshold_adopted": False, "participants": len(roster), "target_sensitivity": protocol["target_sensitivity"],
            "false_start_cap_for_diagnostic": limits[CONFIRMED][1], "routes": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("lock", "prior-inputs", "data-preflight", "historical-model-lock", "coverage-report", "historical-results",
                "historical-confirmation-results", "prior-monitor-plan", "prior-monitoring-cache", "plan", "monitoring-cache",
                "half-hour-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
