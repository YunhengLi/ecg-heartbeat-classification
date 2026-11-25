"""Score causal rhythm-state duration without attributing it to earlier time."""

from collections import defaultdict

import numpy as np

from .af_events import _grid, _intervals, _TIME_EPSILON_S, _union


def annotated_seconds(edges, episodes):
    """Measure AF time in each bin from sorted, disjoint half-open episodes."""
    cumulative = np.r_[0.0, np.cumsum(episodes[:, 1] - episodes[:, 0])]
    count = np.searchsorted(episodes[:, 1], edges, side="right")
    starts = np.r_[episodes[:, 0], np.inf]
    elapsed = cumulative[count] + np.maximum(0.0, edges - starts[count])
    return np.clip(np.diff(elapsed), 0, np.diff(edges))


def state_durations(runs):
    """Return ECG and prediction durations by participant using exact AF overlap."""
    grouped = defaultdict(list)
    for index, run in enumerate(runs):
        key = ("participant", run.subject_id) if run.subject_id is not None else ("run", index)
        grouped[key].append(run)

    result = {}
    for key, participant_runs in grouped.items():
        episodes = _union(np.concatenate([_intervals(run.af_intervals_s) for run in participant_runs]))
        end = -np.inf
        totals = np.zeros(6)
        for run in sorted(participant_runs, key=lambda item: item.start_s):
            edges, decisions, reference, _ = _grid(run)
            if edges[0] < end - _TIME_EPSILON_S:
                raise ValueError("Rhythm-state scoring requires disjoint observation blocks")
            end = edges[-1]
            duration = np.diff(edges)
            af = annotated_seconds(edges, episodes)
            # A row's decision becomes available at its right edge. The first bin starts without a state.
            available = reference & np.r_[False, decisions[:-1] >= 0]
            positive = reference & np.r_[False, decisions[:-1] == 1]
            totals += (duration[reference].sum(), af[reference].sum(), duration[available].sum(),
                       af[available].sum(), af[positive].sum(), (duration - af)[positive].sum())
        result[key] = totals
    return result


def state_summary(durations):
    people = np.stack(list(durations.values()))
    _, person_observed_af, person_scored, person_scored_af, person_true_positive, person_false_positive = people.T
    known = person_scored > 0
    error = (person_true_positive[known] + person_false_positive[known] - person_scored_af[known]) / person_scored[known]
    observed, af, scored, scored_af, positive_af, positive_non_af = people.sum(axis=0)
    return {"participants_or_independent_runs": len(people), "participants_with_ecg_af_time": int((person_observed_af > 0).sum()),
            "ecg_observed_seconds": float(observed),
            "ecg_af_seconds": float(af), "scoreable_seconds": float(scored), "af_seconds_with_state": float(scored_af),
            "af_seconds_with_positive_state": float(positive_af), "non_af_seconds_with_positive_state": float(positive_non_af),
            "scoreable_coverage": float(scored / observed) if observed else None,
            "af_time_sensitivity": float(positive_af / af) if af else None,
            "non_af_time_false_positive_rate": float(positive_non_af / (observed - af)) if observed > af else None,
            "af_fraction_on_scoreable_time": float(scored_af / scored) if scored else None,
            "predicted_af_fraction_on_scoreable_time": float((positive_af + positive_non_af) / scored) if scored else None,
            "median_participant_absolute_af_fraction_error": float(np.median(np.abs(error))) if len(error) else None}
