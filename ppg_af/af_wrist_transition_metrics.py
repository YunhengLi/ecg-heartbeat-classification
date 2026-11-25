"""Measure paired false starts before complete development AF episodes."""

from collections import defaultdict

import numpy as np

from .af_events import _coverage, _event_at_decision
from .af_wrist_vote_compare import _alert_times, _align_runs


_LOOKBACK_S = 30


def _lookback_counts(times, onsets):
    right = np.searchsorted(times, onsets, side="right")
    left = np.searchsorted(times, onsets - _LOOKBACK_S, side="left")
    return right - left


def _verified_onsets(coverage, episodes):
    whole, _ = _coverage(episodes, coverage)
    onsets = episodes[whole, 0]
    preceding_ends = np.r_[-np.inf, episodes[:-1, 1]][whole] if len(episodes) else np.empty(0)
    lookback = np.column_stack((onsets - _LOOKBACK_S, onsets))
    observed, _ = _coverage(lookback, coverage)
    clean = preceding_ends <= lookback[:, 0]
    return onsets[observed & clean], len(onsets), int((~observed).sum()), int((observed & ~clean).sum())


def _by_participant(runs):
    groups = defaultdict(list)
    for run in runs:
        if run.subject_id is None:
            raise ValueError("Counting overlapping alerts requires participant IDs")
        groups[run.subject_id].append(run)
    for subject, values in groups.items():
        samples, = _align_runs(values, values)
        coverage, episodes, times, _ = _alert_times(samples)
        yield subject, coverage, episodes, times[_event_at_decision(times, episodes) < 0]


def pre_onset_alert_burden(wrist_runs, episode_runs):
    """Count distinct definite false starts across wrist and verified pre-onset scopes."""
    wrist = {subject: times for subject, _, _, times in _by_participant(wrist_runs)}
    complete = supported = missing_ecg = prior_af = pre_onset = overlap = 0
    for subject, coverage, episodes, times in _by_participant(episode_runs):
        onsets, all_complete, unobserved, not_clean = _verified_onsets(coverage, episodes)
        complete += all_complete
        supported += len(onsets)
        missing_ecg += unobserved
        prior_af += not_clean
        # A verified non-AF lookback cannot cross a preceding AF episode. Its
        # intervals are disjoint, so each eligible start enters this set once.
        next_onset = np.r_[onsets, np.inf][np.searchsorted(onsets, times, side="left")]
        eligible = times[times >= next_onset - _LOOKBACK_S]
        pre_onset += len(eligible)
        if subject in wrist:
            overlap += len(np.intersect1d(eligible, wrist[subject]))
    wrist_count = sum(map(len, wrist.values()))
    return {
        "complete_episodes": complete,
        "supported_episodes": supported,
        "excluded_insufficient_ecg_coverage": missing_ecg,
        "excluded_prior_af": prior_af,
        "wrist_false_starts": wrist_count,
        "pre_onset_false_starts": pre_onset,
        "shared_wrist_and_pre_onset_starts": overlap,
        "distinct_false_starts": wrist_count + pre_onset - overlap,
    }


def compare_pre_onset_alert_runs(baseline_episodes, candidate_episodes):
    """Count definite ECG-non-AF starts up to 30 seconds before AF onset.

    Pass paired replayed episode runs on identical ECG and PPG clocks. Each
    eligible episode is completely observed and has 30 seconds of continuous,
    ECG-verified non-AF before onset. Missing PPG does not remove an otherwise
    eligible episode. A decision at onset uses the ECG state immediately before
    that timestamp, as in the primary endpoint. This diagnostic does not change
    which starts count toward the primary AF detection endpoint.
    """
    complete = supported = missing_ecg = prior_af = 0
    baseline_total = candidate_total = shared_exact = 0
    gained = lost = both = neither = 0
    for samples in _align_runs(baseline_episodes, candidate_episodes):
        coverage, episodes, left_times, right_times = _alert_times(samples)
        onsets, all_complete, unobserved, not_clean = _verified_onsets(coverage, episodes)
        complete += all_complete
        missing_ecg += unobserved
        prior_af += not_clean
        supported += len(onsets)

        left_false = left_times[_event_at_decision(left_times, episodes) < 0]
        right_false = right_times[_event_at_decision(right_times, episodes) < 0]
        left_counts = _lookback_counts(left_false, onsets)
        right_counts = _lookback_counts(right_false, onsets)
        baseline_total += int(left_counts.sum())
        candidate_total += int(right_counts.sum())
        shared_exact += int(_lookback_counts(np.intersect1d(left_false, right_false), onsets).sum())

        had_start, has_start = left_counts > 0, right_counts > 0
        gained += int((~had_start & has_start).sum())
        lost += int((had_start & ~has_start).sum())
        both += int((had_start & has_start).sum())
        neither += int((~had_start & ~has_start).sum())

    return {
        "complete_episodes": complete,
        "supported_episodes": supported,
        "excluded_insufficient_ecg_coverage": missing_ecg,
        "excluded_prior_af": prior_af,
        "pre_onset_starts": {
            "baseline": baseline_total,
            "candidate": candidate_total,
            "shared_exact": shared_exact,
            "baseline_only_exact": baseline_total - shared_exact,
            "candidate_only_exact": candidate_total - shared_exact,
        },
        "episodes_with_pre_onset_start": {
            "baseline": both + lost,
            "candidate": both + gained,
            "gained": gained,
            "lost": lost,
            "both": both,
            "neither": neither,
        },
    }
