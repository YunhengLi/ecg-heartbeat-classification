"""Measure paired false starts before complete development AF episodes."""

import numpy as np

from .af_events import _coverage, _event_at_decision
from .af_wrist_vote_compare import _alert_times, _align_runs


_LOOKBACK_S = 30


def _lookback_counts(times, onsets):
    right = np.searchsorted(times, onsets, side="right")
    left = np.searchsorted(times, onsets - _LOOKBACK_S, side="left")
    return right - left


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
        whole, _ = _coverage(episodes, coverage)
        if not whole.any():
            continue
        onsets = episodes[whole, 0]
        preceding_ends = np.r_[-np.inf, episodes[:-1, 1]][whole]
        lookback = np.column_stack((onsets - _LOOKBACK_S, onsets))
        observed, _ = _coverage(lookback, coverage)
        clean = preceding_ends <= lookback[:, 0]
        complete += len(onsets)
        missing_ecg += int((~observed).sum())
        prior_af += int((observed & ~clean).sum())
        onsets = onsets[observed & clean]
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
