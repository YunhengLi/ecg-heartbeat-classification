"""Calibrate wrist AF alert thresholds on validation subjects and score held-out runs."""

from dataclasses import dataclass
import math

import numpy as np

from .af_events import AFRun, _grid, _intervals, _union, confirm_alerts, evaluate_af


_NO_ALERT_THRESHOLD = float(np.nextafter(1.0, np.inf))


@dataclass(frozen=True)
class AFScoredRun:
    """A contiguous ECG-clock run with one probability or NaN per window."""

    start_s: float
    window_s: float
    af_intervals_s: np.ndarray
    scores: np.ndarray
    reference_valid: np.ndarray | None = None
    ppg_available: np.ndarray | None = None
    subject_id: str | None = None

    def decisions(self, threshold):
        """Keep unavailable windows on the clock when applying a threshold."""
        scores = np.asarray(self.scores, dtype=float)
        if (scores.ndim != 1 or not len(scores) or np.isinf(scores).any()
                or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Run scores must be probabilities in [0, 1] or NaN")
        decisions = np.where(np.isnan(scores), -1, scores >= threshold).astype(np.int8)
        return AFRun(
            start_s=self.start_s,
            window_s=self.window_s,
            af_intervals_s=self.af_intervals_s,
            decisions=decisions,
            reference_valid=self.reference_valid,
            ppg_available=self.ppg_available,
            subject_id=self.subject_id,
        )


def _remaining_non_af_time(runs, *, analyzable_only=False):
    """Compute non-AF seconds in the ECG-valid windows, including partial windows."""
    annotations = {}
    grids = []
    for index, run in enumerate(runs):
        key = ("subject", run.subject_id) if run.subject_id is not None else ("run", index)
        annotations.setdefault(key, []).append(_intervals(run.af_intervals_s))
        edges, decisions, reference, _ = _grid(run)
        if analyzable_only:
            reference = reference & (decisions >= 0)
        grids.append((key, edges, reference))

    durations = {}
    for key, intervals in annotations.items():
        episodes = _union(np.concatenate(intervals))
        durations[key] = (episodes, np.r_[0.0, np.cumsum(episodes[:, 1] - episodes[:, 0])])

    total = 0.0
    for key, edges, reference in grids:
        episodes, cumulative = durations[key]
        # Cumulative AF time at each window edge, including a partially elapsed episode.
        count = np.searchsorted(episodes[:, 1], edges, side="right")
        starts = np.r_[episodes[:, 0], np.inf]
        annotated = cumulative[count] + np.maximum(0.0, edges - starts[count])
        total += float((np.diff(edges) - np.diff(annotated))[reference].sum())
    return max(0.0, total)


def _evaluate_scopes(representative, episodes, *, allow_daily_rate, non_af_seconds, analyzable_non_af_seconds):
    burden = evaluate_af(representative, representative=allow_daily_rate) if representative else None
    if episodes is None and burden is not None:
        episode_metrics = {**burden, "representative_timeline": False, "false_alerts_per_observed_day": None}
    else:
        episode_metrics = evaluate_af(episodes) if episodes else None
    if burden is not None:
        observed = burden["ecg_reference_seconds"]
        burden["representative_timeline"] = True
        burden["ecg_non_af_seconds"] = non_af_seconds
        burden["analyzable_non_af_seconds"] = analyzable_non_af_seconds
        burden["false_alerts_per_observed_hour"] = (
            3600 * burden["false_alert_starts"] / observed if observed and analyzable_non_af_seconds > 0 else None
        )
        if analyzable_non_af_seconds <= 0:
            burden["false_alerts_per_observed_day"] = None
    return {"episode_metrics": episode_metrics, "burden_metrics": burden}


def score_decisions(representative_runs, *, episode_runs=None, allow_daily_rate=False):
    """Pool held-out AFRun decisions, keeping episode and burden cohorts explicit.

    The burden runs must have been selected without ECG rhythm information.
    Supply independent episode runs when the representative sample has limited
    episode coverage. Their time never enters the burden denominator. This
    function omits a per-day rate unless explicitly requested for a study that
    supports daily extrapolation.
    """
    if not isinstance(allow_daily_rate, (bool, np.bool_)):
        raise ValueError("allow_daily_rate must be boolean")
    representative = tuple(representative_runs)
    episodes = None if episode_runs is None else tuple(episode_runs)
    return _evaluate_scopes(
        representative, episodes, allow_daily_rate=allow_daily_rate,
        non_af_seconds=_remaining_non_af_time(representative),
        analyzable_non_af_seconds=_remaining_non_af_time(representative, analyzable_only=True),
    )


def _at_threshold(runs, threshold, confirmation_windows):
    return confirm_alerts((run.decisions(threshold) for run in runs), confirmation_windows)


def score_operating_point(representative_runs, threshold, *, episode_runs=None, allow_daily_rate=False,
                          confirmation_windows=1):
    """Apply a frozen threshold and report episode sensitivity and alert burden."""
    if not math.isfinite(threshold) or not 0 <= threshold <= _NO_ALERT_THRESHOLD:
        raise ValueError("Threshold must be in [0, 1] or the finite no-alert threshold")
    representative = _at_threshold(representative_runs, threshold, confirmation_windows)
    episodes = None if episode_runs is None else _at_threshold(episode_runs, threshold, confirmation_windows)
    return score_decisions(representative, episode_runs=episodes, allow_daily_rate=allow_daily_rate)


def select_operating_point(representative_runs, *, episode_runs=None, target_sensitivity=0.8, max_thresholds=101,
                           confirmation_windows=1):
    """Select the lowest validation alert burden among thresholds meeting the target.

    Search every unique validation probability on small samples. When there
    are too many unique values, use evenly spaced ranks of the sorted unique
    values. Always check a no-alert policy. When none meets the target, keep
    the best achieved sensitivity and make that shortfall explicit.
    """
    if not math.isfinite(target_sensitivity) or not 0 <= target_sensitivity <= 1:
        raise ValueError("Target sensitivity must be in [0, 1]")
    if not isinstance(max_thresholds, int) or max_thresholds < 3:
        raise ValueError("max_thresholds must be an integer of at least three")
    representative = tuple(representative_runs)
    episodes = None if episode_runs is None else tuple(episode_runs)
    no_alert_representative = _at_threshold(representative, _NO_ALERT_THRESHOLD, confirmation_windows)
    no_alert_episodes = None if episodes is None else _at_threshold(episodes, _NO_ALERT_THRESHOLD, confirmation_windows)
    non_af_seconds = _remaining_non_af_time(no_alert_representative)
    analyzable_non_af_seconds = _remaining_non_af_time(no_alert_representative, analyzable_only=True)

    def evaluate(representative_decisions, episode_decisions):
        return _evaluate_scopes(representative_decisions, episode_decisions, allow_daily_rate=False,
                                non_af_seconds=non_af_seconds, analyzable_non_af_seconds=analyzable_non_af_seconds)

    baseline = evaluate(no_alert_representative, no_alert_episodes)
    episode_reference = baseline["episode_metrics"]
    unmet_support = []
    if episode_reference is None or not episode_reference["complete_reference_episodes"]:
        unmet_support.append("no_complete_reference_episode")
    if non_af_seconds <= 0:
        unmet_support.append("no_representative_ecg_non_af_time")
    elif analyzable_non_af_seconds <= 0:
        unmet_support.append("no_scored_representative_non_af_time")
    if unmet_support:
        return {"status": "insufficient_reference", "threshold": None, "target_sensitivity": target_sensitivity,
                "threshold_candidates": 0, "validation": baseline, "reasons": unmet_support}

    pooled = np.concatenate([np.asarray(run.scores, dtype=float) for run in representative + (episodes or ())])
    candidates = np.unique(pooled[np.isfinite(pooled)])
    if len(candidates) >= max_thresholds:
        indices = np.linspace(0, len(candidates) - 1, max_thresholds - 1, dtype=int)
        candidates = candidates[indices]
    best = None
    for threshold in np.r_[candidates, _NO_ALERT_THRESHOLD]:
        metrics = evaluate(
            _at_threshold(representative, threshold, confirmation_windows),
            None if episodes is None else _at_threshold(episodes, threshold, confirmation_windows),
        )
        sensitivity = metrics["episode_metrics"]["complete_episode_recall"]
        false_alerts = metrics["burden_metrics"]["false_alert_starts"]
        met_target = sensitivity >= target_sensitivity
        # Within the target use minimum false alerts, then higher sensitivity.
        key = (0, false_alerts, -sensitivity, -threshold) if met_target else (1, -sensitivity, false_alerts, -threshold)
        if best is None or key < best[0]:
            best = (key, float(threshold), metrics)
    key, threshold, metrics = best
    return {"status": "target_met" if key[0] == 0 else "target_not_met", "threshold": threshold,
            "target_sensitivity": target_sensitivity, "threshold_candidates": len(candidates) + 1,
            "validation": metrics}


def compare_operating_points(test_metrics_by_route, *, target_sensitivity=0.8, sensitivity_tolerance=0.1,
                             validation_target_met=None):
    """Compare held-out alert burdens only when achieved sensitivities are comparable."""
    if (not math.isfinite(target_sensitivity) or not 0 <= target_sensitivity <= 1
            or not math.isfinite(sensitivity_tolerance) or not 0 <= sensitivity_tolerance <= 1):
        raise ValueError("Target sensitivity and tolerance must be in [0, 1]")
    if len(test_metrics_by_route) < 2:
        raise ValueError("At least two held-out routes are needed for a comparison")
    if validation_target_met is not None and set(validation_target_met) != set(test_metrics_by_route):
        raise ValueError("Validation target status is required for the same routes")

    sensitivities, rates, supports = {}, {}, set()
    for route, result in test_metrics_by_route.items():
        episode = result["episode_metrics"] or {}
        burden = result["burden_metrics"] or {}
        sensitivities[route] = episode.get("complete_episode_recall")
        rates[route] = burden.get("false_alerts_per_observed_hour")
        supports.add((episode.get("complete_reference_episodes"), burden.get("ecg_reference_seconds"),
                      burden.get("ecg_non_af_seconds"), burden.get("analyzable_non_af_seconds")))
    summary = {"target_sensitivity": target_sensitivity, "sensitivity_tolerance": sensitivity_tolerance,
               "held_out_episode_sensitivity": sensitivities, "false_alerts_per_observed_hour": rates,
               "lowest_observed_burden_routes": []}
    if any(value is None for value in (*sensitivities.values(), *rates.values())):
        return {**summary, "status": "insufficient_reference"}
    if len(supports) != 1:
        return {**summary, "status": "different_reference_support"}
    lowest, highest = min(sensitivities.values()), max(sensitivities.values())
    if highest - lowest > sensitivity_tolerance + 1e-12:
        return {**summary, "status": "held_out_sensitivities_differ"}
    if validation_target_met is not None and not all(validation_target_met.values()):
        return {**summary, "status": "validation_target_not_met"}
    if lowest + sensitivity_tolerance + 1e-12 < target_sensitivity:
        return {**summary, "status": "held_out_below_target"}
    best_rate = min(rates.values())
    return {**summary, "status": "comparable_held_out_sensitivity",
            "lowest_observed_burden_routes": [route for route, value in rates.items() if value == best_rate]}
