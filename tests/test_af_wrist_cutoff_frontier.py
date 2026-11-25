"""Compare the exact threshold frontier with causal alert replay."""

import unittest

import numpy as np

from ppg_af.af_wrist_cutoff_frontier import cutoff_frontier
from ppg_af.af_wrist_operating import AFScoredRun, _NO_ALERT_THRESHOLD, score_decisions
from ppg_af.af_wrist_vote_study import _IMMEDIATE


def _run(person, start, scores, intervals=(), valid=None):
    values = np.asarray(scores, dtype=float)
    return AFScoredRun(start, 10, np.asarray(intervals).reshape(-1, 2), values, valid, np.isfinite(values), person)


def _legacy(runs, history, cutoff):
    def prior(run):
        score = history[run.subject_id, run.start_s]
        if score is None:
            return (None,)
        return (-1 if np.isnan(score) else int(score >= cutoff),)

    replay, _ = _IMMEDIATE((run.decisions(cutoff) for run in runs), prior_decisions=prior)
    return replay


def _counts(point):
    return point["detected_complete_episodes"], point["false_starts"]


class CutoffFrontierTests(unittest.TestCase):
    def test_interval_counts_match_brute_force_at_every_score_boundary(self):
        episodes_a = [(10, 30), (50, 65), (82, 95)]
        episodes = (
            _run("a", 20, [0.8, 0.8], episodes_a),
            _run("a", 80, [0.9], episodes_a),
            _run("a", 0, [0.8, 0.2], episodes_a),
            _run("a", 50, [0.9, 0.2], episodes_a),
            _run("b", 0, [0.8, 0.2], [(2, 16)]),
        )
        episode_history = {("a", 0): np.nan, ("a", 20): 0.99, ("a", 50): None, ("a", 80): np.nan, ("b", 0): np.nan}
        wear_a = episodes_a[:2]
        wear = (
            _run("a", 20, [0.8, 0.95], wear_a),
            _run("a", 0, [0.7, 0.2], wear_a),
            _run("a", 60, [0.9, np.nan, 0.75], wear_a),
            _run("b", 0, [0.75, 0.55, 0.55, 0.95], valid=np.array([True, True, False, True])),
        )
        wear_history = {("a", 0): np.nan, ("a", 20): 0.99, ("a", 60): None, ("b", 0): 0.55}
        marks = np.r_[0.0, 0.4, 1.0, _NO_ALERT_THRESHOLD, *(run.scores for run in (*episodes, *wear)),
                      [score for score in (*episode_history.values(), *wear_history.values()) if score is not None]]
        marks = np.unique(marks[np.isfinite(marks)])
        marks = np.unique(np.r_[marks, np.nextafter(marks[:-1], np.inf)])
        target, burden_cap = 2 / 3, 1
        points = {}
        for cutoff in marks:
            with self.subTest(cutoff=cutoff):
                expected = score_decisions(_legacy(wear, wear_history, cutoff),
                                           episode_runs=_legacy(episodes, episode_history, cutoff))
                truth, burden = expected["episode_metrics"], expected["burden_metrics"]
                points[cutoff] = truth["detected_complete_episodes"], burden["false_alert_starts"]
                report, _ = cutoff_frontier(episodes, episode_history, wear, wear_history, cutoff, target, burden_cap)

                self.assertEqual(report["complete_episodes"], truth["complete_reference_episodes"])
                self.assertEqual(report["ecg_observed_wear_seconds"], burden["ecg_reference_seconds"])
                locked = report["at_locked_cutoff"]
                self.assertEqual(_counts(locked), points[cutoff])
                self.assertEqual(locked["sensitivity"], truth["complete_episode_recall"])
                self.assertAlmostEqual(locked["false_starts_per_ecg_observed_hour"], burden["false_alerts_per_observed_hour"])

        # A new alert at onset is false, one at offset counts, and a possible
        # start from unknown preroll stays out of the definite-start counts.
        self.assertEqual(points[0.4], (2, 2))
        report, oracle = cutoff_frontier(episodes, episode_history, wear, wear_history, 0.4, target, burden_cap)
        tradeoffs = set(points.values())
        nondominated = {point for point in tradeoffs if not any(other[0] >= point[0] and other[1] <= point[1]
                                                               and other != point for other in tradeoffs)}
        self.assertEqual({_counts(point) for point in report["nondominated_tradeoffs"]}, nondominated)
        self.assertEqual(_counts(report["maximum_sensitivity"]), min(tradeoffs, key=lambda point: (-point[0], point[1])))
        self.assertEqual(_counts(report["maximum_sensitivity"]), points[oracle])
        self.assertEqual(_counts(report["maximum_sensitivity_within_false_start_cap"]),
                         min((point for point in tradeoffs if point[1] <= burden_cap), key=lambda point: (-point[0], point[1])))
        self.assertEqual(_counts(report["least_burden_at_sensitivity_target"]),
                         min((point for point in tradeoffs if point[0] >= 2), key=lambda point: (point[1], -point[0])))


if __name__ == "__main__":
    unittest.main()
