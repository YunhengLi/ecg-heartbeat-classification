"""Check causal votes, alert starts, and missing history."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun, evaluate_af
from ppg_af.af_wrist_vote import apply_two_of_three_alerts


class TwoOfThreeAlertTests(unittest.TestCase):
    def test_current_positive_can_bridge_one_negative_and_starts_only_once(self):
        run = AFRun(0, 10, np.array([[33, 55]]), np.array([0, 1, 0, 1, 1, 0, 0, 1, 0]))
        (result,), unknown = apply_two_of_three_alerts([run], prior_decisions=lambda _: (-1,))

        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(result.decisions, [0, 0, 0, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result.alert_start_status, [0, 0, 0, 1, 0, 0, 0, 0, 0])
        metrics = evaluate_af([result])
        self.assertEqual(metrics["detected_complete_episodes"], 1)
        self.assertEqual(metrics["median_complete_episode_alert_delay_seconds"], 7)
        self.assertEqual(metrics["false_alert_starts"], 0)

    def test_abstention_clears_prior_votes(self):
        run = AFRun(0, 10, np.empty((0, 2)), np.array([1, -1, 1, 0, 1]))
        (result,), unknown = apply_two_of_three_alerts([run], prior_decisions=lambda _: (-1,))

        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(result.decisions, [0, -1, 0, 0, 1])
        np.testing.assert_array_equal(result.alert_start_status, [0, 0, 0, 0, 1])

    def test_warm_alert_continues_and_adjacent_runs_reuse_state(self):
        first = AFRun(0, 10, np.empty((0, 2)), np.array([1, 1]), subject_id="a")
        adjacent = AFRun(20, 10, np.empty((0, 2)), np.array([0, 1]), subject_id="a")
        separate = AFRun(60, 10, np.empty((0, 2)), np.array([1]), subject_id="a")
        requested = []

        def history(run):
            requested.append(run.start_s)
            return (1, 0, 1) if run.start_s == 0 else (-1,)

        results, unknown = apply_two_of_three_alerts([adjacent, separate, first], prior_decisions=history)

        self.assertEqual(requested, [0, 60])
        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(results[0].alert_start_status, [0, 1])
        np.testing.assert_array_equal(results[1].alert_start_status, [0])
        np.testing.assert_array_equal(results[2].alert_start_status, [0, 0])
        np.testing.assert_array_equal(results[2].decisions, [1, 1])

    def test_unknown_predecessor_can_make_start_uncertain(self):
        run = AFRun(0, 10, np.array([[15, 29]]), np.array([1, 1, 1]))
        (result,), unknown = apply_two_of_three_alerts([run], prior_decisions=lambda _: (1, None))

        self.assertEqual(unknown, 1)
        np.testing.assert_array_equal(result.decisions, [0, 1, 1])
        np.testing.assert_array_equal(result.alert_start_status, [-1, -1, 0])
        metrics = evaluate_af([result])
        self.assertEqual(metrics["detected_complete_episodes"], 0)
        self.assertEqual(metrics["complete_episodes_with_definite_or_possible_alert"], 1)
        self.assertEqual(metrics["ambiguous_non_af_alert_start_timestamps"], 1)
