"""Test alert replay when an evaluation block starts mid-streak."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun, evaluate_af
from ppg_af.af_wrist_history import apply_alert_history


class AlertHistoryTests(unittest.TestCase):
    def test_already_active_alert_does_not_restart_inside_evaluation(self):
        run = AFRun(0, 30, np.array([[80, 150]]), np.array([1, 0, 1, 1, 0]))
        (warm,), unknown = apply_alert_history(
            [run], confirmation_windows=2, prior_decisions=lambda _: (1, 1),
        )

        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(warm.decisions, [1, 0, 0, 1, 0])
        np.testing.assert_array_equal(warm.alert_start_status, [0, 0, 0, 1, 0])
        result = evaluate_af([warm])
        self.assertEqual(result["detected_complete_episodes"], 1)
        self.assertEqual(result["median_complete_episode_alert_delay_seconds"], 40)
        self.assertEqual(result["false_alert_starts"], 0)
        self.assertEqual(result["ecg_reference_seconds"], 150)
        self.assertEqual(result["analyzable_seconds"], 150)

    def test_unknown_history_keeps_possible_starts_separate_from_definite_starts(self):
        run = AFRun(0, 30, np.array([[50, 70]]), np.array([1, 1, 0]))
        (warm,), unknown = apply_alert_history(
            [run], confirmation_windows=2, prior_decisions=lambda _: (None,),
        )

        self.assertEqual(unknown, 1)
        np.testing.assert_array_equal(warm.alert_start_status, [-1, -1, 0])
        result = evaluate_af([warm])
        self.assertEqual(result["detected_complete_episodes"], 0)
        self.assertEqual(result["complete_episodes_with_definite_or_possible_alert"], 1)
        self.assertEqual(result["false_alert_starts"], 0)
        self.assertEqual(result["ambiguous_non_af_alert_start_timestamps"], 1)

    def test_abstention_resets_state_and_adjacent_blocks_carry_it(self):
        first = AFRun(0, 30, np.empty((0, 2)), np.array([1]), subject_id="a")
        next_block = AFRun(30, 30, np.empty((0, 2)), np.array([1]), subject_id="a")
        later = AFRun(90, 30, np.empty((0, 2)), np.array([1]), subject_id="a")
        requested = []

        def history(run):
            requested.append(run.start_s)
            return (None, -1)

        results, unknown = apply_alert_history(
            [next_block, first, later], confirmation_windows=2, prior_decisions=history,
        )
        self.assertEqual(requested, [0, 90])
        self.assertEqual(unknown, 0)
        self.assertEqual([result.alert_start_status.item() for result in results], [1, 0, 0])
        self.assertEqual(evaluate_af(results)["false_alert_starts"], 1)
