"""Check causal confirmation and alert scoring at window boundaries."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun, confirm_alerts
from ppg_af.af_wrist_operating import AFScoredRun, score_operating_point, select_operating_point


class ConfirmationTests(unittest.TestCase):
    def test_alert_can_start_only_at_the_end_of_second_positive_window(self):
        run = AFScoredRun(0, 30, np.array([[40, 80]]), np.array([0.9, 0.8, 0.1]))
        immediate = score_operating_point([run], 0.5)
        delayed = score_operating_point([run], 0.5, confirmation_windows=2)
        self.assertEqual(immediate["episode_metrics"]["detected_complete_episodes"], 0)
        self.assertEqual(delayed["episode_metrics"]["detected_complete_episodes"], 1)
        self.assertEqual(delayed["episode_metrics"]["median_complete_episode_alert_delay_seconds"], 20)
        self.assertEqual(delayed["burden_metrics"]["false_alert_starts"], 0)
        self.assertEqual(delayed["burden_metrics"]["analyzable_seconds"], 90)

    def test_calibration_uses_the_same_confirmation_rule(self):
        run = AFScoredRun(0, 30, np.array([[40, 80]]), np.array([0.9, 0.8, 0.1]))
        immediate = select_operating_point([run])
        delayed = select_operating_point([run], confirmation_windows=2)
        self.assertEqual(immediate["status"], "target_not_met")
        self.assertEqual(delayed["status"], "target_met")
        self.assertEqual(delayed["threshold"], 0.8)

    def test_contiguity_abstentions_and_subjects_control_the_streak(self):
        missing = AFRun(0, 30, np.empty((0, 2)), np.array([1, -1, 1]), subject_id="a")
        contiguous = AFRun(90, 30, np.empty((0, 2)), np.array([1, 0, 1]), subject_id="a")
        separate = AFRun(210, 30, np.empty((0, 2)), np.array([1]), subject_id="a")
        other_subject = AFRun(90, 30, np.empty((0, 2)), np.array([1]), subject_id="b")

        results = confirm_alerts([contiguous, missing, separate, other_subject], 2)
        for result, expected in zip(results, ([1, 0, 0], [0, -1, 0], [0], [0]), strict=True):
            np.testing.assert_array_equal(result.decisions, expected)
        np.testing.assert_array_equal(missing.decisions, [1, -1, 1])

    def test_confirmation_uses_model_decisions_without_reference_labels(self):
        run = AFRun(0, 30, np.empty((0, 2)), np.array([1, 1]), reference_valid=np.array([False, True]))
        result, = confirm_alerts([run], 2)
        np.testing.assert_array_equal(result.decisions, [0, 1])
        np.testing.assert_array_equal(result.reference_valid, [False, True])


if __name__ == "__main__":
    unittest.main()
