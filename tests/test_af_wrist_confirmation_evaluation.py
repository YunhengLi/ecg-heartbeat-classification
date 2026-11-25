"""Keep paired wear estimates on the full, fixed reference clock."""

from dataclasses import replace
import unittest

import numpy as np

from ppg_af.af_events import AFRun
from ppg_af.af_wrist_confirmation_evaluation import summarize_episodes, summarize_wear


class ConfirmationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.people = ["with-recording", "without-recording"]
        self.draws = np.array([[0, 1], [0, 0], [1, 1]])
        self.wear = AFRun(0, 10, np.empty((0, 2)), np.array([1, 0, 0, 0]),
                          np.array([True, True, True, False]), subject_id=self.people[0])

    def test_abstention_reduces_coverage_without_removing_observed_time(self):
        candidate = replace(self.wear, decisions=np.array([-1, 0, 0, 0]))
        report = summarize_wear([self.wear], [candidate], self.people, self.draws)
        self.assertEqual(report["scheduled_wrist_seconds"], 40)
        self.assertEqual(report["ecg_observed_wrist_seconds"], 30)
        self.assertEqual(report["original"]["monitorable_seconds"], 20)
        self.assertEqual(report["confirmation"]["monitorable_seconds"], 10)
        self.assertEqual(report["original"]["false_starts_per_observed_hour"], 120)
        self.assertEqual(report["confirmation"]["false_starts_per_observed_hour"], 0)
        self.assertEqual(report["paired_changes"]["coverage_of_observed_time"]["valid_draws"], 2)

    def test_rejects_different_reference_intervals(self):
        candidate = replace(self.wear, reference_valid=np.array([True, False, True, True]))
        with self.assertRaisesRegex(ValueError, "identical reference"):
            summarize_wear([self.wear], [candidate], self.people, self.draws)

    def test_missing_ppg_decisions_leave_complete_episode_in_denominator(self):
        episode = replace(self.wear, af_intervals_s=np.array([[5, 18]]), alert_start_status=np.array([1, 0, 0, 0]))
        candidate = replace(episode, decisions=np.array([-1, 0, 0, 0]), alert_start_status=np.zeros(4, dtype=int))
        report = summarize_episodes([episode], [candidate], self.people, self.draws)
        self.assertEqual(report["complete_ecg_episodes"], 1)
        self.assertEqual(report["original"]["detected"], 1)
        self.assertEqual(report["confirmation"]["detected"], 0)
