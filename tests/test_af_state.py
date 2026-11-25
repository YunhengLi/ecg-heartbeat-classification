import unittest

import numpy as np

from ppg_af.af_events import AFRun
from ppg_af.af_state import state_durations, state_summary


class RhythmStateDurationTests(unittest.TestCase):
    def test_state_applies_after_decision_and_keeps_missing_time(self):
        run = AFRun(0, 10, np.array([[7, 13], [18, 22]]), np.array([1, -1, 0]))
        report = state_summary(state_durations([run]))
        self.assertEqual(report["ecg_af_seconds"], 10)
        self.assertEqual(report["scoreable_seconds"], 10)
        self.assertEqual(report["af_seconds_with_state"], 5)
        self.assertEqual(report["af_seconds_with_positive_state"], 5)
        self.assertEqual(report["non_af_seconds_with_positive_state"], 5)
        self.assertEqual(report["af_time_sensitivity"], 0.5)
        self.assertEqual(report["predicted_af_fraction_on_scoreable_time"], 1)

    def test_a_new_block_starts_without_prior_state(self):
        first = AFRun(0, 10, np.empty((0, 2)), np.array([1]), subject_id="example")
        second = AFRun(10, 10, np.array([[10, 25]]), np.array([1, 0]), subject_id="example",
                       reference_valid=np.array([True, False]))
        report = state_summary(state_durations([first, second]))
        self.assertEqual(report["ecg_af_seconds"], 10)
        self.assertEqual(report["af_seconds_with_positive_state"], 0)
        self.assertEqual(report["scoreable_seconds"], 0)
        self.assertIsNone(report["af_fraction_on_scoreable_time"])
