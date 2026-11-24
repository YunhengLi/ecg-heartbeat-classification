"""Check causal window resampling and motion projection on known signals."""

import unittest

import numpy as np

from ppg_af.af_wrist_exercise import SOURCE_RATE, clocked_window, remove_linear_motion
from ppg_af.af_wrist_features import PPG_RATE, WINDOW_SECONDS


class WristExerciseTests(unittest.TestCase):
    def test_window_excludes_samples_at_or_after_decision(self):
        times = np.arange(40 * SOURCE_RATE) / SOURCE_RATE
        pulse = np.sin(2 * np.pi * times)
        changed = np.where(times < 30, pulse, 1000)

        expected = clocked_window(pulse, times, 30)
        np.testing.assert_array_equal(clocked_window(changed, times, 30), expected)
        self.assertEqual(expected.shape, (WINDOW_SECONDS * PPG_RATE,))
        self.assertTrue(np.isfinite(expected).all())

    def test_gap_leaves_scheduled_window_unavailable(self):
        times = np.arange(40 * SOURCE_RATE) / SOURCE_RATE
        available = (times < 5) | (times >= 5.2)
        observed = times[available]
        pulse = np.sin(2 * np.pi * observed)

        self.assertTrue(np.isnan(clocked_window(pulse, observed, 30)).all())
        self.assertTrue(np.isfinite(clocked_window(pulse, observed, 40)).all())

    def test_filter_removes_linear_artifact_and_leaves_stationary_input(self):
        times = np.arange(WINDOW_SECONDS * PPG_RATE) / PPG_RATE
        clean = np.sin(2 * np.pi * 1.3 * times)
        artifact = np.sin(2 * np.pi * 2.8 * times)
        motion = np.column_stack((artifact, np.zeros((len(times), 5))))
        contaminated = clean + 2 * artifact

        np.testing.assert_array_equal(remove_linear_motion(clean, np.zeros_like(motion)), clean)
        filtered = remove_linear_motion(contaminated, motion)
        self.assertLess(np.mean((filtered - clean) ** 2), 0.1 * np.mean((contaminated - clean) ** 2))
