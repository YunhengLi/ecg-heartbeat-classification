"""Keep unverified peaks from creating artificial long intervals."""

import unittest

import numpy as np

from ppg_af.features import _interval_features


class PulseConsensusTests(unittest.TestCase):
    def test_intervals_on_both_sides_of_a_rejected_peak_stay_unavailable(self):
        peaks = np.array([10, 100, 190, 280])
        reliable = np.array([True, False, True, True])
        present = np.ones(300, dtype=bool)

        result = _interval_features(peaks, 100, present, verified_peaks=reliable)

        self.assertAlmostEqual(result[1], 0.9)
        self.assertAlmostEqual(result[-1], 1 / 3)
        self.assertEqual(result[4], 0.0)
