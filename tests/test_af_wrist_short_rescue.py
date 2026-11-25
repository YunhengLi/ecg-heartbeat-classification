"""Check that the optional short trigger preserves causal scoring support."""

import unittest

import numpy as np

from ppg_af.af_wrist_short_rescue import short_trigger


class ShortRescueTests(unittest.TestCase):
    def test_only_adds_a_positive_after_a_qualified_rise(self):
        confirmed = np.array([0.1, 0.1, 0.1, 0.8, np.nan, 0.1, 0.1, 0.1])
        original_cutoff = np.array([0.6] * 7 + [np.nextafter(1.0, np.inf)])
        short = np.array([0.3, 0.8, 0.9, 0.8, 0.8, 0.8, np.nan, 0.8])
        short_cutoff = np.full(8, 0.65)
        quality = np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
        predecessor = np.array([-1, 0, 0, 0, 0, 6, 0, 0])
        scores, rows = short_trigger(confirmed, original_cutoff, short, short_cutoff, quality, predecessor)
        np.testing.assert_array_equal(rows, [1])
        np.testing.assert_array_equal(scores, [0.1, 0.6, 0.1, 0.8, np.nan, 0.1, 0.1, 0.1])

    def test_short_threshold_requires_margin_and_does_not_repeat_a_positive(self):
        original = np.full(4, 0.1)
        cutoff = np.full(4, 0.5)
        short = np.array([0.45, 0.55, 0.80, 0.90])
        scored, rows = short_trigger(original, cutoff, short, cutoff, np.full(4, 0.7), np.arange(4) - 1)
        np.testing.assert_array_equal(rows, [])
        np.testing.assert_array_equal(scored, original)
