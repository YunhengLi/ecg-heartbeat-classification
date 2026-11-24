"""Verify permutation expectations and adjacency rules for RR features."""

from itertools import permutations
import unittest

import numpy as np

from ppg_af.af_wrist_interval_order import order_features


class IntervalOrderTests(unittest.TestCase):
    def test_exact_null_matches_all_permutations(self):
        intervals = np.array([0.55, 0.6, 1.0, 1.05])
        valid = np.ones(len(intervals), dtype=bool)
        _, expected = order_features(intervals, valid)
        samples = [order_features(np.asarray(sequence), valid)[0] for sequence in permutations(intervals)]
        np.testing.assert_allclose(np.mean(samples, axis=0), expected)

    def test_gap_blocks_an_apparent_adjacent_transition(self):
        intervals = np.array([0.6, 0.8, 1.0])
        observed, _ = order_features(intervals, np.array([True, False, True]))
        np.testing.assert_array_equal(observed, [0.0, 0.0])
