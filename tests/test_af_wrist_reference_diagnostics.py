"""Check sealed-preroll boundaries and reference-assisted beat repairs."""

import unittest

import numpy as np

from ppg_af.af_wrist_beat_error_decomposition import START, interval_statistics, match_indices, repair_sequences
from ppg_af.af_wrist_preroll_provenance import _history_states, _origin


class PrerollProvenanceTests(unittest.TestCase):
    def test_unobserved_confirmation_predecessor_is_not_a_reset(self):
        states = _history_states(np.array([0.75, 0.80]), np.array([0.30, 0.30]), np.ones(2, dtype=bool), 0.5, True)
        self.assertEqual(states, [None, 1])
        self.assertIsNone(_origin(states, [-10, 0]))

    def test_known_gap_can_reset_a_positive_tail(self):
        states = _history_states(np.array([np.nan, 0.80]), np.array([0.30, 0.50]), np.ones(2, dtype=bool), 0.5, True)
        self.assertEqual(states, [-1, 1])
        self.assertEqual(_origin(states, [-10, 0]), (0.0, -1))


class BeatErrorDecompositionTests(unittest.TestCase):
    def test_repairs_only_touch_the_selected_error_type(self):
        reference = np.array([100, 200, 300, 400])
        optical = np.array([105, 197, 370])
        indices = match_indices(reference, optical, 10)
        original, timing, occurrence, both = repair_sequences(reference, optical, *indices)
        np.testing.assert_array_equal(original, optical)
        np.testing.assert_array_equal(timing, [100, 200, 370])
        np.testing.assert_array_equal(occurrence, [105, 197, 300, 400])
        np.testing.assert_array_equal(both, reference)

    def test_pnn50_excludes_an_exact_fifty_millisecond_change(self):
        self.assertEqual(interval_statistics(START + np.array([10, 210, 435]))[2], 0.0)
        self.assertEqual(interval_statistics(START + np.array([10, 210, 436]))[2], 1.0)
