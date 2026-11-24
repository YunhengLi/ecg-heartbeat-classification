import unittest

import numpy as np

from ppg_af.af_wrist_features import PPG_RATE
from ppg_af.af_wrist_sequence_features import SEQUENCE_BIN_SECONDS, _sort_bins, _window_features


def _pulse(frequency):
    times = np.arange(SEQUENCE_BIN_SECONDS * PPG_RATE) / PPG_RATE
    return np.sin(2 * np.pi * frequency * times)


class WristSequenceFeatureTest(unittest.TestCase):
    def test_tick_features_exclude_archive_padding_and_keep_bins_independent(self):
        padding = np.zeros(PPG_RATE)
        window = np.r_[padding, _pulse(1.0), _pulse(1.2), _pulse(1.4), padding]
        original = _window_features(window).reshape(3, 4)
        np.testing.assert_array_equal(original[:, -1], np.ones(3))

        window[:PPG_RATE] = 57
        window[-PPG_RATE:] = -42
        np.testing.assert_array_equal(_window_features(window).reshape(3, 4), original)

        window[-11 * PPG_RATE:-PPG_RATE] = _pulse(1.7)
        updated = _window_features(window).reshape(3, 4)
        np.testing.assert_array_equal(updated[:2], original[:2])
        self.assertNotEqual(updated[-1, 0], original[-1, 0])

    def test_flat_bin_has_explicit_missing_flag(self):
        padding = np.zeros(PPG_RATE)
        flat = np.zeros(SEQUENCE_BIN_SECONDS * PPG_RATE)
        matrix = _window_features(np.r_[padding, _pulse(1.0), flat, _pulse(1.0), padding]).reshape(3, 4)
        np.testing.assert_array_equal(matrix[:, -1], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(matrix[1], np.zeros(4))

    def test_sorted_control_preserves_tuples_without_their_time_order(self):
        bins = np.array([[0.9, 0.2, 0.2, 1.0], [0.7, 0.3, 0.2, 1.0], [0.0, 0.0, 0.0, 0.0]])
        chronological = np.array([bins.ravel(), bins[[2, 1, 0]].ravel(), np.full(bins.size, np.nan)])
        control = _sort_bins(chronological).reshape(len(chronological), *bins.shape)
        np.testing.assert_array_equal(control[0], bins[[2, 0, 1]])
        np.testing.assert_array_equal(control[1], control[0])
        self.assertTrue(np.isnan(control[-1]).all())


if __name__ == "__main__":
    unittest.main()
