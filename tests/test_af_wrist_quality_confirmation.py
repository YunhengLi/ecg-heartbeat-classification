"""Keep conditional alert confirmation causal and preserve missingness."""

import unittest

import numpy as np

from ppg_af.af_wrist_quality_confirmation import confirm_scores


class ConditionalConfirmationTests(unittest.TestCase):
    def test_only_current_low_quality_uses_the_prior_raw_score(self):
        scores = np.array([0.3, 0.8, 0.7, np.nan, 0.9, 0.8])
        previous = np.arange(len(scores)) - 1
        low_quality = np.array([True, False, True, True, True, True])

        changed = confirm_scores(scores, previous, low_quality)

        np.testing.assert_array_equal(changed, [np.nan, 0.8, 0.7, np.nan, np.nan, 0.8])
