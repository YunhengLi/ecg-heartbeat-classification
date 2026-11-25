"""Guard training labels and time separation for the new wrist negative pairs."""

import unittest

import numpy as np

from ppg_af.af_wrist_hard_negatives import RMSSD, pick_pairs


class HardNegativeTests(unittest.TestCase):
    def test_pairs_never_include_AF_or_unobserved_ECG(self):
        features = np.ones((30, 10))
        features[:, RMSSD] = np.arange(30)
        valid = np.ones(30, dtype=bool)
        valid[11] = False
        episodes = np.array([[140, 180], [350, 380]])

        pairs = pick_pairs(100, valid, episodes, features, np.random.default_rng(2026))

        self.assertEqual(len(pairs), 2)
        for segment, (control, hard) in enumerate(pairs):
            self.assertGreaterEqual(abs(hard - control), 3)
            self.assertGreater(hard, control)
            for selected in (control, hard):
                self.assertGreaterEqual(selected - 2, segment * 15)
                self.assertLess(selected, (segment + 1) * 15)
                self.assertTrue(valid[selected - 2:selected + 1].all())
                end = 100 + 10 * (selected + 1)
                self.assertFalse(np.any((episodes[:, 0] < end) & (episodes[:, 1] > end - 30)))
