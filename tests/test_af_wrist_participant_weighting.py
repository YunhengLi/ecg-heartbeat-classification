"""Verify the per-participant balance within each rhythm."""

import unittest

import numpy as np

from ppg_af.af_wrist_cv import _within_rhythm_participant_weights


class ParticipantWeightingTests(unittest.TestCase):
    def test_each_rhythm_and_each_participant_have_equal_loss_weight(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        subjects = np.array(["a", "a", "b", "a", "b", "b", "c"])
        weights = _within_rhythm_participant_weights(labels, subjects)

        self.assertAlmostEqual(float(weights[labels == 0].sum()), 3.5)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 3.5)
        for label, count in ((0, 2), (1, 3)):
            for subject in np.unique(subjects[labels == label]):
                rows = (labels == label) & (subjects == subject)
                self.assertAlmostEqual(float(weights[rows].sum()), 3.5 / count)
