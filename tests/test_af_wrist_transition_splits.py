"""Test that extra training labels cannot redefine held-out participant groups."""

import unittest
from unittest.mock import patch

import numpy as np

from ppg_af import af_wrist_cv


class SupplementalFitTests(unittest.TestCase):
    def test_original_split_and_training_only_augmentation(self):
        subjects = np.repeat(list("abcdef"), 3)
        labels = np.array([0, 0, -1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 0, -1, 1, 1, -1])
        supplemental = np.full(len(labels), -1)
        supplemental[[2, 5, 14, 17]] = [1, 0, 1, 0]
        sequence = np.arange(len(labels), dtype=float)
        features = {"template": np.column_stack((sequence, sequence % 5))}

        def calibrate(scores, rows):
            return {"threshold": 0.5, "status": "synthetic"}

        common = {"n_splits": 3, "inner_splits": 2, "calibration_selector": calibrate}
        with patch.object(af_wrist_cv, "_nested_folds", wraps=af_wrist_cv._nested_folds) as baseline_split:
            baseline = af_wrist_cv.train_oof(features, labels, subjects, **common)
        with (patch.object(af_wrist_cv, "_nested_folds", wraps=af_wrist_cv._nested_folds) as candidate_split,
              patch.object(af_wrist_cv, "_fit_and_predict", wraps=af_wrist_cv._fit_and_predict) as predict):
            candidate = af_wrist_cv.train_oof(features, labels, subjects, supplemental_fit_labels=supplemental, **common)

        np.testing.assert_array_equal(baseline["fold_assignment"], candidate["fold_assignment"])
        for original, augmented in zip(baseline_split.call_args_list, candidate_split.call_args_list, strict=True):
            for left, right in zip(original.args[:3], augmented.args[:3], strict=True):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(original.args[3]["template"], augmented.args[3]["template"])
            self.assertEqual(original.args[4:], augmented.args[4:])

        used = set()
        for call in predict.call_args_list:
            _, fitted_labels, fit_rows, score_rows = call.args
            self.assertTrue(set(subjects[fit_rows]).isdisjoint(subjects[score_rows]))
            extra = fit_rows[supplemental[fit_rows] >= 0]
            np.testing.assert_array_equal(fitted_labels[extra], supplemental[extra])
            used.update(extra)
        self.assertEqual(used, {2, 5, 14, 17})

    def test_supplemental_labels_do_not_enable_unsupported_outer_folds(self):
        subjects = np.repeat(list("abcd"), 2)
        labels = np.array([1, -1, 1, -1, 0, -1, 0, -1])
        supplemental = np.array([-1, -1, -1, -1, -1, 1, -1, 1])
        matrix = np.arange(len(labels), dtype=float)[:, None]
        matrix[2] = np.nan
        features = {"template": matrix}

        baseline = af_wrist_cv.train_oof(features, labels, subjects, n_splits=2)
        candidate = af_wrist_cv.train_oof(features, labels, subjects, n_splits=2,
                                          supplemental_fit_labels=supplemental)
        supported = np.isfinite(baseline["routes"]["template"]["scores"])
        self.assertEqual(int(supported.sum()), 3)
        np.testing.assert_array_equal(supported, np.isfinite(candidate["routes"]["template"]["scores"]))
