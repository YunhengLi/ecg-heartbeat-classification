import unittest

import numpy as np

from ppg_af.af_wrist_cv import train_oof


class FoldFeatureTransformTest(unittest.TestCase):
    def test_identity_transform_receives_only_actual_disjoint_nested_fit_and_score_rows(self):
        people = np.repeat(np.array(list("abcdef")), 3)
        labels = np.tile([0, 1, -1], 6)
        matrix = np.column_stack((np.arange(len(labels), dtype=float), labels))
        matrix[labels < 0] = np.nan
        calls = []

        def transform(original, fit_rows, score_rows):
            self.assertFalse(original.flags.writeable)
            np.testing.assert_array_equal(original, matrix)
            self.assertTrue(np.isfinite(matrix[fit_rows]).all())
            participants = set(people[fit_rows])
            self.assertTrue(participants.isdisjoint(people[score_rows]))
            self.assertEqual(set(fit_rows), set(np.flatnonzero(np.isin(people, sorted(participants)) & (labels >= 0))))
            calls.append(len(participants))
            return original

        def calibrate(scores, rows):
            np.testing.assert_array_equal(np.isfinite(scores), labels[rows] >= 0)
            return {"threshold": 0.5, "status": "fixed"}

        options = {"n_splits": 3, "inner_splits": 2, "calibration_selector": calibrate}
        before = train_oof({"template": matrix}, labels, people, **options)
        after = train_oof({"template": matrix}, labels, people, fold_feature_transform=transform, **options)
        self.assertCountEqual(calls, [4, 4, 4, 2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(before["fold_assignment"], after["fold_assignment"])
        for key in ("scores", "decisions"):
            np.testing.assert_array_equal(before["routes"]["template"][key], after["routes"]["template"][key])

        def change_availability(original, fit_rows, score_rows):
            changed = original.copy()
            changed[score_rows[0]] = np.nan
            return changed

        with self.assertRaisesRegex(ValueError, "preserve matrix shape and finite fit and score rows"):
            train_oof({"template": matrix}, labels, people, fold_feature_transform=change_availability, **options)


if __name__ == "__main__":
    unittest.main()
