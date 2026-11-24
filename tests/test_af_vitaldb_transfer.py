"""Check that challenging negatives only replace windows inside each source participant."""

from collections import Counter
import unittest

import numpy as np

from ppg_af.af_vitaldb_preflight import AF_OR_FLUTTER
from ppg_af.af_vitaldb_transfer import training_rows
from ppg_af.af_wrist_mimic_transfer import _fit_source


class VitalDBTransferTests(unittest.TestCase):
    def test_replacement_preserves_positive_rows_and_per_person_class_counts(self):
        ectopy = "Patterned Ventricular Ectopy"
        rhythms = np.array(["N", "N", "N", "N", ectopy, ectopy, AF_OR_FLUTTER, AF_OR_FLUTTER, "N", "N"])
        people = np.array([0] * 8 + [1, 1])
        endings = 30 * np.arange(1, len(rhythms) + 1)
        positive = rhythms == AF_OR_FLUTTER

        rows = training_rows(rhythms, people, endings)

        for label in (True, False):
            per_person = [Counter(people[selected[positive[selected] == label]]) for selected in rows.values()]
            self.assertEqual(*per_person)
        self.assertEqual(sum(rhythms[rows["sinus"]] == ectopy), 0)
        self.assertEqual(sum(rhythms[rows["ectopy"]] == ectopy), 2)
        np.testing.assert_equal(rows["sinus"][positive[rows["sinus"]]], rows["ectopy"][positive[rows["ectopy"]]])

    def test_constant_source_features_do_not_change_inference_input_width(self):
        matrix = np.array([[0.1, 1.0], [0.2, 1.0], [0.8, 1.0], [0.9, 1.0]])
        model = _fit_source(matrix, np.array([0, 0, 1, 1]), np.array([0, 1, 2, 3]), drop_constant_features=True)

        np.testing.assert_equal(model.named_steps["variancethreshold"].get_support(), [True, False])
        scores = model.predict_proba(matrix)[:, 1]
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all(np.diff(scores) > 0))
