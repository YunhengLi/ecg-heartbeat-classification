from contextlib import nullcontext
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ppg_af.af_wrist_training_expansion import _check_alignment


class TestTrainingExpansionAlignment(unittest.TestCase):
    def check(self, starts, labels=None):
        starts = np.asarray(starts, dtype=float)
        labels = np.zeros(len(starts), dtype=int) if labels is None else np.asarray(labels)
        original = {"start_s": np.array([300.0])}
        history = {"input_start_s": np.array([210.0]), "decision_end_s": np.array([240.0])}
        with patch("ppg_af.af_wrist_training_expansion.np.load", side_effect=(nullcontext(original), nullcontext(history))):
            _check_alignment(np.full(len(starts), "1"), labels, starts, starts + 30, Path("original"), Path("fine"))

    def test_padding_overlap_with_predecessor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "predecessor"):
            self.check([179.0])
        self.check([178.0])

    def test_time_bin_cap_applies_separately_to_each_rhythm(self):
        with self.assertRaisesRegex(ValueError, "five-minute"):
            self.check([30.0, 90.0])
        self.check([30.0, 90.0], [0, 1])

    def test_added_windows_cannot_share_padding(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self.check([30.0, 61.0], [0, 1])


if __name__ == "__main__":
    unittest.main()
