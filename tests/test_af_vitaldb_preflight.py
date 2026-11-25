"""Check that the source adapter keeps reference and waveform gaps explicit."""

import gzip
from io import BytesIO
import unittest

import numpy as np

from ppg_af.af_vitaldb_preflight import decision_windows, load_track, stable_reference


class VitalDBPreflightTests(unittest.TestCase):
    def test_stable_spans_end_at_missing_reference_or_rhythm_transition(self):
        times = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10])
        labels = np.array(["N", "N", "AFIB/AFL", "AFIB/AFL", "AFIB/AFL", "AFIB/AFL", "AFIB/AFL", "N", "N"])
        usable = np.array([True, True, True, True, False, True, True, True, True])

        self.assertEqual(stable_reference(times, labels, usable), [(0, 1, "N"), (2, 3, "AFIB/AFL"), (9, 10, "N")])
        self.assertEqual(list(decision_windows([(0.5, 52.0, "AFIB/AFL"), (52.1, 90.0, "N")])),
                         [(40, "AFIB/AFL"), (50, "AFIB/AFL"), (90, "N")])

    def test_track_csv_keeps_missing_rows_on_case_clock(self):
        track = BytesIO(gzip.compress(b"Time,SNUADC/PLETH\n0,\n0.002,2.0\n,\n0.006,4.0\n"))
        np.testing.assert_equal(load_track(track, "SNUADC/PLETH"), [np.nan, 2.0, np.nan, 4.0])
