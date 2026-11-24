"""Check exposure sampling and measured predecessor state on the new wrist clock."""

import unittest

import numpy as np

from ppg_af.af_wrist_monitor_extension import _person_runs
from ppg_af.af_wrist_monitor_plan import choose_monitor_blocks
from ppg_af.afppgecg import WristSegment


class SupplementalMonitorTests(unittest.TestCase):
    def test_planning_keeps_all_native_signal_support_disjoint(self):
        wearer = (WristSegment(0, 0, 6200, "beat"),)
        excluded = np.array([[1000, 2400], [3500, 3900]])

        plan = choose_monitor_blocks(wearer, 6500, excluded, 2026)

        self.assertEqual(plan, choose_monitor_blocks(wearer, 6500, excluded, 2026))
        self.assertEqual(len(plan), 3)
        for item in plan:
            start, stop = item["input_support_s"]
            self.assertFalse(np.any((start < excluded[:, 1]) & (stop > excluded[:, 0])))
            self.assertEqual(item["evaluation_s"][1] - item["evaluation_s"][0], 300)
        for before, after in zip(plan[:-1], plan[1:], strict=True):
            self.assertLessEqual(before["input_support_s"][1], after["input_support_s"][0])

    def test_preceding_positive_window_does_not_start_a_new_alert(self):
        matrix = np.ones((1, 37, 1))
        matrix[0, 8, 0] = -1
        archive = (np.array([100]), np.ones((1, 30), dtype=bool), np.empty((0, 2)), {"baseline": matrix},
                   np.ones((1, 37), dtype=bool))
        model = {"mean": [0.0], "scale": [1.0], "coef": [1.0], "intercept": 0.0}

        (result,) = _person_runs("participant", archive, model, 0.5, "baseline")

        self.assertEqual(result.start_s, 100)
        self.assertEqual(result.decisions.shape, (30,))
        np.testing.assert_equal(np.flatnonzero(result.alert_start_status), [2])
