"""Check the external wrist clock and chest ECG consensus gate."""

from datetime import time
from types import SimpleNamespace
import unittest

import numpy as np

from ppg_af.af_wrist_move import activity_clock, consensus_heart_rate, ECG_SOURCE_RATE


def _record(clock, first, second):
    annotations = [SimpleNamespace(text="run", onset=first, duration=500),
                   SimpleNamespace(text="walk_after", onset=second, duration=60)]
    return SimpleNamespace(starttime=clock, annotations=annotations)


class MoveReferenceTests(unittest.TestCase):
    def test_shared_events_refine_the_header_clock_and_disagreement_blocks_use(self):
        chest = _record(time(12, 0, 3), 10.0, 530.0)
        wrist = _record(time(12, 0, 0), 13.4, 533.4)

        offset, activity = activity_clock(chest, wrist)
        self.assertAlmostEqual(offset, 3.4)
        self.assertEqual(activity, [("run", 10, 510), ("walk_after", 530, 590)])
        self.assertIsNone(activity_clock(chest, _record(time(12, 0, 0), 13.4, 533.5)))

    def test_reference_requires_both_chest_channels_to_locate_the_same_beats(self):
        times = np.arange(30 * ECG_SOURCE_RATE) / ECG_SOURCE_RATE
        position = (times + 0.5) % 1 - 0.5
        gel = np.exp(-0.5 * ((position - 0.2) / 0.015) ** 2)
        dry = -np.roll(gel, 5)

        self.assertAlmostEqual(consensus_heart_rate(np.stack((gel, dry))), 60.0)
        self.assertTrue(np.isnan(consensus_heart_rate(np.stack((gel, np.roll(dry, 180))))))
