import unittest

from ppg_af.af_wrist_low_quality_stage import _optimal_nonoverlap


def _candidate(start, quality):
    return start, quality, None, None


class LowQualityStageTest(unittest.TestCase):
    def test_joint_optimum_can_skip_an_individually_better_candidate(self):
        options = [(0, [_candidate(270, 0.10), _candidate(240, 0.30)]),
                   (1, [_candidate(330, 0.30), _candidate(300, 0.05)])]
        result = _optimal_nonoverlap(options)
        self.assertEqual([row[0] for row in result], [240, 300])

    def test_equal_total_quality_prefers_the_earliest_chronological_starts(self):
        options = [(0, [_candidate(60, 0.25), _candidate(30, 0.25)]),
                   (1, [_candidate(330, 0.25), _candidate(300, 0.25)])]
        result = _optimal_nonoverlap(options)
        self.assertEqual([row[0] for row in result], [30, 300])


if __name__ == "__main__":
    unittest.main()
