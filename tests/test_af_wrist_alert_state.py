"""Exercise causal alert entry and release on the frozen decision clock."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun
from ppg_af.af_wrist_alert_state import replay_alert_state, replay_score_block
from ppg_af.af_wrist_clock_study import MONITOR_KIND, ScoringGrid
from ppg_af.af_wrist_train import Timeline
from ppg_af.af_wrist_vote_study import _IMMEDIATE, _held_out_replay


def _block(scores, history=(), **margins):
    run = AFRun(0, 10, np.empty((0, 2)), np.zeros(len(scores), dtype=np.int8))
    return replay_score_block(run, scores, history, 0.5, **margins)


class AlertStateTests(unittest.TestCase):
    def test_borderline_entry_requires_a_previous_positive(self):
        run, unknown = _block([0.55, 0.55, 0.10, 0.59, 0.60, 0.55], entry_margin=0.10)

        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(run.decisions, [0, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(run.alert_start_status, [0, 1, 0, 0, 1, 0])

    def test_release_holds_one_shallow_negative_and_closes_on_a_clear_negative(self):
        run, unknown = _block([0.65, 0.46, 0.45, 0.65, 0.10], release_margin=0.05)

        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(run.decisions, [1, 1, 0, 1, 0])
        np.testing.assert_array_equal(run.alert_start_status, [1, 0, 0, 1, 0])

    def test_borderline_active_continuation_and_missing_input_reset(self):
        scores = [0.65, 0.47, 0.55, np.nan, 0.55, 0.55]
        run, _ = _block(scores, entry_margin=0.10, release_margin=0.05)

        np.testing.assert_array_equal(run.decisions, [1, 1, 1, -1, 0, 1])
        np.testing.assert_array_equal(run.alert_start_status, [1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(run.decisions >= 0, np.isfinite(scores))

    def test_only_unobserved_history_can_produce_possible_starts(self):
        scores = [0.55, 0.55, 0.10, 0.60]
        ambiguous, unknown = _block(scores, (None,), entry_margin=0.10)
        known, known_unknown = _block(scores, (np.nan,), entry_margin=0.10)

        self.assertEqual((unknown, known_unknown), (1, 0))
        np.testing.assert_array_equal(ambiguous.decisions, [0, 1, 0, 1])
        np.testing.assert_array_equal(known.decisions, ambiguous.decisions)
        np.testing.assert_array_equal(ambiguous.alert_start_status, [-1, -1, 0, 1])
        np.testing.assert_array_equal(known.alert_start_status, [0, 1, 0, 1])

    def test_unobserved_active_state_can_carry_one_shallow_negative(self):
        scores = [0.47, 0.47, 0.65]
        run, unknown = _block(scores, (None,), release_margin=0.05)
        prefix, _ = _block(scores[:1], (None,), release_margin=0.05)

        self.assertEqual(unknown, 1)
        np.testing.assert_array_equal(run.decisions, [0, 0, 1])
        np.testing.assert_array_equal(run.alert_start_status, [0, 0, 1])
        np.testing.assert_array_equal(prefix.decisions, run.decisions[:1])
        np.testing.assert_array_equal(prefix.alert_start_status, run.alert_start_status[:1])

    def test_grid_carries_adjacent_runs_in_time_order_and_rewarms_gapped_runs(self):
        grid, scores, cutoffs = self._grid()
        _, episodes, monitored = replay_alert_state(grid, scores, cutoffs, entry_margin=0.10, release_margin=0.05)

        for runs in (episodes, monitored):
            self.assertEqual([run.start_s for run in runs], [20, 60, 0])
            np.testing.assert_array_equal([run.decisions for run in runs], [[1, 1], [0, 1], [1, 1]])
            np.testing.assert_array_equal([run.alert_start_status for run in runs], [[0, 0], [0, 1], [1, 0]])

    def test_zero_margins_reproduce_legacy_replay(self):
        grid, scores, cutoffs = self._grid()
        scores[6] = np.nan
        grid.unknown_source[6] = True
        cutoffs[2:4] = 0.57

        reference, reference_episodes, reference_monitored = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
        summary, episodes, monitored = replay_alert_state(grid, scores, cutoffs)

        self.assertEqual(summary, reference)
        for actual, expected in zip((*episodes, *monitored), (*reference_episodes, *reference_monitored), strict=True):
            np.testing.assert_array_equal(actual.decisions, expected.decisions)
            np.testing.assert_array_equal(actual.alert_start_status, expected.alert_start_status)

        block, unknown = _block([0.55, np.nan, 0.65, 0.55], (None,))
        self.assertEqual(unknown, 1)
        np.testing.assert_array_equal(block.decisions, [1, -1, 1, 1])
        np.testing.assert_array_equal(block.alert_start_status, [-1, 0, 1, 0])

    @staticmethod
    def _grid():
        scores = np.array([0.65, 0.47, 0.55, 0.47, 0.55, 0.55, 0.20, 0.20, 0.20])
        timelines = (
            Timeline("a", MONITOR_KIND, np.array([2, 3]), 20, np.empty((0, 2))),
            Timeline("a", MONITOR_KIND, np.array([4, 5]), 60, np.empty((0, 2))),
            Timeline("a", MONITOR_KIND, np.array([0, 1]), 0, np.empty((0, 2))),
        )
        table = {"reference_valid": np.ones(len(scores), dtype=bool), "ppg_available": np.ones(len(scores), dtype=bool)}
        history = {("a", start): np.array([row]) for start, row in ((0, 6), (20, 7), (60, 8))}
        grid = ScoringGrid(10, table, np.full(len(scores), "a"), timelines, history, np.zeros(len(scores), dtype=bool))
        return grid, scores, np.full(len(scores), 0.5)


if __name__ == "__main__":
    unittest.main()
