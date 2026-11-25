"""Check event overlap and alternate calibration without changing the AF target."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun
from ppg_af.af_wrist_operating import select_scored_threshold
from ppg_af.af_wrist_transition_metrics import compare_pre_onset_alert_runs, pre_onset_alert_burden


def _run(subject, count, alert_ends, episodes, *, unobserved=()):
    decisions = np.zeros(count, dtype=np.int8)
    decisions[np.asarray(alert_ends, dtype=int) // 10 - 1] = 1
    reference = np.ones(count, dtype=bool)
    reference[list(unobserved)] = False
    return AFRun(0.0, 10.0, np.asarray(episodes), decisions, reference_valid=reference,
                 subject_id=subject, alert_start_status=decisions.copy())


class OnsetCalibrationTests(unittest.TestCase):
    def test_both_lookback_boundaries_and_exact_participant_overlap(self):
        episode = _run("a", 12, (20, 50, 90), ((50, 70),))
        wrist_a = _run("a", 4, (20,), ((50, 70),))
        wrist_b = _run("b", 4, (20,), ())
        wrist, evaluation = (wrist_a, wrist_b), (episode, wrist_b)
        result = pre_onset_alert_burden(wrist, evaluation)
        self.assertEqual(result["complete_episodes"], 1)
        self.assertEqual(result["supported_episodes"], 1)
        self.assertEqual(result["pre_onset_false_starts"], 2)
        self.assertEqual(result["wrist_false_starts"], 2)
        self.assertEqual(result["shared_wrist_and_pre_onset_starts"], 1)
        self.assertEqual(result["distinct_false_starts"], 3)
        paired = compare_pre_onset_alert_runs(evaluation, evaluation)
        self.assertEqual(paired["pre_onset_starts"]["shared_exact"], 2)

    def test_pre_onset_ecg_gap_and_prior_af_exclude_lookbacks(self):
        run = _run("a", 12, (20, 70), ((30, 45), (70, 90)), unobserved=(0,))
        result = pre_onset_alert_burden((), (run,))
        self.assertEqual(result["complete_episodes"], 2)
        self.assertEqual(result["excluded_insufficient_ecg_coverage"], 1)
        self.assertEqual(result["excluded_prior_af"], 1)
        self.assertEqual(result["supported_episodes"], 0)
        self.assertEqual(result["pre_onset_false_starts"], 0)

    def test_custom_burden_changes_rank_without_lowering_fallback_sensitivity(self):
        pool = (0.3, 0.5, 0.8)

        def evaluate(cutoff):
            sensitivity, wrist, extra = {0.3: (0.75, 0, 2), 0.5: (0.75, 1, 0),
                                         0.8: (0.5, 0, 0)}.get(cutoff, (0.0, 0, 0))
            return {"episode_metrics": {"complete_reference_episodes": 4, "complete_episode_recall": sensitivity},
                    "burden_metrics": {"false_alert_starts": wrist, "ecg_non_af_seconds": 30,
                                       "analyzable_non_af_seconds": 30},
                    "distinct_false_starts": wrist + extra}

        for target, status in ((0.8, "target_not_met"), (0.7, "target_met")):
            with self.subTest(target=target):
                before = select_scored_threshold(pool, evaluate, target_sensitivity=target)
                after = select_scored_threshold(pool, evaluate, target_sensitivity=target,
                                                burden_rank=lambda metrics: metrics["distinct_false_starts"])
                self.assertEqual((before["threshold"], after["threshold"]), (0.3, 0.5))
                self.assertEqual((before["threshold_candidates"], after["threshold_candidates"]), (4, 4))
                self.assertEqual((before["status"], after["status"]), (status, status))
                self.assertEqual(before["validation"]["episode_metrics"], after["validation"]["episode_metrics"])


if __name__ == "__main__":
    unittest.main()
