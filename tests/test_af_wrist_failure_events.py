"""Check diagnostic labels without changing the reported AF endpoint."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun, evaluate_af
from ppg_af.af_wrist_failure_events import audit_events
from ppg_af.af_wrist_history import apply_alert_history


class FailureEventTests(unittest.TestCase):
    def test_separates_preexisting_state_from_a_new_in_episode_start(self):
        run = AFRun(0, 10, np.array([[13, 28]]), np.array([1, 1, 0, 1, 0, 0]))

        def history(_):
            return (1,)

        confirmed, _ = apply_alert_history([run], confirmation_windows=1, prior_decisions=history)
        events = audit_events([run], confirmed, confirmation_windows=1, prior_decisions=history)
        episode, = events["complete_episodes"]
        false_start, = events["false_alerts"]
        self.assertEqual(episode["primary"], "already_active_at_onset")
        self.assertEqual(episode["active_count"], 1)
        self.assertEqual(evaluate_af(confirmed)["detected_complete_episodes"], 0)
        self.assertEqual(false_start["time_s"], 40)
        self.assertEqual(false_start["af_in_trailing_input_s"], 15)
        self.assertEqual(false_start["previous_reset_kind"], "negative")
        self.assertEqual(false_start["seconds_since_previous_reset"], 10)
        self.assertEqual(false_start["active_duration_s"], 10)
        self.assertFalse(false_start["right_censored"])

    def test_distinguishes_abstention_below_threshold_and_incomplete_confirmation(self):
        run = AFRun(0, 10, np.array([[0, 10], [20, 30], [40, 50]]), np.array([-1, 0, 0, 0, 1, 0]))

        def history(_):
            return (0,)

        confirmed, _ = apply_alert_history([run], confirmation_windows=2, prior_decisions=history)
        events = audit_events([run], confirmed, confirmation_windows=2, prior_decisions=history)
        self.assertEqual([episode["primary"] for episode in events["complete_episodes"]],
                         ["no_score", "below_cutoff", "confirmation_not_reached"])
        self.assertEqual(events["false_alerts"], [])
        self.assertEqual(evaluate_af(confirmed)["detected_complete_episodes"], 0)
