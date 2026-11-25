"""Check interval and alert-edge consistency on the Holter reference clock."""

import unittest

import numpy as np

from ppg_af.af_events import AFRun
from ppg_af.af_shdb_alerts import Holter, replay_record
from ppg_af.af_shdb_ecg import RATE, interval_matrix
from ppg_af.af_wrist_history import apply_alert_history
from ppg_af.features import _interval_features


class HolterReferenceTests(unittest.TestCase):
    def test_regular_measured_intervals_match_original_features(self):
        samples = np.arange(20, 30 * RATE, 160)
        observed = interval_matrix(samples, np.array([30 * RATE]), 30)[0]
        original = _interval_features(samples, RATE, np.ones(30 * RATE, dtype=bool))
        np.testing.assert_allclose(observed, original, atol=1e-7)

    def test_vectorized_record_starts_match_historical_immediate_policy(self):
        probabilities = np.array([0.9, 0.9, np.nan, 0.9, 0.1, 0.9])
        available = np.isfinite(probabilities)
        observed = np.ones(6, dtype=bool)
        episodes = np.empty((0, 2))
        stub = Holter("recording", "participant", np.empty((2, 6, 7)), np.full(6, -1), observed, episodes, 20, available)
        raw_decisions = np.where(available, probabilities >= 0.5, -1)
        original = AFRun(20, 10, episodes, raw_decisions, observed, available, "recording")
        (expected,), _ = apply_alert_history((original,), confirmation_windows=1, prior_decisions=lambda _: ())
        result = replay_record(stub, probabilities, 0.5)
        np.testing.assert_array_equal(result.alert_start_status, expected.alert_start_status)
        np.testing.assert_array_equal(result.decisions, expected.decisions)
