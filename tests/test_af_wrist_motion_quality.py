"""Check that native motion coherence reduces power from an observed artifact."""

import unittest

import numpy as np

from ppg_af.af_wrist_features import PPG_RATE, WINDOW_SECONDS
from ppg_af.af_wrist_motion_quality import ACCELERATION_RATE, residual_pulse_power


class MotionQualityTests(unittest.TestCase):
    def test_residual_power_removes_motion_coherent_frequency(self):
        ppg_time = np.arange(PPG_RATE * WINDOW_SECONDS) / PPG_RATE
        motion_time = np.arange(ACCELERATION_RATE * WINDOW_SECONDS) / ACCELERATION_RATE
        pulse = np.sin(2 * np.pi * 1.5 * ppg_time) + 2 * np.sin(2 * np.pi * 2.5 * ppg_time)
        motion = np.column_stack((np.sin(2 * np.pi * 2.5 * motion_time), np.zeros((len(motion_time), 2))))

        self.assertAlmostEqual(residual_pulse_power(pulse, np.zeros_like(motion)), 1.0)
        self.assertLess(residual_pulse_power(pulse, motion), 0.35)
