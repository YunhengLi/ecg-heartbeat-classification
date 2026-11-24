"""Regression checks for independent RDDM sampling and feature provenance."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from ppg_af.af_wrist_features import FEATURE_VERSION_KEY, RDDM_FEATURE_VERSION, _current_cache
from ppg_af.af_wrist_train import load_features
from ppg_af.rddm_window import generate_window_seeded_ecg


class DiffusionFixture:
    n_T = 3
    oneover_sqrta = (1.0, 1.01, 1.02, 1.03)
    mab_over_sqrtmab = (0.0, 0.11, 0.12, 0.13)
    sqrt_beta_t = (0.0, 0.21, 0.22, 0.23)

    @staticmethod
    def region_model(signal, condition, time):
        return signal * 0.9 + condition * time[:, None, None] * 0.1

    @staticmethod
    def eps_model(signal, condition, time):
        return (signal - condition) * time[:, None, None]


def _write_archive_fixture(destination, subject, version=None):
    fields = {
        "subject_id": subject, "start_s": [0.0], "run_id": ["wear"], "run_kind": ["monitor_wear"],
        "stable_label": [0], "reference_valid": [True], "ppg_available": [True],
        "af_intervals_s": np.empty((0, 2)), "direct_ppg": np.zeros((1, 9)),
        "pulse_template": np.zeros((1, 7)), "rddm": np.zeros((1, 7)),
    }
    if version is not None:
        fields[FEATURE_VERSION_KEY] = version
    np.savez(destination, **fields)


class WindowSamplingTests(unittest.TestCase):
    def test_window_order_and_global_rng_are_independent(self):
        time = np.arange(3200) / 100
        windows = np.stack([np.sin(2 * np.pi * rate * time) for rate in (0.9, 1.2, 1.5)])
        models = (DiffusionFixture(), torch.nn.Identity(), torch.nn.Identity())
        initial = torch.random.get_rng_state()
        original = generate_window_seeded_ecg(windows[:2], 100, models, batch_size=1, device="cpu")
        expanded = generate_window_seeded_ecg(windows[[2, 1, 0]], 100, models, batch_size=3, device="cpu")
        np.testing.assert_array_equal(original, expanded[[2, 1]])
        self.assertTrue(torch.equal(initial, torch.random.get_rng_state()))

    def test_legacy_features_are_invalidated_and_mixing_rejected(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source, legacy, seeded = root / "source.mat", root / "001.npz", root / "002.npz"
            source.touch()
            _write_archive_fixture(legacy, "001")
            _write_archive_fixture(seeded, "002", RDDM_FEATURE_VERSION)
            self.assertFalse(_current_cache(source, legacy))
            self.assertTrue(_current_cache(source, seeded))
            with self.assertRaisesRegex(ValueError, "versions are mixed"):
                load_features(root)


if __name__ == "__main__":
    unittest.main()
