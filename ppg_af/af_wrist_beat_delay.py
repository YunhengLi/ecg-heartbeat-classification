"""Calibrate the optical pulse-to-R-peak delay using source participants only."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .af_wrist_beat_cnn import EDGE, SAMPLES, load_move, load_shimmer, summarize
from .af_wrist_features import PPG_RATE
from .af_wrist_holdout import _write_json
from .af_wrist_waveform_study import ARCHIVE_ASSUMED_DELAY_MS
from .tcn_metrics import _peak_score
from .template import _pulse_peaks


DELAYS_MS = tuple(range(0, 601, 20))
ARCHIVE_DELAY = ARCHIVE_ASSUMED_DELAY_MS * PPG_RATE // 1000
MASK = np.r_[np.zeros(EDGE, dtype=bool), np.ones(SAMPLES - 2 * EDGE, dtype=bool), np.zeros(EDGE, dtype=bool)]


def delay_counts(reference, pulses):
    """Counts let each fold select a delay without using any held-out ECG."""
    counts = np.empty((len(reference), len(DELAYS_MS), 3), dtype=np.int64)
    for row, (actual, optical) in enumerate(zip(reference, pulses, strict=True)):
        for col, milliseconds in enumerate(DELAYS_MS):
            scores = _peak_score(actual, optical - milliseconds * PPG_RATE // 1000, MASK, PPG_RATE)
            counts[row, col] = scores.reference_count, scores.predicted_count, len(scores.errors_ms)
    return counts


def calibrate(counts, participants):
    """Choose the delay with highest mean participant F1 on the fitting set."""
    grouped = np.stack([counts[participants == participant].sum(axis=0) for participant in np.unique(participants)])
    support = grouped[..., 0] + grouped[..., 1]
    f1 = np.divide(2 * grouped[..., 2], support, out=np.zeros_like(support, dtype=float), where=support > 0).mean(axis=0)
    selected = min(range(len(DELAYS_MS)), key=lambda index: (-f1[index], abs(DELAYS_MS[index] - ARCHIVE_ASSUMED_DELAY_MS)))
    return DELAYS_MS[selected]


def evaluate(move_dir, shimmer_dir):
    source, _ = load_move(move_dir)
    source_pulses = [_pulse_peaks(window, PPG_RATE)[0] for window in source.pulse]
    counts = delay_counts(source.reference, source_pulses)
    out_of_fold = [None] * len(source_pulses)
    delays = Counter()
    for trained, held in GroupKFold(n_splits=5).split(source.pulse, groups=source.participants):
        selected = calibrate(counts[trained], source.participants[trained])
        delays[selected] += 1
        for row in held:
            out_of_fold[row] = source_pulses[row] - selected * PPG_RATE // 1000
    source_summary = summarize(source, {"archive_fixed": [pulse - ARCHIVE_DELAY for pulse in source_pulses],
                                        "fitted_delay": out_of_fold})
    frozen_delay = calibrate(counts, source.participants)
    external = load_shimmer(shimmer_dir)
    pulses = [_pulse_peaks(window, PPG_RATE)[0] for window in external.pulse]
    external_summary = summarize(external, {"archive_fixed": [pulse - ARCHIVE_DELAY for pulse in pulses],
                                            "fitted_delay": [pulse - frozen_delay * PPG_RATE // 1000 for pulse in pulses]})
    return {"study": "Source-calibrated fixed PPG delay for ECG R-peak timing", "delay_grid_ms": DELAYS_MS,
            "source_fold_delay_ms_frequency": dict(sorted(delays.items())), "frozen_delay_ms": frozen_delay,
            "scientisst_move_grouped": source_summary, "shimmer_frozen_transfer": external_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move-dir", type=Path, required=True)
    parser.add_argument("--shimmer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args.move_dir, args.shimmer_dir))


if __name__ == "__main__":
    main()
