"""Align raw trailing PPG input evidence to the evaluated decision rows."""

from collections import defaultdict

import numpy as np

from .af_wrist_features import PPG_RATE, WINDOW_SECONDS


def load_input_evidence(grid, raw_dir, feature_dir):
    """Return input status and PPG quality arrays keyed by participant and run start.

    Status describes the entire trailing feature input, including when the
    decision cadence uses cells shorter than the 30-second feature window.
    Quality is NaN when unavailable or the input is incomplete or flat.
    """
    blocks_by_subject = defaultdict(list)
    for block in grid.timelines:
        blocks_by_subject[block.subject].append(block)

    evidence = {}
    for subject, blocks in blocks_by_subject.items():
        with np.load(raw_dir / f"{subject}.npz", allow_pickle=False) as raw, np.load(
            feature_dir / f"{subject}.npz", allow_pickle=False,
        ) as features:
            starts = raw["start_s"]
            kinds = raw["run_kind"]
            run_ids = raw["run_id"]
            if (raw["subject_id"].item() != subject or raw["ppg_fs"].item() != PPG_RATE
                    or any(not np.array_equal(raw[field], features[field])
                           for field in ("subject_id", "start_s", "run_kind", "run_id", "reference_valid"))):
                raise ValueError(f"Input evidence clock does not match features for participant {subject}")
            finer = grid.seconds != WINDOW_SECONDS
            evaluation = raw["evaluation"] if finer else np.ones(len(starts), dtype=bool)
            if finer and (raw["feature_window_seconds"].item() != WINDOW_SECONDS
                          or raw["decision_stride_seconds"].item() != grid.seconds
                          or not np.array_equal(starts + grid.seconds, raw["decision_end_s"])
                          or not np.array_equal(starts + grid.seconds - WINDOW_SECONDS, raw["input_start_s"])):
                raise ValueError(f"Trailing feature window does not match decision clock for participant {subject}")
            waveforms = raw["ppg"]
            quality = features["quality"] if "quality" in features else np.full(len(starts), np.nan)
            if (waveforms.shape != (len(starts), (WINDOW_SECONDS + 2) * PPG_RATE)
                    or quality.shape != starts.shape):
                raise ValueError(f"Input evidence has the wrong dimensions for participant {subject}")
            unknown = raw["input_status"] == "unknown" if finer else np.zeros(len(starts), dtype=bool)
            run_starts = raw["run_start_s"] if finer else None
            for block in blocks:
                expected = block.start + grid.seconds * np.arange(len(block.rows))
                matching = evaluation & (kinds == block.kind) & (starts >= expected[0]) & (starts <= expected[-1])
                if finer:
                    matching &= run_starts == block.start
                rows = np.flatnonzero(matching)
                rows = rows[np.argsort(starts[rows])]
                if (not np.array_equal(starts[rows], expected) or len(np.unique(run_ids[rows])) != 1
                        or not np.array_equal(raw["reference_valid"][rows], grid.table["reference_valid"][block.rows])):
                    raise ValueError(f"Input evidence does not align to the evaluation run for participant {subject}")

                core = waveforms[rows, PPG_RATE:-PPG_RATE]
                finite = np.isfinite(core).all(axis=1)
                flat = np.zeros(len(rows), dtype=bool)
                flat[finite] = np.ptp(core[finite], axis=1) == 0
                usable = finite & ~flat & ~unknown[rows]
                status = np.full(len(rows), "missing_or_nonfinite", dtype="U24")
                status[flat] = "flatline"
                status[usable] = "finite_nonflat"
                status[unknown[rows]] = "unknown_input_support"
                evidence[(subject, block.start)] = {
                    "status": status,
                    "quality": np.where(usable, quality[rows], np.nan),
                }
    return evidence
