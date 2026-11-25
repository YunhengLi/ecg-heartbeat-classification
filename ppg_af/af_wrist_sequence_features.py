"""Summarize successive pulse segments within an existing causal PPG input."""

import numpy as np

from .af_wrist_features import PPG_RATE, RDDM_FEATURE_VERSION, ROUTES, WINDOW_SECONDS, read_feature_version
from .af_wrist_holdout import _shared_features
from .af_wrist_stride import CORE, STRIDE
from .features import ECG_FEATURE_NAMES, _interval_features
from .template import _pulse_peaks


SEQUENCE_BIN_SECONDS = STRIDE
SEQUENCE_BIN_COUNT = WINDOW_SECONDS // SEQUENCE_BIN_SECONDS
SEQUENCE_BIN_FEATURE_NAMES = (
    "median_interval_s", "interval_iqr_over_median", "successive_interval_rmssd_over_median", "interval_support",
)
SEQUENCE_FEATURE_NAMES = tuple(
    f"{earlier}_to_{earlier - SEQUENCE_BIN_SECONDS}_seconds_ago_{name}"
    for earlier in range(WINDOW_SECONDS, 0, -SEQUENCE_BIN_SECONDS)
    for name in SEQUENCE_BIN_FEATURE_NAMES
)
ORDER_INVARIANT_FEATURE_NAMES = tuple(
    f"sorted_bin_{rank}_{name}"
    for rank in range(1, SEQUENCE_BIN_COUNT + 1)
    for name in SEQUENCE_BIN_FEATURE_NAMES
)
_INTERVAL_COLUMNS = [ECG_FEATURE_NAMES.index(name) for name in SEQUENCE_BIN_FEATURE_NAMES[:-1]]
_ROW_FIELDS = ("subject_id", "start_s", "run_id", "run_kind", "stable_label", "reference_valid")
_STRIDE_FIELDS = ("run_start_s", "input_start_s", "decision_end_s", "evaluation", "ppg_available", "input_status")


def _window_features(padded_window):
    """Give each ten-second bin independent pulse estimates and a support flag."""
    bins = padded_window[CORE].reshape(SEQUENCE_BIN_COUNT, SEQUENCE_BIN_SECONDS * PPG_RATE)
    features = np.zeros((SEQUENCE_BIN_COUNT, len(SEQUENCE_BIN_FEATURE_NAMES)))
    for pulse, row in zip(bins, features, strict=True):
        peaks, usable = _pulse_peaks(pulse, PPG_RATE)
        if len(peaks) < 3:
            continue
        # A successive interval change requires two adjacent supported intervals.
        missing = np.r_[0, np.cumsum(~usable)]
        valid = np.diff(missing[peaks]) <= 0.1 * np.diff(peaks)
        if not np.any(valid[:-1] & valid[1:]):
            continue
        row[:-1] = _interval_features(peaks, PPG_RATE, usable)[_INTERVAL_COLUMNS]
        row[-1] = 1.0
    return features.ravel()


def _sort_bins(sequence):
    """Preserve whole bin tuples while removing their positions on the clock."""
    bins = sequence.reshape(-1, SEQUENCE_BIN_COUNT, len(SEQUENCE_BIN_FEATURE_NAMES))
    median, iqr, rmssd, support = bins.transpose(2, 0, 1)
    by_rhythm = np.lexsort((support, median, iqr, rmssd), axis=1)
    return np.take_along_axis(bins, by_rhythm[:, :, None], axis=1).reshape(sequence.shape)


def _archive_sequence(grid, shared, start, subject, raw_path, feature_path, *, finer):
    with (np.load(raw_path, allow_pickle=False) as raw, np.load(feature_path, allow_pickle=False) as source):
        fields = (*_ROW_FIELDS, *_STRIDE_FIELDS) if finer else _ROW_FIELDS
        if (str(raw["subject_id"].item()) != subject or int(raw["ppg_fs"].item()) != PPG_RATE
                or read_feature_version(source) != RDDM_FEATURE_VERSION
                or any(not np.array_equal(raw[name], source[name]) for name in fields)):
            raise ValueError(f"Raw PPG and feature clocks disagree in {raw_path.name}")
        if finer and (raw["decision_stride_seconds"].item() != STRIDE
                      or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                      or not np.array_equal(raw["decision_end_s"] - WINDOW_SECONDS, raw["input_start_s"])
                      or not np.array_equal(raw["start_s"] + STRIDE, raw["decision_end_s"])):
            raise ValueError(f"Unexpected PPG decision timing in {raw_path.name}")

        count = len(raw["start_s"])
        stop = start + count
        rows = slice(start, stop)
        sources_unknown = raw["input_status"] == "unknown" if finer else np.zeros(count, dtype=bool)
        availability = raw["ppg_available"] if finer else source["ppg_available"]
        labels = np.full(count, -1) if finer else source["stable_label"]
        metadata = {"run_kind": raw["run_kind"], "reference_valid": raw["reference_valid"],
                    "stable_label": labels, "ppg_available": availability}
        if (not np.all(grid.subjects[rows] == subject) or len(grid.subjects[rows]) != count
                or not np.array_equal(sources_unknown, grid.unknown_source[rows])
                or any(not np.array_equal(values, grid.table[name][rows]) for name, values in metadata.items())):
            raise ValueError(f"PPG sequence rows do not match the existing scoring grid in {raw_path.name}")
        for route in ROUTES:
            values = source[route]
            values[sources_unknown] = np.nan
            if not np.array_equal(values, grid.table[route][rows], equal_nan=True):
                raise ValueError(f"Route features do not match the scoring grid in {raw_path.name}")

        pulses = raw["ppg"]
        if pulses.shape != (count, (WINDOW_SECONDS + 2) * PPG_RATE):
            raise ValueError(f"Unexpected PPG window shape in {raw_path.name}")
        core = pulses[:, CORE]
        selected = shared[rows]
        if ((finer and np.any(raw["input_status"][selected] != "available"))
                or not np.isfinite(core[selected]).all() or np.any(np.ptp(core[selected], axis=1) <= 0)):
            raise ValueError(f"A scoreable row lacks its complete trailing PPG window in {raw_path.name}")
    matrix = np.full((count, len(SEQUENCE_FEATURE_NAMES)), np.nan)
    for row in np.flatnonzero(selected):
        matrix[row] = _window_features(pulses[row])
    return matrix, stop


def build_sequence_features(grid, development_raw, development_features, stride_raw, stride_features):
    """Return chronological features, sorted control features and support counts.

    Each row uses its own trailing PPG core, excluding the archive padding.
    Original 30-second fit rows, finer evaluation rows and precursor rows all
    use the same feature calculation without looking up neighboring rows.
    """
    if grid.seconds != STRIDE:
        raise ValueError("Use the original 10-second scoring grid")
    shared = _shared_features(grid.table)
    participants = np.unique(grid.subjects)
    offset = 0
    blocks = []
    for finer, raw_dir, feature_dir in ((False, development_raw, development_features),
                                       (True, stride_raw, stride_features)):
        for subject in participants:
            name = f"{subject}.npz"
            matrix, offset = _archive_sequence(grid, shared, offset, subject, raw_dir / name, feature_dir / name,
                                               finer=finer)
            blocks.append(matrix)
    sequence = np.concatenate(blocks)
    if not np.array_equal(np.isfinite(sequence).all(axis=1), shared):
        raise ValueError("Sequence features changed the existing score support")

    fit = (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    evaluation = np.concatenate([timeline.rows for timeline in grid.timelines])
    history = np.concatenate(list(grid.history_rows.values()))
    support = sequence[:, len(SEQUENCE_BIN_FEATURE_NAMES) - 1::len(SEQUENCE_BIN_FEATURE_NAMES)] == 1.0
    summary = {"chronological_columns": list(SEQUENCE_FEATURE_NAMES),
               "order_invariant_columns": list(ORDER_INVARIANT_FEATURE_NAMES)}
    for name, rows in (("classifier_fit", np.flatnonzero(fit)), ("evaluation", evaluation),
                       ("preceding_history", history)):
        eligible = rows[shared[rows]]
        covered = support[eligible]
        summary[name] = {"shared_rows": len(eligible),
                         "supported_rows_by_bin_oldest_first": covered.sum(axis=0).tolist(),
                         "rows_with_all_bins_supported": int(covered.all(axis=1).sum())}
    return sequence, _sort_bins(sequence), summary
