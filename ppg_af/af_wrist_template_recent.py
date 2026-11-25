"""Add a causal recent beat-irregularity feature to the existing template ECG."""

import numpy as np

from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, RDDM_FEATURE_VERSION, WINDOW_SECONDS, read_feature_version
from .af_wrist_holdout import _shared_features
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_stride import STRIDE
from .features import ECG_FEATURE_NAMES, _interval_features, ecg_peaks


RECENT_SECONDS = 10
RECENT_FEATURE = "recent_minus_full_normalized_interval_rmssd"
_RMSSD_COLUMN = ECG_FEATURE_NAMES.index("successive_interval_rmssd_over_median")
_ROW_FIELDS = ("subject_id", "start_s", "run_kind", "run_id", "reference_valid")


def _recent_rmssd(signal, peaks, rate):
    """Use two adjacent supported intervals wholly inside the latest 10 seconds."""
    start = len(signal) - RECENT_SECONDS * rate
    present = np.isfinite(signal[start:])
    recent_peaks = peaks[peaks >= start] - start
    if len(recent_peaks) < 3:
        return None
    missing = np.r_[0, np.cumsum(~present)]
    valid = np.diff(missing[recent_peaks]) <= 0.1 * np.diff(recent_peaks)
    if not np.any(valid[:-1] & valid[1:]):
        return None
    return _interval_features(recent_peaks, rate, present)[_RMSSD_COLUMN]


def _archive_features(raw_path, feature_path, template, *, finer):
    with (np.load(raw_path, allow_pickle=False) as raw, np.load(feature_path, allow_pickle=False) as original):
        if (str(raw["subject_id"].item()) != raw_path.stem or int(raw["ppg_fs"]) != PPG_RATE
                or read_feature_version(original) != RDDM_FEATURE_VERSION
                or any(not np.array_equal(raw[name], original[name]) for name in _ROW_FIELDS)):
            raise ValueError(f"Raw PPG and template features disagree in {raw_path.name}")
        if finer and (raw["decision_stride_seconds"].item() != STRIDE
                      or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                      or not np.array_equal(raw["decision_end_s"] - WINDOW_SECONDS, raw["input_start_s"])
                      or not np.array_equal(raw["start_s"] + STRIDE, raw["decision_end_s"])):
            raise ValueError(f"Unexpected decision timing in {raw_path.name}")
        pulses = raw["ppg"]
        before = original["pulse_template"].copy()
        if finer:
            before[raw["input_status"] == "unknown"] = np.nan
        if (pulses.shape != (len(before), (WINDOW_SECONDS + 2) * PPG_RATE)
                or before.shape != (len(raw["start_s"]), len(ECG_FEATURE_NAMES))):
            raise ValueError(f"Unexpected pulse or feature shape in {raw_path.name}")

    extra = np.full(len(before), np.nan)
    recent_supported = np.zeros(len(before), dtype=bool)
    pad = 2 * PPG_RATE
    for row in np.flatnonzero(np.isfinite(before).all(axis=1)):
        core = pulses[row, PPG_RATE:-PPG_RATE]
        if not np.isfinite(core).all() or np.ptp(core) <= 0:
            raise ValueError(f"Existing template feature lacks its PPG input in {raw_path.name}")
        # Match the original template reconstruction. The copy-pad uses only
        # samples inside the trailing 30 seconds, including at new decision times.
        pulse = np.pad(core, (pad, 0), mode="reflect")
        reconstructed = template.predict(pulse)[pad:]
        peaks = ecg_peaks(reconstructed, PPG_RATE)
        complete = _interval_features(peaks, PPG_RATE, np.isfinite(reconstructed))
        if not np.array_equal(complete, before[row]):
            raise ValueError(f"Reconstructed 30-second template features changed in {raw_path.name}")
        recent = _recent_rmssd(reconstructed, peaks, PPG_RATE)
        # Use the full-window statistic when recent interval support is too
        # sparse. The resulting zero contrast preserves the original scoreability.
        extra[row] = recent - complete[_RMSSD_COLUMN] if recent is not None else 0.0
        recent_supported[row] = recent is not None
    return before, extra, recent_supported


def append_recent_template_feature(grid, development_raw, development_features, stride_raw, stride_features, *, ptt_dir):
    """Return an augmented template matrix and aggregate sample support for one locked grid."""
    if grid.seconds != STRIDE:
        raise ValueError("This comparison uses the original 10-second decision grid")
    participants = np.unique(grid.subjects)
    _, template = fit_ptt_calibration(ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    blocks, subjects = [], []
    for finer, raw_dir, feature_dir in (
        (False, development_raw, development_features),
        (True, stride_raw, stride_features),
    ):
        for subject in participants:
            name = f"{subject}.npz"
            block = _archive_features(raw_dir / name, feature_dir / name, template, finer=finer)
            blocks.append(block)
            subjects.extend([subject] * len(block[0]))
    original, extra, support = [np.concatenate(parts) for parts in zip(*blocks, strict=True)]
    if (not np.array_equal(subjects, grid.subjects)
            or not np.array_equal(original, grid.table["pulse_template"], equal_nan=True)):
        raise ValueError("The augmented feature order differs from the baseline decision grid")
    shared = _shared_features(grid.table)
    fit = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    scored = np.zeros(len(original), dtype=bool)
    history = np.zeros(len(original), dtype=bool)
    for block in grid.timelines:
        scored[block.rows] = True
    for rows in grid.history_rows.values():
        history[rows] = True
    counts = {
        name: {"shared_rows": int(rows.sum()), "recent_interval_supported": int(np.sum(rows & support)),
               "full_window_fallback": int(np.sum(rows & ~support))}
        for name, rows in (
            ("classifier_fit", fit), ("evaluation", scored & shared), ("preceding_history", history & shared),
        )
    }
    return np.column_stack((original, extra)), counts
