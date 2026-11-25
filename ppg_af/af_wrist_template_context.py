"""Compare 10-second and 30-second PPG-derived template ECG for wrist AF."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS, read_feature_version
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_measured_ecg_context import FOLDS, SEED, SHORT_SECONDS, _diagnostics, _verify_original_groups
from .af_wrist_quality_study import build_quality_matrices, quality_interaction_features
from .af_wrist_sequence_study import _comparison
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import ECG_FEATURE_NAMES, ecg_features


_ROW_FIELDS = ("subject_id", "start_s", "run_id", "run_kind", "reference_valid")
_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def _pulse_features(template, core):
    pad = 2 * PPG_RATE
    reflected = np.pad(core, (pad, 0), mode="reflect")
    return ecg_features(template.predict(reflected)[pad:], PPG_RATE)


def _archive_window(grid, raw_path, feature_path, participant, offset, *, finer, version):
    with (np.load(raw_path, allow_pickle=False) as raw, np.load(feature_path, allow_pickle=False) as feature):
        if (str(raw["subject_id"].item()) != participant or raw["ppg_fs"].item() != PPG_RATE
                or read_feature_version(feature) != version
                or any(not np.array_equal(raw[key], feature[key]) for key in _ROW_FIELDS)):
            raise ValueError("Native PPG and cached template features must refer to the same rows")
        original = feature[ROUTE]
        if finer:
            end = raw["decision_end_s"]
            if (raw["decision_stride_seconds"].item() != grid.seconds
                    or raw["feature_window_seconds"].item() != WINDOW_SECONDS
                    or not np.array_equal(end - WINDOW_SECONDS, raw["input_start_s"])
                    or not np.array_equal(end - grid.seconds, raw["start_s"])):
                raise ValueError("Finer PPG must use the locked trailing input clock")
            original[raw["input_status"] == "unknown"] = np.nan
        core = raw["ppg"][:, PPG_RATE:-PPG_RATE]
        rows = slice(offset, offset + len(original))
        if (original.shape != (len(core), len(ECG_FEATURE_NAMES))
                or core.shape != (len(original), WINDOW_SECONDS * PPG_RATE)
                or not np.all(grid.subjects[rows] == participant)
                or not np.array_equal(original, grid.table[ROUTE][rows], equal_nan=True)
                or any(not np.array_equal(raw[key], grid.table[key][rows]) for key in ("run_kind", "reference_valid"))):
            raise ValueError("Native PPG rows must align with the full decision grid")
    return core, original, rows


def short_template_features(grid, development_raw, development_features, stride_raw, stride_features, ptt_data,
                            version):
    """Verify stored 30-second templates and rebuild from only the last 10 seconds."""
    _, template = fit_ptt_calibration(ptt_data, PPG_RATE, subject_count=PTT_SUBJECTS)
    matrix = np.full_like(grid.table[ROUTE], np.nan)
    shared = _shared_features(grid.table)
    status, offset = Counter(), 0
    for finer, raw_dir, feature_dir in ((False, development_raw, development_features),
                                       (True, stride_raw, stride_features)):
        for participant in np.unique(grid.subjects):
            name = f"{participant}.npz"
            core, original, rows = _archive_window(grid, raw_dir / name, feature_dir / name, participant, offset,
                                                   finer=finer, version=version)
            for row in np.flatnonzero(shared[rows]):
                pulse = core[row]
                if (not np.isfinite(pulse).all() or np.ptp(pulse) <= 0
                        or not np.array_equal(_pulse_features(template, pulse), original[row])):
                    raise ValueError("Original 30-second pulse features must reproduce exactly")
                recent = pulse[-SHORT_SECONDS * PPG_RATE:]
                if np.ptp(recent) <= 0:
                    status["flat_recent_ppg"] += 1
                    continue
                values = _pulse_features(template, recent)
                if not np.isfinite(values).all() or values[_SUPPORT] <= 0:
                    status["no_supported_recent_template_intervals"] += 1
                    continue
                matrix[offset + row] = values
                status["scoreable_recent_template"] += 1
            offset = rows.stop
    if sum(status.values()) != int(shared.sum()) or offset != len(grid.subjects):
        raise ValueError("Short template support did not reconcile")
    return matrix, dict(sorted(status.items())), template.delay_samples


def _compare_pair(grid, common_grid, original, short, lock, previous, scopes):
    full_fit = fit_template(grid, original, lock, DEFAULT_C, seed=SEED)
    full_summary = _replay(grid, full_fit)[0]
    if any(previous["summary"][key] != value for key, value in full_summary.items()):
        raise ValueError("The original full-support 30-second model did not reproduce")
    if ("inner_calibration_status_counts" in previous
            and previous["inner_calibration_status_counts"] != _calibration_status(full_fit)):
        raise ValueError("The original quality model's inner calibration did not reproduce")
    baseline = full_fit if grid is common_grid else fit_template(common_grid, original, lock, DEFAULT_C, seed=SEED)
    candidate = fit_template(common_grid, short, lock, DEFAULT_C, seed=SEED)
    verify_fit_support(baseline, candidate)
    before, after = _replay(common_grid, baseline), _replay(common_grid, candidate)
    return (full_summary,
            {"summary": before[0], "diagnostics": _diagnostics(common_grid, baseline, before, WINDOW_SECONDS, scopes)},
            {**_comparison(before, after, lock), "diagnostics":
             _diagnostics(common_grid, candidate, after, SHORT_SECONDS, scopes)})


def evaluate_context(grid, lock, short_inputs, quality_matrices, clock_previous, quality_previous):
    short, status, template_delay_samples = short_inputs
    original = grid.table[ROUTE]
    shared = _shared_features(grid.table)
    common = shared & np.isfinite(short).all(axis=1)
    _verify_original_groups(grid, shared, common)
    common_grid = grid if np.array_equal(shared, common) else replace(
        grid, table={**grid.table, ROUTE: np.where(common[:, None], original, np.nan)},
    )
    fit_rows = (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    scopes = {"evaluation": np.concatenate([block.rows for block in grid.timelines]),
              "preceding_history": np.concatenate(list(grid.history_rows.values())),
              "stable_fitting": np.flatnonzero(fit_rows)}
    full, before, after = _compare_pair(grid, common_grid, original, short, lock, {"summary": clock_previous}, scopes)
    print(f"Plain template: {before['summary']['episodes_detected']} and {after['summary']['episodes_detected']} "
          "detections with 30 and 10 seconds of PPG", flush=True)
    quality = quality_matrices["quality"][:, len(ECG_FEATURE_NAMES)]
    quality_short = quality_interaction_features(short, quality)
    quality_full, quality_before, quality_after = _compare_pair(
        grid, common_grid, quality_matrices["quality_interactions"], quality_short, lock, quality_previous, scopes,
    )
    return {
        "scope": "Development-only matched 10s and 30s wrist PPG pulse-template comparison",
        "seed": SEED, "outer_folds": FOLDS, "logistic_C": DEFAULT_C, "decision_stride_seconds": grid.seconds,
        "development_participants": len(np.unique(grid.subjects)), "PTT_template_participants": PTT_SUBJECTS,
        "PTT_template_delay_samples": template_delay_samples, "native_ppg_rate_hz": PPG_RATE,
        "short_support_reasons": status,
        "support_by_scope": {name: {"original_shared_ppg_rows": int(shared[rows].sum()),
                                    "matched_rows": int(common[rows].sum())} for name, rows in scopes.items()},
        "original_30s_full_support": full,
        "pulse_template_30s_common": before, "pulse_template_10s_common": after,
        "exploratory_quality_followup": {
            "timing": "Specified after the plain-template context comparison",
            "native_ppg_quality_window_seconds": WINDOW_SECONDS,
            "original_30s_quality_full_support": quality_full,
            "pulse_template_30s_quality_common": quality_before,
            "pulse_template_10s_quality_common": quality_after,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("ptt-data", "development-raw", "development-features", "stride-raw", "stride-features", "lock",
             "clock-results", "quality-results", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    try:
        lock_bytes = args.lock.read_bytes()
        lock = json.loads(lock_bytes)
        clock = json.loads(args.clock_results.read_bytes())
        quality = json.loads(args.quality_results.read_bytes())
        table, participants, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[name]) for name in ("fit_subjects", "calibration_subjects",
                                                                           "test_roster"))
        version = table[FEATURE_VERSION_KEY]
        if (fit & calibration or (fit | calibration) & reserved or set(participants) != fit | calibration
                or version != lock[FEATURE_VERSION_KEY] or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest()
                or quality["fixed_regularization"]["C"] != DEFAULT_C):
            raise ValueError("Use the original development cohort, decision clock and quality results")
        grid = load_grid(table, participants, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        archives = (args.development_raw, args.development_features, args.stride_raw, args.stride_features)
        short = short_template_features(grid, *archives, args.ptt_data, version)
        quality_matrices = build_quality_matrices(grid, *archives)
        clock_previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        quality_previous = quality["fixed_regularization"]["quality_interactions"]
        report = evaluate_context(grid, lock, short, quality_matrices, clock_previous, quality_previous)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Pulse-template context comparison failed: {error}\n")
    followup = report["exploratory_quality_followup"]
    before, after = (followup[f"pulse_template_{seconds}s_quality_common"]["summary"]
                     for seconds in (WINDOW_SECONDS, SHORT_SECONDS))
    print(f"Quality interactions, 30s: {before['episodes_detected']}/{before['complete_episodes']} detections, "
          f"{before['wrist_false_alerts']} false starts. 10s: {after['episodes_detected']}/"
          f"{after['complete_episodes']} detections, {after['wrist_false_alerts']} false starts.")


if __name__ == "__main__":
    main()
