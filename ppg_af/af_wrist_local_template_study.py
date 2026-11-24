"""Compare a fold-local wrist pulse template against the external PTT route."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS
from .af_wrist_waveform_study import participant_template_stats
from .features import ECG_FEATURE_NAMES, ecg_features
from .template import fit_template_from_stats


_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def stage_cores(grid, directory_pairs):
    blocks, people = [], []
    for seconds, (raw, feature) in zip((WINDOW_SECONDS, grid.seconds), directory_pairs, strict=True):
        for subject in np.unique(grid.subjects):
            filename = f"{subject}.npz"
            with (np.load(raw / filename, allow_pickle=False) as waveform,
                  np.load(feature / filename, allow_pickle=False) as reference):
                if (waveform["subject_id"].item() != subject or waveform["ppg_fs"].item() != PPG_RATE
                        or any(not np.array_equal(waveform[key], reference[key])
                               for key in ("subject_id", "start_s", "run_id", "run_kind", "reference_valid"))):
                    raise ValueError("Raw PPG and baseline feature clocks differ")
                if seconds != WINDOW_SECONDS and (waveform["decision_stride_seconds"].item() != seconds
                        or not np.array_equal(waveform["input_start_s"] + WINDOW_SECONDS, waveform["decision_end_s"])
                        or not np.array_equal(waveform["start_s"] + seconds, waveform["decision_end_s"])):
                    raise ValueError("Local ECG generation must use the original causal 30-second clock")
                blocks.append(waveform["ppg"][:, PPG_RATE:-PPG_RATE])
                people.extend([subject] * len(blocks[-1]))
    ppg = np.concatenate(blocks)
    if not np.array_equal(people, grid.subjects) or ppg.shape != (len(people), WINDOW_SECONDS * PPG_RATE):
        raise ValueError("Raw waveform rows do not match the grouped scoring table")
    shared = _shared_features(grid.table)
    if not np.isfinite(ppg[shared]).all() or np.any(np.ptp(ppg[shared], axis=1) <= 0):
        raise ValueError("Every original scoreable row requires a complete causal PPG core")
    return ppg


def native_template_stats(path, receipt, lock_hash, roster):
    pair_hash = sha256(path.read_bytes()).hexdigest()
    if (pair_hash != receipt["private_paired_manifest_sha256"] or receipt["original_protocol_sha256"] != lock_hash
            or receipt["locked_participants"] != len(roster) or receipt["waveform_rate_hz"] != PPG_RATE
            or receipt["window_duration_seconds"] != WINDOW_SECONDS or receipt["independent_waveforms_or_predictions"] != 0):
        raise ValueError("Use the verified native waveform bundle")
    with np.load(path, allow_pickle=False) as pairs:
        groups = pairs["subjects"]
        pulse, ecg = pairs["ppg_core"], pairs["ecg_ppg_clock"]
        if (pairs["source_protocol_sha256"].item() != lock_hash or not set(groups) <= roster
                or groups.shape != (223,) or len(set(groups)) != receipt["paired_participants"]
                or pulse.shape != ecg.shape or pulse.shape != (223, WINDOW_SECONDS * PPG_RATE)
                or not np.isfinite(pulse).all() or not np.isfinite(ecg[:, :-1]).all()):
            raise ValueError("The native template pool must contain the exact paired development windows")
        coverage = Counter(zip(pairs["origin"], pairs["stable_label"], strict=True))
        expected = {(origin, label): receipt["signal_provenance"][origin][name]
                    for origin in ("original", "supplemental") for label, name in ((0, "non_AF"), (1, "AF"))}
        if coverage != expected:
            raise ValueError("The template pool did not reproduce the verified paired coverage")
    return participant_template_stats(groups, pulse, ecg), pair_hash


def evaluate(grid, ppg, stats, lock, prior):
    baseline_matrix = grid.table[ROUTE]
    shared = _shared_features(grid.table)
    baseline_fit = fit_template(grid, baseline_matrix, lock, DEFAULT_C)
    baseline = _replay(grid, baseline_fit)
    if any(prior[key] != value for key, value in baseline[0].items()):
        raise ValueError("The original external-PTT template result failed to reproduce")

    counts = Counter()
    pad = 2 * PPG_RATE

    def locally_reconstructed(features, train_rows, score_rows):
        training_people = set(grid.subjects[train_rows])
        scoring_people = set(grid.subjects[score_rows])
        if training_people & scoring_people:
            raise ValueError("Reconstruction and classifier training must exclude every scoring participant")
        source = [stats[person] for person in sorted(training_people & stats.keys())]
        if not source:
            raise ValueError("No paired ECG/PPG training participants in a classifier fit")
        template = fit_template_from_stats(source)
        counts["model_fits"] += 1
        matrix = features.copy()
        combined = np.r_[train_rows, score_rows]
        for index in combined[shared[combined]]:
            pulse = np.pad(ppg[index], (pad, 0), mode="reflect")
            values = ecg_features(template.predict(pulse)[pad:], PPG_RATE)
            if np.isfinite(values).all() and values[_SUPPORT] > 0:
                matrix[index] = values
                counts["local_feature_rows"] += 1
            else:
                counts["external_template_fallback_rows"] += 1
        return matrix

    candidate_fit = fit_template(grid, baseline_matrix, lock, DEFAULT_C, fold_feature_transform=locally_reconstructed)
    verify_fit_support(baseline_fit, candidate_fit)
    candidate = _replay(grid, candidate_fit)
    result = {"scope": "Exploratory fold-local wrist waveform template for AF event detection",
              "external_PTT_template": {**baseline[0], "inner_calibration": _calibration_status(baseline_fit)},
              "fold_local_wrist_template": {**_compare_with_pre_onset(baseline, candidate, lock),
                                            "inner_calibration": _calibration_status(candidate_fit)},
              "feature_generation_counts": dict(counts)}
    print("Local template", candidate[0]["episodes_detected"], candidate[0]["wrist_false_alerts"], dict(counts), flush=True)
    return result


def _run(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    lock_hash = sha256(lock_bytes).hexdigest()
    clock = json.loads(args.clock_results.read_bytes())
    receipt_bytes = args.waveform_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    table, people, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & reserved or set(people) != (fit | calibration)
            or (len(fit), len(calibration), len(reserved)) != (5, 5, 25)
            or clock["original_lock_sha256"] != lock_hash or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]):
        raise ValueError("Use the verified original development scope and scoring protocol")
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    pulse = stage_cores(grid, ((args.development_raw, args.development_features), (args.stride_raw, args.stride_features)))
    stats, pair_hash = native_template_stats(args.native_pairs, receipt, lock_hash, fit | calibration)
    prior = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
    return {**evaluate(grid, pulse, stats, lock, prior), "native_paired_bundle_sha256": pair_hash,
            "waveform_receipt_sha256": sha256(receipt_bytes).hexdigest(), "source_protocol_sha256": lock_hash,
            "development_participants": len(fit | calibration), "independent_predictions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "native-pairs", "waveform-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new aggregate output path")
    try:
        _write_json(args.output, _run(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Fold-local wrist template study failed: {error}\n")


if __name__ == "__main__":
    main()
