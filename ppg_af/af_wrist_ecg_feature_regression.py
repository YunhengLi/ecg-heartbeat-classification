"""Learn measured ECG timing statistics from paired wrist PPG within each fit."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_wrist_cv import _subject_folds
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_native_beat_transfer import read_pairs
from .af_wrist_quality_study import _compare_with_pre_onset, quality_interaction_features
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, _replay, fit_template, verify_fit_support
from .features import ECG_FEATURE_NAMES, _interval_features, ppg_features


WIDTH = len(ECG_FEATURE_NAMES)
MEASURED_RATE = 500
ALPHA = 8.0
RMSSD = ECG_FEATURE_NAMES.index("successive_interval_rmssd_over_median")
PNN50 = ECG_FEATURE_NAMES.index("successive_interval_pnn50")


def measured_features(peaks):
    support = np.ones(WINDOW_SECONDS * MEASURED_RATE, dtype=bool)
    indices = np.rint(peaks * MEASURED_RATE / PPG_RATE).astype(int)
    usable = np.unique(indices[(indices >= 0) & (indices < len(support))])
    if len(usable) < 3:
        raise ValueError("Each paired fitting window needs valid measured ECG beat intervals")
    return _interval_features(usable, MEASURED_RATE, support)


def learn_statistics(source, targets, people, train):
    keep = np.isin(people, sorted(train))
    if np.sum(keep) < 2 or not np.isfinite(source[keep]).all() or not np.isfinite(targets[keep]).all():
        raise ValueError("Regress ECG statistics using only paired fitting participants with verified inputs")
    x, y = source[keep], targets[keep]
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(x, y)
    bounds = np.min(y, axis=0), np.max(y, axis=0)
    return model, bounds, int(keep.sum())


def estimate(model, bounds, features):
    return np.clip(model.predict(features), *bounds)


def native_diagnostics(grid, inputs, targets, people, rhythm):
    original = inputs[:, :WIDTH]
    predictions = np.full_like(original, np.nan)
    shared = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    labels = np.where(shared, grid.table["stable_label"], -1)
    folds = _subject_folds(labels, grid.subjects, 5, SEEDS[0])
    for fold in range(5):
        hold = np.isin(people, grid.subjects[folds == fold])
        training = set(grid.subjects[folds != fold])
        model, bounds, _ = learn_statistics(inputs, targets, people, training)
        predictions[hold] = estimate(model, bounds, inputs[hold])
    if not np.isfinite(predictions).all():
        raise ValueError("Every verified wrist PPG and ECG pair needs one held-out estimate")

    def group(mask):
        scores = {}
        for name, values in (("pulse_template", original), ("regressed_ecg_statistics", predictions)):
            delta = values[mask] - targets[mask]
            scores[name] = {"median_absolute_error_by_feature": dict(zip(ECG_FEATURE_NAMES, np.median(np.abs(delta), axis=0).tolist(), strict=True)),
                            "median_absolute_combined_irregularity_error": float(np.median(np.abs(delta[:, RMSSD] + delta[:, PNN50])))}
        return {"windows": int(mask.sum()), "methods": scores}

    return {"all": group(np.ones(len(people), dtype=bool)), "stable_AF": group(rhythm == 1), "stable_non_AF": group(rhythm == 0)}


def evaluate(args):
    grid, quality, _, lock, digest, original_result = load_motion_development(args)
    report = json.loads(args.waveform_results.read_bytes())
    people, pulse, _, rhythm, references = read_pairs(args.waveform_pairs, report, digest, set(grid.subjects))
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    template_pairs = _extract(pulse, template)["baseline"]
    optical, q = map(np.asarray, zip(*(ppg_features(window, PPG_RATE) for window in pulse), strict=True))
    inputs = np.column_stack((template_pairs, optical, q))
    target = np.stack([measured_features(peaks) for peaks in references])
    native = native_diagnostics(grid, inputs, target, people, rhythm)
    diagnostics = {}
    shared = _shared_features(grid.table)
    quality_score = quality["quality"][:, -1]
    matrix = quality["quality_interactions"]
    original_inputs = np.column_stack((quality["baseline"], grid.table["direct_ppg"], quality_score))
    augmented_matrix = np.column_stack((matrix, quality["baseline"]))

    for seed in SEEDS:
        baseline = fit_template(grid, matrix, lock, DEFAULT_C, seed=seed)
        reference = _replay(grid, baseline)
        if seed == SEEDS[0] and reference[0] != original_result:
            raise ValueError("The original quality template must reproduce before learning ECG statistics")
        outcomes = {"original": reference[0]}
        for name, initial in (("replace_template_intervals", matrix), ("append_regressed_intervals", augmented_matrix)):
            fits = []

            def transform(current, fitting, scoring):
                trained, held = set(grid.subjects[fitting]), set(grid.subjects[scoring])
                if trained & held:
                    raise ValueError("Paired ECG statistics must fit on participants outside every scoring group")
                regressor, bounds, count = learn_statistics(inputs, target, people, trained)
                fits.append(count)
                rows = np.r_[fitting, scoring]
                rows = rows[shared[rows]]
                reconstructed = estimate(regressor, bounds, original_inputs[rows])
                replacement = current.copy()
                if name == "replace_template_intervals":
                    replacement[rows] = quality_interaction_features(reconstructed, quality_score[rows])
                else:
                    replacement[rows, -WIDTH:] = reconstructed
                return replacement

            evaluated = fit_template(grid, initial, lock, DEFAULT_C, seed=seed, fold_feature_transform=transform)
            verify_fit_support(baseline, evaluated)
            replay = _replay(grid, evaluated)
            outcomes[name] = {**_compare_with_pre_onset(reference, replay, lock), "paired_regression_fits": len(fits),
                              "min_paired_fit_windows": min(fits), "max_paired_fit_windows": max(fits)}
            print(f"{seed} {name}: {replay[0]['episodes_detected']}/{replay[0]['complete_episodes']}, "
                  f"{replay[0]['wrist_false_alerts']} false starts", flush=True)
        diagnostics[str(seed)] = outcomes
    return {"scope": "Fold-local statistical ECG targets from wrist PPG", "paired_training_windows": len(people),
            "paired_participants": len(np.unique(people)), "ridge_alpha": ALPHA,
            "held_out_native_interval_errors": native, "af_comparisons": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development-raw", "development-features", "stride-raw", "stride-features", "clock-results", "quality-results",
                "lock", "waveform-pairs", "waveform-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
