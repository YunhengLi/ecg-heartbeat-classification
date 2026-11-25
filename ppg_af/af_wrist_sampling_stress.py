"""Replay fixed 100 Hz wrist classifiers on synthetic 50 Hz PPG sampling."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_wrist_clock_study import load_grid
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import MONITOR_KIND, _shared_features, _subject_ids, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_quality_study import build_quality_matrices, quality_interaction_features, _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features


FAMILIES = ("baseline", "quality_interactions")
FIT_SEED = 2026
INTERVAL_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def _extract(cores, template):
    pad = 2 * PPG_RATE
    matrix = np.full((len(cores), len(ECG_FEATURE_NAMES)), np.nan)
    quality = np.full(len(cores), np.nan)
    for i, row in enumerate(cores):
        forecast = template.predict(np.pad(row, (pad, 0), mode="reflect"))[pad:]
        values = ecg_features(forecast, PPG_RATE)
        if np.isfinite(values).all() and values[INTERVAL_SUPPORT] > 0:
            matrix[i] = values
            quality[i] = ppg_features(row, PPG_RATE)[1]
    return {"baseline": matrix, "quality_interactions": quality_interaction_features(matrix, quality)}


def _scope_masks(grid):
    count = len(grid.subjects)
    evaluation, history, wrist = (np.zeros(count, dtype=bool) for _ in range(3))
    for timeline in grid.timelines:
        evaluation[timeline.rows] = True
        if timeline.kind == MONITOR_KIND:
            wrist[timeline.rows] = True
    for rows in grid.history_rows.values():
        history[rows] = True
    if np.any(evaluation & history):
        raise ValueError("Evaluation and history clocks must stay disjoint")
    return {"all_grid_rows": np.ones(count, dtype=bool), "all_evaluation": evaluation,
            "wrist_evaluation": wrist, "preceding_history": history,
            "original_stable_fitting": (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")}


def _fixed_scores(grid, original, transformed, fitted):
    route = fitted["routes"][ROUTE]
    before, cutoffs, _ = fold_arrays(fitted, route)
    shared = _shared_features(grid.table)
    after = np.full(len(grid.subjects), np.nan)
    labels = grid.table["stable_label"]
    eligible = shared & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    transformed_supported = np.isfinite(transformed).all(axis=1)
    assignment = fitted["fold_assignment"]
    for fold in range(fitted["n_splits"]):
        held = assignment == fold
        train = ~held & eligible
        native_inputs = held & shared
        if set(grid.subjects[train]) & set(grid.subjects[held]):
            raise ValueError("Keep the original held-out participants separate")
        if not np.array_equal(native_inputs, held & np.isfinite(before)):
            raise ValueError("Original model scoreability changed")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=DEFAULT_C, class_weight="balanced", max_iter=1000))
        model.fit(original[train], labels[train])
        if native_inputs.any() and not np.array_equal(before[native_inputs], model.predict_proba(original[native_inputs])[:, 1]):
            raise ValueError("Original outer model scores must reproduce exactly before stress")
        score = native_inputs & transformed_supported
        if score.any():
            after[score] = model.predict_proba(transformed[score])[:, 1]
    return before, after, cutoffs


def _shift(before, after, cutoffs, scope):
    native = scope & np.isfinite(before)
    stressed = scope & np.isfinite(after)
    if np.any(stressed & ~native):
        raise ValueError("Synthetic resampling cannot create originally unscoreable inputs")
    paired = native & stressed
    delta = after[paired] - before[paired]
    determined = paired & np.isfinite(cutoffs)
    old, new = before[determined] >= cutoffs[determined], after[determined] >= cutoffs[determined]
    return {"grid_rows": int(scope.sum()), "native_finite_scores": int(native.sum()), "stress_finite_scores": int(stressed.sum()),
            "lost_finite_scores": int((native & ~stressed).sum()), "paired_finite_scores": int(paired.sum()),
            "mean_signed_score_change": float(np.mean(delta)) if len(delta) else None,
            "median_absolute_score_change": float(np.median(np.abs(delta))) if len(delta) else None,
            "p90_absolute_score_change": float(np.quantile(np.abs(delta), 0.9)) if len(delta) else None,
            "maximum_absolute_score_change": float(np.max(np.abs(delta))) if len(delta) else None,
            "fixed_cutoff_positive_to_negative": int((old & ~new).sum()),
            "fixed_cutoff_negative_to_positive": int((~old & new).sum())}


def _evaluate(grid, template, cores, existing, lock, receipt):
    shared = _shared_features(grid.table)
    native = _extract(cores[shared], template)
    if any(not np.array_equal(native[k], existing[k][shared]) for k in FAMILIES):
        raise ValueError("Native 100 Hz feature extraction must reproduce every originally scoreable row")
    # Resample each trailing input separately. Linear boundary padding uses
    # only that already observed window's first and last values.
    reduced = resample_poly(cores[shared], 1, 2, axis=1, padtype="line")
    restored = resample_poly(reduced, 2, 1, axis=1, padtype="line")
    if reduced.shape != (int(shared.sum()), WINDOW_SECONDS * PPG_RATE // 2) or restored.shape != cores[shared].shape:
        raise ValueError("Decimation/interpolation must preserve the scheduled fixed-length input period")
    transformed = _extract(restored, template)
    scopes = _scope_masks(grid)
    reports = {}
    fitted = {name: fit_template(grid, existing[name], lock, DEFAULT_C, seed=FIT_SEED) for name in FAMILIES}
    verify_fit_support(*(fitted[name] for name in FAMILIES))
    for name in FAMILIES:
        baseline = _replay(grid, fitted[name])
        if baseline[0] != receipt["fixed_regularization"][name]["summary"]:
            raise ValueError("The published native-quality comparison must reproduce exactly")
        changed = np.full(existing[name].shape, np.nan)
        changed[shared] = transformed[name]
        before, after, cutoffs = _fixed_scores(grid, existing[name], changed, fitted[name])
        stressed = _held_out_replay(grid, after, cutoffs, _IMMEDIATE)
        reports[name] = {"native_100Hz": baseline[0], "synthetic_50Hz_roundtrip": _compare_with_pre_onset(baseline, stressed, lock),
                         "scoring_support_and_shift": {k: _shift(before, after, cutoffs, v) for k, v in scopes.items()}}
        print(f"{name}: {baseline[0]['episodes_detected']} to {stressed[0]['episodes_detected']} detections, "
              f"{baseline[0]['wrist_false_alerts']} to {stressed[0]['wrist_false_alerts']} false starts", flush=True)
    return {"raw_shared_cores_transformed": int(shared.sum()), "native_samples_per_window": cores.shape[1],
            "downsampled_samples_per_window": reduced.shape[1], "restored_samples_per_window": restored.shape[1],
            "first_sample_grid_origin_preserved": True, "filter_padding": "line, computed independently per input",
            "original_fit_input_rate_hz": PPG_RATE, "synthetic_intermediate_rate_hz": PPG_RATE // 2,
            "decision_stride_seconds": STRIDE_SECONDS, "feature_window_seconds": WINDOW_SECONDS, "reports": reports}


def _run(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    receipt_bytes = args.quality_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    table, people, timelines = load_features(args.development_features)
    fit, calibration, original_test = (_subject_ids(lock[k]) for k in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & original_test or set(people) != fit | calibration
            or (len(fit), len(calibration), len(original_test)) != (5, 5, 25)
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]):
        raise ValueError("Use the original locked development participants")
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    cores = stage_cores(grid, ((args.development_raw, args.development_features), (args.stride_raw, args.stride_features)))
    existing = build_quality_matrices(grid, args.development_raw, args.development_features, args.stride_raw, args.stride_features)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    result = _evaluate(grid, template, cores, existing, lock, receipt)
    return {"scope": "Synthetic within-window 100→50→100 Hz sampling sensitivity of fixed wrist development models",
            "source_protocol_sha256": sha256(lock_bytes).hexdigest(), "quality_receipt_sha256": sha256(receipt_bytes).hexdigest(),
            "fit_seed": FIT_SEED, "original_participants": len(set(people)), "independent_predictions": 0,
            "trained_on_resampled_waveforms": False, "cutoffs_recalibrated_after_resampling": False, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock",
                 "quality-receipt", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Supply a new aggregate output")
    try:
        _write_json(args.output, _run(args))
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Synthetic wrist sampling test failed: {error}\n")


if __name__ == "__main__":
    main()
