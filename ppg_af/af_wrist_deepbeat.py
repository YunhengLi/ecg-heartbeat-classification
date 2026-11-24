"""Transfer the published DeepBeat wrist classifier to the locked AF clocks."""

import argparse
from collections import Counter
from hashlib import file_digest
from itertools import islice
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from .af_wrist_clock_study import scoring_policy
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_local_template_study import stage_cores
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_context import _pulse_features
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template, verify_fit_support
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE
from .features import _bandpass
from .ptt_data import iter_ptt_records


REVISION = "f54004a80eaca99460d1e171c8ec203e182dacb9"
MODEL_SHA256 = "6df126f88855388f8706ea6e158b80902752a019b1f3c26d3ceab22991e4eee3"
INPUT_SECONDS = 25
MODEL_RATE = 32
FILTER_HZ = (0.4, 8.0)
PREPROCESSING = ("normalize_after_filter", "normalize_before_filter")


def load_external(path):
    import keras

    if keras.__version__ != "3.12.0" or keras.backend.backend() != "torch":
        raise ValueError("Use Keras 3.12.0 with the torch backend for exact weight inference")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("Use the pinned DeepBeat weight file from the authors' repository")
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    if model.input_shape != (None, INPUT_SECONDS * MODEL_RATE, 1) or model.output_names != ["qa_output", "rhythm_output"]:
        raise ValueError("The external model must have its original rhythm and quality outputs")
    return model


def normalize(rows):
    ranges = np.ptp(rows, axis=1)
    if np.any(ranges <= 0) or not np.isfinite(rows).all():
        raise ValueError("Every matched optical input needs a finite, nonflat 25-second core")
    return (rows - rows.min(axis=1)[:, None]) / ranges[:, None]


def predict_external(model, pulse, order):
    inputs = normalize(pulse) if order == "normalize_before_filter" else pulse
    rows = np.stack([_bandpass(window, PPG_RATE, *FILTER_HZ) for window in inputs])
    samples = resample_poly(rows, MODEL_RATE, PPG_RATE, axis=1, padtype="line")
    if order == "normalize_after_filter":
        samples = normalize(samples)
    quality, rhythm = model.predict(samples.astype(np.float32)[:, :, None], batch_size=32, verbose=0)
    if (quality.shape != (len(pulse), 3) or rhythm.shape != (len(pulse), 2)
            or not np.isfinite(quality).all() or not np.isfinite(rhythm).all()):
        raise ValueError("DeepBeat must emit both scores for every shared optical input")
    margin = (1.0 + rhythm[:, 1].astype(float) - rhythm[:, 0].astype(float)) / 2
    return margin, np.argmax(quality, axis=1)


def quality_counts(classes):
    return dict(zip(("noise", "acceptable", "excellent"), np.bincount(classes, minlength=3).tolist(), strict=True))


def resting_controls(ptt_dir):
    records = tuple(islice(iter_ptt_records(ptt_dir, ("sit",)), PTT_SUBJECTS))
    if len(records) != PTT_SUBJECTS:
        raise ValueError("Use the same public sitting PTT roster for preprocessing controls")
    return np.stack([resample_poly(record.ppg[int(5 * record.fs):int(30 * record.fs)], PPG_RATE, int(record.fs),
                                   padtype="line") for record in records])


def nested_cutoffs(grid, scores, folds, lock):
    cutoff = np.full(len(folds), np.nan)
    status = Counter()
    for fold in np.unique(folds):
        training = folds != fold
        people = set(grid.subjects[training])
        evaluate, pool = scoring_policy(grid, np.where(training, scores, np.nan), people, _IMMEDIATE)
        option = select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                        max_thresholds=max(3, len(pool) + 1))
        status[option["status"]] += 1
        if option["threshold"] is not None:
            cutoff[~training] = option["threshold"]
    return cutoff, dict(sorted(status.items()))


def compare(grid, baseline, scores, cutoffs, lock):
    replay = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    return {**_compare_with_pre_onset(baseline, replay, lock),
            "scoreable_evaluation_rows": sum(int(np.sum(run.decisions >= 0)) for run in replay[1])}


def evaluate(args):
    grid, quality, paths, lock, _, _ = load_motion_development(args)
    shared = _shared_features(grid.table)
    cores = stage_cores(grid, paths)[shared, -INPUT_SECONDS * PPG_RATE:]
    model = load_external(args.weights)
    controls = resting_controls(args.ptt_dir)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    recent = np.full_like(quality["baseline"], np.nan)
    recent[shared] = np.stack([_pulse_features(template, row) for row in cores])
    prior = json.loads(args.confirmation_results.read_bytes())["comparisons"]
    stable = shared & (grid.table["stable_label"] >= 0) & (grid.table["run_kind"] != "enriched_episode")
    predictions = {}
    diagnostics = {}
    for name in PREPROCESSING:
        score, quality_class = predict_external(model, cores, name)
        _, control_quality = predict_external(model, controls, name)
        external = np.full(len(shared), np.nan)
        external[shared] = score
        excellent = np.full(len(shared), False)
        excellent[shared] = quality_class == 2
        predictions[name] = external, np.where(excellent, external, np.nan)
        diagnostics[name] = {"wrist_quality_classes": quality_counts(quality_class), "external_sitting_quality_classes": quality_counts(control_quality),
                             "wrist_stable_auc": float(roc_auc_score(grid.table["stable_label"][stable], external[stable]))}
    results = {}
    for seed in SEEDS:
        fit = fit_template(grid, quality["baseline"], lock, DEFAULT_C, seed=seed)
        baseline = _replay(grid, fit)
        if baseline[0] != prior[str(seed)]["plain_template"]["original"]:
            raise ValueError("The original 30-second plain template must reproduce")
        shorter = fit_template(grid, recent, lock, DEFAULT_C, seed=seed)
        verify_fit_support(fit, shorter)
        shorter_replay = _replay(grid, shorter)
        original_cutoff = np.full(len(shared), np.nextafter(0.5, np.inf))
        routes = {
            "original_30_second_template": baseline[0],
            "same_25_second_template": _compare_with_pre_onset(baseline, shorter_replay, lock),
            "original_template_stable_auc": float(roc_auc_score(grid.table["stable_label"][stable],
                                                               fit["routes"][ROUTE]["scores"][stable])),
        }
        for name, (external, gated) in predictions.items():
            cutoffs, status = nested_cutoffs(grid, external, fit["fold_assignment"], lock)
            routes[name] = {"published_class_rule": compare(grid, baseline, external, original_cutoff, lock),
                            "excellent_quality_and_published_rule": compare(grid, baseline, gated, original_cutoff, lock),
                            "training_group_cutoff": {**compare(grid, baseline, external, cutoffs, lock), "status": status}}
        results[str(seed)] = routes
    return {"scope": "Frozen published DeepBeat cross-device AF alert transfer on the same wrist folds",
            "external_repository_revision": REVISION, "external_model_sha256": MODEL_SHA256,
            "parameters": int(model.count_params()), "input_seconds": INPUT_SECONDS, "sample_rate_hz": MODEL_RATE,
            "local_bandpass_hz": list(FILTER_HZ), "all_scoreable_rows": int(shared.sum()),
            "preprocessing_diagnostics": diagnostics,
            "comparisons": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "quality-results", "confirmation-results", "ptt-dir", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
