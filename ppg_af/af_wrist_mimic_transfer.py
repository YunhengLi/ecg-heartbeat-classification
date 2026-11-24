"""Test frozen ICU PPG transfer on the original wrist AF development folds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_wrist_clock_study import fit_nested_grid, load_grid, scoring_policy
from .af_wrist_cv import _subject_folds, _within_rhythm_participant_weights, train_oof
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_template_independent import _archives
from .af_wrist_template_study import DEFAULT_C, ROUTE
from .af_wrist_train import load_features
from .af_wrist_transition_metrics import compare_pre_onset_alert_runs
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE, _compare, _held_out_replay, _select_matched
from .features import PPG_FEATURE_NAMES, ppg_features
from .mimic_perform_waveform import RATE as SOURCE_RATE, SOURCE_DOI, SOURCES, _read_records


SOURCE_C = 1.0
SOURCE_FOLDS = 5
SEED = 2026
DIRECT = "direct_ppg"
INTERVAL_SUPPORT_COLUMN = PPG_FEATURE_NAMES.index("valid_interval_fraction")


def _load_source(directory, *, with_waveforms=False):
    """Apply the shared MIMIC eligibility rule and optionally retain its inputs."""
    features, labels, subjects = [], [], []
    waveforms = [] if with_waveforms else None
    inventory = {str(label): Counter() for label in (0, 1)}
    samples = WINDOW_SECONDS * SOURCE_RATE
    records = _read_records(directory)
    for subject, (pulse, _, label) in enumerate(records):
        counts = inventory[str(label)]
        counts["participants"] += 1
        before = len(features)
        for start in range(0, len(pulse) - samples + 1, samples):
            counts["all_windows"] += 1
            window = pulse[start:start + samples]
            if not np.isfinite(window).all() or np.ptp(window) <= 0:
                continue
            # Resample inside the decision window, so filter input stops at its endpoint.
            converted = resample_poly(window, PPG_RATE, SOURCE_RATE)
            values, _ = ppg_features(converted, PPG_RATE)
            if np.isfinite(values).all() and values[INTERVAL_SUPPORT_COLUMN] > 0:
                features.append(values)
                labels.append(label)
                subjects.append(subject)
                if waveforms is not None:
                    waveforms.append(converted)
        if len(features) == before:
            raise ValueError("Every source participant needs at least one eligible PPG feature window")
        counts["participants_with_features"] += 1
        counts["feature_windows"] += len(features) - before
    return (np.asarray(features), np.asarray(labels), np.asarray(subjects), inventory,
            np.stack(waveforms) if waveforms is not None else None)


def _source_features(directory):
    features, labels, subjects, inventory, _ = _load_source(directory)
    return features, labels, subjects, inventory


def _fit_source(matrix, labels, subjects, *, drop_constant_features=False):
    n_subjects = len(np.unique(subjects))
    weights = _within_rhythm_participant_weights(labels, subjects) * (n_subjects / len(labels))
    steps = [StandardScaler(), LogisticRegression(C=SOURCE_C, max_iter=1000)]
    if drop_constant_features:
        steps.insert(0, VarianceThreshold())
    model = make_pipeline(*steps)
    model.fit(matrix, labels, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    return model


def _source_auc(matrix, labels, subjects):
    # Preserve the source loader's original participant ID sort order.
    group_names = np.char.zfill(subjects.astype(str), 3)
    folds = _subject_folds(labels, group_names, SOURCE_FOLDS, SEED)
    values = np.full(len(subjects), np.nan)
    for fold in range(SOURCE_FOLDS):
        train = folds != fold
        predictor = _fit_source(matrix[train], labels[train], subjects[train])
        values[~train] = predictor.predict_proba(matrix[~train])[:, 1]
    by_record = [(labels[keep][0], np.median(values[keep])) for subject in np.unique(subjects)
                 for keep in (subjects == subject,)]
    outcome, score = zip(*by_record, strict=True)
    return float(roc_auc_score(outcome, score))


def _calibrate(grid, probabilities, people, lock, template_reference):
    evaluate, pool = scoring_policy(grid, probabilities, people, _IMMEDIATE)
    evaluated = {}

    def record(cutoff):
        metrics = evaluate(cutoff)
        evaluated[cutoff] = metrics
        return metrics

    primary = select_scored_threshold(pool, record, target_sensitivity=lock["target_sensitivity"],
                                      max_thresholds=max(3, len(pool) + 1))
    matched = (_select_matched(evaluated, template_reference, lock["sensitivity_tolerance"])
               if primary["threshold"] is not None else primary)
    return {**primary, "matched_template": matched}


def _fit_fusion(grid, frozen_logit, lock, assignments, template_folds):
    shared = _shared_features(grid.table)
    matrix = np.column_stack((grid.table[ROUTE], frozen_logit))
    if not np.isfinite(matrix[shared]).all():
        raise ValueError("Source fusion must retain every shared wrist feature row")
    features = np.where(shared[:, None], matrix, np.nan)
    fit_rows = shared & (grid.table["run_kind"] != "enriched_episode")
    training_rows = {tuple(np.flatnonzero(assignments != fold["fold"])): fold["operating_point"]
                     for fold in template_folds}

    def calibrate(scores, rows):
        reference = training_rows.get(tuple(rows))
        if reference is None:
            raise ValueError("The candidate calibration participants must match a template training fold")
        probabilities = np.full(len(grid.subjects), np.nan)
        probabilities[rows] = scores
        return _calibrate(grid, probabilities, set(grid.subjects[rows]), lock, reference)

    fitted = train_oof({ROUTE: features}, grid.table["stable_label"], grid.subjects, n_splits=len(template_folds),
                       inner_splits=3, seed=SEED, train_eligible=fit_rows, calibration_selector=calibrate,
                       regularization=DEFAULT_C)
    if not np.array_equal(fitted["fold_assignment"], assignments):
        raise ValueError("The candidate changed the original outer participant folds")
    fields = ("fold", "train_subjects", "test_subjects", "inner_splits", "fit_af_windows", "fit_non_af_windows",
              "held_out_finite_scores")
    route = fitted["routes"][ROUTE]
    for before, after in zip(template_folds, route["folds"], strict=True):
        if any(before[key] != after[key] for key in fields):
            raise ValueError("The candidate changed the original participant-fold support")
    return route


def _comparison(baseline, candidate, lock):
    return {**_compare(baseline, candidate, lock),
            "pre_onset": compare_pre_onset_alert_runs(baseline[1], candidate[1])}


def _replay_candidate(grid, scores, assignments, points, baseline_cutoffs, *, matched):
    cutoffs = np.full(len(assignments), np.nan)
    statuses = Counter()
    for fold, options in enumerate(points):
        selected = options["matched_template"] if matched else options
        statuses[selected["status"]] += 1
        if selected["threshold"] is not None:
            cutoffs[assignments == fold] = selected["threshold"]
    if not np.array_equal(np.isfinite(cutoffs), np.isfinite(baseline_cutoffs)):
        raise ValueError("Candidate cutoff support differs from the original wrist comparison")
    return _held_out_replay(grid, scores, cutoffs, _IMMEDIATE), dict(statuses)


def evaluate(grid, lock, prior, source_model, *, source_name="mimic"):
    baseline_fit = fit_nested_grid(grid, lock, 1)
    assignments = baseline_fit["fold_assignment"]
    baselines, baseline_cutoffs = {}, {}
    for name in (DIRECT, ROUTE):
        route = baseline_fit["routes"][name]
        scores, cutoffs, _ = fold_arrays(baseline_fit, route)
        replay = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
        if any(prior[name][key] != value for key, value in replay[0].items()):
            raise ValueError(f"The historical {name} baseline did not reproduce")
        baselines[name], baseline_cutoffs[name] = replay, cutoffs
    template_folds = baseline_fit["routes"][ROUTE]["folds"]

    shared = _shared_features(grid.table)
    frozen_logits = np.full(len(grid.subjects), np.nan)
    frozen_probabilities = np.full(len(grid.subjects), np.nan)
    frozen_logits[shared] = source_model.decision_function(grid.table[DIRECT][shared])
    frozen_probabilities[shared] = source_model.predict_proba(grid.table[DIRECT][shared])[:, 1]
    if not np.array_equal(np.isfinite(frozen_probabilities),
                          np.isfinite(baseline_fit["routes"][DIRECT]["scores"])):
        raise ValueError("The frozen source score changed wrist feature eligibility")
    source_points = [_calibrate(grid, frozen_probabilities, set(grid.subjects[assignments != fold["fold"]]),
                                lock, fold["operating_point"]) for fold in template_folds]
    print("Calibrated the frozen source scores without outer test subjects", flush=True)
    fusion = _fit_fusion(grid, frozen_logits, lock, assignments, template_folds)
    if not np.array_equal(np.isfinite(fusion["scores"]), np.isfinite(baseline_fit["routes"][ROUTE]["scores"])):
        raise ValueError("The fusion model changed wrist feature eligibility")

    fusion_points = [fold["operating_point"] for fold in fusion["folds"]]
    report = {"wrist_direct_baseline": baselines[DIRECT][0], "wrist_template_baseline": baselines[ROUTE][0]}
    linked = (
        (f"frozen_{source_name}_direct", frozen_probabilities, source_points, DIRECT),
        (f"wrist_template_plus_frozen_{source_name}", fusion["scores"], fusion_points, ROUTE),
    )
    for name, scores, points, primary_baseline in linked:
        for matched in (False, True):
            baseline = ROUTE if matched else primary_baseline
            label = "matched_template_sensitivity" if matched else "80_percent_target"
            replay, statuses = _replay_candidate(grid, scores, assignments, points, baseline_cutoffs[baseline],
                                                 matched=matched)
            report[f"{name}_{label}"] = {**_comparison(baselines[baseline], replay, lock),
                                         "reference_route": baseline, "training_fold_calibration_status_counts": statuses}
            summary = replay[0]
            print(f"{name} ({label}): {summary['episodes_detected']}/{summary['complete_episodes']} detections, "
                  f"{summary['wrist_false_alerts']} false starts", flush=True)
    return {"outer_folds": len(template_folds), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "development-features", "stride-raw", "stride-features", "lock", "clock-results", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new report path.")
    try:
        lock_bytes = args.lock.read_bytes()
        lock, clock = json.loads(lock_bytes), json.loads(args.clock_results.read_bytes())
        base, members, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
        development = fit | calibration
        lock_digest = sha256(lock_bytes).hexdigest()
        if (fit & calibration or development & reserved or set(members) != development
                or (len(fit), len(calibration), len(reserved)) != (5, 5, 25)
                or any(set(_archives(path)) != development for path in (args.stride_raw, args.stride_features))
                or base[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
                or clock["original_lock_sha256"] != lock_digest):
            raise ValueError("Use the original locked development cohort and clock study")
        grid = load_grid(base, members, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        source_x, source_y, source_members, source_counts = _source_features(args.source)
        source_model = _fit_source(source_x, source_y, source_members)
        source_auc = _source_auc(source_x, source_y, source_members)
        print(f"Verified and fit {len(np.unique(source_members))} MIMIC source participants", flush=True)
        results = evaluate(grid, lock, clock["warm_10s"]["nested_development_cv"]["1"], source_model)
        _write_json(args.output, {
            "scope": "MIMIC PERform AF source training and nested wrist development evaluation",
            "source_doi": SOURCE_DOI, "source_md5": {name: digest for name, digest, _, _ in SOURCES},
            "source_label_scope": "Record AF status inherited by nonoverlapping 30-second PPG windows",
            "source_sampling_hz": SOURCE_RATE, "feature_sampling_hz": PPG_RATE, "source_logistic_C": SOURCE_C,
            "source_auc_folds": SOURCE_FOLDS, "grouped_source_record_auroc": source_auc,
            "source_class_inventory": source_counts, "wrist_lock_sha256": lock_digest,
            "wrist_development_participants": len(development), "wrist_independent_inference": False,
            "feature_window_seconds": WINDOW_SECONDS, "decision_stride_seconds": STRIDE_SECONDS,
            "max_inner_folds_for_wrist_model_fits": 3,
            "target_episode_sensitivity": lock["target_sensitivity"],
            "sensitivity_tolerance": lock["sensitivity_tolerance"], **results,
        })
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"MIMIC transfer study failed: {error}\n")


if __name__ == "__main__":
    main()
