"""Lock AF classifiers and alert thresholds before scoring new participants."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .af_events import evaluate_af
from .af_wrist_features import FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION, ROUTES, WINDOW_SECONDS
from .af_wrist_operating import compare_operating_points, score_decisions, score_operating_point, select_operating_point
from .af_wrist_train import (
    _compare_window_points, _score_window_comparison, _stable_operating_point, _timeline_runs, load_features,
)


LOCK_VERSION = 1
TARGET_SENSITIVITY = 0.8
SENSITIVITY_TOLERANCE = 0.1
MONITOR_KIND = "monitor_wear"


def _subject_ids(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("Provide a nonempty list of unique participant IDs")
    return set(values)


def _shared_features(table):
    available = np.logical_and.reduce([np.isfinite(table[name]).all(axis=1) for name in ROUTES])
    return available & table["ppg_available"].astype(bool)


def _fit_model(features, labels, *, regularization=1.0):
    classifier = make_pipeline(StandardScaler(),
                               LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000))
    classifier.fit(features, labels)
    scaler, regression = [step for _, step in classifier.steps]
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": regression.coef_[0].tolist(), "intercept": float(regression.intercept_[0])}


def _predict(features, available, model):
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    intercept = float(model["intercept"])
    if (mean.shape != (features.shape[1],) or scale.shape != mean.shape or coef.shape != mean.shape
            or not np.isfinite(np.r_[mean, scale, coef, intercept]).all() or np.any(scale <= 0)):
        raise ValueError("Frozen model parameters must match the feature width and be finite")
    scores = np.full(len(features), np.nan)
    scores[available] = expit((features[available] - mean) / scale @ coef + intercept)
    return scores


def _episode_support(timelines, subjects, table, available):
    # A positive decision at every shared score endpoint exposes the reachability
    # ceiling before applying a threshold or the new-alert streak policy.
    decisions = np.where(available, 1, -1).astype(np.int8)
    _, episodes = _timeline_runs(timelines, subjects, decisions, table, scored=False, monitor_kind=MONITOR_KIND)
    metrics = evaluate_af(episodes) if episodes else {}
    complete = metrics.get("complete_reference_episodes", 0)
    reachable = metrics.get("complete_episodes_with_positive_decision", 0)
    return {
        "complete_reference_episodes": complete,
        "complete_episodes_with_ecg_valid_shared_score_endpoint": reachable,
        "max_sensitivity_from_endpoint_support": reachable / complete if complete else None,
    }


def freeze(table, subjects, timelines, *, calibration_subjects, test_roster):
    """Fit and calibrate on disjoint sets of development participants."""
    development = set(subjects)
    calibration = _subject_ids(calibration_subjects)
    test = _subject_ids(test_roster)
    if not calibration < development or development & test:
        raise ValueError("Calibration must be a proper development subset, and test IDs must be disjoint")
    fit = development - calibration
    labels = table["stable_label"]
    available = _shared_features(table)
    fit_rows = np.isin(subjects, list(fit)) & available & (labels >= 0) & (table["run_kind"] != "enriched_episode")
    if not np.isin((0, 1), labels[fit_rows]).all():
        raise ValueError("Fit participants must provide usable windows from both rhythms")
    calibration_rows = np.isin(subjects, list(calibration))
    routes = {}
    for name in ROUTES:
        model = _fit_model(table[name][fit_rows], labels[fit_rows])
        scores = _predict(table[name], calibration_rows & available, model)
        monitor, episodes = _timeline_runs(timelines, calibration, scores, table, scored=True, monitor_kind=MONITOR_KIND)
        routes[name] = {
            "model": model,
            "episode_calibration": select_operating_point(
                monitor, episode_runs=episodes, target_sensitivity=TARGET_SENSITIVITY,
                max_thresholds=max(3, int(calibration_rows.sum()) + 1),
            ),
            "window_calibration": _stable_operating_point(
                labels[calibration_rows], scores[calibration_rows], TARGET_SENSITIVITY,
            ),
        }
    return {
        "lock_version": LOCK_VERSION,
        FEATURE_VERSION_KEY: table.get(FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION),
        "monitor_kind": MONITOR_KIND,
        "window_seconds": WINDOW_SECONDS,
        "target_sensitivity": TARGET_SENSITIVITY,
        "sensitivity_tolerance": SENSITIVITY_TOLERANCE,
        "fit_subjects": sorted(fit),
        "calibration_subjects": sorted(calibration),
        "test_roster": sorted(test),
        "fit_af_windows": int(np.sum(fit_rows & (labels == 1))),
        "fit_non_af_windows": int(np.sum(fit_rows & (labels == 0))),
        "calibration_episode_support": _episode_support(timelines, calibration, table, available),
        "routes": routes,
    }


def evaluate(lock, table, subjects, timelines):
    """Score only frozen-roster participants and retain abstentions in coverage."""
    if (lock["lock_version"] != LOCK_VERSION or lock["window_seconds"] != WINDOW_SECONDS
            or lock["monitor_kind"] != MONITOR_KIND or set(lock["routes"]) != set(ROUTES)
            or lock["target_sensitivity"] != TARGET_SENSITIVITY or lock["sensitivity_tolerance"] != SENSITIVITY_TOLERANCE):
        raise ValueError("Unrecognized frozen AF protocol")
    feature_version = table.get(FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION)
    if lock.get(FEATURE_VERSION_KEY, LEGACY_FEATURE_VERSION) != feature_version:
        raise ValueError("Test RDDM feature version must match the frozen models")
    fit = _subject_ids(lock["fit_subjects"])
    calibration = _subject_ids(lock["calibration_subjects"])
    roster = _subject_ids(lock["test_roster"])
    found = set(subjects)
    if fit & calibration or (fit | calibration) & roster or not found <= roster:
        raise ValueError("Test archives must contain only participants from the disjoint frozen test roster")

    available = _shared_features(table)
    test_metrics, window_metrics, calibration_summary = {}, {}, {}
    for name in ROUTES:
        route = lock["routes"][name]
        scores = _predict(table[name], available, route["model"])
        option = route["episode_calibration"]
        threshold = option["threshold"]
        if threshold is None:
            decisions = np.full(len(scores), -1, dtype=np.int8)
            monitor, episodes = _timeline_runs(timelines, found, decisions, table, scored=False, monitor_kind=MONITOR_KIND)
            test_metrics[name] = score_decisions(monitor, episode_runs=episodes)
        else:
            monitor, episodes = _timeline_runs(timelines, found, scores, table, scored=True, monitor_kind=MONITOR_KIND)
            test_metrics[name] = score_operating_point(monitor, threshold, episode_runs=episodes)
        window_comparison = {"scores": scores, "folds": [
            {"fold": 0, "operating_point": {"window_calibration": route["window_calibration"]}},
        ]}
        window_metrics[name], _ = _score_window_comparison(
            window_comparison, np.zeros(len(scores), dtype=int), table["stable_label"],
        )
        calibration_summary[name] = {"episode": option, "window": route["window_calibration"]}

    def compare(names):
        return compare_operating_points(
            {name: test_metrics[name] for name in names},
            target_sensitivity=TARGET_SENSITIVITY, sensitivity_tolerance=SENSITIVITY_TOLERANCE,
            validation_target_met={name: calibration_summary[name]["episode"]["status"] == "target_met" for name in names},
        )

    monitor = table["run_kind"] == MONITOR_KIND
    return {
        "fit_participants": len(fit),
        FEATURE_VERSION_KEY: feature_version,
        "calibration_participants": len(calibration),
        "planned_test_participants": len(roster),
        "test_participants": len(found),
        "planned_test_roster_complete": found == roster,
        "test_windows": int(len(subjects)),
        "monitoring_reference_seconds": int(np.sum(monitor & table["reference_valid"])) * WINDOW_SECONDS,
        "shared_monitoring_feature_windows": int(np.sum(monitor & available)),
        "calibration_episode_support": lock.get("calibration_episode_support"),
        "test_episode_support": _episode_support(timelines, found, table, available),
        "calibration": calibration_summary,
        "test_alert_metrics": test_metrics,
        "test_stable_window_metrics": window_metrics,
        "stable_window_scope": "all staged stable test windows",
        "stable_window_operating_point": _compare_window_points(
            window_metrics, target=TARGET_SENSITIVITY, tolerance=SENSITIVITY_TOLERANCE,
        ),
        "operating_point": compare(ROUTES),
        "pairwise_operating_points": {f"{left}_vs_{right}": compare((left, right)) for left, right in combinations(ROUTES, 2)},
    }


def _write_json(path, data):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="Fit and calibrate only on development participants")
    freeze_parser.add_argument("--development-features", type=Path, required=True)
    freeze_parser.add_argument("--calibration-subject", action="append", required=True)
    freeze_parser.add_argument("--test-roster", type=Path, required=True, help="JSON list of planned new participant IDs")
    freeze_parser.add_argument("--output", type=Path, required=True, help="New private model lock JSON")
    evaluate_parser = commands.add_parser("evaluate", help="Apply a frozen lock to new participants")
    evaluate_parser.add_argument("--lock", type=Path, required=True)
    evaluate_parser.add_argument("--test-features", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True, help="New aggregate report JSON")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            roster = json.loads(args.test_roster.read_text(encoding="utf-8"))
            lock = freeze(*load_features(args.development_features), calibration_subjects=args.calibration_subject, test_roster=roster)
            _write_json(args.output, lock)
            print(f"Frozen lock SHA-256 {sha256(args.output.read_bytes()).hexdigest()}")
            for name, route in lock["routes"].items():
                print(f"{name} calibration {route['episode_calibration']['status']}")
        else:
            contents = args.lock.read_bytes()
            result = evaluate(json.loads(contents), *load_features(args.test_features))
            result["lock_sha256"] = sha256(contents).hexdigest()
            _write_json(args.output, result)
            print(f"Evaluated {result['test_participants']} of {result['planned_test_participants']} planned participants")
            print(f"Operating point {result['operating_point']['status']}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF holdout failed: {error}\n")


if __name__ == "__main__":
    main()
