"""Compare one- and two-window AF confirmation on development data only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_features import FEATURE_VERSION_KEY, ROUTES
from .af_wrist_holdout import MONITOR_KIND, _predict, _shared_features
from .af_wrist_operating import score_operating_point, select_operating_point
from .af_wrist_train import _timeline_runs, evaluate, load_features


RULES = (1, 2)


def _summary(metrics):
    episodes = metrics["episode_metrics"]
    monitor = metrics["burden_metrics"]
    return {
        "complete_episodes": episodes["complete_reference_episodes"],
        "episodes_detected": episodes["detected_complete_episodes"],
        "episode_sensitivity": episodes["complete_episode_recall"],
        "median_delay_seconds_in_detected_episodes": episodes["median_complete_episode_alert_delay_seconds"],
        "wrist_ecg_observed_seconds": monitor["ecg_reference_seconds"],
        "wrist_non_af_seconds": monitor["ecg_non_af_seconds"],
        "wrist_false_alerts": monitor["false_alert_starts"],
        "false_alerts_per_observed_wrist_hour": monitor["false_alerts_per_observed_hour"],
    }


def compare_development(table, subjects, timelines, lock):
    fit = set(lock["fit_subjects"])
    calibration = set(lock["calibration_subjects"])
    reserved = set(lock["test_roster"])
    development = set(subjects)
    if not fit or not calibration or fit & calibration or development != (fit | calibration) or reserved & development:
        raise ValueError("Load exactly the frozen development participants for this study")
    if table[FEATURE_VERSION_KEY] != lock.get(FEATURE_VERSION_KEY):
        raise ValueError("The development features must match the frozen feature version")

    calibration_rows = np.isin(subjects, sorted(calibration))
    available = _shared_features(table) & calibration_rows
    target = lock["target_sensitivity"]
    max_thresholds = max(3, int(calibration_rows.sum()) + 1)
    fixed, recalibrated = {}, {}
    for name in ROUTES:
        route = lock["routes"][name]
        cutoff = route["episode_calibration"]["threshold"]
        if cutoff is None:
            raise ValueError(f"The frozen development model has no cutoff for {name}")
        scores = _predict(table[name], available, route["model"])
        monitor, episodes = _timeline_runs(timelines, calibration, scores, table, scored=True, monitor_kind=MONITOR_KIND)
        fixed[name], recalibrated[name] = {}, {}
        for windows in RULES:
            metrics = score_operating_point(monitor, cutoff, episode_runs=episodes, confirmation_windows=windows)
            fixed[name][str(windows)] = {"threshold": cutoff, **_summary(metrics)}
            selected = select_operating_point(monitor, episode_runs=episodes, target_sensitivity=target,
                                              max_thresholds=max_thresholds, confirmation_windows=windows)
            recalibrated[name][str(windows)] = {
                "threshold": selected["threshold"], "status": selected["status"],
                "threshold_candidates": selected["threshold_candidates"], **_summary(selected["validation"]),
            }
        original = _summary(route["episode_calibration"]["validation"])
        if any(fixed[name]["1"][key] != value for key, value in original.items()):
            raise ValueError(f"The original development result did not replay for {name}")

    grouped = {}
    for windows in RULES:
        nested = evaluate(table, subjects, timelines, folds=5, inner_folds=3, seed=2026, target=target,
                          tolerance=lock["sensitivity_tolerance"], monitor_kind=MONITOR_KIND,
                          confirmation_windows=windows, exhaustive_thresholds=True)
        grouped[str(windows)] = {
            "operating_point_status": nested["operating_point"]["status"],
            "routes": {
                name: {
                    **_summary(metrics),
                    "outer_folds_with_inner_threshold_at_target": sum(
                        fold["operating_point"]["status"] == "target_met" for fold in nested["fold_calibration"][name]
                    ),
                }
                for name, metrics in nested["test_alert_metrics"].items()
            },
        }
    return {
        "scope": "Development-only comparison with unchanged ECG event boundaries",
        FEATURE_VERSION_KEY: lock[FEATURE_VERSION_KEY],
        "development_participants": len(development),
        "fixed_split_fit_participants": len(fit),
        "fixed_split_calibration_participants": len(calibration),
        "target_episode_sensitivity": target,
        "confirmation_windows": list(RULES),
        "fixed_thresholds": fixed,
        "recalibrated_thresholds": recalibrated,
        "nested_development_cv": {"outer_folds": 5, "max_inner_folds": 3, "seed": 2026, "rules": grouped},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        original = args.lock.read_bytes()
        report = compare_development(*load_features(args.development_features), json.loads(original))
        report["original_lock_sha256"] = sha256(original).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF alert study failed: {error}\n")
    print(f"Compared confirmation rules on {report['development_participants']} development participants")


if __name__ == "__main__":
    main()
