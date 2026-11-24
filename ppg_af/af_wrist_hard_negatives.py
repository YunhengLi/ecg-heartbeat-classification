"""Compare matched ordinary and irregular ECG-confirmed non-AF training inputs."""

import argparse
from pathlib import Path

import numpy as np

from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_monitor_extension import _archive_features, read_plan
from .af_wrist_monitor_plan import per_person_seed
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _replay, fit_template
from .af_wrist_training_expansion import _fit_expanded
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import ECG_FEATURE_NAMES


RMSSD = ECG_FEATURE_NAMES.index("successive_interval_rmssd_over_median")
PNN50 = ECG_FEATURE_NAMES.index("successive_interval_pnn50")
ARM_NAMES = ("matched_random_non_af", "high_irregularity_non_af")
DEVELOPMENT_MONITOR_PLAN_SHA = "4eaff4417dbb3e1ec0a0d5d80fd0ce844229877b111776efbbb7351cfe828a4d"


def pick_pairs(start, valid, episodes, features, rng):
    """Select one irregular and one random nonoverlapping negative per half block."""
    count = len(valid)
    width = WINDOW_SECONDS // STRIDE_SECONDS
    ends = start + STRIDE_SECONDS * np.arange(1, count + 1)
    starts = ends - WINDOW_SECONDS
    allowed = np.isfinite(features).all(axis=1)
    allowed[:width - 1] = False
    allowed[width - 1:] &= np.lib.stride_tricks.sliding_window_view(valid, width).all(axis=1)
    overlaps = (episodes[:, 0] < ends[:, None]) & (episodes[:, 1] > starts[:, None])
    allowed &= ~overlaps.any(axis=1)
    rank = features[:, RMSSD] + features[:, PNN50]
    selected = []
    for first in (0, count // 2):
        candidates = np.flatnonzero(allowed[first:first + count // 2]) + first
        candidates = candidates[candidates >= first + width - 1]
        if len(candidates) < 2:
            continue
        ranked = candidates[np.argsort(-rank[candidates], kind="stable")]
        for hard in ranked:
            alternate = candidates[np.abs(candidates - hard) >= width]
            if len(alternate):
                selected.append((int(rng.choice(alternate)), int(hard)))
                break
    return selected


def addition_matrices(schedule, digest, cache, template):
    people = []
    picked = {name: [] for name in ARM_NAMES}
    for person, blocks in schedule.items():
        if not blocks:
            continue
        rng = np.random.default_rng(per_person_seed(f"negative-pairs:{digest}", person))
        starts, valid, episodes, matrix, _ = _archive_features(cache / f"{person}.npz", person, blocks, digest, template)
        for start, observed, feature in zip(starts, valid, matrix["quality_interactions"], strict=True):
            feature = feature[-len(observed):]
            for control, hard in pick_pairs(start, observed, episodes, feature, rng):
                picked[ARM_NAMES[0]].append(feature[control])
                picked[ARM_NAMES[1]].append(feature[hard])
                people.append(person)
    if not people:
        raise ValueError("The locked monitoring intervals have no matched non-AF training pairs")
    return np.asarray(people), {name: np.stack(rows) for name, rows in picked.items()}


def evaluate(args):
    schedule, receipt, digest = read_plan(args.plan, args.lock)
    if digest != DEVELOPMENT_MONITOR_PLAN_SHA:
        raise ValueError("Use the previously verified, disjoint 150-minute development plan")
    grid, quality, _, lock, _, original = load_motion_development(args)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    people, new = addition_matrices(schedule, digest, args.monitoring_cache, template)
    if set(people) - set(grid.subjects):
        raise ValueError("Only original development participants can supply training negatives")
    original_features = quality["quality_interactions"]
    labels = np.zeros(len(people), dtype=int)
    results = {}
    for seed in SEEDS:
        baseline = fit_template(grid, original_features, lock, DEFAULT_C, seed=seed)
        replay = _replay(grid, baseline)
        if seed == SEEDS[0] and replay[0] != original:
            raise ValueError("Reproduce the original wrist AF benchmark before comparing new training negatives")
        compared = {"original": replay[0]}
        for name, extra in new.items():
            augmented = _fit_expanded(grid, original_features, extra, people, labels, lock, seed)
            route = augmented["routes"][ROUTE]
            before = baseline["routes"][ROUTE]
            if (not np.array_equal(baseline["fold_assignment"], augmented["fold_assignment"])
                    or not np.array_equal(np.isfinite(route["scores"]), np.isfinite(before["scores"]))
                    or not np.array_equal(route["decisions"] >= 0, before["decisions"] >= 0)):
                raise ValueError("Training negatives must preserve the original grouped evaluation support")
            for base_fold, augmented_fold in zip(before["folds"], route["folds"], strict=True):
                held = grid.subjects[baseline["fold_assignment"] == base_fold["fold"]]
                expected = int(np.count_nonzero(~np.isin(people, held)))
                if (base_fold["fit_af_windows"] != augmented_fold["fit_af_windows"]
                        or augmented_fold["fit_non_af_windows"] - base_fold["fit_non_af_windows"] != expected):
                    raise ValueError("Only new negative rows from training participants may enter each fit")
            prediction = _replay(grid, augmented)
            compared[name] = _compare_with_pre_onset(replay, prediction, lock)
            print(f"{seed} {name}: {prediction[0]['episodes_detected']}/{prediction[0]['complete_episodes']} detections, "
                  f"{prediction[0]['wrist_false_alerts']} false starts", flush=True)
        results[str(seed)] = compared
    return {"scope": "Matched additional non-AF wrist training under the original nested participant folds",
            "monitor_plan_sha256": digest, "original_planned_monitor_blocks": receipt["scheduled_blocks"],
            "training_windows_per_arm": len(people), "training_participants": len(np.unique(people)),
            "median_selection_score": {name: float(np.median(values[:, RMSSD] + values[:, PNN50])) for name, values in new.items()},
            "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "lock", "monitoring-cache", "development-raw", "development-features", "stride-raw",
                 "stride-features", "clock-results", "quality-results", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
