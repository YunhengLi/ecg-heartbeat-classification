"""Test additional verified stable training windows on frozen wrist AF folds."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_cv import train_oof
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_operating import select_scored_threshold
from .af_wrist_quality_study import build_quality_matrices, quality_interaction_features
from .af_wrist_sequence_study import _comparison
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _IMMEDIATE
from .features import ECG_FEATURE_NAMES, ecg_features, ppg_features


MAX_WINDOWS_PER_RHYTHM_AND_PARTICIPANT = 8
_INTERVAL_SUPPORT = ECG_FEATURE_NAMES.index("valid_interval_fraction")


def _disjoint_intervals(start, end, original_raw, stride_raw):
    with (np.load(original_raw, allow_pickle=False) as reference, np.load(stride_raw, allow_pickle=False) as finer):
        baseline_start, fine_start = reference["start_s"], finer["input_start_s"]
        occupied_start = np.r_[baseline_start, fine_start] - 1
        occupied_end = np.r_[baseline_start + WINDOW_SECONDS, finer["decision_end_s"]] + 1
    if np.any((start[:, None] - 1 < occupied_end) & (occupied_start < end[:, None] + 1)):
        raise ValueError("New training time overlaps a staged development input or predecessor")
    order = np.argsort(start)
    if np.any(start[order[1:]] - 1 < end[order[:-1]] + 1):
        raise ValueError("Additional training windows must not overlap")


def _check_alignment(people, labels, start, end, development_raw, stride_raw):
    if not np.array_equal(end, start + WINDOW_SECONDS) or not np.isfinite(start).all():
        raise ValueError("Additional training rows must occupy exactly 30 seconds on the source clock")
    for person in np.unique(people):
        rows = people == person
        path = f"{person}.npz"
        _disjoint_intervals(start[rows], end[rows], development_raw / path, stride_raw / path)
        for rhythm in (0, 1):
            group = start[rows & (labels == rhythm)]
            bins = np.unique(np.floor(group / 300))
            if len(group) > MAX_WINDOWS_PER_RHYTHM_AND_PARTICIPANT or len(bins) != len(group):
                raise ValueError("Respect the per-rhythm participant cap and five-minute sampling spacing")


def load_supplement(path, grid, lock_sha256, development_raw, stride_raw, ptt):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        if (archive["source_protocol_sha256"].item() != lock_sha256
                or archive["ppg_fs"].item() != PPG_RATE
                or archive["ecg_reference_fs"].item() != 500 or archive["ppg_flank_seconds"].item() != 1
                or archive["feature_window_seconds"].item() != WINDOW_SECONDS):
            raise ValueError("Additional training windows must be pinned to the same development protocol")
        people, labels, starts, ends, pulses = (archive[name] for name in
                                               ("subjects", "stable_label", "start_s", "decision_end_s", "ppg_with_context"))
    if (people.ndim != 1 or not len(people) or any(x.shape != people.shape for x in (labels, starts, ends))
            or not np.isin(labels, (0, 1)).all() or not set(people) <= set(grid.subjects)
            or pulses.shape != (len(people), (WINDOW_SECONDS + 2) * PPG_RATE)):
        raise ValueError("Use only verified labels and paired windows from the locked development participants")
    _check_alignment(people, labels, starts, ends, development_raw, stride_raw)

    _, template = fit_ptt_calibration(ptt, PPG_RATE, subject_count=PTT_SUBJECTS)
    pad = 2 * PPG_RATE
    accepted, template_rows, quality = [], [], []
    for index, padded in enumerate(pulses):
        pulse = padded[PPG_RATE:-PPG_RATE]
        if not np.isfinite(pulse).all() or np.ptp(pulse) == 0:
            continue
        forecast = template.predict(np.pad(pulse, (pad, 0), mode="reflect"))[pad:]
        values = ecg_features(forecast, PPG_RATE)
        if not np.isfinite(values).all() or values[_INTERVAL_SUPPORT] <= 0:
            continue
        accepted.append(index)
        template_rows.append(values)
        quality.append(ppg_features(pulse, PPG_RATE)[1])

    if set(labels[accepted]) != {0, 1}:
        raise ValueError("Additional fitting data need usable pulse-template features for both rhythms")
    matrix = np.asarray(template_rows)
    quality_matrix = quality_interaction_features(matrix, np.asarray(quality))
    inventory = {"sha256": digest, "selected_windows": len(labels), "feature_supported_windows": len(accepted),
                 "by_rhythm": {name: _support(people[accepted], labels[accepted], value)
                               for name, value in (("non_AF", 0), ("AF", 1))}}
    return people[accepted], labels[accepted], {"baseline": matrix, "quality_interactions": quality_matrix}, inventory


def _support(people, labels, value):
    counts = sorted(Counter(people[labels == value]).values(), reverse=True)
    return {"windows": sum(counts), "participants": len(counts), "top_two_participants_windows": sum(counts[:2])}


def _fit_expanded(grid, original_matrix, extra_matrix, extra_people, extra_labels, lock, seed):
    shared = _shared_features(grid.table)
    labels = grid.table["stable_label"]
    n = len(labels)
    if not np.isfinite(original_matrix[shared]).all() or not np.isfinite(extra_matrix).all():
        raise ValueError("Additional training must preserve all original score support")
    features = np.vstack((np.where(shared[:, None], original_matrix, np.nan), extra_matrix))
    extra = len(extra_labels)
    original_fit = shared & (grid.table["run_kind"] != "enriched_episode")

    def calibrate(scores, rows):
        original_rows = rows < n
        full = np.full(n, np.nan)
        full[rows[original_rows]] = scores[original_rows]
        evaluate, pool = scoring_policy(grid, full, set(grid.subjects[rows[original_rows]]), _IMMEDIATE)
        return select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                       max_thresholds=max(3, len(pool) + 1))

    fitted = train_oof(
        {ROUTE: features}, np.r_[labels, np.full(extra, -1)], np.r_[grid.subjects, extra_people],
        n_splits=5, inner_splits=3, seed=seed, train_eligible=np.r_[original_fit, np.zeros(extra, dtype=bool)],
        calibration_selector=calibrate, regularization=DEFAULT_C,
        supplemental_fit_labels=np.r_[np.full(n, -1), extra_labels],
    )
    assigned = fitted["fold_assignment"]
    route = fitted["routes"][ROUTE]
    for key in ("scores", "decisions"):
        route[key] = route[key][:n]
    for fold in route["folds"]:
        rows = assigned[:n] == fold["fold"]
        fold["held_out_finite_scores"] = int(np.count_nonzero(np.isfinite(route["scores"][rows])))
        if set(extra_people[assigned[n:] != fold["fold"]]) & set(grid.subjects[rows]):
            raise ValueError("A held-out participant's additional windows entered a training fold")
    fitted["fold_assignment"] = assigned[:n]
    return fitted


def evaluate_expansion(grid, existing, extra, lock, original_summary, seed):
    people, labels, matrices, receipt = extra
    report = {}
    for name, additions in matrices.items():
        original = fit_template(grid, existing[name], lock, DEFAULT_C, seed=seed)
        augmented = _fit_expanded(grid, existing[name], additions, people, labels, lock, seed)
        if not np.array_equal(original["fold_assignment"], augmented["fold_assignment"]):
            raise ValueError("Additional training changed the original grouped folds")
        before, after = (fitted["routes"][ROUTE] for fitted in (original, augmented))
        if (not np.array_equal(np.isfinite(before["scores"]), np.isfinite(after["scores"]))
                or not np.array_equal(before["decisions"] >= 0, after["decisions"] >= 0)):
            raise ValueError("Additional training changed evaluation support")
        added_by_fold = []
        for a, b in zip(before["folds"], after["folds"], strict=True):
            extra_train = ~np.isin(people, grid.subjects[original["fold_assignment"] == a["fold"]])
            differences = {rhythm: b[f"fit_{rhythm}_windows"] - a[f"fit_{rhythm}_windows"]
                           for rhythm in ("af", "non_af")}
            if (a["inner_splits"] != b["inner_splits"]
                    or differences != {"af": int(np.count_nonzero(extra_train & (labels == 1))),
                                       "non_af": int(np.count_nonzero(extra_train & (labels == 0)))}):
                raise ValueError("Additional counts must agree with participant-fold membership")
            added_by_fold.append(differences)
        baseline, candidate = _replay(grid, original), _replay(grid, augmented)
        if name == "baseline" and seed == 2026 and any(original_summary[key] != value for key, value in baseline[0].items()):
            raise ValueError("Original template ECG metrics did not reproduce")
        report[name] = {"original": baseline[0], "augmented": _comparison(baseline, candidate, lock),
                        "added_training_by_fold": added_by_fold,
                        "original_calibration": _calibration_status(original),
                        "augmented_calibration": _calibration_status(augmented)}
        print(f"{name}: {baseline[0]['episodes_detected']} to {candidate[0]['episodes_detected']} detections; "
              f"{baseline[0]['wrist_false_alerts']} to {candidate[0]['wrist_false_alerts']} false starts", flush=True)
    shared = _shared_features(grid.table) & (grid.table["run_kind"] != "enriched_episode")
    combined_people, combined_labels = np.r_[grid.subjects[shared], people], np.r_[grid.table["stable_label"][shared], labels]
    return {"scope": "Additional stable development training windows on unchanged held-out wrist data",
            "fold_seed": seed, "outer_folds": 5, "additional_training": receipt,
            "stable_training_support": {name: {"original": _support(grid.subjects[shared], grid.table["stable_label"][shared], value),
                                               "augmented": _support(combined_people, combined_labels, value)}
                                        for name, value in (("non_AF", 0), ("AF", 1))},
            "comparisons": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-raw", "development-features", "stride-raw", "stride-features", "lock", "clock-results",
                 "supplement", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[2026], help="One or more prespecified fold seeds")
    args = parser.parse_args()
    if len(set(args.seed)) != len(args.seed):
        parser.error("Each evaluation seed must be distinct")
    try:
        lock_raw = args.lock.read_bytes()
        lock = json.loads(lock_raw)
        lock_digest = sha256(lock_raw).hexdigest()
        clock = json.loads(args.clock_results.read_bytes())
        table, people, timelines = load_features(args.development_features)
        fit, calibration, reserved = (_subject_ids(lock[key]) for key in
                                      ("fit_subjects", "calibration_subjects", "test_roster"))
        if (fit & calibration or (fit | calibration) & reserved or set(people) != (fit | calibration)
                or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY] or clock["original_lock_sha256"] != lock_digest):
            raise ValueError("Use the original locked development cohort and clock study")
        grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
        existing = build_quality_matrices(grid, args.development_raw, args.development_features,
                                          args.stride_raw, args.stride_features)
        extra = load_supplement(args.supplement, grid, lock_digest, args.development_raw, args.stride_raw, args.ptt_dir)
        previous = clock["warm_10s"]["nested_development_cv"]["1"][ROUTE]
        reports = [evaluate_expansion(grid, existing, extra, lock, previous, seed) for seed in args.seed]
        result = reports[0] if len(reports) == 1 else {"scope": "Development split sensitivity of added training windows",
                                                       "trials": reports}
        _write_json(args.output, result)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist AF training expansion failed: {error}\n")


if __name__ == "__main__":
    main()
