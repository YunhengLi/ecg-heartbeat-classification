"""Transfer a frozen AFDB beat-annotation classifier to measured and template ECG."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import wfdb

from .af_wrist_clock_study import load_grid, scoring_policy
from .af_wrist_cv import _within_rhythm_participant_weights
from .af_wrist_failure_study import fold_arrays
from .af_wrist_features import FEATURE_VERSION_KEY, WINDOW_SECONDS
from .af_wrist_holdout import _shared_features, _subject_ids, _write_json
from .af_wrist_measured_ecg import _DOI as WRIST_DOI, measured_features
from .af_wrist_operating import select_scored_threshold
from .af_wrist_sequence_study import _comparison
from .af_wrist_template_study import DEFAULT_C, ROUTE, _calibration_status, _replay, fit_template, verify_fit_support
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS, _held_out_replay, _IMMEDIATE
from .features import ECG_FEATURE_NAMES, _interval_features


SOURCE_RATE = 250
SOURCE_STRIDE_SECONDS = 60
SOURCE_MAX_BEAT_GAP_SECONDS = 3
ECG_GAP_BLOCK_SAMPLES = 2560
UNREADABLE_BLOCKS = {"04043": (39,), "08405": (1067,), "08434": (648, 857, 894)}
RHYTHMS = {"(N": 0, "(AFIB": 1, "(AFL": -1, "(J": -1}
SOURCE_DOI = "10.13026/C2MW2D"
_SOURCE_MANIFEST_SHA256 = "8f3ab7b38bf0591235680a855d6a7f00fe8fdaf2e457fa4c00edad2c735e0d9a"
_SOURCE_LICENSE = "ODC-By-1.0"


def _verify_sources(directory):
    manifest_bytes = (directory / "SHA256SUMS.txt").read_bytes()
    if sha256(manifest_bytes).hexdigest() != _SOURCE_MANIFEST_SHA256:
        raise ValueError("Use the checksum manifest from the pinned AFDB v1.0.0 release")
    manifest = dict(line.split(maxsplit=1)[::-1] for line in manifest_bytes.decode("ascii").splitlines())
    roster = (directory / "RECORDS").read_text("ascii").splitlines()
    if len(roster) != 25 or len(set(roster)) != len(roster) or any(len(record) != 5 or not record.isascii()
                                                                  or not record.isdigit() for record in roster):
        raise ValueError("Expected the original 25-record AFDB roster")
    names = ("RECORDS", "notes.txt", "ANNOTATORS", *(f"{person}.{suffix}" for person in roster
                                                   for suffix in ("hea", "atr", "qrs")))
    payload = len(manifest_bytes)
    for name in names:
        path = directory / name
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != manifest[name]:
                raise ValueError(f"AFDB release checksum mismatch for {name}")
        payload += path.stat().st_size
    license_html = (directory / "license.html").read_bytes()
    if b"Open Data Commons Attribution License" not in license_html:
        raise ValueError("Store the official AFDB license page as license.html")
    return roster, payload + len(license_html)


def source_data(directory):
    roster, payload = _verify_sources(directory)

    features, labels, people = [], [], []
    counters = Counter(excluded_unstable_or_other_rhythm=0, excluded_unreadable_block=0, excluded_missing_beat_support=0)
    width = WINDOW_SECONDS * SOURCE_RATE
    step = SOURCE_STRIDE_SECONDS * SOURCE_RATE
    present = np.ones(width, dtype=bool)
    for person in roster:
        record = str(directory / person)
        header, beats, rhythm = wfdb.rdheader(record), wfdb.rdann(record, "qrs"), wfdb.rdann(record, "atr")
        if header.fs != SOURCE_RATE or getattr(beats, "fs", SOURCE_RATE) != SOURCE_RATE:
            raise ValueError("AFDB annotations must use the documented 250 Hz clock")
        peaks = np.asarray(beats.sample, dtype=np.int64)
        if (len(peaks) < 3 or peaks[0] < 0 or np.any(np.diff(peaks) <= 0)
                or set(beats.symbol) != {"N"}):
            raise ValueError("Expected the unaudited beat-timestamp annotations without beat types")
        # The two headers without voltage have zero declared samples. Limit them
        # to the available QRS timeline rather than inventing a terminal clock.
        end = header.sig_len if header.sig_len else int(peaks[-1]) + 1
        rhythm_rows = [(int(sample), note.strip().rstrip("\x00"))
                       for sample, symbol, note in zip(rhythm.sample, rhythm.symbol, rhythm.aux_note, strict=True)
                       if symbol == "+"]
        if (not rhythm_rows or any(note not in RHYTHMS for _, note in rhythm_rows)
                or any(right[0] <= left[0] for left, right in zip(rhythm_rows, rhythm_rows[1:]))):
            raise ValueError("Unrecognized or unordered AFDB rhythm changes")
        boundaries = np.array([sample for sample, _ in rhythm_rows])
        codes = [RHYTHMS[note] for _, note in rhythm_rows]
        # The release notes list unreadable blocks without documenting whether
        # the count starts at zero. Exclude both possible positions.
        gaps = [((block - 1) * ECG_GAP_BLOCK_SAMPLES, (block + 1) * ECG_GAP_BLOCK_SAMPLES)
                for block in UNREADABLE_BLOCKS.get(person, ())]
        for start in range(step, end - width - SOURCE_RATE + 1, step):
            lo, hi = start - SOURCE_RATE, start + width + SOURCE_RATE
            counters["time_spaced_candidates"] += 1
            interval = np.searchsorted(boundaries, lo, side="right") - 1
            after = np.searchsorted(boundaries, hi, side="left") - 1
            if interval < 0 or interval != after or codes[interval] < 0:
                counters["excluded_unstable_or_other_rhythm"] += 1
                continue
            if any(lo < right and left < hi for left, right in gaps):
                counters["excluded_unreadable_block"] += 1
                continue
            first, last = np.searchsorted(peaks, [start, start + width])
            local = peaks[first:last] - start
            if len(local) < 3 or np.max(np.diff(np.r_[0, local, width])) > SOURCE_MAX_BEAT_GAP_SECONDS * SOURCE_RATE:
                counters["excluded_missing_beat_support"] += 1
                continue
            features.append(_interval_features(local, SOURCE_RATE, present))
            labels.append(codes[interval])
            people.append(person)

    matrix, labels, people = map(np.asarray, (features, labels, people))
    if not np.isfinite(matrix).all() or set(labels) != {0, 1}:
        raise ValueError("Source training requires finite timing features with both stable rhythms")
    summary = {"records_in_release": len(roster), "records_with_supported_source_windows": len(set(people)),
               "source_hz": SOURCE_RATE, "source_decision_spacing_seconds": SOURCE_STRIDE_SECONDS,
               "source_files_payload_bytes": payload,
               "duration_seconds": WINDOW_SECONDS, "max_beat_gap_seconds": SOURCE_MAX_BEAT_GAP_SECONDS,
               "stable_window_candidates": dict(sorted(counters.items())),
               "stable_N": {"windows": int(np.count_nonzero(labels == 0)), "records": len(set(people[labels == 0]))},
               "stable_AF": {"windows": int(np.count_nonzero(labels == 1)), "records": len(set(people[labels == 1]))},
               "source_feature_names": list(ECG_FEATURE_NAMES),
               "source_constant_features": [name for name, values in zip(ECG_FEATURE_NAMES, matrix.T, strict=True)
                                            if np.ptp(values) == 0]}
    return matrix, labels, people, summary


def _stable_auc(grid, scores):
    labels = grid.table["stable_label"]
    stable = np.isfinite(scores) & (labels >= 0) & (grid.table["run_kind"] != "enriched_episode")
    return float(roc_auc_score(labels[stable], scores[stable]))


def _transfer(grid, baseline_fit, matrix, model, lock):
    shared = _shared_features(grid.table)
    source_scores = np.full(len(shared), np.nan)
    source_scores[shared] = model.predict_proba(matrix[shared])[:, 1]
    reference = baseline_fit["routes"][ROUTE]
    ref_scores, ref_cutoffs, _ = fold_arrays(baseline_fit, reference)
    if not np.array_equal(np.isfinite(source_scores), np.isfinite(ref_scores)):
        raise ValueError("Source model and wrist comparator must share every available input")
    source_cutoffs = np.full(len(shared), np.nan)
    assignment = baseline_fit["fold_assignment"]
    statuses = Counter()
    for fold in range(baseline_fit["n_splits"]):
        training = assignment != fold
        people = set(grid.subjects[training])
        if people & set(grid.subjects[~training]):
            raise ValueError("Threshold calibration must exclude all outer test participants")
        evaluate, pool = scoring_policy(grid, np.where(training, source_scores, np.nan), people, _IMMEDIATE)
        choice = select_scored_threshold(pool, evaluate, target_sensitivity=lock["target_sensitivity"],
                                         max_thresholds=max(3, len(pool) + 1))
        statuses[choice["status"]] += 1
        if choice["threshold"] is not None:
            source_cutoffs[~training] = choice["threshold"]
    if not np.array_equal(np.isfinite(source_cutoffs), np.isfinite(ref_cutoffs)):
        raise ValueError("Source-only transfer changed calibrated availability")
    reference_replay = _replay(grid, baseline_fit)
    source_replay = _held_out_replay(grid, source_scores, source_cutoffs, _IMMEDIATE)
    evaluation = np.concatenate([block.rows for block in grid.timelines])
    return {"wrist_fit": {"summary": reference_replay[0], "stable_window_AUROC": _stable_auc(grid, ref_scores),
                          "calibration_status_counts": _calibration_status(baseline_fit),
                          "positive_evaluation_decisions": int(np.sum(ref_scores[evaluation] >= ref_cutoffs[evaluation]))},
            "source_fit_wrist_cutoffs": {**_comparison(reference_replay, source_replay, lock),
                                         "stable_window_AUROC": _stable_auc(grid, source_scores),
                                         "calibration_status_counts": dict(sorted(statuses.items())),
                                         "positive_evaluation_decisions": int(np.sum(source_scores[evaluation]
                                                                                     >= source_cutoffs[evaluation]))}}


def evaluate_transfer(args):
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    clock = json.loads(args.clock_results.read_bytes())
    inventory = json.loads(args.source_inventory.read_bytes())
    previous = json.loads(args.measured_control_results.read_bytes())
    table, people, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[name]) for name in ("fit_subjects", "calibration_subjects", "test_roster"))
    if (fit & calibration or (fit | calibration) & reserved or set(people) != fit | calibration
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY]
            or clock["original_lock_sha256"] != sha256(lock_bytes).hexdigest() or inventory["doi"] != WRIST_DOI):
        raise ValueError("Use the original locked v6 development cohort")
    source_matrix, labels, people_source, training = source_data(args.afdb_dir)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=DEFAULT_C, max_iter=1000))
    weights = _within_rhythm_participant_weights(labels, people_source)
    model.fit(source_matrix, labels, logisticregression__sample_weight=weights)
    print(f"Fitted AFDB-only C=1 model on {len(labels)} stable source windows", flush=True)
    grid = load_grid(table, people, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    source = {entry["key"]: entry for entry in inventory["files"]}
    raw_ecg, status = measured_features(grid, source, args.source_dir, args.cache_dir, args.development_raw, args.stride_raw)
    shared = _shared_features(grid.table) & np.isfinite(raw_ecg).all(axis=1)
    template = np.where(shared[:, None], grid.table[ROUTE], np.nan)
    comparison_grid = replace(grid, table={**grid.table, ROUTE: template})
    variants = {"measured_ECG": raw_ecg, "PPG_template_ECG": template}
    wrist_fits = {name: fit_template(comparison_grid, matrix, lock, DEFAULT_C) for name, matrix in variants.items()}
    verify_fit_support(*wrist_fits.values())
    outcomes = {name: _transfer(comparison_grid, wrist_fits[name], matrix, model, lock) for name, matrix in variants.items()}
    reference = {"measured_ECG": previous["measured_ecg_on_common_footprint"]["summary"],
                 "PPG_template_ECG": previous["template_on_common_footprint"]}
    for name, result in outcomes.items():
        baseline = result["wrist_fit"]["summary"]
        if any(reference[name][key] != value for key, value in baseline.items()):
            raise ValueError("The original measured and template wrist comparators must reproduce exactly")
    return {
        "scope": "Development-only transfer of AFDB beat timing to matched ECG and PPG-template inputs",
        "source_doi": SOURCE_DOI, "source_license": _SOURCE_LICENSE, "source_training": training,
        "logistic_C": DEFAULT_C, "target_wrist_participants": len(set(people)), "target_outer_folds": 5,
        "target_device_feature_footprint": {"raw_ECG_status": status, "common_rows": int(shared.sum())},
        "transfer_by_signal": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("afdb-dir", "source-inventory", "source-dir", "cache-dir", "development-raw", "development-features",
             "stride-raw", "stride-features", "lock", "clock-results", "measured-control-results", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    try:
        report = evaluate_transfer(args)
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AFDB transfer failed: {error}\n")
    for name, results in report["transfer_by_signal"].items():
        replay = results["source_fit_wrist_cutoffs"]["summary"]
        print(f"{name}: detected {replay['episodes_detected']}/{replay['complete_episodes']} episodes, "
              f"{replay['wrist_false_alerts']} wrist false starts", flush=True)


if __name__ == "__main__":
    main()
