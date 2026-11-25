"""Compare sinus and ectopy source training on the locked wrist AF development folds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from .af_vitaldb_preflight import (
    AF_OR_FLUTTER, ANNOTATIONS, ECTOPY, PILOT_GROUPS, RATE, _waveform_support,
    decision_windows, load_native_case, plan_cases, read_annotations,
)
from .af_wrist_clock_study import load_grid
from .af_wrist_cv import _subject_folds, _within_rhythm_participant_weights
from .af_wrist_features import FEATURE_VERSION_KEY, PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _subject_ids, _write_json
from .af_wrist_mimic_transfer import _fit_source, evaluate
from .af_wrist_template_independent import _archives
from .af_wrist_train import load_features
from .af_wrist_vote_study import STRIDE_SECONDS
from .features import PPG_FEATURE_NAMES, ppg_features


SOURCE_GROUP_SIZES = (6, 12, 12)
SOURCE_FOLDS = 5
SEED = 2026
MIN_CLOCK_AGREEMENT = 0.95
MAX_WINDOWS_PER_RHYTHM = 32
INTERVAL_SUPPORT = PPG_FEATURE_NAMES.index("valid_interval_fraction")
SINUS = "N"
SOURCE_RHYTHMS = {AF_OR_FLUTTER, SINUS, *ECTOPY}


def _source_rows(directory, native_cache, plan, *, extra_features=None):
    features, labels, patients, endings = [], [], [], []
    clocks = Counter()
    for patient, item in enumerate(plan["pilot"]):
        case = item["case_id"]
        times, _, clean, spans = read_annotations(directory, case)
        waveforms, _ = load_native_case(native_cache / f"{case}.vital")
        report = _waveform_support(waveforms, times, clean, spans)
        clocks["checked"] += 1
        if report["matched_peak_fraction"] < MIN_CLOCK_AGREEMENT:
            clocks["below_minimum_r_peak_agreement"] += 1
            continue

        added = 0
        for end, label in decision_windows(spans):
            if end % WINDOW_SECONDS != 0 or label not in SOURCE_RHYTHMS:
                continue
            start = (end - WINDOW_SECONDS) * RATE
            stop = end * RATE
            ecg, pulse = (signal[start:stop] for signal in waveforms)
            if not np.isfinite(ecg).all() or not np.isfinite(pulse).all() or np.ptp(pulse) <= 0:
                continue
            core = resample_poly(pulse.astype(float), PPG_RATE, RATE, padtype="line")
            values, _ = ppg_features(core, PPG_RATE)
            if not np.isfinite(values).all() or values[INTERVAL_SUPPORT] <= 0:
                continue
            if extra_features is not None:
                values = np.r_[values, extra_features(core)]
            features.append(values)
            labels.append(label)
            patients.append(patient)
            endings.append(end)
            added += 1
        if added == 0:
            clocks["no_eligible_rhythm_features"] += 1
        if (patient + 1) % 5 == 0:
            print(f"Processed {patient + 1}/{len(plan['pilot'])} native cases", flush=True)
    return np.stack(features), np.array(labels), np.array(patients), np.array(endings), dict(clocks)


def _evenly_spaced(rows, count):
    if len(rows) < count or count < 0:
        raise ValueError("Choose no more windows than are available")
    return rows[np.linspace(0, len(rows) - 1, count, dtype=int)]


def training_rows(rhythms, patients, endings):
    """Swap negative windows within each person and preserve both fit denominators."""
    sinus_rows, ectopy_rows = [], []
    for patient in np.unique(patients):
        within = np.flatnonzero(patients == patient)
        within = within[np.argsort(endings[within], kind="stable")]
        sinus = within[rhythms[within] == SINUS]
        af = within[rhythms[within] == AF_OR_FLUTTER]
        ectopy = within[np.isin(rhythms[within], sorted(ECTOPY))]
        positive = _evenly_spaced(af, min(len(af), MAX_WINDOWS_PER_RHYTHM))
        negative = _evenly_spaced(sinus, min(len(sinus), MAX_WINDOWS_PER_RHYTHM))
        replace = min(len(ectopy), len(negative) // 2)
        mixed = np.r_[_evenly_spaced(negative, len(negative) - replace), _evenly_spaced(ectopy, replace)]
        sinus_rows.extend(np.r_[positive, negative])
        ectopy_rows.extend(np.r_[positive, mixed])
    return {"sinus": np.asarray(sinus_rows, dtype=int), "ectopy": np.asarray(ectopy_rows, dtype=int)}


def _source_auc(probabilities, rhythms, patients, negative_rhythms):
    rows = np.isin(rhythms, [AF_OR_FLUTTER, *negative_rhythms])
    labels = (rhythms[rows] == AF_OR_FLUTTER).astype(int)
    groups = patients[rows]
    weights = _within_rhythm_participant_weights(labels, groups)
    return {"participant_weighted_auc": float(roc_auc_score(labels, probabilities[rows], sample_weight=weights)),
            "positive_windows": int(labels.sum()), "negative_windows": int((1 - labels).sum()),
            "positive_participants": len(np.unique(groups[labels == 1])),
            "negative_participants": len(np.unique(groups[labels == 0]))}


def source_models(features, rhythms, patients, endings):
    training = training_rows(rhythms, patients, endings)
    if np.array_equal(np.sort(training["sinus"]), np.sort(training["ectopy"])):
        raise ValueError("The ectopy comparison requires valid within-person replacements")
    labels = (rhythms == AF_OR_FLUTTER).astype(int)
    groups = np.char.zfill(patients.astype(str), 3)
    assignment_labels = np.where(rhythms == AF_OR_FLUTTER, 1, np.where(rhythms == SINUS, 0, -1))
    folds = _subject_folds(assignment_labels, groups, SOURCE_FOLDS, SEED)
    models, reports = {}, {}
    for name, rows in training.items():
        predictions = np.full(len(labels), np.nan)
        for fold in range(SOURCE_FOLDS):
            train = rows[folds[rows] != fold]
            test = folds == fold
            if len(np.unique(labels[train])) != 2 or set(groups[train]) & set(groups[test]):
                raise ValueError("Each source fit needs both classes and disjoint people")
            model = _fit_source(features[train], labels[train], groups[train], drop_constant_features=True)
            predictions[test] = model.predict_proba(features[test])[:, 1]
        if not np.isfinite(predictions).all():
            raise ValueError("Source folds must score every common comparison row")
        reports[name] = {"fit_windows_by_reference_rhythm": dict(Counter(rhythms[rows])),
                         "positive_fit_participants": len(np.unique(groups[rows[labels[rows] == 1]])),
                         "negative_fit_participants": len(np.unique(groups[rows[labels[rows] == 0]])),
                         "grouped_af_or_flutter_vs_sinus": _source_auc(predictions, rhythms, groups, [SINUS]),
                         "grouped_af_or_flutter_vs_ectopy": _source_auc(predictions, rhythms, groups, sorted(ECTOPY))}
        models[name] = _fit_source(features[rows], labels[rows], groups[rows], drop_constant_features=True)
    return models, reports


def _development_grid(args):
    lock_bytes = args.lock.read_bytes()
    lock, clock = json.loads(lock_bytes), json.loads(args.clock_results.read_bytes())
    table, members, timelines = load_features(args.development_features)
    fit, calibration, reserved = (_subject_ids(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    people = fit | calibration
    digest = sha256(lock_bytes).hexdigest()
    if (fit & calibration or people & reserved or set(members) != people
            or (len(fit), len(calibration), len(reserved)) != (5, 5, 25)
            or any(set(_archives(path)) != people for path in (args.stride_raw, args.stride_features))
            or table[FEATURE_VERSION_KEY] != lock[FEATURE_VERSION_KEY] or clock["original_lock_sha256"] != digest):
        raise ValueError("Use the original locked wrist development cohort and clock study")
    grid = load_grid(table, members, timelines, args.stride_raw, args.stride_features, seconds=STRIDE_SECONDS)
    return grid, lock, clock["warm_10s"]["nested_development_cv"]["1"], digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations-dir", "track-list", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--private-plan", action="store_true")
    for key in ("native-cache", "development-features", "stride-raw", "stride-features", "lock", "clock-results"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new output path.")
    if not args.private_plan and not all((args.native_cache, args.development_features, args.stride_raw,
                                         args.stride_features, args.lock, args.clock_results)):
        parser.error("Supply the native source cache and all locked wrist development inputs")
    try:
        plan = plan_cases(args.annotations_dir, args.track_list, cases_per_group=SOURCE_GROUP_SIZES,
                          nonoverlapping_only=True)
        if args.private_plan:
            _write_json(args.output, plan)
            return
        grid, lock, prior, lock_digest = _development_grid(args)
        features, rhythms, patients, endings, clocks = _source_rows(args.annotations_dir, args.native_cache, plan)
        models, source = source_models(features, rhythms, patients, endings)
        wrist = {name: evaluate(grid, lock, prior, model, source_name=f"vitaldb_{name}") for name, model in models.items()}
        _write_json(args.output, {"scope": "Native surgical PPG source training with grouped wrist development transfer",
                                  "source": ANNOTATIONS, "source_annotation_label": AF_OR_FLUTTER,
                                  "native_waveform_rate_hz": RATE, "feature_sample_rate_hz": PPG_RATE,
                                  "source_window_seconds": WINDOW_SECONDS, "source_grouped_folds": SOURCE_FOLDS,
                                  "planned_cases_by_stratum": dict(zip(PILOT_GROUPS, SOURCE_GROUP_SIZES, strict=True)),
                                  "source_participants_with_features": len(np.unique(patients)),
                                  "eligible_source_windows_by_reference_rhythm": dict(Counter(rhythms)),
                                  "minimum_native_r_peak_agreement": MIN_CLOCK_AGREEMENT, "source_clock_checks": clocks,
                                  "maximum_fit_windows_per_participant_rhythm": MAX_WINDOWS_PER_RHYTHM,
                                  "source_group_seed": SEED, "source_track_list_sha256": plan["track_listing_sha256"],
                                  "wrist_original_lock_sha256": lock_digest, "wrist_development_participants": 10,
                                  "wrist_independent_inference": False, "source_grouped_classification": source,
                                  "wrist_development_transfer": wrist})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"VitalDB source transfer failed: {error}\n")


if __name__ == "__main__":
    main()
