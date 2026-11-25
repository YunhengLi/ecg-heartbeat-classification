"""Check observed pulse order on the previously verified VitalDB rhythm sample."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .af_vitaldb_preflight import plan_cases
from .af_vitaldb_transfer import SOURCE_GROUP_SIZES, _source_rows, source_models
from .af_wrist_features import PPG_RATE
from .af_wrist_holdout import _write_json
from .af_wrist_interval_order import interval_order_from_peaks
from .af_wrist_pulse_consensus import _find
from .features import PPG_FEATURE_NAMES, _bandpass, _plateaus, _prepare


def optical_order(signal):
    values, present = _prepare(signal, PPG_RATE, minimum_fs=4.0)
    detected = _find(_bandpass(values, PPG_RATE, 0.4, 8.0))
    usable = present & ~_plateaus(values, present, PPG_RATE)
    return np.asarray(interval_order_from_peaks(detected, usable)).ravel()


def evaluate(args):
    plan = plan_cases(args.annotations_dir, args.track_list, cases_per_group=SOURCE_GROUP_SIZES, nonoverlapping_only=True)
    old = json.loads(args.historical_transfer.read_bytes())
    if plan["track_listing_sha256"] != old["source_track_list_sha256"]:
        raise ValueError("Use the same frozen surgical waveform catalog as the original comparison")
    features, rhythm, people, endings, clocks = _source_rows(args.annotations_dir, args.native_cache, plan,
                                                             extra_features=optical_order)
    if dict(Counter(rhythm)) != old["eligible_source_windows_by_reference_rhythm"] or clocks != old["source_clock_checks"]:
        raise ValueError("Surgical pulse-order comparisons must retain the original clock and rhythm availability")
    base = features[:, :len(PPG_FEATURE_NAMES)]
    observed = features[:, len(PPG_FEATURE_NAMES):len(PPG_FEATURE_NAMES) + 2]
    null = features[:, -2:]
    matrices = {"original_features": base, "distribution_only": np.column_stack((base, null)),
                "observed_pulse_order": np.column_stack((base, observed))}
    results = {name: source_models(values, rhythm, people, endings)[1] for name, values in matrices.items()}
    if results["original_features"] != old["source_grouped_classification"]:
        raise ValueError("The original surgical grouped analysis must reproduce")
    return {"scope": "Compare order-dependent pulse features on fixed surgical AF-or-flutter and ectopy windows",
            "eligible_reference_windows": old["eligible_source_windows_by_reference_rhythm"],
            "participants_with_features": len(np.unique(people)), "comparisons": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations-dir", "track-list", "native-cache", "historical-transfer", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
