"""Evaluate pulse interval reliability from paired red and infrared PPG."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_move import _nearest_fraction
from .af_wrist_pulse_consensus import AGREEMENT_MS, PNN50, RMSSD, VALID, _find, corroborated_peaks, pulse_consensus
from .features import _bandpass, _interval_features, _plateaus, _prepare, ecg_peaks


REVISION = "0bd11f9dd0fa8a9329ea113b388007d9c7ae7984"
SOURCE_RATE = 500
PATIENTS_PER_CENTER = 8
WINDOWS_PER_PATIENT = 5
MIN_ECG_MATCH_RATE = 0.90
METHODS = ("red", "infrared", "red_two_filters", "red_infrared_consensus")
BOOTSTRAP_REPETITIONS = 10_000
SEED = 2026


def reference_intervals(ecg):
    from wfdb.processing import xqrs_detect

    if not np.isfinite(ecg).all() or np.ptp(ecg) <= 0 or _plateaus(ecg, np.ones(len(ecg), dtype=bool), SOURCE_RATE).any():
        return None
    original = ecg_peaks(ecg, SOURCE_RATE) / SOURCE_RATE
    comparison = xqrs_detect(ecg.astype(float), SOURCE_RATE, verbose=False) / SOURCE_RATE
    if (min(len(original), len(comparison)) < 3
            or min(_nearest_fraction(original, comparison), _nearest_fraction(comparison, original)) < MIN_ECG_MATCH_RATE):
        return None
    rr = np.diff(original)
    rates = 60 / np.array([np.median(rr), np.median(np.diff(comparison))])
    if abs(rates[0] - rates[1]) > 5:
        return None
    differences = np.diff(rr)
    irregularity = np.sqrt(np.mean(np.square(differences))) / np.median(rr) + np.mean(np.abs(differences) > 0.05)
    return rates[0], irregularity


def pulse_view(signal):
    if not np.isfinite(signal).all() or np.ptp(signal) <= 0:
        return None
    values = resample_poly(signal, PPG_RATE, SOURCE_RATE, padtype="line")
    prepared = _prepare(values, PPG_RATE, minimum_fs=4.0)
    if prepared is None:
        return None
    adjusted, present = prepared
    wave = _bandpass(adjusted, PPG_RATE, 0.4, 8.0)
    small, center, large = np.percentile(wave, (10, 50, 90))
    polarity = -1 if center - small > large - center else 1
    peaks = _find(polarity * wave)
    return polarity * values, peaks, present & ~_plateaus(adjusted, present, PPG_RATE)


def evaluate_window(window):
    reference = reference_intervals(window[0])
    if reference is None:
        return None
    red, infrared = map(pulse_view, window[1:])
    result = {}
    if red is not None:
        original, single = pulse_consensus(red[0])
        result.update(red=original, red_two_filters=single)
    if infrared is not None:
        result["infrared"] = _interval_features(infrared[1], PPG_RATE, infrared[2])
    if red is not None and infrared is not None:
        accepted = corroborated_peaks(red[1], infrared[1])
        accepted &= infrared[2][red[1]]
        result["red_infrared_consensus"] = _interval_features(red[1], PPG_RATE, red[2], verified_peaks=accepted)
    return reference, {method: features for method, features in result.items() if features[VALID] > 0}


def evaluate(directory):
    files = sorted(directory.glob("*.npz"))
    if len(files) != 4 * PATIENTS_PER_CENTER:
        raise ValueError("Use the complete preregistered four-center PeriWave sample")
    sources = Counter()
    member_digests = set()
    records = []
    red_support = []
    coverage = Counter()
    for participant, path in enumerate(files):
        with np.load(path, allow_pickle=False) as stored:
            observed, center, member = stored["waveforms"], stored["center"].item(), stored["member_digest"].item()
            if (stored["release_revision"].item() != REVISION or center not in "ABCD" or member in member_digests
                    or observed.shape[1:] != (3, SOURCE_RATE * WINDOW_SECONDS) or len(observed) > WINDOWS_PER_PATIENT):
                raise ValueError("A staged sample does not match the pinned, unique paired waveform plan")
        sources[center] += 1
        member_digests.add(member)
        coverage["source_waveform_windows"] += len(observed)
        for window in observed:
            result = evaluate_window(window)
            if result is None:
                continue
            reference, features = result
            coverage["ecg_consensus_windows"] += 1
            coverage.update(features.keys())
            if "red" in features:
                red_support.append((center, participant, reference, {
                    "red": features["red"], "dual_with_red_fallback": features.get("red_infrared_consensus", features["red"]),
                }))
            if set(features) != set(METHODS):
                continue
            records.append((center, participant, reference, features))
    if sources != dict.fromkeys("ABCD", PATIENTS_PER_CENTER) or not records:
        raise ValueError("Four centers and some common reference and optical support are required")

    def summarize(selected, methods=METHODS):
        if not selected:
            return {"paired_windows": 0}
        rates, irregularity = np.asarray([row[2] for row in selected]).T
        metrics = {}
        for name in methods:
            matrix = np.stack([row[3][name] for row in selected])
            predicted = matrix[:, RMSSD] + matrix[:, PNN50]
            delta = predicted - irregularity
            metrics[name] = {"median_absolute_irregularity_error": float(np.median(np.abs(delta))),
                             "median_irregularity_bias": float(np.median(delta)),
                             "median_absolute_heart_rate_error_bpm": float(np.median(np.abs(60 / matrix[:, 1] - rates))),
                             "median_accepted_interval_fraction": float(np.median(matrix[:, VALID]))}
        return {"participants": len({row[1] for row in selected}), "paired_windows": len(selected), "methods": metrics}

    def paired_difference(selected, candidate):
        reference = np.asarray([row[2][1] for row in selected])
        people = np.asarray([row[1] for row in selected])
        errors = {}
        for name in ("red", candidate):
            values = np.stack([row[3][name] for row in selected])
            errors[name] = np.abs(values[:, RMSSD] + values[:, PNN50] - reference)
        delta = errors[candidate] - errors["red"]
        groups = [delta[people == participant] for participant in np.unique(people)]
        random = np.random.default_rng(SEED)
        draws = random.integers(len(groups), size=(BOOTSTRAP_REPETITIONS, len(groups)))
        averages = [np.mean(np.concatenate([groups[index] for index in sample])) for sample in draws]
        tolerance = 1e-9
        return {"paired_median_absolute_error_change": float(np.median(delta)), "participant_resamples": BOOTSTRAP_REPETITIONS,
                "paired_mean_absolute_error_change": float(np.mean(delta)),
                "paired_mean_95_percentile_interval": [float(value) for value in np.quantile(averages, [0.025, 0.975])],
                "windows_with_lower_error": int(np.sum(delta < -tolerance)), "windows_with_higher_error": int(np.sum(delta > tolerance)),
                "windows_with_unchanged_error": int(np.sum(np.abs(delta) <= tolerance))}

    return {"scope": "External dual-wavelength finger PPG timing benchmark during surgery", "release_revision": REVISION,
            "scheduled_windows": len(files) * WINDOWS_PER_PATIENT, "participants_by_center": dict(sources),
            "peak_agreement_ms": AGREEMENT_MS, "coverage": dict(coverage), "all": summarize(records),
            "by_center": {center: summarize([row for row in records if row[0] == center]) for center in sorted(sources)},
            "paired_dual_vs_red": paired_difference(records, "red_infrared_consensus"),
            "exploratory_red_fallback": {**summarize(red_support, ("red", "dual_with_red_fallback")),
                                         "paired_vs_red": paired_difference(red_support, "dual_with_red_fallback")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args.staged))


if __name__ == "__main__":
    main()
