"""Compare optical and electrical rhythm timing around verified ICU AF offsets."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_mimic_bashar_stage import RADIUS_SECONDS
from .af_periwave_benchmark import reference_intervals
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_pulse_consensus import PNN50, RMSSD, VALID
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_context import _pulse_features
from .af_wrist_template_independent import _locked_model
from .features import ppg_features


FIELDS = ("measured_ECG_irregularity", "direct_PPG_irregularity", "template_irregularity", "frozen_wrist_AF_score")


def optical_and_electrical_rows(ecg, ppg, rate, template, model, coverage):
    span = WINDOW_SECONDS * rate
    results = {"before_offset": [], "after_offset": []}
    for start in range(0, len(ecg), span):
        phase = "before_offset" if start < RADIUS_SECONDS * rate else "after_offset"
        coverage[(phase, "scheduled")] += 1
        reference = reference_intervals(ecg[start:start + span], sampling_rate=rate, xqrs_rate=250)
        if reference is None:
            continue
        coverage[(phase, "ECG_consensus")] += 1
        native = ppg[start:start + span]
        if not np.isfinite(native).all() or np.ptp(native) <= 0:
            continue
        pulse = resample_poly(native, PPG_RATE, rate, padtype="line")
        direct, _ = ppg_features(pulse, PPG_RATE)
        reconstructed = _pulse_features(template, pulse)
        if direct[VALID] <= 0 or reconstructed[VALID] <= 0:
            continue
        probability = _predict(reconstructed[None], np.ones(1, dtype=bool), model)[0]
        results[phase].append((reference[1], direct[RMSSD] + direct[PNN50],
                               reconstructed[RMSSD] + reconstructed[PNN50], probability))
        coverage[(phase, "matched")] += 1
    return results


def evaluate(args):
    receipt = json.loads(args.staging_results.read_bytes())
    protocol = _read_pinned(args.protocol_lock, PROTOCOL_SHA)
    frozen, _, _, _ = _locked_model(_read_pinned(args.wrist_model_lock, BASELINE_SHA), protocol)
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    seen, coverage, pairs = set(), Counter(), []
    for path in sorted(args.signal_cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            ecg, ppg = stored["ecg"], stored["ppg"]
            rate, record = int(stored["sampling_rate_hz"].item()), stored["private_record_digest"].item()
            if (stored["plan_sha256"].item() != receipt["plan_sha256"] or stored["boundary"].item() != "offset"
                    or record in seen or ecg.shape != (2 * RADIUS_SECONDS * rate,) or ppg.shape != ecg.shape):
                raise ValueError("Use only the original complete paired 4-minute offset windows")
        seen.add(record)
        samples = optical_and_electrical_rows(ecg, ppg, rate, template, frozen["model"], coverage)
        if all(samples.values()):
            pairs.append(tuple(np.median(values, axis=0) for values in samples.values()))
    if len(seen) != receipt["status"]["both_waves_staged"] or not pairs:
        raise ValueError("Offset comparisons require all staged files and at least one paired record")
    before, after = np.asarray(pairs).transpose(1, 0, 2)
    features = {name: {"median_before": float(np.median(before[:, col])), "median_after": float(np.median(after[:, col])),
                       "records_with_higher_values_before": int(np.sum(before[:, col] > after[:, col]))}
                for col, name in enumerate(FIELDS)}
    return {"scope": "Descriptive native ICU electrical and optical rhythm timing around annotated AF offsets",
            "staging": receipt, "records_with_common_before_and_after_signal": len(pairs),
            "coverage": {f"{phase}_{status}": count for (phase, status), count in sorted(coverage.items())},
            "wrist_model_sha256": BASELINE_SHA, "matched_record_summary": features}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("staging-results", "signal-cache", "protocol-lock", "wrist-model-lock", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
