"""Compare native wrist movement at false alert starts and nearby non-alert times."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .af_events import _grid, _intervals, _union
from .af_state import annotated_seconds
from .af_wrist_confirmation_retrospective import replay_on_schedule
from .af_wrist_exercise import BASELINE_SHA, PROTOCOL_SHA
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS, WINDOW_SECONDS
from .af_wrist_holdout import _write_json
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import EARLIER_S
from .af_wrist_motion_quality import optical_motion_features
from .af_wrist_quality_retrospective import _read_pinned
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_state_probability import HALF_HOUR_RESULTS_SHA
from .af_wrist_template_independent import _locked_model
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import ACCEL_FS


FEATURES = ("PPG_quality", "pulse_band_entropy", "motion_weighted_pulse_power_fraction", "log_relative_acceleration_variation")
MAX_CONTROL_SECONDS = 120
NATIVE_RATE = int(ACCEL_FS)
INPUT_BINS = WINDOW_SECONDS // STRIDE_SECONDS
BOOTSTRAPS = 10_000


def source_inputs(optical_path, acceleration_path, run, digest):
    with np.load(optical_path, allow_pickle=False) as optical, np.load(acceleration_path, allow_pickle=False) as motion:
        raw, acceleration = optical["raw_ppg"], motion["accel"]
        if (optical["plan_sha256"].item() != digest or motion["plan_sha256"].item() != digest
                or motion["subject_id"].item() != run.subject_id or optical["subject_id"].item() != run.subject_id
                or motion["start_s"].item() != run.start_s or optical["start_s"].tolist() != [run.start_s]
                or sha256(raw.tobytes()).hexdigest() != motion["frozen_optical_sha256"].item()):
            raise ValueError("Native acceleration must match the sealed PPG input and participant clock")
        if len(acceleration):
            phase = motion["accel_minus_ppg_s"].item()
            if acceleration.shape != ((BLOCK_SECONDS + EARLIER_S) * NATIVE_RATE, 3) or not -1e-9 <= phase <= 1 / PPG_RATE + 1e-9:
                raise ValueError("Motion samples do not cover the planned wrist input cadence")
    return raw[0], acceleration


def matched_pairs(cases, controls):
    free = set(np.flatnonzero(controls))
    max_distance = MAX_CONTROL_SECONDS // STRIDE_SECONDS
    for case in np.flatnonzero(cases):
        options = [control for control in free if abs(control - case) <= max_distance]
        if options:
            control = min(options, key=lambda test: (abs(test - case), test))
            free.difference_update(range(control - INPUT_BINS + 1, control + INPUT_BINS))
            yield case, control


def audit_block(left, right, raw, acceleration, groups, coverage):
    if not len(acceleration):
        return
    coverage["blocks_with_native_motion"] += 1
    edges, decisions, valid, _ = _grid(left)
    af = annotated_seconds(edges, _union(_intervals(left.af_intervals_s)))
    stable = np.convolve((valid & (af <= 1e-8)).astype(int), np.ones(INPUT_BINS, dtype=int), mode="valid")
    clean_reference = np.r_[np.zeros(INPUT_BINS - 1, dtype=bool), stable == INPUT_BINS]
    warm_rows = len(raw) - len(decisions)
    available = np.isfinite(raw[warm_rows:]).all(axis=1) & (np.ptp(raw[warm_rows:], axis=1) > 0)
    motion = []
    for align in range(warm_rows, len(raw)):
        begin = align * STRIDE_SECONDS * NATIVE_RATE
        motion.append(acceleration[begin:begin + WINDOW_SECONDS * NATIVE_RATE])
    available &= np.isfinite(np.asarray(motion)).all(axis=(1, 2))
    eligible = clean_reference & available
    alerted_left, alerted_right = left.alert_start_status == 1, right.alert_start_status == 1
    alerted = (alerted_left | alerted_right).astype(int)
    near_alert = np.convolve(alerted, np.ones(2 * INPUT_BINS - 1, dtype=int), mode="same") > 0
    control = eligible & ~near_alert & (decisions >= 0) & (right.decisions >= 0)
    categories = {"plain_false_starts": alerted_left, "confirmation_false_starts": alerted_right,
                  "plain_only_exact_starts": alerted_left & ~alerted_right,
                  "shared_exact_starts": alerted_left & alerted_right}
    cached = {}
    for label, selected in categories.items():
        cases = eligible & selected
        coverage[f"{label}_with_clean_reference_and_motion"] += int(cases.sum())
        for case, negative in matched_pairs(cases, control):
            for row in (case, negative):
                if row not in cached:
                    upsampled = resample_poly(motion[row], PPG_RATE, NATIVE_RATE, axis=0, padtype="line")
                    cached[row] = optical_motion_features(raw[warm_rows + row], upsampled)
            groups[label].append((left.subject_id, cached[case], cached[negative]))


def summarize_pairs(rows):
    if not rows:
        return {"matched_pairs": 0}
    people, cases, controls = map(np.asarray, zip(*rows, strict=True))
    contributors = np.unique(people)
    paired = cases - controls
    means = np.asarray([paired[people == person].mean(axis=0) for person in contributors])
    draws = np.random.default_rng(2026).integers(len(means), size=(BOOTSTRAPS, len(means)))
    bounds = np.quantile(means[draws].mean(axis=1), [0.025, 0.975], axis=0)
    fields = {name: {"median_case": float(np.median(cases[:, index])), "median_control": float(np.median(controls[:, index])),
                     "mean_paired_difference_by_participant": float(means[:, index].mean()),
                     "participant_bootstrap_95_interval": bounds[:, index].tolist()}
              for index, name in enumerate(FEATURES)}
    return {"matched_pairs": len(rows), "participants": len(contributors), "features": fields}


def evaluate(args):
    protocol = _read_pinned(args.lock, PROTOCOL_SHA)
    model_lock = _read_pinned(args.historical_model_lock, BASELINE_SHA)
    model, roster, _, _ = _locked_model(model_lock, protocol)
    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    original = _read_pinned(args.half_hour_results, HALF_HOUR_RESULTS_SHA, public=True)
    if set(schedule) != roster or digest != original["plan_sha256"]:
        raise ValueError("The motion audit must use the original sealed wrist plan and frozen model")
    _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    baseline, candidate = replay_on_schedule(schedule, digest, args.monitoring_cache, template, model,
                                             block_seconds=BLOCK_SECONDS)
    if baseline[0] != original["original"] or candidate[0] != original["conditional_confirmation"]:
        raise ValueError("Both earlier alert policies must reproduce before auditing motion")
    groups, coverage = defaultdict(list), Counter()
    for left, right in zip(baseline[1], candidate[1], strict=True):
        filename = f"{left.subject_id}.npz"
        pulse, accel = source_inputs(args.monitoring_cache / filename, args.acceleration_cache / filename, left, digest)
        audit_block(left, right, pulse, accel, groups, coverage)
    return {"scope": "Match observed wrist motion at fixed false starts to same-block non-alert controls",
            "reference_plan_sha256": digest, "coverage": dict(coverage), "maximum_control_distance_seconds": MAX_CONTROL_SECONDS,
            "groups": {key: summarize_pairs(groups[key]) for key in sorted(groups)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "historical-model-lock", "prior-inputs", "prior-monitor-plan", "plan", "half-hour-results",
                 "monitoring-cache", "acceleration-cache", "ptt-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.output, evaluate(args))
