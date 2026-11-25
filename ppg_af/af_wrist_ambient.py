"""Test native ambient-light coherence on the original wrist AF folds."""

import argparse
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np

from .af_wrist_acceleration_stage import SIDECAR_FOLDERS, scoreable_rows, verified_wrist_bounds
from .af_wrist_exercise import HEART_RATE_BAND_HZ, motion_weighted_spectrum
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development
from .af_wrist_quality_study import _compare_with_pre_onset
from .af_wrist_template_study import DEFAULT_C, _replay, fit_template, verify_fit_support
from .afppgecg import DEFAULT_MAX_REFERENCE_RR_SECONDS, WristSubject, _read_vector, _vector_length


class NativeAmbient:
    """Locate ambient light on the already verified green PPG sample clock."""

    def __init__(self, ecg, wrist, person):
        self.subject = WristSubject(ecg, wrist, person, DEFAULT_MAX_REFERENCE_RR_SECONDS, False)
        self.wrist = wrist
        self.references = _read_vector(wrist["PPG_AMBIENT"])
        if len(self.references) != len(self.subject.segments):
            raise ValueError("The source optical channels have different segment counts")
        self.datasets = {}

    def sources_for(self, start, method, green):
        segment, first, last = verified_wrist_bounds(self.subject, start, method, green)
        if segment.index not in self.datasets:
            self.datasets[segment.index] = self.wrist[self.references[segment.index]]
        ambient = self.datasets[segment.index]
        green_source = self.subject._ppg_data[segment.index]
        if _vector_length(ambient) != _vector_length(green_source):
            return None
        return ambient, first, last


def save_ambient(target, rows, green, ambient, lock_digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, rows=rows, ambient=ambient, source_protocol_sha256=lock_digest,
                        staged_green_core_sha256=sha256(green.tobytes()).hexdigest())
    temp.replace(target)


def stage_ambient(source_dir, paths, sidecars, people, lock_digest):
    for person in people:
        with (h5py.File(source_dir / f"{person}_ECG.mat", "r") as ecg,
              h5py.File(source_dir / f"{person}_PPG.mat", "r") as wrist):
            native = NativeAmbient(ecg, wrist, person)
            for stride, ((raw, features), folder) in enumerate(zip(paths, SIDECAR_FOLDERS, strict=True)):
                name = f"{person}.npz"
                rows, starts, methods, green = scoreable_rows(raw / name, features / name, stride=bool(stride))
                ambient = np.full(green.shape, np.nan)
                for i, (start, method, pulse) in enumerate(zip(starts, methods, green, strict=True)):
                    clip = native.sources_for(float(start), method, pulse)
                    if clip is not None:
                        source, first, last = clip
                        values = _read_vector(source, slice(first, last))
                        if np.isfinite(values).all() and np.ptp(values) > 0:
                            ambient[i] = values
                save_ambient(sidecars / folder / name, rows, green, ambient, lock_digest)


def ambient_residual_power(pulse, ambient):
    frequencies, power, weighted = motion_weighted_spectrum(pulse, ambient[:, None])
    keep = (frequencies >= HEART_RATE_BAND_HZ[0]) & (frequencies <= HEART_RATE_BAND_HZ[1])
    total = power[keep].sum()
    return float(np.clip(weighted[keep].sum() / total, 0, 1)) if total > 0 else 0.0


def _ambient_rows(raw_file, feature_file, sidecar_file, lock_digest, *, stride):
    rows, _, _, green = scoreable_rows(raw_file, feature_file, stride=stride)
    with (np.load(sidecar_file, allow_pickle=False) as sidecar,
          np.load(feature_file, allow_pickle=False) as feature):
        ambient = sidecar["ambient"]
        if (not np.array_equal(sidecar["rows"], rows) or ambient.shape != green.shape
                or str(sidecar["source_protocol_sha256"].item()) != lock_digest
                or sha256(green.tobytes()).hexdigest() != str(sidecar["staged_green_core_sha256"].item())):
            raise ValueError("Ambient rows must match the exact original green PPG and feature clock")
        features = np.zeros((len(feature["start_s"]), 2))
    present = np.isfinite(ambient).all(axis=1) & (np.ptp(ambient, axis=1) > 0)
    features[rows[present], 0] = 1
    for i in np.flatnonzero(present):
        features[rows[i], 1] = ambient_residual_power(green[i], ambient[i])
    return features


def build_ambient_features(grid, paths, sidecars, lock_digest):
    blocks, people = [], []
    for stride, ((raw, feature), folder) in enumerate(zip(paths, SIDECAR_FOLDERS, strict=True)):
        for person in np.unique(grid.subjects):
            name = f"{person}.npz"
            values = _ambient_rows(raw / name, feature / name, sidecars / folder / name, lock_digest, stride=bool(stride))
            blocks.append(values)
            people.extend([person] * len(values))
    if not np.array_equal(people, grid.subjects):
        raise ValueError("Ambient light must preserve the original participant and window order")
    side_features = np.concatenate(blocks)
    present = side_features[:, 0] > 0
    shared = _shared_features(grid.table)
    if np.any(present & ~shared) or not present.any():
        raise ValueError("Ambient features need support on the original PPG scoreable decisions")
    study = {"original_ppg_scoreable_rows": int(shared.sum()), "scoreable_with_ambient": int(present.sum()),
             "participants_with_ambient": len(np.unique(grid.subjects[present])),
             "ambient_residual_power_quantiles": np.quantile(side_features[present, 1], [0, .25, .5, .75, 1]).tolist(),
             "ambient_presence_varies_on_scoreable_rows": bool(np.any(~present[shared]))}
    return side_features, study


def evaluate_ambient(grid, original, side_features, support, lock, prior):
    comparison = {}
    availability_only = np.column_stack((original, side_features[:, 0]))
    coherent_power = np.column_stack((original, side_features))
    for seed in SEEDS:
        fit = fit_template(grid, original, lock, DEFAULT_C, seed=seed)
        baseline = _replay(grid, fit)
        if seed == SEEDS[0] and baseline[0] != prior:
            raise ValueError("The original template quality result must reproduce before ambient comparison")
        controls = {"ambient_coherence": coherent_power}
        if support["ambient_presence_varies_on_scoreable_rows"]:
            controls["ambient_presence"] = availability_only
        evaluated = {"original": baseline[0]}
        for name, matrix in controls.items():
            candidate = fit_template(grid, matrix, lock, DEFAULT_C, seed=seed)
            verify_fit_support(fit, candidate)
            result = _compare_with_pre_onset(baseline, _replay(grid, candidate), lock)
            evaluated[name] = result
            summary = result["summary"]
            print(f"{seed} {name}: {summary['episodes_detected']}/{summary['complete_episodes']} detections, "
                  f"{summary['wrist_false_alerts']} false starts", flush=True)
        comparison[str(seed)] = evaluated
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development-raw", "development-features", "stride-raw", "stride-features", "ambient-sidecars",
                "lock", "clock-results", "quality-results", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Optionally stage ambient from local native MAT files before evaluation")
    args = parser.parse_args()
    try:
        grid, quality, paths, lock, lock_digest, prior = load_motion_development(args)
        if args.source_dir is not None:
            stage_ambient(args.source_dir, paths, args.ambient_sidecars, np.unique(grid.subjects), lock_digest)
        side_features, support = build_ambient_features(grid, paths, args.ambient_sidecars, lock_digest)
        comparison = evaluate_ambient(grid, quality["quality_interactions"], side_features, support, lock, prior)
        _write_json(args.output, {"scope": "Ambient-light coherence on the original AF wrist development cohort",
                                  "development_participants": len(np.unique(grid.subjects)), "outer_folds": 5,
                                  "coherence_band_hz": HEART_RATE_BAND_HZ, "support": support, "comparisons": comparison})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist ambient-light study failed: {error}\n")


if __name__ == "__main__":
    main()
