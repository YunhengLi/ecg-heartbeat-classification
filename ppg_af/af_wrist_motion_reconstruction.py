"""Compare original and motion-filtered template ECG on the locked AF clock."""

import argparse
from pathlib import Path

import numpy as np

from .af_wrist_acceleration_stage import SIDECAR_FOLDERS
from .af_wrist_features import PPG_RATE, PTT_SUBJECTS
from .af_wrist_holdout import _shared_features, _write_json
from .af_wrist_motion_quality import SEEDS, load_motion_development, verified_motion_rows
from .af_wrist_motion_waveform import motion_filtered_ppg
from .af_wrist_quality_study import _compare_with_pre_onset, quality_interaction_features
from .af_wrist_sampling_stress import _extract
from .af_wrist_smoke import fit_ptt_calibration
from .af_wrist_template_study import DEFAULT_C, _replay, fit_template, verify_fit_support


def transformed_intervals(grid, paths, sidecar_root, lock_digest, template, original):
    transformed = original.copy()
    counts = {"original_scoreable_windows": int(_shared_features(grid.table).sum()), "windows_with_acceleration": 0,
              "motion_filtered_windows": 0, "fallback_to_original_after_filtering": 0}
    position = 0
    for stride, ((raw, features), folder) in enumerate(zip(paths, SIDECAR_FOLDERS, strict=True)):
        for person in np.unique(grid.subjects):
            filename = f"{person}.npz"
            count, rows, cores, motion, present = verified_motion_rows(
                raw / filename, features / filename, sidecar_root / folder / filename, lock_digest, stride=bool(stride),
            )
            if not np.all(grid.subjects[position:position + count] == person):
                raise ValueError("Motion-filtered PPG rows must stay on the original subject clock")
            selected = position + rows[present]
            position += count
            if not len(selected):
                continue
            recovered = _extract(cores[present], template)["baseline"]
            if not np.array_equal(recovered, original[selected]):
                raise ValueError("The unchanged template must reproduce every native interval feature")
            filtered = np.stack([motion_filtered_ppg(pulse, acc)[0] for pulse, acc in zip(cores[present], motion[present], strict=True)])
            values = _extract(filtered, template)["baseline"]
            supported = np.isfinite(values).all(axis=1)
            transformed[selected[supported]] = values[supported]
            counts["windows_with_acceleration"] += len(selected)
            counts["motion_filtered_windows"] += int(supported.sum())
            counts["fallback_to_original_after_filtering"] += int((~supported).sum())
    if position != len(original) or not np.array_equal(np.isfinite(transformed).all(axis=1), np.isfinite(original).all(axis=1)):
        raise ValueError("Filtering must preserve the original PPG score support")
    return transformed, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development-raw", "development-features", "stride-raw", "stride-features", "acceleration-sidecars",
                "lock", "clock-results", "quality-results", "ptt-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    try:
        grid, quality, paths, lock, lock_digest, original = load_motion_development(args)
        _, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
        intervals, support = transformed_intervals(grid, paths, args.acceleration_sidecars, lock_digest, template,
                                                   quality["baseline"])
        candidate = quality_interaction_features(intervals, quality["quality"][:, -1])
        comparisons = {}
        for seed in SEEDS:
            base = fit_template(grid, quality["quality_interactions"], lock, DEFAULT_C, seed=seed)
            fitted = fit_template(grid, candidate, lock, DEFAULT_C, seed=seed)
            verify_fit_support(base, fitted)
            original_replay, candidate_replay = _replay(grid, base), _replay(grid, fitted)
            if seed == SEEDS[0] and original_replay[0] != original:
                raise ValueError("The original quality and template replay must reproduce")
            comparisons[str(seed)] = {"original": original_replay[0],
                                      "motion_filtered": _compare_with_pre_onset(original_replay, candidate_replay, lock)}
            summary = candidate_replay[0]
            print(f"{seed}: {summary['episodes_detected']}/{summary['complete_episodes']} detections, "
                  f"{summary['wrist_false_alerts']} false starts", flush=True)
        _write_json(args.output, {"scope": "Grouped AF replay from phase-preserving motion-filtered template ECG",
                                  "support": support, "native_ppg_quality_retained": True,
                                  "development_participants": len(np.unique(grid.subjects)), "outer_folds": 5,
                                  "comparisons": comparisons})
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Wrist motion reconstruction study failed: {error}\n")


if __name__ == "__main__":
    main()
