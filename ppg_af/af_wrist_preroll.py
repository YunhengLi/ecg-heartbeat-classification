"""Stage state-only wrist PPG before the locked development evaluation runs."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .af_wrist_cohort import CONTEXT, WINDOW, in_wrist_segment
from .afppgecg import PPG_FS, _read_vector, _sample_bounds, load_subject


EVALUATION_KINDS = ("monitor", "monitor_wear", "enriched_episode")
HISTORY_WINDOWS = 3
FLANK = int(CONTEXT * PPG_FS)
WIDTH = int(WINDOW * PPG_FS) + 2 * FLANK


def _native_context(subject, start):
    stop = start + WINDOW
    padded = np.full(WIDTH, np.nan)
    segment = in_wrist_segment(subject, start, stop)
    if segment is None:
        return padded, "absent", (0, 0), "none", ""
    green = subject._ppg_data[segment.index]
    first, last = _sample_bounds(start, stop, segment.start_time, PPG_FS, green.size)
    if last - first != WIDTH - 2 * FLANK:
        raise ValueError("A native PPG core must contain exactly 30 seconds")
    lo, hi = max(0, first - FLANK), min(green.size, last + FLANK)
    left, right = first - lo, hi - last
    padded[FLANK - left:WIDTH - FLANK + right] = _read_vector(green, slice(lo, hi))
    return padded, segment.sync_method, (left, right), "raw_mat", ""


def stage_subject(source, destination, ecg, ppg):
    """Accept a local MAT pair or seekable streams, writing one feature-ready NPZ."""
    with np.load(source, allow_pickle=False) as archive:
        subject_id = str(archive["subject_id"].item())
        if subject_id != source.stem or archive["ppg_fs"].item() != PPG_FS:
            raise ValueError(f"Unexpected participant or PPG sample rate in {source.name}")
        starts = archive["start_s"]
        kinds = archive["run_kind"]
        names = archive["run_id"]
        methods = archive["sync_method"]
        waveforms = archive["ppg"]
        intervals = archive["af_intervals_s"]
    if (waveforms.shape != (len(starts), WIDTH) or len(set(starts)) != len(starts)
            or np.any(np.remainder(starts, WINDOW) != 0)):
        raise ValueError(f"Expected unique 30-second window starts and staged PPG in {source.name}")

    existing = {
        int(start): (signal, method, (FLANK, FLANK),
                     "staged_evaluation" if kind in EVALUATION_KINDS else "staged_fit", name)
        for start, signal, method, kind, name in zip(starts, waveforms, methods, kinds, names, strict=True)
        if method in {"beat", "coarse"}
    }
    runs = []
    for name in np.unique(names):
        rows = names == name
        run_kinds = np.unique(kinds[rows])
        clock = np.sort(starts[rows])
        if len(run_kinds) != 1 or np.any(np.diff(clock) != WINDOW):
            raise ValueError(f"Run {name} must be contiguous and use one scope")
        if run_kinds[0] in EVALUATION_KINDS:
            runs.append((name, run_kinds[0], int(clock[0])))
    if not runs:
        raise ValueError(f"No evaluation runs in {source.name}")

    rows = []
    with load_subject(ecg, ppg, subject_id=subject_id, include_acceleration=False) as subject:
        for name, kind, cutoff in sorted(runs, key=lambda run: run[2]):
            for start in range(cutoff - HISTORY_WINDOWS * WINDOW, cutoff, WINDOW):
                if start not in existing:
                    existing[start] = _native_context(subject, start)
                rows.append((name, kind, cutoff, start, *existing[start]))
    names, kinds, cutoffs, starts, waveforms, methods, flanks, sources, source_runs = zip(*rows, strict=True)
    waveforms = np.stack(waveforms)
    cores = waveforms[:, FLANK:-FLANK]
    finite = np.isfinite(cores).all(axis=1)
    usable = finite.copy()
    usable[finite] = np.ptp(cores[finite], axis=1) > 0
    status = np.where(np.asarray(methods) == "absent", "no_wrist_core",
                      np.where(usable, "finite_nonflat", np.where(finite, "flat", "nonfinite")))

    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary, subject_id=subject_id, ppg_fs=int(PPG_FS), af_intervals_s=intervals,
        run_id=names, run_kind=kinds, run_start_s=cutoffs, start_s=starts, ppg=waveforms,
        sync_method=methods, native_context_samples=flanks, preroll_status=status,
        preroll_source=sources, preroll_source_run_id=source_runs, core_usable=usable,
        stable_label=np.full(len(rows), -1, dtype=np.int8), reference_valid=np.zeros(len(rows), dtype=bool),
    )
    temporary.replace(destination)
    return len(rows), int(usable.sum())


def _development_ids(lock):
    groups = [lock.get(key) for key in ("fit_subjects", "calibration_subjects", "test_roster")]
    if lock.get("window_seconds") != WINDOW or any(
        not isinstance(group, list) or not group or any(
            not isinstance(subject, str) or not re.fullmatch(r"[0-9]{3}", subject) for subject in group
        ) or len(group) != len(set(group)) for group in groups
    ):
        raise ValueError("Expected a locked protocol with unique AFPPGECG IDs and 30-second windows")
    fit, calibration, test = map(set, groups)
    if fit & calibration or (fit | calibration) & test:
        raise ValueError("The fit, calibration, and test rosters must be disjoint")
    return fit | calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "mat-dir", "lock", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    members = _development_ids(json.loads(args.lock.read_text(encoding="utf-8")))
    sources = {path.stem: path for path in args.cohort.glob("*.npz") if not path.name.endswith(".partial.npz")}
    if set(sources) != members:
        parser.error("The cohort must contain exactly the locked fit and calibration participants")
    if args.output.resolve() in {args.cohort.resolve(), args.mat_dir.resolve()}:
        parser.error("Use a separate directory for the staged history")
    if any(not (args.mat_dir / f"{subject}_{signal}.mat").is_file()
           for subject in members for signal in ("ECG", "PPG")):
        parser.error("The MAT directory must contain both native files for each development participant")

    args.output.mkdir(parents=True, exist_ok=True)
    totals = [
        stage_subject(sources[subject], args.output / f"{subject}.npz", args.mat_dir / f"{subject}_ECG.mat",
                      args.mat_dir / f"{subject}_PPG.mat")
        for subject in sorted(members)
    ]
    rows, usable = map(sum, zip(*totals, strict=True))
    print(f"Staged {rows} state-only windows for {len(members)} development participants, {usable} with usable raw PPG")


if __name__ == "__main__":
    main()
