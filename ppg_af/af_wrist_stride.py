"""Build a cache-only 10-second alert grid over the frozen evaluation blocks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .af_wrist_features import (
    FEATURE_VERSION_KEY, PPG_RATE, PTT_SUBJECTS, RDDM_FEATURE_VERSION, ROUTES, WINDOW_SECONDS, extract_features,
    read_feature_version,
)


KINDS = ("monitor", "monitor_wear", "enriched_episode")
STRIDE = 10
CORE = slice(PPG_RATE, -PPG_RATE)
CELLS_PER_PARENT = WINDOW_SECONDS // STRIDE
PREROLL_PARENTS = 3
CELL_SAMPLES = PPG_RATE * STRIDE


def selected_rows(archive, run_id):
    rows = np.flatnonzero(archive["run_id"] == run_id)
    return rows[np.argsort(archive["start_s"][rows])]


def reusable_features(path, source):
    with np.load(path, allow_pickle=False) as cached:
        if read_feature_version(cached) != RDDM_FEATURE_VERSION:
            raise ValueError(f"Unexpected feature version in {path}")
        for field in ("subject_id", "start_s", "run_id", "run_kind"):
            np.testing.assert_array_equal(cached[field], source[field])
        np.testing.assert_array_equal(cached["ppg_available"], np.isfinite(source["ppg"][:, CORE]).all(1))
        return {field: cached[field] for field in (*ROUTES, "quality")}


def stage_run(original, preroll, rows, prior):
    run_id, = np.unique(original["run_id"][rows])
    kind, = np.unique(original["run_kind"][rows])
    start, stop = original["start_s"][rows[0]], original["start_s"][rows[-1]] + WINDOW_SECONDS
    if kind not in KINDS or len(prior) != PREROLL_PARENTS:
        raise ValueError("Expected an evaluation run with a three-parent preroll")
    np.testing.assert_array_equal(preroll["run_start_s"][prior], np.full(len(prior), start))
    np.testing.assert_array_equal(preroll["run_kind"][prior], np.full(len(prior), kind))
    parents = np.arange(start - WINDOW_SECONDS * PREROLL_PARENTS, stop, WINDOW_SECONDS)
    np.testing.assert_array_equal(parents, np.r_[preroll["start_s"][prior], original["start_s"][rows]])

    pulse = np.concatenate((preroll["ppg"][prior], original["ppg"][rows]))
    methods = np.r_[preroll["sync_method"][prior], original["sync_method"][rows]]
    unknown_parent = np.r_[preroll["preroll_status"][prior] == "cache_missing", np.zeros(len(rows), dtype=bool)]
    parent_ref = np.r_[np.zeros(len(prior), dtype=bool), original["reference_valid"][rows]]
    cells = pulse[:, CORE].reshape(-1, CELL_SAMPLES)
    current_ppg = np.isfinite(cells).all(1)
    cell_unknown = np.repeat(unknown_parent, CELLS_PER_PARENT)

    # The shared two seconds in the original 32-second padded rows certify
    # one wrist time grid. The duplicated samples are only an offline check.
    # None of the exterior context goes into feature extraction at time t.
    overlap = 2 * PPG_RATE
    left, right = pulse[:-1, -overlap:], pulse[1:, :overlap]
    verified = np.isfinite(left).all(1) & (left == right).all(1) & (methods[:-1] == methods[1:])

    valid_cells = np.arange(CELLS_PER_PARENT - 1, len(cells))
    decision_end = parents[0] + STRIDE * (valid_cells + 1)
    evaluation = decision_end > start
    reference = np.repeat(parent_ref, CELLS_PER_PARENT)[valid_cells]
    np.testing.assert_array_equal(reference[evaluation], np.repeat(original["reference_valid"][rows], CELLS_PER_PARENT))
    expected_warmup = np.arange(start - 2 * WINDOW_SECONDS, start + STRIDE, STRIDE)
    np.testing.assert_array_equal(decision_end[~evaluation], expected_warmup)

    count = len(valid_cells)
    windows = np.full((count, (WINDOW_SECONDS + 2) * PPG_RATE), np.nan, dtype=pulse.dtype)
    status = np.full(count, "unavailable", dtype="<U11")
    cached_row = np.full(count, -1, dtype=int)
    cached_prior = np.zeros(count, dtype=bool)
    derived_methods = methods[valid_cells // CELLS_PER_PARENT]
    for index, right_cell in enumerate(valid_cells):
        first = right_cell - CELLS_PER_PARENT + 1
        first_parent, last_parent = first // CELLS_PER_PARENT, right_cell // CELLS_PER_PARENT
        support = slice(first, right_cell + 1)
        if first_parent == last_parent:
            parent = first_parent
            cached_prior[index] = parent < len(prior)
            cached_row[index] = prior[parent] if cached_prior[index] else rows[parent - len(prior)]
            windows[index, CORE] = pulse[parent, CORE]
        elif verified[first_parent]:
            windows[index, CORE] = cells[support].reshape(-1)

        if (current_ppg[support] | cell_unknown[support]).all():
            if cell_unknown[support].any() or (first_parent != last_parent and not verified[first_parent]):
                status[index] = "unknown"
            elif np.ptp(windows[index, CORE]) > 0:
                status[index] = "available"

    return {
        "run_id": np.full(count, run_id),
        "run_kind": np.full(count, kind),
        "run_start_s": np.full(count, start),
        "start_s": decision_end - STRIDE,
        "decision_end_s": decision_end,
        "input_start_s": decision_end - WINDOW_SECONDS,
        "evaluation": evaluation,
        "reference_valid": reference,
        "stable_label": np.full(count, -1, dtype=np.int8),
        "ppg_available": current_ppg[valid_cells],
        "input_status": status,
        "sync_method": derived_methods,
        "cached_row": cached_row,
        "cached_prior": cached_prior,
        "ppg": windows,
    }


def stage_subject(path, preroll_path):
    """Build one subject's raw decision cells without loading model weights."""
    with (np.load(path, allow_pickle=False) as original,
          np.load(preroll_path, allow_pickle=False) as preroll):
        for field in ("subject_id", "ppg_fs", "af_intervals_s"):
            np.testing.assert_array_equal(original[field], preroll[field])
        if int(original["ppg_fs"].item()) != PPG_RATE or str(original["subject_id"].item()) != path.stem:
            raise ValueError(f"Wrong subject or sample rate in {path}")
        ids, kinds = original["run_id"], original["run_kind"]
        expected = set(np.unique(ids[np.isin(kinds, KINDS)]))
        if not expected or set(np.unique(preroll["run_id"])) != expected:
            raise ValueError(f"Preroll run coverage changed for {path.stem}")
        runs = [stage_run(original, preroll, selected_rows(original, rid), selected_rows(preroll, rid))
                for rid in sorted(expected)]
        data = {field: np.concatenate([run[field] for run in runs]) for field in runs[0]}
        baseline = data["evaluation"] & (data["cached_row"] >= 0)
        data["stable_label"][baseline] = original["stable_label"][data["cached_row"][baseline]]
        return {
            **data,
            "subject_id": original["subject_id"], "ppg_fs": original["ppg_fs"],
            "af_intervals_s": original["af_intervals_s"],
            "feature_window_seconds": np.array(WINDOW_SECONDS), "decision_stride_seconds": np.array(STRIDE),
        }


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def feature_rows(data, baseline, preroll, calibration, *, batch_size, device):
    matching = data["cached_row"] >= 0
    new = np.flatnonzero(~matching & (data["input_status"] == "available"))
    scores = {key: np.full((len(matching), *items.shape[1:]), np.nan)
              for key, items in baseline.items()}
    scores["quality"][:] = 0.0
    for prior, source in ((False, baseline), (True, preroll)):
        mask = matching & (data["cached_prior"] == prior)
        old_rows = data["cached_row"][mask]
        for key, values in source.items():
            scores[key][mask] = values[old_rows]
    if len(new):
        delay, template, models = calibration
        generated, available, quality = extract_features(data["ppg"][new], template, delay, models,
                                                         batch_size=batch_size, device=device)
        if not available.all():
            raise ValueError("An off-grid row marked available lacks finite PPG")
        for key, values in {**generated, "quality": quality}.items():
            scores[key][new] = values
    for key in ROUTES:
        scoreable = np.isfinite(scores[key]).all(1)
        if np.any(scoreable & (~data["ppg_available"] | (data["input_status"] != "available"))):
            raise ValueError(f"{key} produced a score for missing input")
    return {
        **{key: value for key, value in data.items() if key != "ppg"},
        **scores,
        FEATURE_VERSION_KEY: RDDM_FEATURE_VERSION,
    }


def locked_sources(cohort, preroll, lock_path, *, independent=False):
    lock = json.loads(lock_path.read_bytes())
    fit, calibration, test = (set(lock[key]) for key in ("fit_subjects", "calibration_subjects", "test_roster"))
    development = fit | calibration
    sources = sorted(path for path in cohort.glob("*.npz") if not path.name.endswith(".partial.npz"))
    history = {path.stem for path in preroll.glob("*.npz") if not path.name.endswith(".partial.npz")}
    found = {path.stem for path in sources}
    matches = found <= test if independent else found == development
    if (not fit or not calibration or fit & calibration or development & test
            or not found or not matches or history != found):
        raise ValueError("Original input and history must match the selected participant roster")
    return sources


def _load_calibration(args):
    import torch

    from .af_wrist_smoke import fit_ptt_calibration
    from .rddm import load_author_model

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("The installed PyTorch cannot access CUDA")
    delay, template = fit_ptt_calibration(args.ptt_dir, PPG_RATE, subject_count=PTT_SUBJECTS)
    models = load_author_model(args.upstream, args.weights, args.device)
    return delay, template, models


def _build(args):
    sources = locked_sources(args.cohort, args.preroll, args.lock, independent=args.independent)
    calibration = _load_calibration(args) if args.feature_output is not None else None
    totals = Counter()
    for path in sources:
        history_path = args.preroll / path.name
        data = stage_subject(path, history_path)
        scored = data["evaluation"]
        new = data["cached_row"] < 0
        for name, mask in (("scored", scored), ("warmup", ~scored)):
            totals[f"{name}_cells"] += int(mask.sum())
            totals[f"{name}_ecg_valid_cells"] += int((mask & data["reference_valid"]).sum())
            totals[f"{name}_ppg_present_cells"] += int((mask & data["ppg_available"]).sum())
            for state in ("available", "unavailable", "unknown"):
                totals[f"{name}_input_{state}"] += int((mask & (data["input_status"] == state)).sum())
            totals[f"{name}_new_inference_rows"] += int((mask & new & (data["input_status"] == "available")).sum())
        save_npz(args.output / path.name, data)
        if calibration is not None:
            with (np.load(path, allow_pickle=False) as original,
                  np.load(history_path, allow_pickle=False) as preroll):
                baseline = reusable_features(args.cohort_features / path.name, original)
                history = reusable_features(args.preroll_features / path.name, preroll)
            bundle = feature_rows(data, baseline, history, calibration, batch_size=args.batch_size, device=args.device)
            save_npz(args.feature_output / path.name, bundle)
    print(json.dumps(dict(totals), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "preroll", "lock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--independent", action="store_true", help="Stage only the decoded independent roster members")
    parser.add_argument("--feature-output", type=Path, help="Also generate the three AF feature routes")
    for name in ("cohort-features", "preroll-features", "ptt-dir", "upstream", "weights"):
        parser.add_argument(f"--{name}", type=Path, help="Required with --feature-output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Both batch size and threads must be positive")
    feature_inputs = (args.cohort_features, args.preroll_features, args.ptt_dir, args.upstream, args.weights)
    if args.feature_output is not None and None in feature_inputs:
        parser.error("--feature-output requires both feature caches, --ptt-dir, --upstream, and --weights")
    inputs = {path.resolve() for path in (args.cohort, args.preroll, *feature_inputs) if path is not None}
    outputs = [path.resolve() for path in (args.output, args.feature_output) if path is not None]
    if set(outputs) & inputs or len(set(outputs)) != len(outputs):
        parser.error("Input and output directories must be distinct")
    try:
        _build(args)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"AF stride staging failed: {error}\n")


if __name__ == "__main__":
    main()
