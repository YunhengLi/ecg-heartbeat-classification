"""Audit reference and raw PPG support on the locked independent roster."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .af_events import AFRun, _coverage, _event_at_decision, _intervals, _union
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_holdout import MONITOR_KIND
from .af_wrist_operating import _remaining_non_af_time
from .af_wrist_stride import KINDS, STRIDE, stage_subject


def _locked_roster(protocol_lock):
    lock = json.loads(protocol_lock.read_bytes())
    if (lock.get("window_seconds") != WINDOW_SECONDS
            or lock.get("decision_stride_seconds", STRIDE) != STRIDE
            or lock.get("monitor_kind") != MONITOR_KIND):
        raise ValueError("Coverage requires the locked wrist monitoring and decision grid")
    groups = []
    for role in ("fit_subjects", "calibration_subjects", "test_roster"):
        identifiers = lock[role]
        if (not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(value, str) or not value for value in identifiers)
                or len(identifiers) != len(set(identifiers))):
            raise ValueError(f"Invalid or duplicate participant identifiers in {role}")
        groups.append(set(identifiers))
    fit, calibration, roster = groups
    if fit & calibration or (fit | calibration) & roster:
        raise ValueError("Fit, calibration and independent participant rosters must be disjoint")
    return roster


def _baseline(source):
    with np.load(source, allow_pickle=False) as archive:
        if str(archive["subject_id"].item()) != source.stem or int(archive["ppg_fs"].item()) != PPG_RATE:
            raise ValueError(f"Unexpected participant or PPG rate in {source.name}")
        use = np.isin(archive["run_kind"], KINDS)
        if not np.any(use):
            raise ValueError(f"No evaluation blocks in {source.name}")
        signal = archive["ppg"][use]
        if signal.shape != (int(use.sum()), (WINDOW_SECONDS + 2) * PPG_RATE):
            raise ValueError(f"Unexpected original PPG window shape in {source.name}")
        core = signal[:, PPG_RATE:-PPG_RATE]
        finite = np.isfinite(core).all(axis=1)
        usable = finite.copy()
        usable[finite] = np.ptp(core[finite], axis=1) > 0
        return {
            "run_id": archive["run_id"][use],
            "run_kind": archive["run_kind"][use],
            "decision_end_s": archive["start_s"][use] + WINDOW_SECONDS,
            "evaluation": np.ones(int(use.sum()), dtype=bool),
            "reference_valid": archive["reference_valid"][use],
            "input_status": np.where(usable, "available", "unavailable"),
            "af_intervals_s": archive["af_intervals_s"],
        }


def _spans(item, stride, mask):
    end = item["decision_end_s"][mask]
    return _union(np.column_stack((end - stride, end)))


def _subject_support(subject, item, stride):
    end = item["decision_end_s"]
    evaluation = item["evaluation"]
    reference = item["reference_valid"]
    states = item["input_status"]
    usable = states == "available"
    monitor = evaluation & (item["run_kind"] == MONITOR_KIND)
    episodes = _union(_intervals(item["af_intervals_s"]))
    complete, partial = _coverage(episodes, _spans(item, stride, evaluation & reference))
    monitor_complete, _ = _coverage(episodes, _spans(item, stride, monitor & reference))
    supported = _event_at_decision(end[evaluation & reference & usable], episodes)
    with_input = np.isin(np.arange(len(episodes)), supported[supported >= 0])

    runs, monitor_runs = [], []
    preceding_final = Counter()
    for run_id in np.unique(item["run_id"]):
        belongs = item["run_id"] == run_id
        rows = np.flatnonzero(belongs & evaluation)
        rows = rows[np.argsort(end[rows])]
        if not len(rows) or np.any(np.diff(end[rows]) != stride):
            raise ValueError(f"Evaluation blocks must use the original regular grid in {subject}")
        history = np.flatnonzero(belongs & ~evaluation)
        if len(history):
            latest = history[np.argmax(end[history])]
            if end[latest] != end[rows[0]] - stride or reference[history].any():
                raise ValueError(f"History must end at the evaluation boundary in {subject}")
            preceding_final[str(states[latest])] += 1
        kinds = np.unique(item["run_kind"][rows])
        if len(kinds) != 1:
            raise ValueError(f"Each evaluation block must have one scope in {subject}")
        run = AFRun(float(end[rows[0]] - stride), stride, episodes,
                    np.where(usable[rows], 0, -1), reference[rows], usable[rows], subject)
        runs.append(run)
        if kinds[0] == MONITOR_KIND:
            monitor_runs.append(run)
    ordered = sorted(runs, key=lambda run: run.start_s)
    for left, right in zip(ordered, ordered[1:]):
        if left.start_s + len(left.decisions) * stride > right.start_s:
            raise ValueError(f"Evaluation blocks overlap in {subject}")

    return {
        "complete_reference_episodes": int(complete.sum()),
        "partial_reference_episodes": int(partial.sum()),
        "complete_reference_episodes_in_wrist_monitoring": int(monitor_complete.sum()),
        "complete_reference_episodes_with_usable_raw_input_endpoint": int((complete & with_input).sum()),
        "reference_observed_seconds": int((evaluation & reference).sum()) * stride,
        "monitoring_observed_seconds": int((monitor & reference).sum()) * stride,
        "monitoring_observed_non_af_seconds": _remaining_non_af_time(monitor_runs),
        "monitoring_raw_input_opportunity_seconds": int((monitor & reference & usable).sum()) * stride,
        "monitoring_raw_input_opportunity_non_af_seconds": _remaining_non_af_time(monitor_runs, analyzable_only=True),
        "prehistory_input_states": dict(Counter(map(str, states[~evaluation]))),
        "last_prehistory_input_states_by_block": dict(preceding_final),
        "evaluation_input_states": dict(Counter(map(str, states[evaluation]))),
        "monitoring_blocks": len(monitor_runs),
        "evaluation_blocks": len(runs),
    }


def _sum_support(reports, stride):
    empty = {
        "decision_end_s": np.empty(0),
        "evaluation": np.empty(0, dtype=bool),
        "reference_valid": np.empty(0, dtype=bool),
        "run_id": np.empty(0, dtype=str),
        "run_kind": np.empty(0, dtype=str),
        "input_status": np.empty(0, dtype=str),
        "af_intervals_s": np.empty((0, 2)),
    }
    combined = _subject_support("", empty, stride)
    for report in reports:
        for key, value in report.items():
            if isinstance(value, dict):
                combined[key] = dict(Counter(combined[key]) + Counter(value))
            else:
                combined[key] += value
    return {"decision_stride_seconds": stride, **combined}


def coverage_report(independent_raw, protocol_lock, *, preroll=None):
    """Count reference and native input support before opening independent predictions."""
    roster = _locked_roster(protocol_lock)
    if not independent_raw.is_dir() or (preroll is not None and not preroll.is_dir()):
        raise ValueError("Supply existing directories for independent raw recordings and history")
    sources = sorted(path for path in independent_raw.glob("*.npz") if not path.name.endswith(".partial.npz"))
    found = {path.stem for path in sources}
    if not found <= roster:
        raise ValueError("Use only archives from the locked independent roster")
    baseline = _sum_support((_subject_support(source.stem, _baseline(source), WINDOW_SECONDS)
                             for source in sources), WINDOW_SECONDS)
    report = {
        "prespecified_roster_size": len(roster),
        "decoded_roster_members": len(found),
        "roster_members_without_decoded_archives": len(roster - found),
        "feature_window_seconds": WINDOW_SECONDS,
        "original_30_second_grid": baseline,
    }
    if preroll is not None:
        finer = _sum_support((_subject_support(source.stem, stage_subject(source, preroll / source.name), STRIDE)
                              for source in sources), STRIDE)
        invariant_counts = ("complete_reference_episodes", "partial_reference_episodes",
                            "complete_reference_episodes_in_wrist_monitoring", "reference_observed_seconds",
                            "monitoring_observed_seconds", "evaluation_blocks", "monitoring_blocks")
        if any(finer[key] != baseline[key] for key in invariant_counts) or not np.isclose(
            finer["monitoring_observed_non_af_seconds"], baseline["monitoring_observed_non_af_seconds"],
            rtol=0, atol=1e-8,
        ):
            raise ValueError("Ten-second staging changed the locked ECG reference denominator")
        report["ten_second_grid"] = finer
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("independent-raw", "protocol-lock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preroll", type=Path, help="Native history archives for the 10-second raw input audit")
    args = parser.parse_args()
    try:
        report = coverage_report(args.independent_raw, args.protocol_lock, preroll=args.preroll)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (AssertionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Independent coverage audit failed: {error}\n")
    print(f"Audited reference coverage for {report['decoded_roster_members']} independent participants")


if __name__ == "__main__":
    main()
