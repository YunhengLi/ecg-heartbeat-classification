"""Stage deterministic wrist AF training and monitoring windows from local MAT files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .afppgecg import _read_vector, load_subject


WINDOW = 30
CONTEXT = 1

def rng_for(subject_id):
    digest = sha256(f"AFPPGECG-v6:{subject_id}:schedule-20261124".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))

def in_wrist_segment(subject, start, stop):
    # The supplied segments are nonoverlapping in observed recordings. Pick the
    # first complete overlap if a release timestamp rounded a common endpoint.
    return next((s for s in subject.segments if s.start_time <= start and s.stop_time >= stop), None)

def monitor_runs(subject, random, count, width):
    # Draw one uniformly positioned run from each section of the ECG timeline.
    # Neither annotation contents nor the PPG sample values enter selection.
    length = int(subject.ecg_duration_seconds) // WINDOW
    edges = np.linspace(0, length, count + 1, dtype=int)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if hi - lo >= width // WINDOW:
            start = WINDOW * int(random.integers(lo, hi - width // WINDOW + 1))
            yield (f"monitor-{index}", "monitor", start, start + width)

def overlaps(start, stop, runs):
    return any(start - CONTEXT < existing_stop + CONTEXT and stop + CONTEXT > existing_start - CONTEXT
               for _, _, existing_start, existing_stop in runs)


def wear_monitor_runs(subject, random, existing, width):
    """Draw one uniformly placed run within wrist recording timestamps."""
    if width == 0:
        return []
    choices = set()
    for segment in subject.segments:
        start = max(0.0, segment.start_time) + CONTEXT
        stop = min(subject.ecg_duration_seconds, segment.stop_time) - width - CONTEXT
        first, last = int(np.ceil(start / WINDOW)), int(np.floor(stop / WINDOW))
        choices.update(elapsed for elapsed in range(first * WINDOW, (last + 1) * WINDOW, WINDOW)
                       if not overlaps(elapsed, elapsed + width, existing))
    if not choices:
        return []
    start = int(random.choice(sorted(choices)))
    return [("monitor-wear-0", "monitor_wear", start, start + width)]

def _monitoring_schedule(subject, random, monitor_count, monitor_seconds, wear_seconds):
    runs = list(monitor_runs(subject, random, monitor_count, monitor_seconds))
    return runs + wear_monitor_runs(subject, random, runs, wear_seconds)


def episode_runs(subject, random, episodes, reference, existing, count, max_seconds):
    choices = []
    for onset, offset in episodes:
        start = WINDOW * int(np.floor((onset - WINDOW) / WINDOW))
        stop = WINDOW * int(np.ceil((offset + WINDOW) / WINDOW))
        if offset - onset < 30 or stop - start > max_seconds or start < CONTEXT or stop + CONTEXT > subject.ecg_duration_seconds:
            continue
        if not in_wrist_segment(subject, start - CONTEXT, stop + CONTEXT):
            continue
        inside_episode = (reference[:, 2] == 1) & (reference[:, 0] <= onset) & (reference[:, 1] >= offset)
        if inside_episode.any() and not overlaps(start, stop, existing):
            choices.append((start, stop))
    chosen = []
    for index in random.permutation(len(choices)):
        if len(chosen) >= count:
            break
        start, stop = choices[index]
        if not overlaps(start, stop, chosen):
            chosen.append((f"episode-{len(chosen)}", "enriched_episode", start, stop))
    return chosen

def survey_subject(subject, *, monitor_count=1, monitor_seconds=150, wear_seconds=300, max_episode_run_seconds=480):
    """Check annotation and timestamp support without reading waveform samples."""
    if (monitor_count < 1 or monitor_seconds < WINDOW or monitor_seconds % WINDOW
            or wear_seconds < 0 or wear_seconds % WINDOW or max_episode_run_seconds < WINDOW):
        raise ValueError("Invalid monitoring counts or run durations")
    reference = subject.reference_rhythm_intervals()
    episodes = subject.annotated_af_episodes()
    random = rng_for(subject.subject_id)
    runs = _monitoring_schedule(subject, random, monitor_count, monitor_seconds, wear_seconds)
    candidates = episode_runs(subject, random, episodes, reference, runs, len(episodes), max_episode_run_seconds)
    return {
        "annotated_af_episodes": len(episodes),
        "wrist_monitoring_block_possible": any(kind == "monitor_wear" for _, kind, _, _ in runs),
        "timestamp_episode_candidates": len(candidates),
    }


def fit_runs(subject, random, reference, existing, count):
    choices = {0: [], 1: []}
    for start, stop, label in reference:
        for segment in subject.segments:
            lo = max(start, segment.start_time) + CONTEXT
            hi = min(stop, segment.stop_time) - WINDOW - CONTEXT
            first, last = int(np.ceil(lo / WINDOW)), int(np.floor(hi / WINDOW))
            if first <= last:
                choices[int(label)].extend(range(WINDOW * first, WINDOW * last + 1, WINDOW))
    chosen = []
    for label, candidates in choices.items():
        selected = 0
        for start in random.permutation(candidates):
            if selected >= count:
                break
            stop = int(start) + WINDOW
            if not overlaps(start, stop, existing + chosen):
                chosen.append((f"fit-{label}-{selected}", "enriched_fit", int(start), stop))
                selected += 1
    return chosen

def core_label(reference, start, stop, ecg):
    first = np.searchsorted(reference[:, 1], start, side="right")
    last = np.searchsorted(reference[:, 0], stop)
    inside = reference[first:last]
    coverage = np.maximum(0, np.minimum(inside[:, 1], stop) - np.maximum(inside[:, 0], start)).sum()
    valid = np.isfinite(ecg).all() and abs(coverage - (stop - start)) < 1e-7
    before = np.searchsorted(reference[:, 1], start - CONTEXT, side="right")
    rhythm = reference[before:before + 1]
    stable = (valid and len(rhythm) and rhythm[0, 0] <= start - CONTEXT and rhythm[0, 1] >= stop + CONTEXT)
    return int(rhythm[0, 2]) if stable else -1, bool(valid)

def build_rows(subject, runs, reference):
    rows = []
    for run_id, run_kind, start, stop in runs:
        # ECG-only windows stay on the prospective time grid, including periods
        # outside every wrist segment. Read the target waveform only once per run.
        voltage = _read_vector(subject._voltage, slice(500 * start, 500 * stop))
        for elapsed in range(start, stop, WINDOW):
            end = elapsed + WINDOW
            ecg = voltage[500 * (elapsed - start):500 * (end - start)]
            stable_label, valid = core_label(reference, elapsed, end, ecg)
            segment = in_wrist_segment(subject, elapsed - CONTEXT, end + CONTEXT)
            if segment is None:
                ppg, accel, method = np.full(3200, np.nan), np.full((1500, 3), np.nan), "absent"
            else:
                pulse = subject.read_segment(segment.index, elapsed - CONTEXT - segment.start_time,
                                             end + CONTEXT - segment.start_time)
                ppg, accel, method = pulse.ppg, pulse.accel[50:-50], pulse.sync_method
            if ppg.shape != (3200,) or ecg.shape != (15000,) or accel.shape != (1500, 3):
                raise ValueError(f"Unexpected window shapes for {subject.subject_id} at {elapsed}")
            rows.append((ppg, ecg, accel, elapsed, run_id, run_kind, stable_label, valid, method))
    return rows

def stage_subject(
    subject, output, *, monitor_count=1, monitor_seconds=150, fit_per_class=2,
    episode_count=1, max_episode_run_seconds=270, wear_seconds=0,
):
    """Stage label-independent monitoring and distinct ECG-selected windows."""
    if (monitor_count < 1 or monitor_seconds < WINDOW or monitor_seconds % WINDOW
            or wear_seconds < 0 or wear_seconds % WINDOW
            or min(fit_per_class, episode_count) < 0 or max_episode_run_seconds < WINDOW):
        raise ValueError("Invalid window counts or run durations")

    reference = subject.reference_rhythm_intervals()
    episodes = subject.annotated_af_episodes()
    random = rng_for(subject.subject_id)
    runs = _monitoring_schedule(subject, random, monitor_count, monitor_seconds, wear_seconds)
    runs += episode_runs(subject, random, episodes, reference, runs, episode_count, max_episode_run_seconds)
    runs += fit_runs(subject, random, reference, runs, fit_per_class)
    rows = build_rows(subject, runs, reference)
    if not rows:
        raise ValueError("No windows fit this subject and schedule")
    names = ("ppg", "ecg", "accel", "start_s", "run_id", "run_kind", "stable_label", "reference_valid", "sync_method")
    fields = {name: np.asarray(column) for name, column in zip(names, zip(*rows, strict=True), strict=True)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, subject_id=subject.subject_id, af_intervals_s=episodes,
                            ppg_fs=100, ecg_fs=500, accel_fs=50, **fields)
    temporary.replace(output)
    return {
        "monitored_windows": int(np.sum(fields["run_kind"] == "monitor")),
        "wear_monitored_windows": int(np.sum(fields["run_kind"] == "monitor_wear")),
        "enriched_episode_windows": int(np.sum(fields["run_kind"] == "enriched_episode")),
        "fit_windows": int(np.sum(fields["run_kind"] == "enriched_fit")),
        "labeled_af_windows": int(np.sum(fields["stable_label"] == 1)),
        "labeled_non_af_windows": int(np.sum(fields["stable_label"] == 0)),
        "ppg_present_windows": int(np.sum(np.isfinite(fields["ppg"]).any(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg", type=Path, required=True)
    parser.add_argument("--ppg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor-count", type=int, default=1)
    parser.add_argument("--monitor-seconds", type=int, default=150)
    parser.add_argument("--wear-seconds", type=int, default=0)
    parser.add_argument("--fit-per-class", type=int, default=2)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--max-episode-run-seconds", type=int, default=270)
    parser.add_argument("--survey-only", action="store_true", help="Write candidate counts as JSON using annotations and timestamps")
    args = parser.parse_args()
    try:
        with load_subject(args.ecg, args.ppg, include_acceleration=False) as subject:
            if args.survey_only:
                support = survey_subject(
                    subject, monitor_count=args.monitor_count, monitor_seconds=args.monitor_seconds,
                    wear_seconds=args.wear_seconds, max_episode_run_seconds=args.max_episode_run_seconds,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(support, indent=2) + "\n", encoding="utf-8")
                return
            summary = stage_subject(
                subject, args.output, monitor_count=args.monitor_count, monitor_seconds=args.monitor_seconds,
                fit_per_class=args.fit_per_class, episode_count=args.episode_count,
                max_episode_run_seconds=args.max_episode_run_seconds, wear_seconds=args.wear_seconds,
            )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Unable to stage wrist AF windows: {exc}\n")
    print(f"Staged {summary['monitored_windows']} calendar, {summary['wear_monitored_windows']} wrist, "
          f"{summary['fit_windows']} fit, "
          f"and {summary['enriched_episode_windows']} episode windows")


if __name__ == "__main__":
    main()
