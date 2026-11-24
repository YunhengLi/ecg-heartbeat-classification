"""Stage accelerometry on the already sealed half-hour wrist clock."""

import argparse
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np

from .af_wrist_acceleration_stage import NativeAcceleration, read_acceleration
from .af_wrist_features import PPG_RATE, WINDOW_SECONDS
from .af_wrist_long_monitor import BLOCK_SECONDS, verify_long_schedule
from .af_wrist_monitor_extension import EARLIER_S
from .af_wrist_retrospective_monitor import verify_schedule
from .af_wrist_vote_study import STRIDE_SECONDS
from .afppgecg import ACCEL_FS


def stage_block(ecg, wrist, person, block, digest, frozen_raw, output):
    start, stop = block["evaluation_s"]
    if stop - start != BLOCK_SECONDS:
        raise ValueError("Use one sealed half-hour block per participant")
    with np.load(frozen_raw, allow_pickle=False) as archived:
        expected = archived["raw_ppg"]
        if (archived["subject_id"].item() != person or archived["plan_sha256"].item() != digest
                or archived["start_s"].tolist() != [start] or expected.shape !=
                (1, (BLOCK_SECONDS + EARLIER_S - WINDOW_SECONDS) // STRIDE_SECONDS + 1, WINDOW_SECONDS * PPG_RATE)):
            raise ValueError("The acceleration staging clock must match the frozen optical archive")
    block_motion = np.empty((0, 3))
    phase = np.nan
    anchored = np.flatnonzero(np.isfinite(expected[0]).all(axis=1))
    if len(anchored) >= 2:
        with h5py.File(ecg, "r") as reference_file, h5py.File(wrist, "r") as wrist_file:
            native = NativeAcceleration(reference_file, wrist_file, person)
            sync = native.subject.segments[block["native_wrist_segment"]].sync_method
            first_row, last_row = anchored[0], anchored[-1]
            starts = start - EARLIER_S + STRIDE_SECONDS * anchored[[0, -1]]
            first = native.sources_for(float(starts[0]), sync, expected[0, first_row])
            last = native.sources_for(float(starts[-1]), sync, expected[0, last_row])
            if first is not None and last is not None:
                axes, first_sample, _, phase = first
                other_axes, _, last_sample, other_phase = last
                first_sample -= int((starts[0] - (start - EARLIER_S)) * ACCEL_FS)
                last_sample += int((stop - (starts[-1] + WINDOW_SECONDS)) * ACCEL_FS)
                if (any(a.name != b.name for a, b in zip(axes, other_axes, strict=True))
                        or last_sample - first_sample != int((BLOCK_SECONDS + EARLIER_S) * ACCEL_FS)
                        or not np.isclose(phase, other_phase, atol=1e-8, rtol=0)):
                    raise ValueError("The two optical anchors must bound one consistent native acceleration clock")
                block_motion = read_acceleration(axes, first_sample, last_sample)
                if len(block_motion) != last_sample - first_sample:
                    raise ValueError("The native accelerometer must cover every staged half-hour sample")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{person}.npz"
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, accel=block_motion, accel_minus_ppg_s=phase, start_s=start, subject_id=person,
                        plan_sha256=digest, frozen_optical_sha256=sha256(expected.tobytes()).hexdigest())
    temporary.replace(path)
    return len(block_motion) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "lock", "prior-inputs", "prior-monitor-plan", "source-dir", "monitoring-cache", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    previous, _, previous_digest = verify_schedule(args.prior_monitor_plan, args.lock, args.prior_inputs)
    schedule, _, digest = verify_long_schedule(args.plan, args.lock, args.prior_inputs, previous, previous_digest)
    for person, blocks in schedule.items():
        if blocks:
            stage_block(args.source_dir / f"{person}_ECG.mat", args.source_dir / f"{person}_PPG.mat", person, blocks[0],
                        digest, args.monitoring_cache / f"{person}.npz", args.output)


if __name__ == "__main__":
    main()
