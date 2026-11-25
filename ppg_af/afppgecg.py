"""Read synchronized wrist PPG and reference ECG from AFPPGECG v6."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
import math
from pathlib import Path
import re
from typing import BinaryIO

import h5py
import numpy as np


ECG_FS = 500.0
PPG_FS = 100.0
ACCEL_FS = 50.0
SECONDS_PER_DAY = 86400
DEFAULT_MAX_REFERENCE_RR_SECONDS = 3.0


@dataclass(frozen=True)
class WristSegment:
    """One PPG segment on the ECG recording clock, in seconds."""

    index: int
    start_time: float
    duration_seconds: float
    sync_method: str

    @property
    def stop_time(self) -> float:
        return self.start_time + self.duration_seconds


@dataclass(frozen=True)
class WristSlice:
    """Native-rate samples and reference annotations for a half-open window.

    All times are in seconds relative to the first ECG sample. The three start
    times identify the first returned sample on each original sample grid.
    PPG is green-channel ADC units. Acceleration is samples by XYZ axes, in
    g. The ECG uses the original unit in ``ecg_unit``. Missing samples remain NaN.
    Each R peak carries the source annotation for the following RR interval.
    ``rhythm_intervals`` holds [start, stop, AF flag] rows for usable reference.
    """

    subject_id: str
    segment_index: int
    sync_method: str
    start_time: float
    stop_time: float
    ppg_start_time: float
    ecg_start_time: float
    accel_start_time: float
    ppg: np.ndarray
    ecg: np.ndarray
    accel: np.ndarray
    r_peak_times: np.ndarray
    af_beat_labels: np.ndarray
    rhythm_intervals: np.ndarray
    ecg_unit: str
    ppg_fs: float = PPG_FS
    ecg_fs: float = ECG_FS
    accel_fs: float = ACCEL_FS

    @property
    def ecg_times(self) -> np.ndarray:
        return self.ecg_start_time + np.arange(len(self.ecg)) / self.ecg_fs

    @property
    def af_intervals(self) -> np.ndarray:
        """Reference AF intervals as [start, stop) rows on the ECG clock."""
        return self.rhythm_intervals[self.rhythm_intervals[:, 2] == 1, :2]

    @property
    def reference_intervals(self) -> np.ndarray:
        """Labeled ECG coverage as [start, stop) rows on the ECG clock."""
        return self.rhythm_intervals[:, :2]


def _sample_bounds(
    start: float, stop: float, origin: float, fs: float, count: int
) -> tuple[int, int]:
    def first_at_or_after(time: float) -> int:
        # ECG-relative timestamps can be several days from zero. Allow only
        # the floating-point roundoff expected when converting to sample units.
        tolerance = 4 * math.ulp(max(abs(time), abs(origin))) * fs
        return min(count, max(0, math.ceil((time - origin) * fs - tolerance)))

    return first_at_or_after(start), first_at_or_after(stop)


def _merge_rhythm_intervals(peaks: np.ndarray, labels: np.ndarray, max_rr: float) -> np.ndarray:
    supported = (np.diff(peaks) <= max_rr) & np.isin(labels, [0, 1])
    starts = peaks[:-1][supported]
    stops = peaks[1:][supported]
    labels = labels[supported]
    if not len(starts):
        return np.empty((0, 3))
    boundaries = np.flatnonzero((starts[1:] != stops[:-1]) | (labels[1:] != labels[:-1])) + 1
    first = np.r_[0, boundaries]
    last = np.r_[boundaries - 1, len(starts) - 1]
    return np.column_stack((starts[first], stops[last], labels[first]))


def _vector_selection(dataset: h5py.Dataset, selection: slice) -> tuple | slice:
    if dataset.ndim == 1:
        return selection
    if dataset.ndim == 2 and dataset.shape[0] == 1:
        return (0, selection)
    if dataset.ndim == 2 and dataset.shape[1] == 1:
        return (selection, 0)
    raise ValueError(f"Expected a MATLAB vector at {dataset.name}, got {dataset.shape}")


def _read_vector(dataset: h5py.Dataset, selection: slice = slice(None)) -> np.ndarray:
    return np.asarray(dataset[_vector_selection(dataset, selection)])


def _vector_length(dataset: h5py.Dataset) -> int:
    _vector_selection(dataset, slice(None))
    return dataset.size


def _matlab_text(dataset: h5py.Dataset) -> str:
    if dataset.ndim == 0:
        return bytes(dataset[()]).decode("utf-8").strip()
    values = _read_vector(dataset)
    if values.dtype.kind in "iu":
        return "".join(chr(value) for value in values if value).strip()
    if values.dtype.kind == "S":
        return b"".join(values).decode("utf-8").strip()
    raise ValueError(f"Expected MATLAB text at {dataset.name}, got {dataset.dtype}")


def _scalar(dataset: h5py.Dataset) -> float:
    if dataset.size != 1:
        raise ValueError(f"Expected a MATLAB scalar at {dataset.name}, got {dataset.shape}")
    return float(np.asarray(dataset[()]).item())


def _clock_seconds(day: h5py.Dataset, time: h5py.Dataset) -> Decimal:
    day_number = (
        float(_matlab_text(day))
        if day.attrs.get("MATLAB_class") in (b"char", "char")
        else _scalar(day)
    )
    hour_text, minute_text, second_text = _matlab_text(time).split(":")
    hours, minutes, seconds = int(hour_text), int(minute_text), Decimal(second_text)
    if not day_number.is_integer() or not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"Invalid recording day or time in {day.name} and {time.name}")
    return int(day_number) * SECONDS_PER_DAY + hours * 3600 + minutes * 60 + seconds


def _cell(record: h5py.File, name: str) -> tuple[h5py.Dataset, ...]:
    entries = _read_vector(record[name])
    return tuple(record[entry] for entry in entries)


def _subject_id(ecg_path: str | Path | BinaryIO, ppg_path: str | Path | BinaryIO, explicit: str | None) -> str:
    ids = [] if explicit is None else [explicit]
    for source, signal in ((ecg_path, "ECG"), (ppg_path, "PPG")):
        name = Path(source).name if isinstance(source, (str, Path)) else ""
        match = re.fullmatch(fr"([0-9]{{3}})_{signal}\.mat", name)
        if match:
            ids.append(match[1])
    if len(set(ids)) > 1:
        raise ValueError(f"ECG and PPG paths refer to different subjects: {ids}")
    if not ids:
        raise ValueError("Pass subject_id when inputs do not have the AFPPGECG filenames")
    return ids[0]


def _check_nominal_rates(record: h5py.File, expected: tuple[float, ...]) -> None:
    cells = _cell(record, "signalHeader/samples_in_record")
    rates = tuple(_scalar(cell) for cell in cells[: len(expected)])
    if rates != expected:
        raise ValueError(f"Expected sampling rates {expected} in {record.filename}, got {rates}")


class WristSubject:
    """One paired AFPPGECG v6 subject, backed by open HDF5 files."""

    def __init__(self, ecg: h5py.File, ppg: h5py.File, subject_id: str, max_reference_rr: float,
                 include_acceleration: bool = True):
        self.subject_id = subject_id
        self._ecg = ecg
        self._max_reference_rr = max_reference_rr
        self._voltage = ecg["ECG"]
        self.ecg_duration_seconds = _vector_length(self._voltage) / ECG_FS
        _check_nominal_rates(ecg, (ECG_FS,))
        _check_nominal_rates(ppg, (PPG_FS, PPG_FS, ACCEL_FS, ACCEL_FS, ACCEL_FS))
        self.ecg_unit = _matlab_text(_cell(ecg, "signalHeader/physical_dimension")[0])
        wrist_units = tuple(_matlab_text(cell) for cell in _cell(ppg, "signalHeader/physical_dimension"))
        if wrist_units != ("ADC unit", "ADC unit", "g", "g", "g"):
            raise ValueError(f"Unexpected wrist signal units in {subject_id}: {wrist_units}")

        self._ppg_data = _cell(ppg, "PPG_GREEN")
        if include_acceleration:
            axes = [_cell(ppg, f"Accelerometer_{axis}") for axis in "XYZ"]
            self._accel_data = tuple(zip(*axes, strict=True))
        else:
            self._accel_data = (None,) * len(self._ppg_data)
        days = _cell(ppg, "recording_startday")
        times = _cell(ppg, "recording_starttime")
        methods = _cell(ppg, "sync_method")
        ecg_clock = _clock_seconds(ecg["recording_startday"], ecg["recording_starttime"])
        segments = []
        sources = zip(self._ppg_data, self._accel_data, days, times, methods, strict=True)
        for index, (green, accel, day, time, method) in enumerate(sources):
            start = float(_clock_seconds(day, time) - ecg_clock)
            duration = _vector_length(green) / PPG_FS
            if accel is not None:
                accel_counts = {_vector_length(axis) for axis in accel}
                if len(accel_counts) != 1 or abs(accel_counts.pop() / ACCEL_FS - duration) > 1 / ACCEL_FS + 1e-9:
                    raise ValueError(f"Inconsistent sample counts in {subject_id} segment {index}")
            sync_method = _matlab_text(method).lower()
            if sync_method not in {"beat", "coarse"}:
                raise ValueError(f"Unexpected sync_method in {subject_id} segment {index}: {sync_method!r}")
            if duration <= 0 or start >= self.ecg_duration_seconds or start + duration <= 0:
                raise ValueError(f"No usable ECG overlap in {subject_id} segment {index}")
            segments.append(WristSegment(index, start, duration, sync_method))
        self.segments = tuple(segments)

    @cached_property
    def _reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # QRSindex contains 1-based MATLAB indices. In the original annotation
        # format label i belongs to the interval from peak i to peak i + 1.
        indices = _read_vector(self._ecg["QRSindex"]).astype(float)
        labels = _read_vector(self._ecg["AF_annotation"]).astype(float)
        rr = _read_vector(self._ecg["rr"]).astype(float)
        peaks = (indices - 1) / ECG_FS
        if (not np.isfinite(indices).all() or np.any(indices != np.floor(indices))
                or np.any(np.diff(peaks) <= 0) or np.any(peaks < 0)
                or np.any(peaks >= self.ecg_duration_seconds)):
            raise ValueError(f"Invalid QRSindex values in subject {self.subject_id}")
        if rr.shape != labels.shape or len(labels) != max(len(peaks) - 1, 0):
            raise ValueError(f"Expected one AF annotation per RR interval in {self.subject_id}")
        annotated = _merge_rhythm_intervals(peaks, labels, math.inf)
        # A missing RR or unsupported label contributes no reference coverage.
        usable = np.where(np.isfinite(rr) & (rr > 0), labels, np.nan)
        intervals = _merge_rhythm_intervals(peaks, usable, self._max_reference_rr)
        beat_labels = np.r_[labels, np.nan] if len(peaks) else labels
        return peaks, beat_labels, intervals, annotated[annotated[:, 2] == 1, :2]

    def reference_rhythm_intervals(
        self, start_time: float = 0.0, stop_time: float | None = None
    ) -> np.ndarray:
        """Return [start, stop, AF flag] intervals supported by ECG annotations.

        Adjacent intervals with the same flag are merged. The default spans
        the entire ECG recording, including time outside wrist PPG segments.
        Intervals longer than the selected reference RR limit count as gaps.
        """
        stop = self.ecg_duration_seconds if stop_time is None else stop_time
        if not (0 <= start_time < stop <= self.ecg_duration_seconds):
            raise ValueError("Reference time bounds must be within the ECG recording")
        _, _, intervals, _ = self._reference
        first = np.searchsorted(intervals[:, 1], start_time, side="right")
        last = np.searchsorted(intervals[:, 0], stop)
        result = intervals[first:last].copy()
        if len(result):
            result[0, 0] = max(result[0, 0], start_time)
            result[-1, 1] = min(result[-1, 1], stop)
        return result

    def annotated_af_episodes(self) -> np.ndarray:
        """Return full-record AF episode bounds from consecutive AF labels.

        The episode endpoints come from the underlying ECG labels. Use
        ``reference_rhythm_intervals`` to check whether an elapsed span is
        supported by closely spaced ECG peaks for evaluation.
        """
        return self._reference[3].copy()

    def read_segment(
        self, segment_index: int, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> WristSlice:
        """Return a lazy read of the requested PPG-segment-relative seconds."""
        if not 0 <= segment_index < len(self.segments):
            raise IndexError(f"PPG segment index out of range: {segment_index}")
        segment = self.segments[segment_index]
        stop_seconds = segment.duration_seconds if stop_seconds is None else stop_seconds
        if not (0 <= start_seconds < stop_seconds <= segment.duration_seconds):
            raise ValueError("Window bounds must be within the selected PPG segment")
        start = max(0.0, segment.start_time + start_seconds)
        stop = min(self.ecg_duration_seconds, segment.start_time + stop_seconds)
        if start >= stop:
            raise ValueError("Requested PPG window has no ECG overlap")

        green = self._ppg_data[segment_index]
        axes = self._accel_data[segment_index]
        ppg_lo, ppg_hi = _sample_bounds(start, stop, segment.start_time, PPG_FS, _vector_length(green))
        ecg_lo, ecg_hi = _sample_bounds(start, stop, 0.0, ECG_FS, _vector_length(self._voltage))
        accel_length = math.ceil(segment.duration_seconds * ACCEL_FS) if axes is None else _vector_length(axes[0])
        acc_lo, acc_hi = _sample_bounds(start, stop, segment.start_time, ACCEL_FS, accel_length)
        accel = (np.full((acc_hi - acc_lo, 3), np.nan) if axes is None
                 else np.column_stack([_read_vector(axis, slice(acc_lo, acc_hi)) for axis in axes]))
        peaks, labels, _, _ = self._reference
        first, last = np.searchsorted(peaks, [ecg_lo / ECG_FS, ecg_hi / ECG_FS])
        return WristSlice(
            subject_id=self.subject_id,
            segment_index=segment_index,
            sync_method=segment.sync_method,
            start_time=start,
            stop_time=stop,
            ppg_start_time=segment.start_time + ppg_lo / PPG_FS,
            ecg_start_time=ecg_lo / ECG_FS,
            accel_start_time=segment.start_time + acc_lo / ACCEL_FS,
            ppg=_read_vector(green, slice(ppg_lo, ppg_hi)),
            ecg=_read_vector(self._voltage, slice(ecg_lo, ecg_hi)),
            accel=accel,
            r_peak_times=peaks[first:last].copy(),
            af_beat_labels=labels[first:last].copy(),
            rhythm_intervals=self.reference_rhythm_intervals(start, stop),
            ecg_unit=self.ecg_unit,
        )


@contextmanager
def load_subject(
    ecg_path: str | Path | BinaryIO,
    ppg_path: str | Path | BinaryIO,
    *,
    subject_id: str | None = None,
    max_reference_rr: float = DEFAULT_MAX_REFERENCE_RR_SECONDS,
    include_acceleration: bool = True,
) -> Iterator[WristSubject]:
    """Open an AFPPGECG v6 pair and read signals at their original rates.

    Inputs are standard local paths or seekable binary streams. Supply the
    subject identifier when using streams. Use matching files from v6. No
    signal resampling or interpolation is performed. ECG and green wrist PPG
    use the release's native units, as do wrist acceleration X, Y, Z. Set
    ``include_acceleration=False`` for ECG/PPG-only analyses. Acceleration
    arrays then contain NaNs on the nominal wrist time grid.
    """
    if not math.isfinite(max_reference_rr) or max_reference_rr <= 0:
        raise ValueError("The maximum annotated RR duration must be finite and positive")
    subject = _subject_id(ecg_path, ppg_path, subject_id)
    with h5py.File(ecg_path, "r") as ecg, h5py.File(ppg_path, "r") as ppg:
        yield WristSubject(ecg, ppg, subject, max_reference_rr, include_acceleration)
