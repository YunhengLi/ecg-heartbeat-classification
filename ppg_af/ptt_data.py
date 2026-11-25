"""Read paired waveforms and verified R peaks from PhysioNet PTT PPG v1.1.0."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import wfdb


ACTIVITIES = ("sit", "walk", "run")
RECORD_NAME = re.compile(r"s([1-9][0-9]*)_(sit|walk|run)")
PPG_CHANNEL = "pleth_3"
ECG_CHANNEL = "ecg"


@dataclass(frozen=True)
class PTTRecord:
    """A paired ECG and green fingertip PPG record on its original time grid."""

    subject_id: str
    activity: str
    fs: float
    ppg: np.ndarray
    ecg: np.ndarray
    r_peaks: np.ndarray


def iter_ptt_records(
    data_dir: str | Path, activities: Iterable[str] | None = None
) -> Iterator[PTTRecord]:
    """Read local WFDB records, optionally restricted to specific activities.

    R peaks are native sample indices from the manually verified annotations.
    The dataset uses WFDB's N marker for these peaks without giving a separate
    rhythm diagnosis. ECG is read in mV and PPG in the provided nominal units.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"PTT PPG data directory not found: {data_dir}")

    selected = set(ACTIVITIES if activities is None else activities)
    if not selected or not selected.issubset(ACTIVITIES):
        raise ValueError(f"Expected one or more PTT activities from {ACTIVITIES}")

    records = []
    for header in data_dir.glob("*.hea"):
        match = RECORD_NAME.fullmatch(header.stem)
        if match and match[2] in selected:
            records.append((int(match[1]), match[2], header.with_suffix("")))

    if not records:
        raise FileNotFoundError(f"No requested PTT PPG WFDB records found in {data_dir}")

    for subject, activity, path in sorted(records):
        for suffix in (".dat", ".atr"):
            file_path = path.with_suffix(suffix)
            if not file_path.is_file():
                raise FileNotFoundError(f"PTT PPG record {path.name} is missing {file_path.name}")

        record = wfdb.rdrecord(str(path), channel_names=[PPG_CHANNEL, ECG_CHANNEL], return_res=32)
        if record.fs != 500 or record.sig_name != [PPG_CHANNEL, ECG_CHANNEL]:
            raise ValueError(f"Unexpected waveform sampling rate or channel layout: {path.name}")

        annotation = wfdb.rdann(str(path), "atr")
        if any(symbol != "N" for symbol in annotation.symbol):
            raise ValueError(f"Unexpected WFDB annotation marker: {path.name}")
        r_peaks = np.asarray(annotation.sample, dtype=np.int64)
        if not len(r_peaks) or r_peaks[0] < 0 or r_peaks[-1] >= record.sig_len:
            raise ValueError(f"Missing or out-of-range R-peak annotations: {path.name}")
        if np.any(np.diff(r_peaks) <= 0):
            raise ValueError(f"R-peak annotations are not strictly increasing: {path.name}")

        ppg, ecg = record.p_signal.T
        yield PTTRecord(f"s{subject}", activity, float(record.fs), ppg, ecg, r_peaks)
