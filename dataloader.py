import wfdb
import numpy as np


def load_ecg(record_id):

    # 1. Load raw ECG data
    record = wfdb.rdrecord(
        record_id,
        pn_dir="mitdb"
    )

    annotation = wfdb.rdann(
        record_id,
        "atr",
        pn_dir="mitdb"
    )

    # 2. Select MLII channel
    if "MLII" not in record.sig_name:
        raise ValueError(
            f"Record {record_id} has no MLII channel"
        )

    channel_index = record.sig_name.index("MLII")

    signal = record.p_signal[:, channel_index]

    # 3. Prepare containers
    X = []
    y = []

    before = 100
    after = 150

    # 4. Extract heartbeats
    for position, symbol in zip(
        annotation.sample,
        annotation.symbol,
        strict=True
    ):

        if symbol not in ["N", "A"]:
            continue

        start = position - before
        end = position + after

        if start < 0 or end > len(signal):
            continue

        heartbeat = signal[start:end]

        X.append(heartbeat)
        y.append(symbol)

    # 5. Convert to NumPy arrays
    X = np.asarray(X).reshape(-1, before + after)
    y = np.asarray(y)

    return X, y


# Test the function
if __name__ == "__main__":

    X, y = load_ecg("101")

    print("X shape:", X.shape)
    print("y shape:", y.shape)