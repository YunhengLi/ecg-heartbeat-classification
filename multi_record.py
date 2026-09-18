import wfdb
import numpy as np

record_id = "101"

record = wfdb.rdrecord(
    record_id,
    pn_dir="mitdb"
)

annotation = wfdb.rdann(
    record_id,
    "atr",
    pn_dir="mitdb"
)

print("Record:", record_id)

print("Signal shape:", record.p_signal.shape)

print("Channels:", record.sig_name)

print("Sampling frequency:", record.fs)

print("Number of annotations:", len(annotation.sample))


# Count heartbeat labels in Record 101

classes, counts = np.unique(
    annotation.symbol,
    return_counts=True
)

for cls, count in zip(classes, counts):
    if cls in ["N", "A"]:
        print(f"{cls}: {count} annotations")
