import wfdb
import numpy as np
import matplotlib.pyplot as plt

record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb",
    sampto=3600
)

print(record.fs)
print(record.p_signal.shape)
print(record.sig_name)

signal = record.p_signal[:, 0]

print(signal.shape)
print(signal[:10])


time = np.arange(len(signal)) / record.fs


plt.figure(figsize=(12, 4))

plt.plot(time, signal)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal - Record 100")

plt.grid(True)

plt.show()


annotation = wfdb.rdann(
    "100",
    "atr",
    pn_dir="mitdb",
    sampto=3600
)

print("Annotation positions:")
print(annotation.sample[:10])

print("Annotation symbols:")
print(annotation.symbol[:10])


# Convert annotation positions to time
annotation_time = annotation.sample / record.fs

print("Annotation times:")
print(annotation_time[:10])


plt.figure(figsize=(12, 4))

plt.plot(time, signal)

for position, symbol in zip(
    annotation.sample,
    annotation.symbol
):

    if symbol not in ["N", "A"]:
        continue

    x = position / record.fs
    y = signal[position]

    plt.scatter(x, y, color="red")

    plt.text(x, y + 0.1, symbol)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal with Heartbeat Labels")

plt.grid(True)
plt.show()


# Extract abnormal heartbeat

center = 2044

before = 100
after = 150

heartbeat = signal[
    center - before : center + after
]

print("Heartbeat shape:", heartbeat.shape)


# Create a time axis relative to the annotation

heartbeat_time = (
    np.arange(len(heartbeat)) - before
) / record.fs

plt.figure(figsize=(10, 4))

plt.plot(heartbeat_time, heartbeat)

plt.axvline(
    x=0,
    color="red",
    linestyle="--",
    label="Annotation position"
)

plt.xlabel("Time relative to annotation (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Extracted Heartbeat - Label A")

plt.legend()
plt.grid(True)
plt.show()


#Extract all valid heartbeats

X = []
y = []

before = 100
after = 150

for position, symbol in zip(
    annotation.sample,
    annotation.symbol
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

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First label:", y[0])
print("First heartbeat shape:", X[0].shape)


#Check class distribution

classes, counts = np.unique(
    y,
    return_counts=True
)

print("Classes:", classes)
print("Counts:", counts)


total = len(y)

for cls, count in zip(classes, counts):

    percentage = count / total * 100

    print(
        f"{cls}: {count} samples "
        f"({percentage:.2f}%)"
    )