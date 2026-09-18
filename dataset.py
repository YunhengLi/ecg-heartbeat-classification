import wfdb
import numpy as np

# Load the full ECG record
record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb"
)

# Load all annotations
annotation = wfdb.rdann(
    "100",
    "atr",
    pn_dir="mitdb"
)

print("Signal shape:", record.p_signal.shape)
print("Sampling frequency:", record.fs)
print("Number of annotations:", len(annotation.sample))


# Select MLII channel
signal = record.p_signal[:, 0]

X = []
y = []

before = 100
after = 150

for position, symbol in zip(
    annotation.sample,
    annotation.symbol,
    strict=True
):

    # Keep only N and A
    if symbol not in ["N", "A"]:
        continue

    start = position - before
    end = position + after

    # Skip incomplete windows
    if start < 0 or end > len(signal):
        continue

    heartbeat = signal[start:end]

    X.append(heartbeat)
    y.append(symbol)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


classes, counts = np.unique(
    y,
    return_counts=True
)

total = len(y)

for cls, count in zip(classes, counts):

    percentage = count / total * 100

    print(
        f"{cls}: {count} samples "
        f"({percentage:.2f}%)"
    )

    
#Majority-class baseline

# Predict N for every heartbeat
y_pred = np.full(
    len(y),
    "N"
)

print("Predictions:", y_pred[:10])
print("Actual labels:", y[:10])


correct = np.sum(y_pred == y)

accuracy = correct / len(y)

print("Correct predictions:", correct)
print("Accuracy:", accuracy)


# Find all actual A heartbeats
actual_A = (y == "A")

# Find correctly predicted A heartbeats
correct_A = (y == "A") & (y_pred == "A")

total_A = np.sum(actual_A)
true_positive_A = np.sum(correct_A)

recall_A = true_positive_A / total_A

print("Total A:", total_A)
print("Correctly predicted A:", true_positive_A)
print("Recall A:", recall_A)