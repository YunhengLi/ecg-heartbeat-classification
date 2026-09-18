import numpy as np
from dataloader import load_ecg


X_100, y_100 = load_ecg("100")

X_101, y_101 = load_ecg("101")

print("Record 100:", X_100.shape)

print("Record 101:", X_101.shape)


X = np.concatenate(
    [X_100, X_101],
    axis=0
)

y = np.concatenate(
    [y_100, y_101],
    axis=0
)

print("Combined X:", X.shape)

print("Combined y:", y.shape)


groups_100 = np.full(
    len(y_100),
    "100"
)

groups_101 = np.full(
    len(y_101),
    "101"
)

groups = np.concatenate(
    [groups_100, groups_101],
    axis=0
)

print("Groups shape:", groups.shape)

print("First 5 groups:", groups[:5])
print("Last 5 groups:", groups[-5:])


assert len(X) == len(y) == len(groups)

print("Dataset alignment: OK")

for record_id in ["100", "101"]:

    mask = groups == record_id

    record_labels = y[mask]

    classes, counts = np.unique(
        record_labels,
        return_counts=True
    )

    print(f"\nRecord {record_id}")

    for cls, count in zip(classes, counts):
        print(f"{cls}: {count}")