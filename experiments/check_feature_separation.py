import random
import pandas as pd
import numpy as np

from src.cim import cim


df = pd.read_csv("data/features/train_features.csv")

shape_cols = [f"shape_{i}" for i in range(23)]
color_cols = [f"color_{i}" for i in range(63)]

S = df[shape_cols].values
C = df[color_cols].values
labels = df["name"].values

random.seed(42)

same = []
different = []

n = len(df)

while len(same) < 200 or len(different) < 200:
    i, j = random.sample(range(n), 2)

    pair = (
        cim(S[i], S[j], 1.0),
        cim(C[i], C[j], 1.0),
    )

    if labels[i] == labels[j] and len(same) < 200:
        same.append(pair)

    elif labels[i] != labels[j] and len(different) < 200:
        different.append(pair)


same_shape = np.mean([x[0] for x in same])
same_color = np.mean([x[1] for x in same])

diff_shape = np.mean([x[0] for x in different])
diff_color = np.mean([x[1] for x in different])

print("SAME class")
print("Shape:", round(same_shape, 4))
print("Color:", round(same_color, 4))

print()

print("DIFFERENT class")
print("Shape:", round(diff_shape, 4))
print("Color:", round(diff_color, 4))



print()
print("DISTANCE PERCENTILES")

same_shape_values = [x[0] for x in same]
diff_shape_values = [x[0] for x in different]

same_color_values = [x[1] for x in same]
diff_color_values = [x[1] for x in different]

for p in [10, 25, 50, 75, 90, 95]:
    print(
        f"{p}% | "
        f"same shape={np.percentile(same_shape_values, p):.4f} | "
        f"diff shape={np.percentile(diff_shape_values, p):.4f} | "
        f"same color={np.percentile(same_color_values, p):.4f} | "
        f"diff color={np.percentile(diff_color_values, p):.4f}"
    )