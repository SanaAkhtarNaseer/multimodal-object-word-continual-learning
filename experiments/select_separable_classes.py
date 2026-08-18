import itertools
import pandas as pd
import numpy as np

from src.cim import cim


TRAIN_CSV = "data/features/train_features.csv"

df = pd.read_csv(TRAIN_CSV)

shape_columns = [f"shape_{i}" for i in range(23)]
color_columns = [f"color_{i}" for i in range(63)]

classes = sorted(df["name"].unique())

class_centers = {}

for class_name in classes:
    class_df = df[df["name"] == class_name]

    shape_center = class_df[shape_columns].to_numpy(
        dtype=float
    ).mean(axis=0)

    color_center = class_df[color_columns].to_numpy(
        dtype=float
    ).mean(axis=0)

    class_centers[class_name] = {
        "shape": shape_center,
        "color": color_center,
    }


pair_results = []

for class_a, class_b in itertools.combinations(classes, 2):

    shape_distance = cim(
        class_centers[class_a]["shape"],
        class_centers[class_b]["shape"],
        1.0,
    )

    color_distance = cim(
        class_centers[class_a]["color"],
        class_centers[class_b]["color"],
        1.0,
    )

    combined_distance = (
        float(shape_distance)
        + float(color_distance)
    ) / 2.0

    pair_results.append(
        {
            "class_a": class_a,
            "class_b": class_b,
            "shape_distance": float(shape_distance),
            "color_distance": float(color_distance),
            "combined_distance": combined_distance,
        }
    )


pair_df = pd.DataFrame(pair_results)

scores = {}

for class_name in classes:

    relevant = pair_df[
        (pair_df["class_a"] == class_name)
        | (pair_df["class_b"] == class_name)
    ]

    scores[class_name] = relevant[
        "combined_distance"
    ].mean()


score_df = pd.DataFrame(
    {
        "class": list(scores.keys()),
        "separation_score": list(scores.values()),
    }
)

score_df = score_df.sort_values(
    by="separation_score",
    ascending=False,
)

print()
print("================================")
print("CLASS SEPARATION RANKING")
print("================================")

print(score_df.to_string(index=False))

# --------------------------------------------------
# Find best GROUP of 6 classes efficiently
# --------------------------------------------------

distance_lookup = {}

for _, row in pair_df.iterrows():
    key = tuple(sorted((row["class_a"], row["class_b"])))
    distance_lookup[key] = row["combined_distance"]


best_group = None
best_score = -1

for group in itertools.combinations(classes, 6):

    distances = [
        distance_lookup[tuple(sorted((a, b)))]
        for a, b in itertools.combinations(group, 2)
    ]

    group_score = np.mean(distances)

    if group_score > best_score:
        best_score = group_score
        best_group = group


print()
print("================================")
print("BEST 6-CLASS GROUP")
print("================================")
print("Classes:", list(best_group))
print("Mean pairwise separation:", round(best_score, 6))

score_df.to_csv(
    "data/features/class_separation_scores.csv",
    index=False,
)
print(
    score_df.head(6)["class"].tolist()
)

score_df.to_csv(
    "data/features/class_separation_scores.csv",
    index=False,
)
