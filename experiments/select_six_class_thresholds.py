import itertools
import pandas as pd

from src.cdsn import CDSN


TRAIN_CSV = "data/features/train_features.csv"
VALIDATION_CSV = "data/features/validation_features.csv"

SELECTED_CLASSES = [
    "apple",
    "banana",
    "beetroot",
    "cucumber",
    "garlic",
    "orange",
]

SHAPE_THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06]
COLOR_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]


def evaluate(model, df, shape_columns, color_columns):

    exact_correct = 0
    association_correct = 0
    ambiguous = 0
    empty = 0
    total_names = 0

    for _, row in df.iterrows():

        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)

        true_name = row["name"]

        result = model.recall_name_from_visual(
            shape_vector,
            color_vector
        )

        names = result["names"]

        if true_name in names:
            association_correct += 1

        if len(names) == 1 and names[0] == true_name:
            exact_correct += 1

        if len(names) > 1:
            ambiguous += 1

        if len(names) == 0:
            empty += 1

        total_names += len(names)

    total = len(df)

    return {
        "exact_accuracy": exact_correct / total if total else 0,
        "association_recall": association_correct / total if total else 0,
        "ambiguity_rate": ambiguous / total if total else 0,
        "empty_rate": empty / total if total else 0,
        "avg_names_returned": total_names / total if total else 0,
    }


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
validation_df = pd.read_csv(VALIDATION_CSV)

shape_columns = [f"shape_{i}" for i in range(23)]
color_columns = [f"color_{i}" for i in range(63)]

train_df = train_df[
    train_df["name"].isin(SELECTED_CLASSES)
].copy()

validation_df = validation_df[
    validation_df["name"].isin(SELECTED_CLASSES)
].copy()

print("Training samples:", len(train_df))
print("Validation samples:", len(validation_df))


# --------------------------------------------------
# Threshold search
# --------------------------------------------------

results = []

for shape_threshold, color_threshold in itertools.product(
    SHAPE_THRESHOLDS,
    COLOR_THRESHOLDS
):

    print()
    print(
        "Testing:",
        "shape =", shape_threshold,
        "color =", color_threshold
    )

    model = CDSN(
        sigma=1.0,
        shape_threshold=shape_threshold,
        color_threshold=color_threshold,
        maintenance_interval=10
    )

    for _, row in train_df.iterrows():

        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)

        model.learn(
            shape_vector=shape_vector,
            color_vector=color_vector,
            name=row["name"]
        )

    metrics = evaluate(
        model,
        validation_df,
        shape_columns,
        color_columns
    )

    result = {
        "shape_threshold": shape_threshold,
        "color_threshold": color_threshold,
        **metrics,
        "shape_nodes": len(model.shape_memory.nodes),
        "color_nodes": len(model.color_memory.nodes),
        "associative_nodes": len(model.associative_memory.nodes),
    }

    results.append(result)

    print(
        "Exact accuracy:",
        round(metrics["exact_accuracy"], 4)
    )

    print(
        "Association recall:",
        round(metrics["association_recall"], 4)
    )

    print(
        "Ambiguity:",
        round(metrics["ambiguity_rate"], 4)
    )

    print(
        "Empty:",
        round(metrics["empty_rate"], 4)
    )


# --------------------------------------------------
# Rank configurations
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "exact_accuracy",
        "association_recall",
        "ambiguity_rate",
        "empty_rate",
    ],
    ascending=[
        False,
        False,
        True,
        True,
    ]
)


print()
print("========================================")
print("BEST SIX-CLASS THRESHOLDS")
print("========================================")

print(
    results_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


OUTPUT_CSV = (
    "data/features/"
    "six_class_threshold_results.csv"
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:", OUTPUT_CSV)