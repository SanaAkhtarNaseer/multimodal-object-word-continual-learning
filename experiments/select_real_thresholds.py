import itertools
import pandas as pd

from src.cdsn import CDSN


TRAIN_CSV = "data/features/train_features.csv"
VALIDATION_CSV = "data/features/validation_features.csv"

PHASE_1_CLASSES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "carrot",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "lemon",
]

SHAPE_THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07]
COLOR_THRESHOLDS = [0.02, 0.03, 0.04, 0.05]


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    shape_columns = [f"shape_{i}" for i in range(23)]
    color_columns = [f"color_{i}" for i in range(63)]

    return df, shape_columns, color_columns

def evaluate(model, df, shape_columns, color_columns):
    association_correct = 0
    exact_correct = 0
    total = 0

    total_names_returned = 0
    ambiguous_predictions = 0
    empty_predictions = 0

    for _, row in df.iterrows():
        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)
        true_name = row["name"]

        result = model.recall_name_from_visual(
            shape_vector,
            color_vector
        )

        predicted_names = result["names"]

        # Correct name appears somewhere in retrieved associations
        if true_name in predicted_names:
            association_correct += 1

        # Stronger criterion:
        # exactly one name returned and it is correct
        if predicted_names == [true_name]:
            exact_correct += 1

        if len(predicted_names) > 1:
            ambiguous_predictions += 1

        if len(predicted_names) == 0:
            empty_predictions += 1

        total_names_returned += len(predicted_names)
        total += 1

    return {
        "association_recall": (
            association_correct / total if total else 0.0
        ),
        "exact_accuracy": (
            exact_correct / total if total else 0.0
        ),
        "ambiguity_rate": (
            ambiguous_predictions / total if total else 0.0
        ),
        "empty_rate": (
            empty_predictions / total if total else 0.0
        ),
        "avg_names_returned": (
            total_names_returned / total if total else 0.0
        ),
    }


train_df, shape_columns, color_columns = load_data(TRAIN_CSV)
validation_df, _, _ = load_data(VALIDATION_CSV)

phase_1_train = train_df[
    train_df["name"].isin(PHASE_1_CLASSES)
].copy()

phase_1_validation = validation_df[
    validation_df["name"].isin(PHASE_1_CLASSES)
].copy()


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

    for _, row in phase_1_train.iterrows():

        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)
        name = row["name"]

        model.learn(
            shape_vector=shape_vector,
            color_vector=color_vector,
            name=name
        )

    metrics = evaluate(
    model,
    phase_1_validation,
    shape_columns,
    color_columns
)

    result = {
        "shape_threshold": shape_threshold,
        "color_threshold": color_threshold,
        "association_recall": metrics["association_recall"],
        "exact_accuracy": metrics["exact_accuracy"],
        "ambiguity_rate": metrics["ambiguity_rate"],
        "empty_rate": metrics["empty_rate"],
        "avg_names_returned": metrics["avg_names_returned"],
        "shape_nodes": len(model.shape_memory.nodes),
        "color_nodes": len(model.color_memory.nodes),
        "associative_nodes": len(model.associative_memory.nodes),
    }
    results.append(result)

    print("Association recall:", round(result["association_recall"], 4))
    print("Exact accuracy:", round(result["exact_accuracy"], 4))
    print("Ambiguity rate:", round(result["ambiguity_rate"], 4))
    print("Empty rate:", round(result["empty_rate"], 4))
    print("Average names returned:", round(result["avg_names_returned"], 4))
    print("Shape nodes:", result["shape_nodes"])
    print("Color nodes:", result["color_nodes"])
    print("Associative nodes:", result["associative_nodes"])


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "exact_accuracy",
        "association_recall",
        "ambiguity_rate",
        "associative_nodes",
    ],
    ascending=[
        False,
        False,
        True,
        True,
    ]
)

print()
print("===================================")
print("BEST THRESHOLD CONFIGURATIONS")
print("===================================")

print(
    results_df.head(10).to_string(index=False)
)

results_df.to_csv(
    "data/features/threshold_search_results.csv",
    index=False
)

print()
print(
    "Saved:",
    "data/features/threshold_search_results.csv"
)