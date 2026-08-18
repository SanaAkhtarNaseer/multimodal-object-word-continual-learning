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

# Best strict configuration from our threshold experiment
SHAPE_THRESHOLD = 0.03
COLOR_THRESHOLD = 0.02


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    shape_columns = [f"shape_{i}" for i in range(23)]
    color_columns = [f"color_{i}" for i in range(63)]

    return df, shape_columns, color_columns


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df, shape_columns, color_columns = load_data(TRAIN_CSV)
validation_df, _, _ = load_data(VALIDATION_CSV)

phase_1_train = train_df[
    train_df["name"].isin(PHASE_1_CLASSES)
].copy()

phase_1_validation = validation_df[
    validation_df["name"].isin(PHASE_1_CLASSES)
].copy()


# --------------------------------------------------
# Train CDSN
# --------------------------------------------------

model = CDSN(
    sigma=1.0,
    shape_threshold=SHAPE_THRESHOLD,
    color_threshold=COLOR_THRESHOLD,
    maintenance_interval=10
)

for _, row in phase_1_train.iterrows():

    shape_vector = row[shape_columns].to_numpy(dtype=float)
    color_vector = row[color_columns].to_numpy(dtype=float)

    model.learn(
        shape_vector=shape_vector,
        color_vector=color_vector,
        name=row["name"]
    )


print()
print("========================================")
print("PHASE 1 MODEL")
print("========================================")
print("Training samples:", len(phase_1_train))
print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print("Associative nodes:", len(model.associative_memory.nodes))


# --------------------------------------------------
# Per-class evaluation
# --------------------------------------------------

results = []

for class_name in PHASE_1_CLASSES:

    class_df = phase_1_validation[
        phase_1_validation["name"] == class_name
    ]

    correct = 0
    wrong = 0
    empty = 0
    ambiguous = 0

    for _, row in class_df.iterrows():

        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)

        result = model.recall_name_from_visual(
            shape_vector,
            color_vector
        )

        predicted_names = result["names"]

        if len(predicted_names) == 0:
            empty += 1

        elif len(predicted_names) > 1:
            ambiguous += 1

            if class_name in predicted_names:
                correct += 1
            else:
                wrong += 1

        elif predicted_names[0] == class_name:
            correct += 1

        else:
            wrong += 1

    total = len(class_df)

    exact_accuracy = (
        correct / total if total else 0.0
    )

    results.append(
        {
            "class": class_name,
            "samples": total,
            "correct": correct,
            "wrong": wrong,
            "empty": empty,
            "ambiguous": ambiguous,
            "accuracy": exact_accuracy,
        }
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
)


print()
print("========================================")
print("PER-CLASS RESULTS")
print("========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

OUTPUT_CSV = "data/features/phase1_class_analysis.csv"

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:", OUTPUT_CSV)