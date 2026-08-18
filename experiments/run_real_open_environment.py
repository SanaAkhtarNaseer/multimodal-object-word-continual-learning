import pandas as pd
import numpy as np

from src.cdsn import CDSN


TRAIN_CSV = "data/features/train_features.csv"
VALIDATION_CSV = "data/features/validation_features.csv"
TEST_CSV = "data/features/test_features.csv"


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

PHASE_2_CLASSES = [
    "mango",
    "onion",
    "orange",
    "pear",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "tomato",
    "watermelon",
]


def load_feature_data(csv_path):
    df = pd.read_csv(csv_path)

    shape_columns = [f"shape_{i}" for i in range(23)]
    color_columns = [f"color_{i}" for i in range(63)]

    return df, shape_columns, color_columns


train_df, shape_columns, color_columns = load_feature_data(TRAIN_CSV)

print("Training samples:", len(train_df))
print("Classes:", train_df["name"].nunique())
print("Shape features:", len(shape_columns))
print("Color features:", len(color_columns))



phase_1_train = train_df[
    train_df["name"].isin(PHASE_1_CLASSES)
].copy()

model = CDSN(
    sigma=1.0,
    shape_threshold=0.08,
    color_threshold=0.06,
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

print()
print("PHASE 1 COMPLETE")
print("Phase 1 samples learned:", len(phase_1_train))
print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print("Associative nodes:", len(model.associative_memory.nodes))


validation_df, _, _ = load_feature_data(VALIDATION_CSV)

phase_1_validation = validation_df[
    validation_df["name"].isin(PHASE_1_CLASSES)
].copy()

correct = 0
total = 0

for _, row in phase_1_validation.iterrows():
    shape_vector = row[shape_columns].to_numpy(dtype=float)
    color_vector = row[color_columns].to_numpy(dtype=float)
    true_name = row["name"]

    result = model.recall_name_from_visual(
        shape_vector,
        color_vector
    )

    predicted_names = result["names"]

    if true_name in predicted_names:
        correct += 1

    total += 1

accuracy = correct / total if total else 0.0

print()
print("PHASE 1 VALIDATION")
print("Samples:", total)
print("Correct:", correct)
print("Visual recall accuracy:", round(accuracy, 4))
