import pandas as pd

from src.cdsn import CDSN


TRAIN_CSV = "data/features/train_features.csv"
VALIDATION_CSV = "data/features/validation_features.csv"
TEST_CSV = "data/features/test_features.csv"

SHAPE_THRESHOLD = 0.03
COLOR_THRESHOLD = 0.04


PHASE_1_CLASSES = [
    "apple",
    "beetroot",
    "garlic",
]

PHASE_2_CLASSES = [
    "banana",
    "cucumber",
    "orange",
]


shape_columns = [f"shape_{i}" for i in range(23)]
color_columns = [f"color_{i}" for i in range(63)]


train_df = pd.read_csv(TRAIN_CSV)
validation_df = pd.read_csv(VALIDATION_CSV)


phase_1_train = train_df[
    train_df["name"].isin(PHASE_1_CLASSES)
].copy()

phase_1_validation = validation_df[
    validation_df["name"].isin(PHASE_1_CLASSES)
].copy()


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
print("================================")
print("PHASE 1 TRAINING COMPLETE")
print("================================")

print("Classes:", PHASE_1_CLASSES)
print("Training samples:", len(phase_1_train))
print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print(
    "Associative nodes:",
    len(model.associative_memory.nodes)
)


# --------------------------------------------------
# PHASE 1 VALIDATION
# --------------------------------------------------

correct = 0
association_correct = 0
empty = 0
ambiguous = 0

for _, row in phase_1_validation.iterrows():

    shape_vector = row[shape_columns].to_numpy(dtype=float)
    color_vector = row[color_columns].to_numpy(dtype=float)

    result = model.recall_name_from_visual(
        shape_vector,
        color_vector
    )

    names = result["names"]
    true_name = row["name"]

    if true_name in names:
        association_correct += 1

    if len(names) == 1 and names[0] == true_name:
        correct += 1

    if len(names) == 0:
        empty += 1

    if len(names) > 1:
        ambiguous += 1


total = len(phase_1_validation)

print()
print("================================")
print("PHASE 1 VALIDATION")
print("================================")

print("Validation samples:", total)
print("Exact correct:", correct)
print("Association correct:", association_correct)

print(
    "Exact accuracy:",
    round(correct / total, 4)
)

print(
    "Association recall:",
    round(association_correct / total, 4)
)

print(
    "Empty rate:",
    round(empty / total, 4)
)

print(
    "Ambiguity rate:",
    round(ambiguous / total, 4)
)


# --------------------------------------------------
# Store Phase 1 baseline
# --------------------------------------------------

phase1_before_accuracy = correct / total if total else 0.0


# --------------------------------------------------
# PHASE 2 TRAINING
# --------------------------------------------------

phase_2_train = train_df[
    train_df["name"].isin(PHASE_2_CLASSES)
].copy()

for _, row in phase_2_train.iterrows():

    shape_vector = row[shape_columns].to_numpy(dtype=float)
    color_vector = row[color_columns].to_numpy(dtype=float)

    model.learn(
        shape_vector=shape_vector,
        color_vector=color_vector,
        name=row["name"]
    )


print()
print("================================")
print("PHASE 2 TRAINING COMPLETE")
print("================================")

print("Classes:", PHASE_2_CLASSES)
print("Training samples:", len(phase_2_train))
print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print(
    "Associative nodes:",
    len(model.associative_memory.nodes)
)


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate_classes(model, df, class_names):

    subset = df[
        df["name"].isin(class_names)
    ].copy()

    exact = 0
    association = 0
    empty = 0
    ambiguous = 0

    for _, row in subset.iterrows():

        shape_vector = row[shape_columns].to_numpy(dtype=float)
        color_vector = row[color_columns].to_numpy(dtype=float)

        result = model.recall_name_from_visual(
            shape_vector,
            color_vector
        )

        names = result["names"]
        true_name = row["name"]

        if true_name in names:
            association += 1

        if len(names) == 1 and names[0] == true_name:
            exact += 1

        if len(names) == 0:
            empty += 1

        if len(names) > 1:
            ambiguous += 1

    total = len(subset)

    return {
        "samples": total,
        "exact_accuracy": exact / total if total else 0.0,
        "association_recall": association / total if total else 0.0,
        "empty_rate": empty / total if total else 0.0,
        "ambiguity_rate": ambiguous / total if total else 0.0,
    }


# --------------------------------------------------
# Evaluate after Phase 2
# --------------------------------------------------

phase1_after = evaluate_classes(
    model,
    validation_df,
    PHASE_1_CLASSES
)

phase2_result = evaluate_classes(
    model,
    validation_df,
    PHASE_2_CLASSES
)

all_classes = PHASE_1_CLASSES + PHASE_2_CLASSES

overall_result = evaluate_classes(
    model,
    validation_df,
    all_classes
)


forgetting = (
    phase1_before_accuracy
    - phase1_after["exact_accuracy"]
)


print()
print("================================")
print("FINAL CONTINUAL-LEARNING RESULTS")
print("================================")

print(
    "Phase 1 accuracy before Phase 2:",
    round(phase1_before_accuracy, 4)
)

print(
    "Phase 1 accuracy after Phase 2:",
    round(phase1_after["exact_accuracy"], 4)
)

print(
    "Phase 2 exact accuracy:",
    round(phase2_result["exact_accuracy"], 4)
)

print(
    "Overall exact accuracy:",
    round(overall_result["exact_accuracy"], 4)
)

print(
    "Forgetting:",
    round(forgetting, 4)
)

print(
    "Overall association recall:",
    round(overall_result["association_recall"], 4)
)

print(
    "Overall empty rate:",
    round(overall_result["empty_rate"], 4)
)

print(
    "Overall ambiguity rate:",
    round(overall_result["ambiguity_rate"], 4)
)

print(
    "Final shape nodes:",
    len(model.shape_memory.nodes)
)

print(
    "Final color nodes:",
    len(model.color_memory.nodes)
)

print(
    "Final associative nodes:",
    len(model.associative_memory.nodes)
)