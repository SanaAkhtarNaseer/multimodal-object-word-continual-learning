import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.cdsn import CDSN


model = CDSN(
    sigma=1.0,
    threshold=0.30,
    maintenance_interval=10
)

# --------------------------------------------------
# Synthetic continual-learning dataset
# --------------------------------------------------

samples = {
    "apple": {
        "shape": np.array([1.0, 2.0, 3.0]),
        "color": np.array([0.2, 0.4, 0.6]),
    },
    "banana": {
        "shape": np.array([5.0, 5.0, 5.0]),
        "color": np.array([0.8, 0.8, 0.2]),
    },
    "pear": {
        "shape": np.array([8.0, 2.0, 1.0]),
        "color": np.array([0.1, 0.8, 0.3]),
    },
    "cherry": {
        "shape": np.array([3.0, 7.0, 4.0]),
        "color": np.array([0.9, 0.2, 0.2]),
    },
    "orange": {
        "shape": np.array([6.0, 3.0, 7.0]),
        "color": np.array([0.9, 0.5, 0.1]),
    },
}


# --------------------------------------------------
# Sequential learning
# --------------------------------------------------

print("\n=== LEARNING ===")

for name, sample in samples.items():
    model.learn(
        shape_vector=sample["shape"],
        color_vector=sample["color"],
        name=name
    )

    print("Learned:", name)


# --------------------------------------------------
# Name -> visual recall accuracy
# --------------------------------------------------

print("\n=== NAME TO VISUAL RECALL ===")

correct_name_to_visual = 0

for name, sample in samples.items():

    result = model.recall_visual_from_name(name)

    success = result is not None

    if success:
        correct_name_to_visual += 1

    print(name, "->", success)


name_to_visual_accuracy = (
    correct_name_to_visual / len(samples)
)


# --------------------------------------------------
# Visual -> name recall accuracy
# --------------------------------------------------

print("\n=== VISUAL TO NAME RECALL ===")

correct_visual_to_name = 0

for name, sample in samples.items():

    result = model.recall_name_from_visual(
        shape_vector=sample["shape"],
        color_vector=sample["color"]
    )

    success = name in result["names"]

    if success:
        correct_visual_to_name += 1

    print(
        name,
        "->",
        result["names"],
        "Correct:",
        success
    )


visual_to_name_accuracy = (
    correct_visual_to_name / len(samples)
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n=== RESULTS ===")

print(
    "Name -> Visual Recall Accuracy:",
    name_to_visual_accuracy
)

print(
    "Visual -> Name Recall Accuracy:",
    visual_to_name_accuracy
)

print("\nShape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print(
    "Associative nodes:",
    len(model.associative_memory.nodes)
)