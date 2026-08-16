import os
import sys
import random
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.cdsn import CDSN


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

NUM_RUNS = 10
SAMPLES_PER_OBJECT = 8
NOISE_STD = 0.03

random.seed(42)
np.random.seed(42)


# --------------------------------------------------
# Base object representations
# Temporary synthetic equivalents of paper features.
# These will later be replaced by the real dataset:
#   shape -> 23 dimensions
#   color -> 63 dimensions
#   name  -> string
# --------------------------------------------------

objects = {
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
# Generate repeated observations / viewpoints
# --------------------------------------------------

def generate_samples():
    samples = []

    for name, features in objects.items():

        for _ in range(SAMPLES_PER_OBJECT):

            shape = (
                features["shape"]
                + np.random.normal(
                    0,
                    NOISE_STD,
                    size=features["shape"].shape
                )
            )

            color = (
                features["color"]
                + np.random.normal(
                    0,
                    NOISE_STD,
                    size=features["color"].shape
                )
            )

            samples.append(
                {
                    "name": name,
                    "shape": shape,
                    "color": color
                }
            )

    return samples


# --------------------------------------------------
# Evaluate one trained model
# --------------------------------------------------

def evaluate(model):

    visual_correct = 0
    auditory_correct = 0

    for name, features in objects.items():

        # Visual -> name
        visual_result = model.recall_name_from_visual(
            features["shape"],
            features["color"]
        )

        if name in visual_result["names"]:
            visual_correct += 1

        # Name -> visual
        auditory_result = model.recall_visual_from_name(name)

        if auditory_result is not None:
            auditory_correct += 1

    total = len(objects)

    return {
        "visual_recall": visual_correct / total,
        "auditory_recall": auditory_correct / total,
        "shape_nodes": len(model.shape_memory.nodes),
        "color_nodes": len(model.color_memory.nodes),
        "associative_nodes": len(model.associative_memory.nodes),
    }


# --------------------------------------------------
# Closed-environment experiment
# --------------------------------------------------

all_results = []

for run in range(NUM_RUNS):

    model = CDSN(
        sigma=1.0,
        threshold=0.30,
        maintenance_interval=10
    )

    samples = generate_samples()

    # Paper-style random presentation
    random.shuffle(samples)

    for sample in samples:

        model.learn(
            shape_vector=sample["shape"],
            color_vector=sample["color"],
            name=sample["name"]
        )

    result = evaluate(model)
    all_results.append(result)

    print(f"\nRun {run + 1}/{NUM_RUNS}")
    print(
        "Visual recall:",
        round(result["visual_recall"], 4)
    )
    print(
        "Auditory recall:",
        round(result["auditory_recall"], 4)
    )
    print(
        "Shape nodes:",
        result["shape_nodes"]
    )
    print(
        "Color nodes:",
        result["color_nodes"]
    )
    print(
        "Associative nodes:",
        result["associative_nodes"]
    )


# --------------------------------------------------
# Aggregate results
# --------------------------------------------------

visual_scores = [
    result["visual_recall"]
    for result in all_results
]

auditory_scores = [
    result["auditory_recall"]
    for result in all_results
]

shape_nodes = [
    result["shape_nodes"]
    for result in all_results
]

color_nodes = [
    result["color_nodes"]
    for result in all_results
]

associative_nodes = [
    result["associative_nodes"]
    for result in all_results
]


print("\n================================")
print("CLOSED ENVIRONMENT SUMMARY")
print("================================")

print(
    "Mean Visual Recall:",
    round(float(np.mean(visual_scores)), 4)
)

print(
    "Mean Auditory Recall:",
    round(float(np.mean(auditory_scores)), 4)
)

print(
    "Mean Shape Nodes:",
    round(float(np.mean(shape_nodes)), 2)
)

print(
    "Mean Color Nodes:",
    round(float(np.mean(color_nodes)), 2)
)

print(
    "Mean Associative Nodes:",
    round(float(np.mean(associative_nodes)), 2)
)