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
# Temporary synthetic object representations
# --------------------------------------------------

objects_phase_1 = {
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
}

objects_phase_2 = {
    "cherry": {
        "shape": np.array([3.0, 7.0, 4.0]),
        "color": np.array([0.9, 0.2, 0.2]),
    },
    "orange": {
        "shape": np.array([6.0, 3.0, 7.0]),
        "color": np.array([0.9, 0.5, 0.1]),
    },
}


all_objects = {
    **objects_phase_1,
    **objects_phase_2
}


# --------------------------------------------------
# Generate noisy viewpoints
# --------------------------------------------------

def generate_samples(objects):
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

    random.shuffle(samples)

    return samples


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, objects):

    visual_correct = 0
    auditory_correct = 0

    for name, features in objects.items():

        visual_result = model.recall_name_from_visual(
            features["shape"],
            features["color"]
        )

        if name in visual_result["names"]:
            visual_correct += 1

        auditory_result = model.recall_visual_from_name(name)

        if auditory_result is not None:
            auditory_correct += 1

    total = len(objects)

    return {
        "visual_recall": visual_correct / total,
        "auditory_recall": auditory_correct / total
    }


# --------------------------------------------------
# Open-ended experiment
# --------------------------------------------------

all_results = []

for run in range(NUM_RUNS):

    model = CDSN(
        sigma=1.0,
        threshold=0.30,
        maintenance_interval=10
    )

    # ==============================================
    # PHASE 1
    # ==============================================

    phase_1_samples = generate_samples(objects_phase_1)

    for sample in phase_1_samples:

        model.learn(
            shape_vector=sample["shape"],
            color_vector=sample["color"],
            name=sample["name"]
        )

    phase_1_before = evaluate(
        model,
        objects_phase_1
    )

    # ==============================================
    # PHASE 2
    # Introduce new concepts
    # ==============================================

    phase_2_samples = generate_samples(objects_phase_2)

    for sample in phase_2_samples:

        model.learn(
            shape_vector=sample["shape"],
            color_vector=sample["color"],
            name=sample["name"]
        )

    # Test OLD knowledge after learning new concepts
    phase_1_after = evaluate(
        model,
        objects_phase_1
    )

    # Test all accumulated knowledge
    all_recall = evaluate(
        model,
        all_objects
    )

    result = {
        "old_visual_before":
            phase_1_before["visual_recall"],

        "old_visual_after":
            phase_1_after["visual_recall"],

        "old_auditory_before":
            phase_1_before["auditory_recall"],

        "old_auditory_after":
            phase_1_after["auditory_recall"],

        "all_visual":
            all_recall["visual_recall"],

        "all_auditory":
            all_recall["auditory_recall"],

        "shape_nodes":
            len(model.shape_memory.nodes),

        "color_nodes":
            len(model.color_memory.nodes),

        "associative_nodes":
            len(model.associative_memory.nodes),
    }

    all_results.append(result)

    print(f"\nRun {run + 1}/{NUM_RUNS}")

    print(
        "Old visual before:",
        result["old_visual_before"]
    )

    print(
        "Old visual after:",
        result["old_visual_after"]
    )

    print(
        "Old auditory before:",
        result["old_auditory_before"]
    )

    print(
        "Old auditory after:",
        result["old_auditory_after"]
    )

    print(
        "All visual recall:",
        result["all_visual"]
    )

    print(
        "All auditory recall:",
        result["all_auditory"]
    )


# --------------------------------------------------
# Aggregate results
# --------------------------------------------------

def mean(key):
    return float(
        np.mean(
            [result[key] for result in all_results]
        )
    )


print("\n================================")
print("OPEN-ENDED ENVIRONMENT SUMMARY")
print("================================")

print(
    "Old Visual Recall Before:",
    round(mean("old_visual_before"), 4)
)

print(
    "Old Visual Recall After:",
    round(mean("old_visual_after"), 4)
)

print(
    "Old Auditory Recall Before:",
    round(mean("old_auditory_before"), 4)
)

print(
    "Old Auditory Recall After:",
    round(mean("old_auditory_after"), 4)
)

print(
    "All Visual Recall:",
    round(mean("all_visual"), 4)
)

print(
    "All Auditory Recall:",
    round(mean("all_auditory"), 4)
)

print(
    "Mean Shape Nodes:",
    round(mean("shape_nodes"), 2)
)

print(
    "Mean Color Nodes:",
    round(mean("color_nodes"), 2)
)

print(
    "Mean Associative Nodes:",
    round(mean("associative_nodes"), 2)
)
