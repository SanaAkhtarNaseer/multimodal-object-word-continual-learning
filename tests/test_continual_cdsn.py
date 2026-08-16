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
# Stage 1: Learn first concept
# --------------------------------------------------

print("\n=== STAGE 1: Learning apple ===")

model.learn(
    shape_vector=np.array([1.0, 2.0, 3.0]),
    color_vector=np.array([0.2, 0.4, 0.6]),
    name="apple"
)

apple_before = model.recall_visual_from_name("apple")

print("Apple recall before new learning:")
print(apple_before)


# --------------------------------------------------
# Stage 2: Learn new concepts sequentially
# --------------------------------------------------

print("\n=== STAGE 2: Learning new concepts ===")

model.learn(
    shape_vector=np.array([5.0, 5.0, 5.0]),
    color_vector=np.array([0.8, 0.8, 0.2]),
    name="banana"
)

model.learn(
    shape_vector=np.array([8.0, 2.0, 1.0]),
    color_vector=np.array([0.1, 0.8, 0.3]),
    name="pear"
)

model.learn(
    shape_vector=np.array([3.0, 7.0, 4.0]),
    color_vector=np.array([0.9, 0.2, 0.2]),
    name="cherry"
)


# --------------------------------------------------
# Stage 3: Recall everything
# --------------------------------------------------

print("\n=== STAGE 3: Testing recall ===")

for name in ["apple", "banana", "pear", "cherry"]:

    result = model.recall_visual_from_name(name)

    print(f"\nRecall {name}:")
    print(result)


# --------------------------------------------------
# Stage 4: Visual -> name recall of old concept
# --------------------------------------------------

print("\n=== STAGE 4: Old concept visual recall ===")

old_apple = model.recall_name_from_visual(
    shape_vector=np.array([1.0, 2.0, 3.0]),
    color_vector=np.array([0.2, 0.4, 0.6])
)

print(old_apple)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n=== MEMORY SUMMARY ===")

print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print("Associative nodes:", len(model.associative_memory.nodes))

for node in model.associative_memory.nodes:
    print(node)