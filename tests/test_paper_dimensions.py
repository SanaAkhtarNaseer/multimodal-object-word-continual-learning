import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.cdsn import CDSN


# ---------------------------------------------------------
# Paper-format feature dimensions
# ---------------------------------------------------------

SHAPE_DIM = 23
COLOR_DIM = 63


# ---------------------------------------------------------
# Create CDSN
# ---------------------------------------------------------

model = CDSN()


# ---------------------------------------------------------
# Create two paper-dimension samples
# ---------------------------------------------------------

apple_shape = np.linspace(0.10, 0.50, SHAPE_DIM)
apple_color = np.linspace(0.20, 0.70, COLOR_DIM)

banana_shape = np.linspace(0.60, 0.90, SHAPE_DIM)
banana_color = np.linspace(0.05, 0.30, COLOR_DIM)


print("=== PAPER DIMENSION TEST ===")

print("\nLearning apple...")

apple_result = model.learn(
    shape_vector=apple_shape,
    color_vector=apple_color,
    name="apple"
)

print("Apple learned.")
print("Shape dimension:", apple_shape.shape)
print("Color dimension:", apple_color.shape)


print("\nLearning banana...")

banana_result = model.learn(
    shape_vector=banana_shape,
    color_vector=banana_color,
    name="banana"
)

print("Banana learned.")
print("Shape dimension:", banana_shape.shape)
print("Color dimension:", banana_color.shape)


# ---------------------------------------------------------
# Test name -> visual recall
# ---------------------------------------------------------

print("\n=== NAME TO VISUAL RECALL ===")

apple_recall = model.recall_visual_from_name("apple")
banana_recall = model.recall_visual_from_name("banana")

print("\nApple:")
print(apple_recall)

print("\nBanana:")
print(banana_recall)


# ---------------------------------------------------------
# Test visual -> name recall
# ---------------------------------------------------------

print("\n=== VISUAL TO NAME RECALL ===")

apple_name = model.recall_name_from_visual(
    apple_shape,
    apple_color
)

banana_name = model.recall_name_from_visual(
    banana_shape,
    banana_color
)

print("Apple visual ->", apple_name)
print("Banana visual ->", banana_name)


# ---------------------------------------------------------
# Memory statistics
# ---------------------------------------------------------

print("\n=== MEMORY SUMMARY ===")

print("Shape nodes:", len(model.shape_memory.nodes))
print("Color nodes:", len(model.color_memory.nodes))
print("Associative nodes:", len(model.associative_memory.nodes))

print("\nPaper-dimension CDSN test completed.")
