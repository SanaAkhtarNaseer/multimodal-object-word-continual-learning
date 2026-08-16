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

sample_1 = model.learn(
    shape_vector=np.array([1.0, 2.0, 3.0]),
    color_vector=np.array([0.2, 0.4, 0.6]),
    name="apple"
)

sample_2 = model.learn(
    shape_vector=np.array([1.1, 2.1, 3.0]),
    color_vector=np.array([0.21, 0.39, 0.61]),
    name="apple"
)

sample_3 = model.learn(
    shape_vector=np.array([5.0, 5.0, 5.0]),
    color_vector=np.array([0.8, 0.8, 0.2]),
    name="banana"
)

print("\nSample 1:")
print(sample_1)

print("\nSample 2:")
print(sample_2)

print("\nSample 3:")
print(sample_3)

print("\nAssociative nodes:")
for node in model.associative_memory.nodes:
    print(node)



print("\nTesting visual-to-name recall...")

visual_recall = model.recall_name_from_visual(
    shape_vector=np.array([1.05, 2.05, 3.0]),
    color_vector=np.array([0.205, 0.395, 0.605])
)

print("Visual recall result:")
print(visual_recall)



print("\nTesting name-to-visual recall...")

apple_visual = model.recall_visual_from_name("apple")
banana_visual = model.recall_visual_from_name("banana")
unknown_visual = model.recall_visual_from_name("orange")

print("\nApple visual recall:")
print(apple_visual)

print("\nBanana visual recall:")
print(banana_visual)

print("\nUnknown visual recall:")
print(unknown_visual)
