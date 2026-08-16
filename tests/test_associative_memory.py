import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.associative_memory import AssociativeMemory


memory = AssociativeMemory()

# Learn a full shape-color-name association
node_1 = memory.learn(
    shape_symbol="s0",
    color_symbol="c0",
    name_symbol="apple"
)

# Same visual pair with another name alias
node_1_again = memory.learn(
    shape_symbol="s0",
    color_symbol="c0",
    name_symbol="red_apple"
)

# Activate using name only
node_from_name = memory.learn(
    name_symbol="apple"
)

# Learn a different object
node_2 = memory.learn(
    shape_symbol="s1",
    color_symbol="c1",
    name_symbol="banana"
)

print("First node ID:", node_1)
print("Same visual pair ID:", node_1_again)
print("Name-only activation ID:", node_from_name)
print("Second node ID:", node_2)

print("\nAssociative nodes:")
for node in memory.nodes:
    print(node)



print("\nTesting recall...")

names = memory.recall_names(
    shape_symbol="s0",
    color_symbol="c0"
)

visual = memory.recall_visual(
    name_symbol="apple"
)

print("Recalled names:", names)
print("Recalled visual:", visual)