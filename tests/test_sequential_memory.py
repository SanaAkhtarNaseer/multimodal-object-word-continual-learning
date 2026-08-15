import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.self_organizing_memory import SelfOrganizingMemory


memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.3,
    learning_rate=0.1
)

samples = [
    np.array([1.0, 2.0, 3.0]),
    np.array([1.1, 2.1, 3.0]),
    np.array([5.0, 5.0, 5.0]),
    np.array([5.1, 4.9, 5.0]),
    np.array([10.0, 10.0, 10.0])
]


for i, sample in enumerate(samples):

    print("\n----------------------")
    print("Sample:", i)
    print("Input:", sample)

    result = memory.learn(sample)

    print("Action:", result["action"])

    # First sample has no previous nodes to compare against
    if result["distance"] if "distance" in result else False:
        pass

    if "first_winner" in result:
        print("First winner:", result["first_winner"])
        print("First distance:", result["first_distance"])
        print("Second winner:", result["second_winner"])
        print("Second distance:", result["second_distance"])

    print("Number of nodes:", len(memory.nodes))
    print("Current edges:", memory.edges)
    print("Win counts:", memory.win_counts)
    print("Bandwidth counts:", memory.bandwidth_counts)
    print("Bandwidths:", memory.bandwidths)
    print("Node errors:", memory.errors)

print("\nTesting node deletion...")

print("Before deletion:")
print("Nodes:", memory.nodes)
print("Edges:", memory.edges)
print("Win counts:", memory.win_counts)
print("Bandwidth counts:", memory.bandwidth_counts)
print("Bandwidths:", memory.bandwidths)
print("Errors:", memory.errors)

# memory.delete_node(1)

# print("\nAfter deleting node 1:")
# print("Nodes:", memory.nodes)
# print("Edges:", memory.edges)
# print("Win counts:", memory.win_counts)
# print("Bandwidth counts:", memory.bandwidth_counts)
# print("Bandwidths:", memory.bandwidths)
# print("Errors:", memory.errors)

deleted_nodes = memory.prune_nodes()

print("\nAutomatic pruning:")
print("Deleted node indices:", deleted_nodes)
print("Remaining nodes:", memory.nodes)
print("Remaining edges:", memory.edges)
print("Remaining errors:", memory.errors)
    

print("\nFinal learned nodes:")

for i, node in enumerate(memory.nodes):
    print(f"Node {i}: {node}")



print("\nTesting highest-CIM edge deletion...")

# Create a fresh memory only for this test
edge_memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.3
)

# Create three nodes
edge_memory.create_node(
    np.array([0.0, 0.0, 0.0])
)

edge_memory.create_node(
    np.array([0.2, 0.2, 0.2])
)

edge_memory.create_node(
    np.array([2.0, 2.0, 2.0])
)

# Manually create two edges connected to Node 0
edge_memory.create_or_update_edge(0, 1)
edge_memory.create_or_update_edge(0, 2)

print("Edges before deletion:", edge_memory.edges)

deleted_edge = edge_memory.delete_highest_cim_edge(0)

print("Deleted edge:", deleted_edge)
print("Edges after deletion:", edge_memory.edges)


print("\nTesting edge intersection detection...")

intersection_memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.3
)

# Four nodes forming an X
intersection_memory.create_node(np.array([0.0, 0.0]))
intersection_memory.create_node(np.array([2.0, 2.0]))
intersection_memory.create_node(np.array([0.0, 2.0]))
intersection_memory.create_node(np.array([2.0, 0.0]))

edge1 = (0, 1)
edge2 = (2, 3)

print(
    "Crossing edges intersect:",
    intersection_memory.edges_intersect(edge1, edge2)
)

# Non-crossing test
edge3 = (0, 2)
edge4 = (1, 3)

print(
    "Non-crossing edges intersect:",
    intersection_memory.edges_intersect(edge3, edge4)
)


print("\nTesting automatic intersection resolution...")

resolution_memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.3
)

resolution_memory.create_node(np.array([0.0, 0.0]))
resolution_memory.create_node(np.array([2.0, 2.0]))
resolution_memory.create_node(np.array([0.0, 2.0]))
resolution_memory.create_node(np.array([2.0, 0.0]))

resolution_memory.create_or_update_edge(0, 1)
resolution_memory.create_or_update_edge(2, 3)

print("Edges before resolution:", resolution_memory.edges)

deleted_edges = resolution_memory.resolve_edge_intersections()

print("Deleted edges:", deleted_edges)
print("Edges after resolution:", resolution_memory.edges)





print("\nTesting periodic maintenance after 10 samples...")

maintenance_memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.30,
    maintenance_interval=10
)

maintenance_samples = [
    np.array([1.0, 1.0, 1.0]),
    np.array([1.1, 1.0, 1.0]),
    np.array([2.0, 2.0, 2.0]),
    np.array([2.1, 2.0, 2.0]),
    np.array([3.0, 3.0, 3.0]),
    np.array([3.1, 3.0, 3.0]),
    np.array([4.0, 4.0, 4.0]),
    np.array([4.1, 4.0, 4.0]),
    np.array([5.0, 5.0, 5.0]),
    np.array([5.1, 5.0, 5.0]),
]

for sample in maintenance_samples:
    maintenance_memory.learn(sample)

print("Samples seen:", maintenance_memory.samples_seen)
print("Nodes after 10 samples:", len(maintenance_memory.nodes))
print("Edges after maintenance:", maintenance_memory.edges)
print("Errors after maintenance:", maintenance_memory.errors)
print("Bandwidths after maintenance:", maintenance_memory.bandwidths)
print("Bandwidth counts:", maintenance_memory.bandwidth_counts)