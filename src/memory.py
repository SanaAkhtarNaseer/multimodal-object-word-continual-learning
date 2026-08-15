import numpy as np
from cim import cim


def find_winner(x, nodes, sigma):
    """
    Find the memory node with the smallest CIM distance
    from the incoming vector x.
    """

    distances = []

    for node in nodes:
        distance = cim(x, node, sigma)
        distances.append(distance)

    winner_index = np.argmin(distances)

    return winner_index, distances

def vigilance_test(winner_distance, threshold):

    return winner_distance <= threshold

def update_winner(winner_vector, input_vector, learning_rate=0.1):
    """
    Move the winner node slightly toward the incoming vector.
    """
    updated_vector = winner_vector + learning_rate * (
        input_vector - winner_vector
    )

    return updated_vector


if __name__ == "__main__":

    # Incoming sample
    input_vector = np.array([10.0, 10.0, 10.0])

    # Existing memory nodes
    nodes = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([8.0, 2.0, 1.0]),
    ]

    sigma = 1.0

    winner_index, distances = find_winner(
        input_vector,
        nodes,
        sigma
    )

    print("Incoming vector:", input_vector)

    for i, distance in enumerate(distances):
        print(f"CIM to Node {i}: {distance}")

    print("\nWinner node:", winner_index)
    print("Winner vector:", nodes[winner_index])
#vigilence test

threshold = 0.30

winner_distance = distances[winner_index]

passed = vigilance_test(
    winner_distance,
    threshold
)

print("\nVigilance threshold:", threshold)
print("Winner distance:", winner_distance)
print("Passed vigilance test:", passed)

# Node updating if vigelence test passed
if passed:
    old_vector = nodes[winner_index].copy()

    nodes[winner_index] = update_winner(
        nodes[winner_index],
        input_vector,
        learning_rate=0.1
    )

    print("\nExisting node updated")
    print("Old vector:", old_vector)
    print("Input vector:", input_vector)
    print("Updated vector:", nodes[winner_index])

else:
    nodes = np.vstack([nodes, input_vector])

    print("\nNew node created")
    print("New node:", input_vector)

print("\nCurrent nodes:")
print(nodes)