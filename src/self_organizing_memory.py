import numpy as np

from cim import cim


class SelfOrganizingMemory:

    def __init__(self, sigma=1.0, threshold=0.30, learning_rate=0.1):
        self.sigma = sigma
        self.threshold = threshold
        self.learning_rate = learning_rate

        # Network starts with no nodes
        self.nodes = []


    def find_winner(self, input_vector):
        """Find the node having the smallest CIM distance."""

        distances = [
            cim(input_vector, node, self.sigma)
            for node in self.nodes
        ]

        winner_index = int(np.argmin(distances))
        winner_distance = distances[winner_index]

        return winner_index, winner_distance


    def update_node(self, winner_index, input_vector):
        """Move the winner node toward the incoming sample."""

        self.nodes[winner_index] = (
            self.nodes[winner_index]
            + self.learning_rate
            * (input_vector - self.nodes[winner_index])
        )


    def create_node(self, input_vector):
        """Create a new node from the incoming sample."""

        self.nodes.append(input_vector.copy())


    def learn(self, input_vector):
        """Learn one sample."""

        input_vector = np.asarray(input_vector, dtype=float)

        # First sample automatically becomes the first node
        if len(self.nodes) == 0:
            self.create_node(input_vector)

            return {
                "action": "created",
                "winner": 0,
                "distance": None
            }

        winner_index, winner_distance = self.find_winner(input_vector)

        # Vigilance test
        if winner_distance <= self.threshold:

            self.update_node(
                winner_index,
                input_vector
            )

            action = "updated"

        else:

            self.create_node(input_vector)
            action = "created"

        return {
            "action": action,
            "winner": winner_index,
            "distance": winner_distance
        }