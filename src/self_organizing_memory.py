import numpy as np
from src.cim import cim


class SelfOrganizingMemory:

    def __init__(self,sigma=1.0,threshold=0.30,learning_rate=0.1,
    maintenance_interval=10
):
        self.sigma = sigma
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.maintenance_interval = maintenance_interval
        self.samples_seen = 0

        # Network starts with no nodes
        self.nodes = []
        self.node_ids = []
        self.next_node_id = 0


        self.edges = {}
        self.win_counts = []
        self.bandwidth_counts = []
        self.bandwidths = []
        self.errors = []


    def find_two_winners(self, input_vector):
        distances = [
        cim(
            input_vector,
            node,
            self.bandwidths[i]
        )
        for i, node in enumerate(self.nodes)
    ]

        sorted_indices = np.argsort(distances)

        sorted_indices = np.argsort(distances)

        first_winner = int(sorted_indices[0])

        if len(self.nodes) > 1:
            second_winner = int(sorted_indices[1])
        else:
            second_winner = None

        first_distance = distances[first_winner]

        if second_winner is not None:
            second_distance = distances[second_winner]
        else:
            second_distance = None

        return first_winner, second_winner, first_distance, second_distance    
        

    def update_node(self, winner_index, input_vector):
        """
        Update the winner node using the incremental
        averaging rule from the CDSN paper.
        """

        m_k = self.win_counts[winner_index]

        old_vector = self.nodes[winner_index]

        updated_vector = (
            (m_k / (m_k + 1)) * old_vector
            +
            (1 / (m_k + 1)) * input_vector
        )

        feature_shift = float(
    np.linalg.norm(input_vector - old_vector)
)

      

        self.nodes[winner_index] = updated_vector

        self.win_counts[winner_index] += 1 
        self.bandwidth_counts[winner_index] += 1

        feature_shift = float(
    np.linalg.norm(input_vector - old_vector)
)

             


    def create_node(self, input_vector, neighbor_indices=None):
        """Create a new node from the incoming sample."""
        self.nodes.append(input_vector.copy())
        self.node_ids.append(self.next_node_id)
        self.next_node_id += 1
        # M_k
        self.win_counts.append(1)
        # N_k
        self.bandwidth_counts.append(1)

        # Initialize sigma_k
        if neighbor_indices:
            neighbor_bandwidths = [
                self.bandwidths[i]
                for i in neighbor_indices
            ]

            new_bandwidth = float(np.mean(neighbor_bandwidths))
        else:
            new_bandwidth = self.sigma

        self.bandwidths.append(new_bandwidth)
        # New nodes start with maximum CIM error
        self.errors.append(1.0)

    def create_or_update_edge(self, node_a, node_b):
        """
        Create an edge between the first and second winner nodes.
        If the edge already exists, increase its age/count.
        """

        # Cannot create an edge without two valid nodes
        if node_a is None or node_b is None:
            return

        # Store the smaller node index first
        edge = tuple(sorted((node_a, node_b)))

        # Update existing edge or create a new one
        if edge in self.edges:
            self.edges[edge] += 1
        else:
            self.edges[edge] = 1

    def update_error(self, winner_index, current_error):
        """
        Update E_CIM for the winner node.

        E_CIM <- min(E_CIM, current_error)
        """

        self.errors[winner_index] = float(
            min(
                self.errors[winner_index],
                current_error
            )
        )


    def prune_nodes(self, protected_indices=None):
        """
        Remove nodes that satisfy the pruning criteria.

        protected_indices:
            Nodes temporarily protected from pruning.
        """

        if len(self.nodes) == 0:
            return []

        if protected_indices is None:
            protected_indices = set()
        else:
            protected_indices = set(protected_indices)

        mean_bandwidth = float(np.mean(self.bandwidths))
        error_threshold = mean_bandwidth ** 2

        nodes_to_delete = []

        for node_index in range(len(self.nodes)):

            if node_index in protected_indices:
                continue

            has_edge = any(
                node_index in edge
                for edge in self.edges
            )

            high_error = (
                self.errors[node_index] > error_threshold
            )

            if (not has_edge) or high_error:
                nodes_to_delete.append(node_index)

        for node_index in reversed(nodes_to_delete):
            self.delete_node(node_index)

        return nodes_to_delete

    

    def delete_highest_cim_edge(self, node_index):
        """
        Delete the edge connected to node_index
        that has the highest CIM distance.

        Returns the deleted edge, or None if the
        node has no connected edges.
        """

        connected_edges = [
            edge
            for edge in self.edges
            if node_index in edge
        ]

        if not connected_edges:
            return None

        edge_scores = {}

        for edge in connected_edges:
            node_a, node_b = edge

            # Use the mean bandwidth of the two endpoint nodes
            edge_sigma = (
                self.bandwidths[node_a]
                + self.bandwidths[node_b]
            ) / 2.0

            score = cim(
                self.nodes[node_a],
                self.nodes[node_b],
                edge_sigma
            )

            edge_scores[edge] = float(score)

        edge_to_delete = max(
            edge_scores,
            key=edge_scores.get
        )

        del self.edges[edge_to_delete]

        return edge_to_delete


    def edges_intersect(self, edge1, edge2):
        """
        Check whether two edges intersect geometrically.
        Edges sharing a node are not considered intersections.
        """

        a, b = edge1
        c, d = edge2

        # Shared endpoint is not considered an intersection
        if len({a, b, c, d}) < 4:
            return False

        p1 = np.asarray(self.nodes[a], dtype=float)
        p2 = np.asarray(self.nodes[b], dtype=float)
        p3 = np.asarray(self.nodes[c], dtype=float)
        p4 = np.asarray(self.nodes[d], dtype=float)

        # For now, intersection is evaluated using
        # the first two dimensions.
        p1 = p1[:2]
        p2 = p2[:2]
        p3 = p3[:2]
        p4 = p4[:2]

        def orientation(p, q, r):
            value = (
                (q[1] - p[1]) * (r[0] - q[0])
                - (q[0] - p[0]) * (r[1] - q[1])
            )

            if np.isclose(value, 0.0):
                return 0

            return 1 if value > 0 else 2

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        return o1 != o2 and o3 != o4



    def resolve_edge_intersections(self):
        """
        Detect intersecting edges and remove the edge
        with the higher CIM distance from each conflict.

        Returns a list of deleted edges.
        """

        deleted_edges = []

        # Work on a fixed copy because self.edges may change
        edge_list = list(self.edges.keys())

        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):

                edge1 = edge_list[i]
                edge2 = edge_list[j]

                # Skip edges that may already have been deleted
                if edge1 not in self.edges or edge2 not in self.edges:
                    continue

                if not self.edges_intersect(edge1, edge2):
                    continue

                a, b = edge1
                c, d = edge2

                sigma1 = (
                    self.bandwidths[a]
                    + self.bandwidths[b]
                ) / 2.0

                sigma2 = (
                    self.bandwidths[c]
                    + self.bandwidths[d]
                ) / 2.0

                score1 = float(
                    cim(
                        self.nodes[a],
                        self.nodes[b],
                        sigma1
                    )
                )

                score2 = float(
                    cim(
                        self.nodes[c],
                        self.nodes[d],
                        sigma2
                    )
                )

                if score1 >= score2:
                    edge_to_delete = edge1
                else:
                    edge_to_delete = edge2

                del self.edges[edge_to_delete]

                deleted_edges.append(edge_to_delete)

        return deleted_edges 



    def delete_node(self, node_index):
        """
        Safely delete one node and all of its associated state.
        """

        # Remove node-specific information
        del self.nodes[node_index]
        del self.node_ids[node_index]


        del self.win_counts[node_index]
        del self.bandwidth_counts[node_index]
        del self.bandwidths[node_index]
        del self.errors[node_index]

        # Rebuild edges because node indices may shift
        new_edges = {}

        for (node_a, node_b), count in self.edges.items():

            # Remove any edge connected to the deleted node
            if node_a == node_index or node_b == node_index:
                continue

            # Shift indices greater than the deleted index
            if node_a > node_index:
                node_a -= 1

            if node_b > node_index:
                node_b -= 1

            edge = tuple(sorted((node_a, node_b)))

            new_edges[edge] = count

        self.edges = new_edges
                


    def learn(self, input_vector):
        """Learn one sample."""

        input_vector = np.asarray(input_vector, dtype=float)
        self.samples_seen += 1


        # First sample automatically becomes the first node
        if len(self.nodes) == 0:
            self.create_node(input_vector)

            return {
                "action": "created",
                "node_id": self.node_ids[0],
                "winner": 0,
                "distance": None
            }

        (first_winner,second_winner,first_distance,second_distance) = self.find_two_winners(input_vector)

        # Vigilance test
        if first_distance <= self.threshold:

            self.update_node(
                first_winner,
                input_vector
            )

            learned_node_id = self.node_ids[first_winner]

            self.update_error(
                first_winner,
                first_distance
            )

            if second_winner is not None:
                self.create_or_update_edge(
                    first_winner,
                    second_winner
                )

                self.resolve_edge_intersections()

            action = "updated"

        else:
            neighbor_indices = [first_winner]

            if second_winner is not None:
                neighbor_indices.append(second_winner)

            self.create_node(
                input_vector,
                neighbor_indices=neighbor_indices
            )
            learned_node_id = self.node_ids[-1]

            action = "created"

        # Periodic topology maintenance
        if (
            self.samples_seen % self.maintenance_interval == 0
            and len(self.nodes) > 2
        ):
          
            self.prune_nodes()
            self.resolve_edge_intersections()  

        return {
    "action": action,
    "node_id": learned_node_id,
    "first_winner": first_winner,
    "second_winner": second_winner,
    "first_distance": first_distance,
    "second_distance": second_distance
}