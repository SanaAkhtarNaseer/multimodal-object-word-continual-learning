from src.self_organizing_memory import SelfOrganizingMemory
from src.concept_memory import ConceptMemory
from src.associative_memory import AssociativeMemory


class CDSN:
    """
    Cognitive Deep Self-Organizing Neural Network.

    Receptive layer:
        - shape
        - color

    Concept layer:
        - shape symbols
        - color symbols
        - name symbols

    Associative layer:
        - binds shape + color + name
    """

    def __init__(
        self,
        sigma=1.0,
        threshold=0.30,
        maintenance_interval=10
    ):
        # -------------------------
        # Receptive layer
        # -------------------------
        self.shape_memory = SelfOrganizingMemory(
            sigma=sigma,
            threshold=threshold,
            maintenance_interval=maintenance_interval
        )

        self.color_memory = SelfOrganizingMemory(
            sigma=sigma,
            threshold=threshold,
            maintenance_interval=maintenance_interval
        )

        # -------------------------
        # Concept layer
        # -------------------------
        self.shape_concepts = ConceptMemory()
        self.color_concepts = ConceptMemory()
        self.name_concepts = ConceptMemory()

        # -------------------------
        # Associative layer
        # -------------------------
        self.associative_memory = AssociativeMemory()


    def learn(self, shape_vector, color_vector, name):
        """
        Learn one multimodal object sample.
        """

        # -------------------------
        # 1. Receptive layer
        # -------------------------
        shape_result = self.shape_memory.learn(shape_vector)
        color_result = self.color_memory.learn(color_vector)

        shape_id = shape_result["node_id"]
        color_id = color_result["node_id"]

        # -------------------------
        # 2. Concept layer
        # -------------------------
        shape_symbol = f"s{shape_id}"
        color_symbol = f"c{color_id}"
        name_symbol = str(name)

        shape_concept_id = self.shape_concepts.learn(shape_symbol)
        color_concept_id = self.color_concepts.learn(color_symbol)
        name_concept_id = self.name_concepts.learn(name_symbol)

        # -------------------------
        # 3. Associative layer
        # -------------------------
        associative_id = self.associative_memory.learn(
            shape_symbol=shape_symbol,
            color_symbol=color_symbol,
            name_symbol=name_symbol
        )

        return {
            "shape_result": shape_result,
            "color_result": color_result,
            "shape_symbol": shape_symbol,
            "color_symbol": color_symbol,
            "name_symbol": name_symbol,
            "shape_concept_id": shape_concept_id,
            "color_concept_id": color_concept_id,
            "name_concept_id": name_concept_id,
            "associative_id": associative_id
        }


    def recall_name_from_visual(self, shape_vector, color_vector):
        """
        Recall learned names from shape and color input
        without modifying the receptive memories.
        """

        # Find closest receptive nodes
        shape_winner, _, shape_distance, _ = (
            self.shape_memory.find_two_winners(shape_vector)
        )

        color_winner, _, color_distance, _ = (
            self.color_memory.find_two_winners(color_vector)
        )

        # Convert receptive nodes to their persistent symbols
        shape_id = self.shape_memory.node_ids[shape_winner]
        color_id = self.color_memory.node_ids[color_winner]

        shape_symbol = f"s{shape_id}"
        color_symbol = f"c{color_id}"

        # Recall through associative memory
        names = self.associative_memory.recall_names(
            shape_symbol,
            color_symbol
        )

        return {
            "shape_symbol": shape_symbol,
            "color_symbol": color_symbol,
            "shape_distance": shape_distance,
            "color_distance": color_distance,
            "names": names
        }    



    def recall_visual_from_name(self, name):
        """
        Recall the visual representation associated with a learned name.
        """

        name_symbol = str(name)

        # Find the associative representation linked to this name
        visual_result = self.associative_memory.recall_visual(name_symbol)

        if visual_result is None:
            return None

        shape_symbol = visual_result["shape"]
        color_symbol = visual_result["color"]

        # Convert symbols such as s0 and c0 back to persistent node IDs
        shape_id = int(shape_symbol[1:])
        color_id = int(color_symbol[1:])

        # Find the current node positions corresponding to those IDs
        try:
            shape_index = self.shape_memory.node_ids.index(shape_id)
            color_index = self.color_memory.node_ids.index(color_id)
        except ValueError:
            return None

        return {
            "name": name_symbol,
            "shape_symbol": shape_symbol,
            "color_symbol": color_symbol,
            "shape_vector": self.shape_memory.nodes[shape_index].copy(),
            "color_vector": self.color_memory.nodes[color_index].copy(),
            "associative_id": visual_result["associative_id"]
        }