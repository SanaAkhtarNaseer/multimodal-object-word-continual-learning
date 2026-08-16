from src.self_organizing_memory import SelfOrganizingMemory
from src.concept_memory import ConceptMemory


class MultimodalMemory:
    """
    Manages a separate self-organizing memory for each modality.
    """

    def __init__(self, modalities, **memory_kwargs):
        self.modalities = list(modalities)

        #Receptive Layer
        self.memories = {
            modality: SelfOrganizingMemory(**memory_kwargs)
            for modality in self.modalities
        }

       #Concept layer
        self.concept_memories = {
            modality: ConceptMemory()
            for modality in self.modalities
        }

    def learn(self, modality, input_vector):
        """
        Learn one input in the selected modality and
        map the receptive-layer winner to a stable concept symbol.
        """

        if modality not in self.memories:
            raise ValueError(f"Unknown modality: {modality}")

        receptive_memory = self.memories[modality]
        result = receptive_memory.learn(input_vector)

        # First sample returns "winner"; later samples return "first_winner"
        if "first_winner" in result:
            winner_index = result["first_winner"]
        else:
            winner_index = result["winner"]

        receptive_id = receptive_memory.node_ids[winner_index]

        receptive_symbol = f"{modality}_{receptive_id}"

        concept_id = self.concept_memories[modality].learn(
            receptive_symbol
        )

        return {
            "modality": modality,
            "receptive_result": result,
            "receptive_id": receptive_id,
            "receptive_symbol": receptive_symbol,
            "concept_id": concept_id
        }    