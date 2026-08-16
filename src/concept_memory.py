class ConceptMemory:
    """
    Incremental symbolic concept layer.
    """

    def __init__(self):
        self.concepts = {}
        self.next_id = 0

    def learn(self, receptive_symbol):
        """
        Learn or activate one symbolic concept.
        """

        if receptive_symbol in self.concepts:
            self.concepts[receptive_symbol]["count"] += 1
            return self.concepts[receptive_symbol]["id"]

        concept_id = self.next_id
        self.next_id += 1

        self.concepts[receptive_symbol] = {
            "id": concept_id,
            "count": 1
        }

        return concept_id