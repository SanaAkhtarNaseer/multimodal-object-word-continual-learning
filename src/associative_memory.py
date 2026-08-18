class AssociativeMemory:
    """
    CDSN associative layer.

    Each associative node links:
        shape concept
        color concept
        one or more name concepts
    """

    def __init__(self):
        self.nodes = []
        self.next_id = 0

    def learn(self, shape_symbol=None, color_symbol=None, name_symbol=None):
        """
        Learn or update an associative representation.

        Inputs may be partial:
        - shape + color
        - name only
        - shape + color + name
        """

        # First try to match using the visual concept pair
        if shape_symbol is not None and color_symbol is not None:

            for node in self.nodes:
                if (
                    node["shape"] == shape_symbol
                    and node["color"] == color_symbol
                ):
                    node["count"] += 1

                    # Allow multiple aliases for one object
                    if (
                        name_symbol is not None
                        and name_symbol not in node["names"]
                    ):
                        node["names"].append(name_symbol)

                    return node["id"]

        # If only a name is available, activate an existing node
        if name_symbol is not None:

            for node in self.nodes:
                if name_symbol in node["names"]:
                    node["count"] += 1
                    return node["id"]

        # We cannot create a visual associative node
        # without shape and color information
        if shape_symbol is None or color_symbol is None:
            return None

        # No matching visual representation exists:
        # create a new associative node
        node_id = self.next_id
        self.next_id += 1

        names = []

        if name_symbol is not None:
            names.append(name_symbol)

        self.nodes.append(
            {
                "id": node_id,
                "shape": shape_symbol,
                "color": color_symbol,
                "names": names,
                "count": 1
            }
        )

        return node_id


    def recall_names(self, shape_symbol, color_symbol):
        """
        Recall auditory/name concepts from visual concepts.

        Associative nodes may be activated by matching symbols
        from one or both visual properties. Exact shape + color
        matches are preferred. If several nodes are activated,
        the most frequently activated node is selected.
        """

        candidates = []

        for node in self.nodes:

            matches = 0

            if node["shape"] == shape_symbol:
                matches += 1

            if node["color"] == color_symbol:
                matches += 1

            if matches > 0:
                candidates.append(
                    {
                        "node": node,
                        "matches": matches
                    }
                )

        if not candidates:
            return []

        winner = max(
            candidates,
            key=lambda item: (
                item["matches"],
                item["node"]["count"]
            )
        )

        return list(winner["node"]["names"])


    def recall_visual(self, name_symbol):
        """
        Recall visual concepts from an auditory/name concept.

        If the same name belongs to multiple associative nodes,
        choose the most frequently activated node.
        """
        matching_nodes = [
            node
            for node in self.nodes
            if name_symbol in node["names"]
        ]

        if not matching_nodes:
            return None

        winner = max(
            matching_nodes,
            key=lambda node: node["count"]
        )

        return {
            "shape": winner["shape"],
            "color": winner["color"],
            "associative_id": winner["id"]
        }