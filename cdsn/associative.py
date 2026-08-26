from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AssociativeNode", "AssociativeLayer"]


@dataclass
class AssociativeNode:
    shape: str | None = None
    color: str | None = None
    names: set[str] = field(default_factory=set)
    activation: int = 0
    # Per-name activation, so recall picks the dominant alias rather than an
    # arbitrary set member.
    name_counts: dict[str, int] = field(default_factory=dict)

    @property
    def visual_key(self) -> tuple[str | None, str | None]:
        return (self.shape, self.color)

    def add_name(self, name: str) -> None:
        if not name:
            return
        self.names.add(name)
        self.name_counts[name] = self.name_counts.get(name, 0) + 1

    def dominant_name(self) -> str | None:
        if not self.name_counts:
            return None
        return max(self.name_counts, key=self.name_counts.get)


class AssociativeLayer:
    """Binds visual (shape, color) symbol pairs to auditory name symbols."""

    def __init__(self) -> None:
        self.nodes: list[AssociativeNode] = []
        self._visual_index: dict[tuple[str | None, str | None], int] = {}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------ bottom-up
    def learn(
        self,
        shape: str | None = None,
        color: str | None = None,
        name: str | None = None,
    ) -> int:
        """Bind whatever symbols are present into an associative node.

        CDSN "operates without the prerequisite of concurrent presentation of
        audio-visual data", so any of the three may be ``None``.
        """
        # Visual pair present -> node identity is the visual pair.
        if shape is not None and color is not None:
            key = (shape, color)
            idx = self._visual_index.get(key)
            if idx is None:
                node = AssociativeNode(shape=shape, color=color)
                self.nodes.append(node)
                idx = len(self.nodes) - 1
                self._visual_index[key] = idx
            self.nodes[idx].activation += 1
            if name:
                self.nodes[idx].add_name(name)
            return idx

        # Auditory-only input -> reinforce every node already carrying this name.
        if name is not None:
            hits = [i for i, n in enumerate(self.nodes) if name in n.names]
            if hits:
                best = max(hits, key=lambda i: self.nodes[i].name_counts.get(name, 0))
                self.nodes[best].activation += 1
                self.nodes[best].add_name(name)
                return best
            node = AssociativeNode(shape=shape, color=color)
            node.add_name(name)
            node.activation = 1
            self.nodes.append(node)
            return len(self.nodes) - 1

        return -1

    # ------------------------------------------------------------- top-down
    def recall_name(self, shape: str, color: str) -> str | None:
        
        idx = self._visual_index.get((shape, color))
        if idx is not None and self.nodes[idx].names:
            return self.nodes[idx].dominant_name()

        # Partial match: prefer nodes agreeing on shape, then on color, and
        # among those take the most frequently activated -- the paper's
        # "most frequently activated node as the winner node".
        for pred in (lambda n: n.shape == shape, lambda n: n.color == color):
            cands = [n for n in self.nodes if pred(n) and n.names]
            if cands:
                return max(cands, key=lambda n: n.activation).dominant_name()
        return None

    def recall_visual(self, name: str) -> tuple[str | None, str | None] | None:
       
        cands = [n for n in self.nodes
                 if name in n.names and (n.shape is not None or n.color is not None)]
        if not cands:
            return None
        # Rank by how often *this name* fired at the node, not by the node's
        # total activation.  A node that absorbed many samples early on would
        # otherwise win every name query regardless of relevance.
        best = max(cands, key=lambda n: (n.name_counts.get(name, 0), n.activation))
        return (best.shape, best.color)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AssociativeLayer(nodes={self.n_nodes})"
