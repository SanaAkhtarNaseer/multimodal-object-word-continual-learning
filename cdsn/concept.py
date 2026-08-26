

from __future__ import annotations

__all__ = ["ConceptLayer"]


class ConceptLayer:

    def __init__(self, prefix: str) -> None:
        self.prefix = str(prefix)
        self.symbols: list[str] = []                 # concept node weights
        self.activation: dict[str, int] = {}         # symbol -> activation count
        self._receptive_to_symbol: dict[int, str] = {}
        self._next_id = 0

    @property
    def n_nodes(self) -> int:
        return len(self.symbols)

    def symbolize(self, receptive_index: int) -> str:
        
        if receptive_index < 0:
            return f"{self.prefix}_none"

        sym = self._receptive_to_symbol.get(receptive_index)
        if sym is None:
            sym = f"{self.prefix}_{self._next_id}"
            self._next_id += 1
            self._receptive_to_symbol[receptive_index] = sym
            self.symbols.append(sym)
            self.activation[sym] = 0
        self.activation[sym] += 1
        return sym

    def lookup(self, receptive_index: int) -> str:
       
        if receptive_index < 0:
            return f"{self.prefix}_none"
        return self._receptive_to_symbol.get(receptive_index,
                                             f"{self.prefix}_unseen")

    def to_receptive(self, symbol: str) -> int:
       
        for idx, sym in self._receptive_to_symbol.items():
            if sym == symbol:
                return idx
        return -1

    def live_symbols(self, live_ids) -> list[str]:
       
        live = set(live_ids)
        return [sym for rid, sym in self._receptive_to_symbol.items() if rid in live]

    def most_activated(self) -> str | None:
        
        if not self.activation:
            return None
        return max(self.activation, key=self.activation.get)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConceptLayer({self.prefix!r}, nodes={self.n_nodes})"
