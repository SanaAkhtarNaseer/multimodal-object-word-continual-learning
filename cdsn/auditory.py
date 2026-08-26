

from __future__ import annotations

__all__ = ["levenshtein", "AuditoryReceptiveLayer"]


def levenshtein(a: str, b: str) -> int:

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,            # deletion
                cur[j - 1] + 1,         # insertion
                prev[j - 1] + (ca != cb),   # substitution
            ))
        prev = cur
    return prev[-1]


class AuditoryReceptiveLayer:

    def __init__(self, max_edit_distance: int = 0) -> None:
        self.max_edit_distance = int(max_edit_distance)
        self.words: list[str] = []
        self.counts: list[int] = []

    @property
    def n_nodes(self) -> int:
        return len(self.words)

    def _match(self, word: str) -> tuple[int, int]:
        
        if not self.words:
            return -1, 10**9
        dists = [levenshtein(word, w) for w in self.words]
        t = min(range(len(dists)), key=dists.__getitem__)
        return t, dists[t]

    def learn(self, word: str) -> int:
      
        word = str(word).strip().lower()
        t, d = self._match(word)
        if t >= 0 and d <= self.max_edit_distance:
            self.counts[t] += 1
            return t
        self.words.append(word)
        self.counts.append(1)
        return len(self.words) - 1

    def predict(self, word: str) -> int:
       
        t, d = self._match(str(word).strip().lower())
        if t < 0:
            return -1
        # At inference we always return the nearest node -- an unseen word maps
        # to its closest known name rather than failing outright.
        return t

    def prototype(self, k: int) -> str:
      
        return self.words[k]

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuditoryReceptiveLayer(nodes={self.n_nodes})"
