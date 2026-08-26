
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .cim import cim_batch, silverman_bandwidth

__all__ = ["TCART"]


class TCART:
    #: Buffered samples required before the vigilance threshold can be
    #: estimated.  Below this the cold-start threshold applies.
    MIN_CALIBRATION_SAMPLES = 8

    def __init__(
        self,
        vigilance: Optional[float] = None,
        vigilance_percentile: float = 12.0,
        sigma_init: float = 0.35,
        lambda_buffer: int = 50,
        edge_max_age: int = 60,
        enable_node_deletion: bool = True,
        deletion_interval: int = 40,
        node_grace_period: int = 30,
    ) -> None:
        self.vigilance = vigilance
        self.vigilance_percentile = float(vigilance_percentile)
        self.sigma_init = float(sigma_init)
        self._auto_threshold: Optional[float] = None
        self.lambda_buffer = int(lambda_buffer)
        self.edge_max_age = int(edge_max_age)
        self.enable_node_deletion = bool(enable_node_deletion)
        self.deletion_interval = int(deletion_interval)
        self.node_grace_period = int(node_grace_period)

        # --- network state -------------------------------------------------
        self.V: list[np.ndarray] = []        # node prototype vectors
        self.sigma: list[float] = []         # per-node kernel bandwidth
        self.M: list[int] = []               # Eq. (8) -- cumulative sample count
        self.N: list[int] = []               # Eq. (9) -- count since last sigma adapt
        self.E_CIM: list[float] = []         # Eq. (10) -- running minimum error
        self.edges: dict[tuple[int, int], int] = {}   # (i, j) i<j -> age

        # Stable, never-reused identifiers.  Positional indices shift whenever
        # topology adjustment deletes a node, which would silently corrupt any
        # downstream index -> symbol mapping in the concept layer.  Every
        # public method therefore speaks in node *ids*, not positions.
        self.ids: list[int] = []
        self.birth: list[int] = []      # sample index at which each node was created
        self._next_id = 0

        self._buffer: deque = deque(maxlen=self.lambda_buffer)
        self._n_seen = 0

    # ------------------------------------------------------------------ utils
    @property
    def n_nodes(self) -> int:
        return len(self.V)

    @property
    def node_matrix(self) -> np.ndarray:
        return np.asarray(self.V, dtype=float) if self.V else np.empty((0, 0))

    def _mean_sigma(self) -> float:
        return float(np.mean(self.sigma)) if self.sigma else self.sigma_init

    def _threshold(self) -> float:
        """The effective vigilance threshold for this step."""
        if self.vigilance is not None:
            return float(self.vigilance)
        if self._auto_threshold is None:
            # Cold start, fewer than MIN_CALIBRATION_SAMPLES seen.  Be *strict*,
            # not permissive: ART cannot merge nodes after the fact, so a
            # permissive warm-up irreversibly collapses the first ~lambda
            # samples into a single node that then dominates every downstream
            # activation count.  Over-segmenting early is recoverable; under-
            # segmenting early is not.
            return 0.05
        return self._auto_threshold

    def _estimate_threshold(self) -> None:
        """Re-estimate the vigilance threshold from buffered pairwise CIMs.

        Called immediately after every bandwidth adaptation so the threshold
        and sigma always describe the same distance scale.
        """
        buf = np.asarray(self._buffer, dtype=float)
        if buf.shape[0] < self.MIN_CALIBRATION_SAMPLES:
            return
        sig = self._mean_sigma()
        # Pairwise CIM over the buffer (buffer is small, so this is cheap).
        dists = []
        for i in range(buf.shape[0]):
            d = cim_batch(buf[i], buf, sig)
            dists.append(np.delete(d, i))
        dists = np.concatenate(dists)
        self._auto_threshold = float(np.percentile(dists, self.vigilance_percentile))

    def _neighbors(self, k: int) -> list[int]:
        return [j for (i, j) in self.edges if i == k] + [
            i for (i, j) in self.edges if j == k
        ]

    @staticmethod
    def _ekey(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    # ------------------------------------------------------- phase 1: winners
    def _select_winners(self, u: np.ndarray) -> tuple[int, int, np.ndarray]:
       
        dists = cim_batch(u, self.node_matrix, self._mean_sigma())
        order = np.argsort(dists)
        k1 = int(order[0])
        k2 = int(order[1]) if len(order) > 1 else -1
        return k1, k2, dists

    # -------------------------------------------------- phase 3: node learning
    def _update_node(self, k: int, u: np.ndarray) -> None:
        
        Mk = self.M[k]
        self.V[k] = (Mk / (Mk + 1.0)) * self.V[k] + (1.0 / (Mk + 1.0)) * u
        self.M[k] = Mk + 1
        self.N[k] += 1

    def _new_node(self, u: np.ndarray, inherit_from: Optional[int] = None) -> int:
       
        if inherit_from is not None and self._neighbors(inherit_from):
            sig = float(np.mean([self.sigma[j] for j in self._neighbors(inherit_from)]))
        elif self.sigma:
            sig = self._mean_sigma()
        else:
            sig = self.sigma_init

        self.V.append(np.array(u, dtype=float, copy=True))
        self.sigma.append(sig)
        self.M.append(1)
        self.N.append(0)
        self.E_CIM.append(1.0)      # Eq. (10): initial error is 1
        self.ids.append(self._next_id)
        self.birth.append(self._n_seen)
        self._next_id += 1
        return len(self.V) - 1

    # ------------------------------------------- bandwidth adaptation (Sigma)
    def _adapt_bandwidths(self) -> None:
        
        if len(self._buffer) < 2:
            return
        h = silverman_bandwidth(np.asarray(self._buffer))
        for k in range(self.n_nodes):
            if self.N[k] > 0:
                self.sigma[k] = h
                self.N[k] = 0       # Eq. (9): N resets at each adaptation
        # Threshold must be re-derived in the new bandwidth's units.
        self._estimate_threshold()

    # ------------------------------------------------- phase 4: topology admin
    def _create_edge(self, k1: int, k2: int) -> None:
        if k1 == k2 or k2 < 0:
            return
        self.edges[self._ekey(k1, k2)] = 0      # age resets on (re)creation

    def _age_edges(self, k: int) -> None:
        for key in list(self.edges):
            if k in key:
                self.edges[key] += 1

    def _delete_old_edges(self) -> None:
        for key, age in list(self.edges.items()):
            if age > self.edge_max_age:
                del self.edges[key]

    def _prune(self) -> None:
       
        if not self.enable_node_deletion or self.n_nodes <= 2:
            return

        thr = self._threshold()
        doomed = set()
        for k in range(self.n_nodes):
            # Grace period: a node created moments ago has not yet had a fair
            # chance to attract samples.  Deleting it immediately (which the
            # literal Eq. (11) sweep does) destroys legitimate rare classes and
            # causes create/delete churn that corrupts downstream symbols.
            if self._n_seen - self.birth[k] < self.node_grace_period:
                continue
            # Eq. (10)/(11): a node whose best-ever match is still poor has no
            # data in its vicinity.  We compare E_CIM against the vigilance
            # threshold rather than mu(Sigma)^2 so the criterion lives on the
            # same scale as the CIM values that produced it.
            if self.M[k] <= 1 and self.E_CIM[k] > thr:
                doomed.add(k)
            elif not self._neighbors(k) and self.M[k] <= 1:
                doomed.add(k)       # isolated node with no support

        if not doomed or len(doomed) >= self.n_nodes - 1:
            return
        self._remove_nodes(sorted(doomed))

    def _remove_nodes(self, idx: list[int]) -> None:
        
        keep = [k for k in range(self.n_nodes) if k not in set(idx)]
        remap = {old: new for new, old in enumerate(keep)}

        self.V = [self.V[k] for k in keep]
        self.sigma = [self.sigma[k] for k in keep]
        self.M = [self.M[k] for k in keep]
        self.N = [self.N[k] for k in keep]
        self.E_CIM = [self.E_CIM[k] for k in keep]
        self.ids = [self.ids[k] for k in keep]
        self.birth = [self.birth[k] for k in keep]

        new_edges = {}
        for (i, j), age in self.edges.items():
            if i in remap and j in remap:
                new_edges[self._ekey(remap[i], remap[j])] = age
        self.edges = new_edges

    # ------------------------------------------------------------------- API
    def learn(self, u: np.ndarray) -> int:
        
        u = np.asarray(u, dtype=float).ravel()
        self._buffer.append(u)
        self._n_seen += 1

        # Cold start: first sample becomes the first node (v_1 = u_1).
        if self.n_nodes == 0:
            k = self._new_node(u)
            self.sigma[k] = self.sigma_init
            return self.ids[k]

        k1, k2, dists = self._select_winners(u)
        thr = self._threshold()
        beta_1 = float(dists[k1])

        if beta_1 <= thr:
            # --- vigilance passed: resonance with the first winner ----------
            self._update_node(k1, u)
            self.E_CIM[k1] = min(self.E_CIM[k1], beta_1)        # Eq. (10)
            self._age_edges(k1)

            # Edge creation requires the *second* winner to also resonate.
            if k2 >= 0 and float(dists[k2]) <= thr:
                self._create_edge(k1, k2)
            winner = k1
        else:
            # --- vigilance failed for all nodes: recruit a new node ---------
            winner = self._new_node(u, inherit_from=k1)

        # Periodic housekeeping.  During warm-up the threshold is recalibrated
        # on every sample, because this is exactly the window in which a stale
        # threshold does irreversible damage to the cluster structure.
        if self._n_seen < self.lambda_buffer:
            self._estimate_threshold()
        elif self._n_seen % self.lambda_buffer == 0:
            self._adapt_bandwidths()

        # Resolve the stable id *before* pruning may shift positions.
        winner_id = self.ids[winner]

        if self._n_seen % self.deletion_interval == 0:
            self._delete_old_edges()
            self._prune()

        return winner_id

    def predict(self, u: np.ndarray) -> int:
       
        if self.n_nodes == 0:
            return -1
        dists = cim_batch(np.asarray(u, dtype=float).ravel(),
                          self.node_matrix, self._mean_sigma())
        return self.ids[int(np.argmin(dists))]

    def position_of(self, node_id: int) -> int:
       
        try:
            return self.ids.index(node_id)
        except ValueError:
            return -1

    def prototype(self, node_id: int) -> np.ndarray | None:
        
        pos = self.position_of(node_id)
        return self.V[pos].copy() if pos >= 0 else None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"TCART(nodes={self.n_nodes}, edges={len(self.edges)}, "
                f"vigilance={self.vigilance}, seen={self._n_seen})")
