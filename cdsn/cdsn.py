from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .associative import AssociativeLayer
from .auditory import AuditoryReceptiveLayer
from .concept import ConceptLayer
from .tcart import TCART

__all__ = ["CDSN", "CDSNConfig", "Sample"]


@dataclass
class Sample:
  
    shape: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    name: Optional[str] = None
    label: Optional[str] = None      # ground truth, evaluation only -- never learned


@dataclass
class CDSNConfig:
  
    shape_vigilance: Optional[float] = None
    color_vigilance: Optional[float] = None
    shape_percentile: float = 2.0
    color_percentile: float = 12.0
    sigma_init: float = 0.35
    lambda_buffer: int = 50
    edge_max_age: int = 60
    deletion_interval: int = 40
    node_grace_period: int = 30
    enable_node_deletion: bool = True
    name_edit_tolerance: int = 0


class CDSN:
    """The full three-layer, dual-pathway cognitive architecture."""

    def __init__(self, config: Optional[CDSNConfig] = None) -> None:
        self.cfg = config or CDSNConfig()
        common = dict(
            sigma_init=self.cfg.sigma_init,
            lambda_buffer=self.cfg.lambda_buffer,
            edge_max_age=self.cfg.edge_max_age,
            deletion_interval=self.cfg.deletion_interval,
            node_grace_period=self.cfg.node_grace_period,
            enable_node_deletion=self.cfg.enable_node_deletion,
        )
        self.rl_shape = TCART(vigilance=self.cfg.shape_vigilance,
                              vigilance_percentile=self.cfg.shape_percentile, **common)
        self.rl_color = TCART(vigilance=self.cfg.color_vigilance,
                              vigilance_percentile=self.cfg.color_percentile, **common)
        self.rl_name = AuditoryReceptiveLayer(
            max_edit_distance=self.cfg.name_edit_tolerance
        )

        # ---- concept layer: symbolic distillation per attribute ------------
        self.cl_shape = ConceptLayer("shape")
        self.cl_color = ConceptLayer("color")
        self.cl_name = ConceptLayer("name")

        # ---- associative layer: cross-modal binding ------------------------
        self.al = AssociativeLayer()

    # ------------------------------------------------------- bottom-up pass
    def learn(self, sample: Sample) -> dict:
        """Present one sample and propagate bottom-up through all three layers."""
        shape_sym = color_sym = name_sym = None

        if sample.shape is not None:
            k = self.rl_shape.learn(sample.shape)
            shape_sym = self.cl_shape.symbolize(k)

        if sample.color is not None:
            k = self.rl_color.learn(sample.color)
            color_sym = self.cl_color.symbolize(k)

        if sample.name is not None:
            k = self.rl_name.learn(sample.name)
            name_sym = self.cl_name.symbolize(k)

        node = self.al.learn(shape=shape_sym, color=color_sym, name=name_sym)
        return {"shape": shape_sym, "color": color_sym,
                "name": name_sym, "associative_node": node}

    

    def fit(self, samples: Sequence[Sample], shuffle: bool = True,
            seed: Optional[int] = None) -> "CDSN":
        """Single-epoch online training, one sample at a time.

        The paper trains "in an online fashion [...] in a single epoch", with
        "a different object from the dataset randomly selected for presentation".
        """
        idx = np.arange(len(samples))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for i in idx:
            self.learn(samples[i])
        return self

    

    # -------------------------------------------------- top-down / recall
    def recall_name_from_vision(self, shape: np.ndarray,
                                color: np.ndarray) -> Optional[str]:
        """Vision -> audition.  Returns the recalled spoken name, or None.

        Bottom-up to the associative layer, then top-down through the auditory
        concept layer to the auditory receptive layer to recover the word.
        """
        ks = self.rl_shape.predict(shape)
        kc = self.rl_color.predict(color)
        s_sym = self.cl_shape.lookup(ks)
        c_sym = self.cl_color.lookup(kc)

        name_sym = self.al.recall_name(s_sym, c_sym)
        if name_sym is None:
            return None
        k_name = self.cl_name.to_receptive(name_sym)
        if k_name < 0:
            return None
        return self.rl_name.prototype(k_name)

    

    def recall_vision_from_name(self, word: str) -> Optional[dict]:
        
        k = self.rl_name.predict(word)
        if k < 0:
            return None
        name_sym = self.cl_name.lookup(k)

        pair = self.al.recall_visual(name_sym)
        if pair is None:
            return None
        s_sym, c_sym = pair

        # Either symbol may legitimately be None when the model was trained on
        # a single visual attribute; -1 means "no receptive node to recall".
        ks = self.cl_shape.to_receptive(s_sym) if s_sym is not None else -1
        kc = self.cl_color.to_receptive(c_sym) if c_sym is not None else -1
        return {
            "shape_symbol": s_sym,
            "color_symbol": c_sym,
            # prototype() returns None if topology adjustment has since retired
            # the node this symbol refers to -- an expected, handled outcome.
            "shape_prototype": self.rl_shape.prototype(ks) if ks >= 0 else None,
            "color_prototype": self.rl_color.prototype(kc) if kc >= 0 else None,
        }

    # ------------------------------------------------------------ reporting
    def node_counts(self) -> dict:
        
        return {
            "receptive": {
                "shape": self.rl_shape.n_nodes,
                "color": self.rl_color.n_nodes,
                "name": self.rl_name.n_nodes,
            },
            # Concept counts are reported over *live* symbols only -- see
            # ConceptLayer.live_symbols.
            "concept": {
                "shape": len(self.cl_shape.live_symbols(self.rl_shape.ids)),
                "color": len(self.cl_color.live_symbols(self.rl_color.ids)),
                "name": self.cl_name.n_nodes,
            },
            "associative": self.al.n_nodes,
        }

    def __repr__(self) -> str:  # pragma: no cover
        c = self.node_counts()
        return (f"CDSN(receptive={c['receptive']}, concept={c['concept']}, "
                f"associative={c['associative']})")
