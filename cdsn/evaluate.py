from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cdsn import CDSN, CDSNConfig, Sample
from .dataset import split_open_ended

__all__ = ["RecallResult", "vision_recall", "auditory_recall",
           "run_closed", "run_open_ended", "run_repeated"]


@dataclass
class RecallResult:
    vision: float
    auditory: float
    nodes: dict

    def __str__(self) -> str:  # pragma: no cover
        return (f"vision={self.vision:6.2%}  auditory={self.auditory:6.2%}  "
                f"nodes={self.nodes}")


def vision_recall(model: CDSN, samples: Sequence[Sample]) -> float:
    """Vision -> name.  Fraction of samples whose name is correctly recalled."""
    correct = 0
    for s in samples:
        recalled = model.recall_name_from_vision(s.shape, s.color)
        if recalled is not None and recalled.strip().lower() == s.name.strip().lower():
            correct += 1
    return correct / max(len(samples), 1)


def auditory_recall(model: CDSN, samples: Sequence[Sample],
                    alias_aware: bool = True) -> float:
  
    # Per-label mean shape, used as the reference set for the nearest-class test.
    by_label: dict[str, list[np.ndarray]] = {}
    name_to_labels: dict[str, set[str]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s.shape)
        name_to_labels.setdefault(s.name.strip().lower(), set()).add(s.label)
    label_means = {lab: np.mean(v, axis=0) for lab, v in by_label.items()}
    labels = list(label_means)
    means = np.asarray([label_means[l] for l in labels])

    correct = 0
    for s in samples:
        out = model.recall_vision_from_name(s.name)
        if out is None or out["shape_prototype"] is None:
            continue
        proto = out["shape_prototype"]
        dists = np.linalg.norm(means - proto[None, :], axis=1)
        retrieved = labels[int(np.argmin(dists))]
        valid = (name_to_labels.get(s.name.strip().lower(), {s.label})
                 if alias_aware else {s.label})
        if retrieved in valid:
            correct += 1
    return correct / max(len(samples), 1)


def _evaluate(model: CDSN, samples: Sequence[Sample]) -> RecallResult:
    return RecallResult(
        vision=vision_recall(model, samples),
        auditory=auditory_recall(model, samples),
        nodes=model.node_counts(),
    )


def run_closed(samples: Sequence[Sample], config: CDSNConfig | None = None,
               seed: int = 0) -> RecallResult:
    """Closed environment: single-epoch online training on all objects."""
    model = CDSN(config)
    model.fit(samples, shuffle=True, seed=seed)
    return _evaluate(model, samples)


def run_open_ended(samples: Sequence[Sample], config: CDSNConfig | None = None,
                   seed: int = 0, n_first: int = 10) -> RecallResult:
    """Open-ended environment: phase 1 (10 objects), then phase 2 (the rest)."""
    phase1, phase2 = split_open_ended(samples, n_first=n_first, seed=seed)
    model = CDSN(config)
    model.fit(phase1, shuffle=True, seed=seed)
    model.fit(phase2, shuffle=True, seed=seed + 1)
    return _evaluate(model, samples)


def run_repeated(samples: Sequence[Sample], n_runs: int = 30,
                 config: CDSNConfig | None = None) -> dict:
   
    out: dict[str, dict] = {}
    for name, fn in (("closed", run_closed), ("open_ended", run_open_ended)):
        vis, aud, shp, col, nam, assoc = [], [], [], [], [], []
        for r in range(n_runs):
            res = fn(samples, config=config, seed=r)
            vis.append(res.vision)
            aud.append(res.auditory)
            shp.append(res.nodes["receptive"]["shape"])
            col.append(res.nodes["receptive"]["color"])
            nam.append(res.nodes["receptive"]["name"])
            assoc.append(res.nodes["associative"])
        out[name] = {
            "vision_mean": float(np.mean(vis)), "vision_std": float(np.std(vis)),
            "auditory_mean": float(np.mean(aud)), "auditory_std": float(np.std(aud)),
            "shape_nodes": float(np.mean(shp)),
            "color_nodes": float(np.mean(col)),
            "name_nodes": float(np.mean(nam)),
            "assoc_nodes": float(np.mean(assoc)),
        }
    return out
