from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .cdsn import Sample
from .features import COLOR_DIM, SHAPE_DIM

__all__ = ["OBJECTS", "make_synthetic_dataset", "load_image_dataset", "split_open_ended"]


OBJECTS: list[tuple[str, str, str, str]] = [
    ("apple",         "apple",         "red",     "round"),
    ("banana",        "banana",        "yellow",  "elongated"),
    ("cherry_tomato", "cherry tomato", "red",     "round"),
    ("chestnut",      "chestnut",      "brown",   "oval"),
    ("pepper",        "pepper",        "red",     "lobed"),
    ("pineapple",     "pineapple",     "green",   "irregular"),
    ("peanut",        "peanut",        "white",   "elongated"),
    ("yellow_pear",   "yellow pear",   "yellow",  "pear"),
    ("lemon",         "lemon",         "yellow",  "oval"),
    ("mango",         "mango",         "yellow",  "oval"),
    ("citrus",        "citrus",        "orange",  "round"),
    ("pear",          "pear",          "green",   "pear"),
    ("pomegranate",   "pomegranate",   "orange",  "round"),
    ("strawberry",    "strawberry",    "red",     "conical"),
    ("tomato",        "tomato",        "red",     "round"),
    ("radish",        "radish",        "pink",    "elongated"),
    ("egg",           "egg",           "white",   "oval"),
    ("onion",         "onion",         "white",   "round"),
    ("green_pear",    "pear",          "green",   "pear"),       # alias pair
    ("potato",        "potato",        "brown",   "irregular"),
    # two aliased items -> 22 presentations over 20 distinct objects
    ("tomato",        "cherry tomato", "red",     "round"),      # alias
    ("yellow_pear",   "pear",          "yellow",  "pear"),       # alias
]

_COLOR_FAMILIES = ["red", "yellow", "green", "orange", "brown", "white", "pink"]
_SHAPE_FAMILIES = ["round", "elongated", "oval", "pear", "lobed", "conical", "irregular"]


def _family_prototype(name: str, families: Sequence[str], dim: int,
                      rng: np.random.Generator) -> np.ndarray:
    """A stable, well-separated prototype vector for a feature family."""
    seed = abs(hash(name)) % (2**32)
    local = np.random.default_rng(seed)
    base = local.random(dim)
    keep = local.choice(dim, size=max(3, dim // 4), replace=False)
    proto = np.zeros(dim)
    proto[keep] = base[keep]
    s = proto.sum()
    return proto / s if s > 0 else proto


def make_synthetic_dataset(
    n_viewpoints: int = 8,
    shape_noise: float = 0.055,
    color_noise: float = 0.030,
    seed: int = 0,
) -> list[Sample]:
   
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []

    for obj, spoken, cfam, sfam in OBJECTS:
        shape_proto = _family_prototype(f"shape::{sfam}::{obj}", _SHAPE_FAMILIES,
                                        SHAPE_DIM, rng)
        color_proto = _family_prototype(f"color::{cfam}", _COLOR_FAMILIES,
                                        COLOR_DIM, rng)

        for _ in range(n_viewpoints):
            shape = np.clip(shape_proto + rng.normal(0, shape_noise, SHAPE_DIM), 0, None)
            color = np.clip(color_proto + rng.normal(0, color_noise, COLOR_DIM), 0, None)
            if shape.sum() > 0:
                shape = shape / shape.sum()
            if color.sum() > 0:
                color = color / color.sum()

            samples.append(Sample(shape=shape, color=color,
                                  name=spoken, label=obj))
    return samples


def load_image_dataset(root: str, views_per_object: int | None = None) -> list[Sample]:
   
    import cv2

    from .features import extract_visual_features

    samples: list[Sample] = []
    for obj in sorted(os.listdir(root)):
        folder = os.path.join(root, obj)
        if not os.path.isdir(folder):
            continue

        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        if views_per_object is not None and 0 < views_per_object < len(files):
            idx = np.linspace(0, len(files) - 1, views_per_object).round().astype(int)
            files = [files[i] for i in sorted(set(idx))]

        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            feats = extract_visual_features(img)
            samples.append(Sample(
                shape=feats["shape"],
                color=feats["color"],
                name=obj.replace("_", " "),
                label=obj,
            ))
    return samples


def split_open_ended(samples: Sequence[Sample], n_first: int = 10,
                     seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    
    labels = sorted({s.label for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    first = set(labels[:n_first])
    phase1 = [s for s in samples if s.label in first]
    phase2 = [s for s in samples if s.label not in first]
    return phase1, phase2
