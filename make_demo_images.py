#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np

# (folder name, BGR colour, shape kind, size ratio)
OBJECTS = [
    ("apple",         (40,  40, 200), "round",     0.80),
    ("banana",        (60, 210, 235), "crescent",  0.95),
    ("cherry_tomato", (50,  60, 215), "round",     0.55),
    ("lemon",         (70, 220, 240), "oval",      0.70),
    ("pear",          (90, 200, 160), "pear",      0.85),
    ("eggplant",      (110, 50,  90), "oval",      0.90),
    ("carrot",        (40, 130, 240), "cone",      0.90),
    ("orange",        (40, 150, 250), "round",     0.78),
    ("cucumber",      (70, 150,  90), "elongated", 0.95),
    ("potato",        (90, 130, 165), "blob",      0.75),
    ("strawberry",    (60,  60, 220), "heart",     0.60),
    ("onion",         (150, 190, 210), "round",    0.72),
    ("pepper",        (50,  70, 210), "pepper",    0.85),
    ("mango",         (60, 180, 245), "oval",      0.82),
    ("radish",        (140, 90, 200), "cone",      0.62),
    ("peanut",        (120, 165, 200), "peanut",   0.55),
    ("lime",          (80, 200, 120), "round",     0.68),
    ("plum",          (110, 60, 120), "round",     0.66),
    ("chestnut",      (60,  95, 135), "blob",      0.58),
    ("pineapple",     (50, 175, 215), "pineapple", 1.00),
]


def _contour_points(kind: str, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)

    if kind == "round":
        r = np.ones_like(t)
    elif kind == "oval":
        r = 1.0 / np.sqrt((np.cos(t) / 0.72) ** 2 + (np.sin(t) / 1.0) ** 2)
    elif kind == "elongated":
        r = 1.0 / np.sqrt((np.cos(t) / 0.34) ** 2 + (np.sin(t) / 1.0) ** 2)
    elif kind == "pear":
        r = 1.0 + 0.30 * np.cos(t) - 0.18 * np.cos(2 * t)
    elif kind == "crescent":
        r = 1.0 + 0.42 * np.cos(t) - 0.30 * np.cos(2 * t) + 0.12 * np.cos(3 * t)
    elif kind == "cone":
        r = 1.0 / (1.0 + 0.85 * np.abs(np.sin(t / 2.0)))
    elif kind == "heart":
        r = 1.0 - 0.28 * np.cos(2 * t) + 0.18 * np.sin(t)
    elif kind == "peanut":
        r = 0.62 + 0.40 * np.abs(np.cos(t))
    elif kind == "pepper":
        r = 1.0 + 0.16 * np.cos(3 * t) - 0.12 * np.cos(t)
    elif kind == "pineapple":
        r = 1.0 + 0.10 * np.cos(6 * t) - 0.14 * np.cos(t)
    else:  # blob
        r = 1.0 + 0.13 * np.cos(3 * t) + 0.07 * np.cos(5 * t)

    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def render(kind: str, color, scale: float, angle_deg: float,
           size: int = 256, rng=None) -> np.ndarray:
    """Render one viewpoint of an object on a black background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    pts = _contour_points(kind)
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = pts @ rot.T

    radius = 0.40 * size * scale
    pts = pts * radius + size / 2.0
    poly = pts.astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(img, [poly], color)

    if rng is not None:
        noise = rng.normal(0, 7, img.shape)
        img = np.clip(img.astype(float) + noise * (img.sum(axis=2, keepdims=True) > 0),
                      0, 255).astype(np.uint8)

    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/demo_images")
    ap.add_argument("--views", type=int, default=8,
                    help="viewpoints PER OBJECT")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for name, color, kind, scale in OBJECTS:
        folder = os.path.join(args.out, name)
        os.makedirs(folder, exist_ok=True)
        for v in range(args.views):
            angle = 360.0 * v / args.views
            img = render(kind, color, scale, angle, size=args.size, rng=rng)
            cv2.imwrite(os.path.join(folder, f"view_{int(angle):03d}.png"), img)
            total += 1

    print(f"Wrote {total} images across {len(OBJECTS)} objects -> {args.out}/")
    print("\nNow run:")
    print(f"    python run_experiments.py --data {args.out} --views {args.views}")
    print(f"    python demo.py --data {args.out}")


if __name__ == "__main__":
    main()
