#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from cdsn import CDSN, CDSNConfig, Sample
from cdsn.dataset import load_image_dataset, make_synthetic_dataset


# ----------------------------------------------------------------- formatting
def hr(char="-", n=64):
    print(char * n)


def train(data_dir: str | None, views: int, seed: int = 0):
    """Load a corpus and train CDSN on it in a single online pass."""
    if data_dir:
        if not os.path.isdir(data_dir):
            sys.exit(f"ERROR: no such directory: {data_dir}\n"
                     f"Run 'python make_demo_images.py' first.")
        print(f"Loading images from {data_dir}/ ...")
        samples = load_image_dataset(data_dir, views_per_object=views)
        if not samples:
            sys.exit(f"ERROR: no images found in {data_dir}/\n"
                     "Expected layout: {data_dir}/<object_name>/<image>.png")
    else:
        print("No --data given; using the synthetic feature corpus.")
        samples = make_synthetic_dataset(seed=seed)

    labels = sorted({s.label for s in samples})
    print(f"  {len(samples)} samples across {len(labels)} objects")
    print(f"  shape={len(samples[0].shape)}-D  color={len(samples[0].color)}-D")

    print("\nTraining (single online pass, one sample at a time) ...")
    model = CDSN(CDSNConfig())
    model.fit(samples, shuffle=True, seed=seed)

    c = model.node_counts()
    print(f"  receptive : shape={c['receptive']['shape']}  "
          f"color={c['receptive']['color']}  name={c['receptive']['name']}")
    print(f"  associative nodes: {c['associative']}")
    return model, samples, labels


# --------------------------------------------------------------------- probes
def probe_image(model: CDSN, path: str, expected: str | None = None):
    """IMAGE -> NAME.  Run the full visual front-end on one image file."""
    import cv2

    from cdsn.features import extract_visual_features

    img = cv2.imread(path)
    if img is None:
        print(f"  could not read image: {path}")
        return None

    feats = extract_visual_features(img)
    recalled = model.recall_name_from_vision(feats["shape"], feats["color"])

    print(f"\n  image    : {os.path.basename(path)}")
    print(f"  recalled : {recalled if recalled else '(nothing — no association)'}")
    if expected:
        ok = recalled is not None and recalled.strip().lower() == expected.lower()
        print(f"  expected : {expected}   {'CORRECT' if ok else 'wrong'}")
    return recalled


def probe_name(model: CDSN, word: str, samples, labels):
    """NAME -> IMAGE.  Recall the visual concept a word refers to."""
    out = model.recall_vision_from_name(word)
    print(f"\n  spoken   : \"{word}\"")

    if out is None:
        print("  recalled : (nothing — that word was never learned)")
        return None

    print(f"  symbols  : shape={out['shape_symbol']}  color={out['color_symbol']}")

    proto = out["shape_prototype"]
    if proto is None:
        # Expected outcome, not a failure: topology adjustment can retire the
        # receptive node a concept symbol still points at.
        print("  prototype: (node was pruned during topology adjustment)")
        return None

    # Identify which object the recalled prototype most resembles.
    by_label: dict[str, list] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s.shape)
    means = {k: np.mean(v, axis=0) for k, v in by_label.items()}
    names = list(means)
    dists = [np.linalg.norm(means[n] - proto) for n in names]
    order = np.argsort(dists)

    print("  nearest objects to the recalled visual prototype:")
    for i in order[:3]:
        print(f"      {names[i]:<16s} distance={dists[i]:.4f}")
    return names[order[0]]


# ----------------------------------------------------------------- interactive
def interactive(model, samples, labels, data_dir):
    hr("=")
    print("INTERACTIVE MODE")
    hr("=")
    print("Commands:")
    print("  name <word>       recall the visual concept for a word")
    print("  image <path>      recall the name for an image file")
    print("  random            test a random image from the corpus")
    print("  objects           list every learned object")
    print("  quit")
    hr()

    rng = np.random.default_rng()
    while True:
        try:
            raw = input("\ncdsn> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        cmd, _, arg = raw.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "objects":
            print("  " + ", ".join(labels))
        elif cmd == "name":
            if arg:
                probe_name(model, arg, samples, labels)
            else:
                print("  usage: name banana")
        elif cmd == "image":
            if arg:
                probe_image(model, arg.strip('"\''))
            else:
                print("  usage: image data/demo_images/apple/view_000.png")
        elif cmd == "random":
            if not data_dir:
                print("  needs --data (a real image corpus)")
                continue
            obj = labels[rng.integers(len(labels))]
            folder = os.path.join(data_dir, obj)
            files = [f for f in sorted(os.listdir(folder))
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            pick = files[rng.integers(len(files))]
            probe_image(model, os.path.join(folder, pick),
                        expected=obj.replace("_", " "))
        else:
            print(f"  unknown command: {cmd}")


def auto_tour(model, samples, labels, data_dir):
    """Scripted demonstration, no user input required."""
    hr("=")
    print("DIRECTION 1: IMAGE -> NAME  (vision recall)")
    hr("=")

    if data_dir:
        correct = total = 0
        for obj in labels[:8]:
            folder = os.path.join(data_dir, obj)
            files = [f for f in sorted(os.listdir(folder))
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            if not files:
                continue
            r = probe_image(model, os.path.join(folder, files[0]),
                            expected=obj.replace("_", " "))
            total += 1
            if r and r.strip().lower() == obj.replace("_", " ").lower():
                correct += 1
        print(f"\n  {correct}/{total} correct on this sample")
    else:
        for s in samples[:6]:
            r = model.recall_name_from_vision(s.shape, s.color)
            ok = r is not None and r.lower() == s.name.lower()
            print(f"  {s.label:<16s} -> {str(r):<16s} {'CORRECT' if ok else 'wrong'}")

    hr("=")
    print("DIRECTION 2: NAME -> IMAGE  (auditory recall)")
    hr("=")
    for word in labels[:5]:
        probe_name(model, word.replace("_", " "), samples, labels)

    hr("=")
    print("ROBUSTNESS: misspelled / unseen words")
    hr("=")
    print("\n  The auditory layer matches with Levenshtein distance, so how it")
    print("  handles a typo depends on name_edit_tolerance (default 0 = exact).")
    for word in ("banan", "aplle", "spaceship"):
        out = model.recall_vision_from_name(word)
        verdict = "matched" if out else "no match (expected at tolerance 0)"
        print(f"    \"{word}\": {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="image corpus directory (root/<object>/<view>.png)")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", default=None, help="test one image file and exit")
    ap.add_argument("--name", default=None, help="test one word and exit")
    ap.add_argument("--auto", action="store_true",
                    help="scripted tour instead of interactive prompt")
    args = ap.parse_args()

    model, samples, labels = train(args.data, args.views, args.seed)

    if args.image:
        probe_image(model, args.image)
        return
    if args.name:
        probe_name(model, args.name, samples, labels)
        return
    if args.auto:
        auto_tour(model, samples, labels, args.data)
        return

    interactive(model, samples, labels, args.data)


if __name__ == "__main__":
    main()
