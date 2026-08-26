from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cdsn import CDSN, CDSNConfig, make_synthetic_dataset, load_image_dataset
from cdsn.evaluate import run_closed, run_open_ended, run_repeated

RESULTS = Path(__file__).parent / "results"


# --------------------------------------------------------------------- helpers
def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def load_data(args):
    if args.data:
        print(f"Loading image dataset from {args.data} ...")
        samples = load_image_dataset(args.data, views_per_object=args.views)
    else:
        print("Using synthetic dataset (20 objects x 8 viewpoints + 2 aliases).")
        samples = make_synthetic_dataset(seed=args.seed)
    labels = sorted({s.label for s in samples})
    print(f"  {len(samples)} samples, {len(labels)} objects, "
          f"shape dim={len(samples[0].shape)}, color dim={len(samples[0].color)}")
    return samples


# ------------------------------------------------------------------ experiments
def experiment_main(samples, runs: int):
    """Tables 2, 4 and 5: node counts and recall in both environments."""
    print(f"\nRunning main experiment ({runs} repeats per environment) ...")
    stats = run_repeated(samples, n_runs=runs)

    recall_rows = []
    node_rows = []
    for env in ("closed", "open_ended"):
        s = stats[env]
        pretty = "Closed" if env == "closed" else "Open-ended"
        recall_rows.append([
            pretty,
            f"{s['vision_mean']:.2%} ± {s['vision_std']:.2%}",
            f"{s['auditory_mean']:.2%} ± {s['auditory_std']:.2%}",
        ])
        node_rows.append([
            pretty,
            f"{s['shape_nodes']:.0f}", f"{s['color_nodes']:.0f}",
            f"{s['name_nodes']:.0f}", f"{s['assoc_nodes']:.0f}",
        ])

    recall_md = (
        "# Recall results\n\n"
        f"Mean ± std over {runs} runs with different presentation orders.\n\n"
        + _md_table(["Environment", "Vision recall", "Auditory recall"], recall_rows)
      
    )
    nodes_md = (
        "# Node counts per layer\n\n"
        f"Receptive-layer node counts, mean over {runs} runs.\n\n"
        + _md_table(["Environment", "Shape", "Color", "Name", "Associative"], node_rows)
        + "\n"
    )
    (RESULTS / "table_recall.md").write_text(recall_md)
    (RESULTS / "table_nodes.md").write_text(nodes_md)
    (RESULTS / "summary.json").write_text(json.dumps(stats, indent=2))

    print("\n" + recall_md)
    print(nodes_md)
    return stats


def experiment_growth(samples, seed: int = 0):
    print("Recording category growth curves ...")
    model = CDSN()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))

    hist = {"shape": [], "color": [], "name": [], "assoc": []}
    for i, idx in enumerate(order, 1):
        model.learn(samples[idx])
        c = model.node_counts()
        hist["shape"].append(c["concept"]["shape"])
        hist["color"].append(c["concept"]["color"])
        hist["name"].append(c["concept"]["name"])
        hist["assoc"].append(c["associative"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        x = range(1, len(order) + 1)
        ax1.plot(x, hist["assoc"], color="black")
        ax1.set_xlabel("Learning iterations")
        ax1.set_ylabel("Number of associative nodes")
        ax1.set_title("(a) Associative Layer")
        ax1.grid(alpha=.3)

        ax2.plot(x, hist["shape"], label="Shape", color="black")
        ax2.plot(x, hist["color"], label="Color", color="red")
        ax2.plot(x, hist["name"], label="Voice", color="tab:blue")
        ax2.set_xlabel("Learning iterations")
        ax2.set_ylabel("Number of categories")
        ax2.set_title("(b) Concept Layer")
        ax2.legend()
        ax2.grid(alpha=.3)

        fig.tight_layout()
        fig.savefig(RESULTS / "growth.png", dpi=140)
        plt.close(fig)
        print(f"  wrote {RESULTS / 'growth.png'}")
    except ImportError:
        print("  matplotlib unavailable; skipping plot")

    return hist


def experiment_ablation(samples, runs: int = 5):
    print(f"\nRunning ablation study ({runs} repeats each) ...")
    variants = {
        "Full model": CDSNConfig(),
        "No node deletion": CDSNConfig(enable_node_deletion=False),
        "Shape channel only": CDSNConfig(),          # handled below
        "Loose vigilance (p=30)": CDSNConfig(shape_percentile=30.0,
                                             color_percentile=30.0),
        "Tight vigilance (p=1)": CDSNConfig(shape_percentile=1.0,
                                            color_percentile=1.0),
    }

    rows = []
    for name, cfg in variants.items():
        vis, aud, nodes = [], [], []
        for r in range(runs):
            if name == "Shape channel only":
                # Drop the colour modality entirely at input time.
                stripped = [type(s)(shape=s.shape, color=None, name=s.name,
                                    label=s.label) for s in samples]
                m = CDSN(cfg)
                m.fit(stripped, shuffle=True, seed=r)
                # Vision recall needs colour; score name recall reachability only.
                from cdsn.evaluate import auditory_recall
                vis.append(float("nan"))
                aud.append(auditory_recall(m, samples))
                nodes.append(m.node_counts()["associative"])
            else:
                res = run_closed(samples, config=cfg, seed=r)
                vis.append(res.vision)
                aud.append(res.auditory)
                nodes.append(res.nodes["associative"])
        finite = [x for x in vis if not np.isnan(x)]
        v = float(np.mean(finite)) if finite else float("nan")
        rows.append([
            name,
            "n/a" if np.isnan(v) else f"{v:.2%}",
            f"{np.mean(aud):.2%}",
            f"{np.mean(nodes):.0f}",
        ])
        print(f"  {name:24s} vision={rows[-1][1]:>8s}  aud={rows[-1][2]:>8s}")

    md = ("# Ablation study\n\n"
          "mechanism; compare against 'Full model'.\n\n"
          + _md_table(["Variant", "Vision recall", "Auditory recall",
                       "Associative nodes"], rows) + "\n")
    (RESULTS / "ablation.md").write_text(md)
    print(f"  wrote {RESULTS / 'ablation.md'}")


def experiment_sensitivity(samples, runs: int = 5):
    print(f"\nRunning vigilance sensitivity sweep ({runs} repeats each) ...")
    rows = []
    for p in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        vis, aud, shp = [], [], []
        for r in range(runs):
            cfg = CDSNConfig(shape_percentile=p, color_percentile=p)
            res = run_closed(samples, config=cfg, seed=r)
            vis.append(res.vision)
            aud.append(res.auditory)
            shp.append(res.nodes["receptive"]["shape"])
        rows.append([f"{p:g}", f"{np.mean(vis):.2%}", f"{np.mean(aud):.2%}",
                     f"{np.mean(shp):.0f}"])
        print(f"  percentile={p:<5g} vision={np.mean(vis):.2%}  "
              f"shape_nodes={np.mean(shp):.0f}")

    md = ("# Vigilance sensitivity\n\n"
          + _md_table(["Vigilance percentile", "Vision recall",
                       "Auditory recall", "Shape nodes"], rows) + "\n")
    (RESULTS / "sensitivity.md").write_text(md)
    print(f"  wrote {RESULTS / 'sensitivity.md'}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5,
                    help="repeats per environment (paper uses 30)")
    ap.add_argument("--data", type=str, default=None,
                    help="directory of real images (ALOI/COIL layout)")
    ap.add_argument("--views", type=int, default=8,
                    help="viewpoints per object when loading real images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--all", action="store_true", help="run every experiment")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    samples = load_data(args)

    experiment_main(samples, runs=args.runs)
    experiment_growth(samples, seed=args.seed)

    if args.ablation or args.all:
        experiment_ablation(samples, runs=min(args.runs, 5))
    if args.sensitivity or args.all:
        experiment_sensitivity(samples, runs=min(args.runs, 5))

    print(f"\nAll outputs written to {RESULTS}/")


if __name__ == "__main__":
    main()
