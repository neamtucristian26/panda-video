import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CLS = ["echomimic", "memo", "liveportrait", "inswapper", "sonic", "hififace", "roop"]
RAMP = LinearSegmentedColormap.from_list(
    "report_teal", ["#FBFDFD", "#DCECEC", "#9CCACC", "#4E9EA3", "#1C6E74", "#0A4449"])
INK, INK_MUTED, SURFACE, RULE = "#161A1F", "#5C666E", "#FFFFFF", "#C7D0D4"

STRINGS = {
    "en": {"pred": "Predicted class", "true": "True class", "cb": "% of true class",
           "acc": "accuracy"},
    "ro": {"pred": "Clasa prezisă", "true": "Clasa reală", "cb": "% din clasa reală",
           "acc": "acuratețe"},
}


def load(prefix, tag):
    with open(f"{prefix}{tag}/results.json") as f:
        return json.load(f)


def main(a):
    global T
    T = STRINGS[a.lang]
    r = load(a.prefix, a.config)
    labels = r["labels"]
    cm = np.array(r["confusion_matrix"], dtype=float)
    M = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    sup = [r["per_class"][c]["support"] for c in labels]
    subtitle = f"{a.config}  ·  {T['acc']} {r['accuracy']:.2f}%"

    order = [labels.index(c) for c in CLS if c in labels]
    names = [labels[i] for i in order]
    M = M[np.ix_(order, order)]
    sup = [sup[i] for i in order]
    n = len(names)

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.imshow(M, cmap=RAMP, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(np.arange(n + 1) - .5, minor=True)
    ax.set_yticks(np.arange(n + 1) - .5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v < 0.05:
                txt, col = "·", "#AAB4B8"
            else:
                txt = f"{v:.1f}"
                col = "#FFFFFF" if v >= 55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if i == j else "normal", color=col)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=10.5, color=INK)
    ax.set_yticklabels([f"{c}\n" + r"$\it{n=}$" + f"{s:,}" for c, s in zip(names, sup)],
                       fontsize=10.5, color=INK)
    ax.set_xlabel(T["pred"], fontsize=11.5, color=INK, labelpad=10)
    ax.set_ylabel(T["true"], fontsize=11.5, color=INK, labelpad=10)
    heading = a.title if a.no_subtitle else f"{a.title}\n{subtitle}"
    ax.set_title(heading, fontsize=12.5, color=INK, pad=14)
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=RAMP,
                      norm=plt.Normalize(0, 100)), ax=ax, fraction=.045, pad=.03)
    cb.set_label(T["cb"], fontsize=10.5, color=INK_MUTED)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=9.5, colors=INK_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(a.output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {a.output}")
    print(f"  diagonal (per-class recall): "
          + ", ".join(f"{c} {M[i,i]:.1f}%" for i, c in enumerate(names)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/")
    ap.add_argument("--config", default="tri_h2048_1024",
                    help="results subdirectory name, i.e. <prefix><config>/results.json")
    ap.add_argument("--title", default="Confusion matrix — closed-set attribution (7 classes)")
    ap.add_argument("--lang", choices=["en","ro"], default="en")
    ap.add_argument("--no_subtitle", action="store_true",
                    help="omit the config/accuracy line under the title")
    ap.add_argument("--output", default="results/figura_confuzie.png")
    main(ap.parse_args())
