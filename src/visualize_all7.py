import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_model
from data_loader import RavenEmbeddingDataset


def subsample_by_class(samples, n_per_class, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for i, s in enumerate(samples):
        by_class.setdefault(s["class_name"], []).append(i)
    keep = []
    for cls, idxs in by_class.items():
        if len(idxs) > n_per_class:
            idxs = rng.sample(idxs, n_per_class)
        keep.extend(idxs)
    return keep


def load_points(metadata_csv, embeddings_dir, modality, n_per_class, allowed_classes=None,
                l2_branches=False, video_key="video"):
    ds = RavenEmbeddingDataset(metadata_csv, embeddings_dir, modality,
                               l2_branches=l2_branches, video_key=video_key)
    if allowed_classes is not None:
        ds.samples = [s for s in ds.samples if s["class_name"] in allowed_classes]
    keep = subsample_by_class(ds.samples, n_per_class)
    embs, classes = [], []
    for i in keep:
        emb, _, cls = ds[i]
        embs.append(emb.numpy())
        classes.append(cls)
    return np.stack(embs), classes


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2,
                      height=np.sqrt(1 - pearson) * 2, **kwargs)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

COLORS = {
    "echomimic": "#377EB8",
    "memo": "#6A51A3",
    "sonic": "#17BECF",
    "liveportrait": "#4DAF4A",
    "inswapper": "#E41A1C",
    "hififace": "#FF7F00",
    "roop": "#A65628",
}
ORDER = ["echomimic", "memo", "sonic", "liveportrait", "inswapper", "hififace", "roop"]

STRINGS = {
    "en": {"before": "Frozen features (before projection)",
           "after": "Proxy-Anchor space (after projection)",
           "sil": "silhouette {a:+.3f} (original space) / {b:+.3f} (2-D)",
           "proxy": "class proxy"},
    "ro": {"before": "Trăsături înghețate (înainte de proiecție)",
           "after": "Spațiul Proxy-Anchor (după proiecție)",
           "sil": "silhouette {a:+.3f} (spațiu original) / {b:+.3f} (2-D)",
           "proxy": "proxy de clasă"},
}


def main(args):
    global T
    T = STRINGS[args.lang]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.metadata_dir, "model_to_label.json")) as f:
        classes = list(json.load(f).keys())

    cap = args.n_per_class if args.n_per_class > 0 else 10**9
    parts, part_cls, part_split = [], [], []
    for sp in args.splits:
        print(f"Loading split '{sp}' (cap {cap}/class) ...")
        r, c = load_points(os.path.join(args.metadata_dir, f"{sp}.csv"),
                           args.embeddings_dir, args.modality, cap,
                           allowed_classes=set(classes),
                           l2_branches=args.l2_branches, video_key=args.video_key)
        parts.append(r); part_cls += c; part_split += [sp] * len(c)
    raw = np.concatenate(parts, axis=0)
    cls = np.array(part_cls)
    split_of = np.array(part_split)
    print(f"  {raw.shape[0]} points, {raw.shape[1]}-d, splits: "
          + ", ".join(f"{s}={int((split_of==s).sum())}" for s in args.splits))

    ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"),
                      map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    lin = sorted((int(k.split(".")[2]), v.shape) for k, v in sd.items()
                 if k.startswith("projection_net.projection.") and k.endswith(".weight")
                 and v.dim() == 2)
    input_dim, hidden_dims, emb_dim = lin[0][1][1], [sh[0] for _, sh in lin[:-1]], lin[-1][1][0]
    if hidden_dims:
        print(f"  inferred architecture: {input_dim} -> {hidden_dims} -> {emb_dim}")
    model = create_model(input_dim=input_dim, hidden_dims=hidden_dims,
                         embedding_dim=emb_dim, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        proj_t, proxies_t = model(torch.from_numpy(raw).float().to(device))
    proj = proj_t.cpu().numpy()
    proxies = proxies_t.cpu().numpy()

    sil_n = min(args.silhouette_sample, len(cls))
    sil_kw = dict(metric="cosine", random_state=42)
    if sil_n < len(cls):
        sil_kw["sample_size"] = sil_n
    sil_raw = silhouette_score(raw, cls, **sil_kw)
    sil_proj = silhouette_score(proj, cls, **sil_kw)
    print(f"  silhouette (cosine, original space, n={sil_n}): "
          f"raw {sil_raw:+.3f} -> projected {sil_proj:+.3f}")

    def reduce(x):
        if args.pca_dim and x.shape[1] > args.pca_dim and x.shape[0] > args.pca_dim:
            return PCA(n_components=args.pca_dim, random_state=42).fit_transform(x)
        return x

    def embed2d(x):
        x = reduce(x)
        if args.method == "umap":
            import umap
            return umap.UMAP(n_components=2, n_neighbors=args.n_neighbors,
                             min_dist=args.min_dist, metric="cosine",
                             random_state=42).fit_transform(x)
        return TSNE(n_components=2, random_state=42, init="pca",
                    perplexity=args.perplexity, n_jobs=args.n_jobs).fit_transform(x)

    cfg = (f"perplexity={args.perplexity:g}" if args.method == "tsne"
           else f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist:g}")
    print(f"Running {args.method.upper()} ({cfg}, PCA->{args.pca_dim}) on {len(raw)} points ...")
    ts_raw = embed2d(raw)
    print(f"Running {args.method.upper()} (projected + proxies) ...")
    comb = embed2d(np.concatenate([proj, proxies]))
    ts_proj, ts_proxy = comb[: len(proj)], comb[len(proj):]

    sil2_raw = silhouette_score(ts_raw, cls, random_state=42,
                                **({"sample_size": sil_n} if sil_n < len(cls) else {}))
    sil2_proj = silhouette_score(ts_proj, cls, random_state=42,
                                 **({"sample_size": sil_n} if sil_n < len(cls) else {}))
    print(f"  silhouette of the 2-D layout: raw {sil2_raw:+.3f}, projected {sil2_proj:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    panels = [
        (axes[0], ts_raw, T["before"] + "\n" + T["sil"].format(a=sil_raw, b=sil2_raw), False),
        (axes[1], ts_proj, T["after"] + "\n" + T["sil"].format(a=sil_proj, b=sil2_proj), True),
    ]
    msize = 7 if len(cls) <= 4000 else (2.2 if len(cls) > 20000 else 4)
    malpha = .45 if len(cls) <= 4000 else (.22 if len(cls) > 20000 else .32)
    for ax, pts, title, show_proxy in panels:
        label_xy = {}
        for c in ORDER:
            m = cls == c
            if not m.any():
                continue
            ax.scatter(pts[m, 0], pts[m, 1], s=msize, c=COLORS[c], alpha=malpha,
                       linewidths=0, label=c if not show_proxy else None)
            if args.style == "ellipses":
                confidence_ellipse(pts[m, 0], pts[m, 1], ax, n_std=2.0,
                                   facecolor=COLORS[c], alpha=.13, edgecolor=COLORS[c], lw=1.4)
            label_xy[c] = (np.median(pts[m, 0]), np.median(pts[m, 1]))

        span = (pts.max(axis=0) - pts.min(axis=0)).mean()
        radius = 0.07 * span
        rng = np.random.RandomState(0)
        anchors = {}
        for c in label_xy:
            own = pts[cls == c]
            cand = own if len(own) <= 150 else own[rng.choice(len(own), 150, replace=False)]
            best, best_s = None, -1e18
            for q in cand:
                d = np.linalg.norm(pts - q, axis=1) < radius
                n_own = int((d & (cls == c)).sum())
                n_oth = int(d.sum()) - n_own
                sc = n_own - 1.5 * n_oth
                if sc > best_s:
                    best_s, best = sc, q
            anchors[c] = np.asarray(best, dtype=float)

        keys = list(anchors)
        minsep = 0.11 * span
        for _ in range(60):
            moved = False
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = anchors[keys[i]], anchors[keys[j]]
                    v = b - a
                    dist = np.linalg.norm(v)
                    if dist < minsep:
                        v = v / dist if dist > 1e-9 else np.array([1.0, 0.0])
                        shift = (minsep - dist) / 2.0
                        anchors[keys[i]] = a - v * shift
                        anchors[keys[j]] = b + v * shift
                        moved = True
            if not moved:
                break

        for c, (lx, ly) in ((c, anchors[c]) for c in label_xy):
            t = ax.text(lx, ly, c, color=COLORS[c], fontsize=10.5, fontweight="bold",
                        ha="center", va="center", zorder=7)
            t.set_path_effects([path_effects.withStroke(linewidth=3.4, foreground="white")])
        if show_proxy:
            for i, c in enumerate(classes):
                ax.scatter(ts_proxy[i, 0], ts_proxy[i, 1], s=330, c=COLORS[c], marker="*",
                           edgecolors="black", linewidths=1.1, zorder=6)
        if args.style == "paper":
            unit = "UMAP" if args.method == "umap" else "t-SNE"
            ax.set_xlabel(f"{unit} 1", fontsize=11.5)
            ax.set_ylabel(f"{unit} 2", fontsize=11.5)
            ax.tick_params(labelsize=9.5, colors="#444444")
        else:
            ax.set_title(title, fontsize=12.5, pad=11)
            ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#BBBBBB")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=COLORS[c], label=c) for c in ORDER]
    handles.append(plt.Line2D([], [], marker="*", ls="", ms=14, color="#555555",
                              markeredgecolor="black", label=T["proxy"]))
    if args.style == "paper":
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
                   fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(args.title, fontsize=14, y=0.985)
        fig.tight_layout(rect=[0, 0.055, 1, 0.97])
    fig.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved {args.output}")

    with open(os.path.splitext(args.output)[0] + "_stats.json", "w") as f:
        json.dump({"silhouette_raw": float(sil_raw), "silhouette_projected": float(sil_proj),
                   "n_points": int(raw.shape[0]), "input_dim": int(raw.shape[1]),
                   "n_per_class": args.n_per_class, "perplexity": args.perplexity,
                   "splits": args.splits, "pca_dim": args.pca_dim,
                   "silhouette_sample": int(sil_n),
                   "silhouette_2d_raw": float(sil2_raw),
                   "silhouette_2d_projected": float(sil2_proj),
                   "method": args.method, "n_neighbors": args.n_neighbors,
                   "min_dist": args.min_dist,
                   "per_split": {s_: int((split_of == s_).sum()) for s_ in args.splits}},
                  f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_dir", default="metadata_all7")
    ap.add_argument("--embeddings_dir",
                    default="data/clip_embeddings,data/dinov3_embeddings,data/raven_embeddings",
                    help="comma-separated; branch order defines the concatenation order")
    ap.add_argument("--l2_branches", action="store_true",
                    help="L2-normalise each branch separately; MUST match training")
    ap.add_argument("--video_key", default="video",
                    choices=["video", "video_hidden", "video_patchmean"])
    ap.add_argument("--checkpoint_dir", default="model/tri_h2048_1024")
    ap.add_argument("--modality", choices=["fusion", "video", "audio"], default="fusion")
    ap.add_argument("--embedding_dim", type=int, default=512)
    ap.add_argument("--n_per_class", type=int, default=300,
                    help="max clips per class per split; 0 = no cap")
    ap.add_argument("--splits", nargs="+", default=["test_indomain"],
                    choices=["train", "validation", "test_indomain"])
    ap.add_argument("--pca_dim", type=int, default=50,
                    help="PCA dimensionality before t-SNE; 0 disables")
    ap.add_argument("--silhouette_sample", type=int, default=5000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--perplexity", type=float, default=30,
                    help="t-SNE perplexity; higher emphasises global structure")
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--n_neighbors", type=int, default=30, help="UMAP n_neighbors")
    ap.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    ap.add_argument("--title", default="Closed-set generator attribution (7 classes)")
    ap.add_argument("--lang", choices=["en","ro"], default="en")
    ap.add_argument("--style", choices=["paper","ellipses"], default="paper",
                    help="paper: plain scatter + labelled axes, as in the source "
                         "paper's Fig. 1; ellipses: 2-sigma confidence ellipses")
    ap.add_argument("--output", default="results/figura_tsne_all7.png")
    main(ap.parse_args())
