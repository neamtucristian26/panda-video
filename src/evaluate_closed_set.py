import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from data_loader import RavenEmbeddingDataset
from model import create_model
from utils import get_device


@torch.no_grad()
def predict(model, loader, device):
    preds, labels = [], []
    for embeddings, y, _ in tqdm(loader, desc="Inference", leave=False):
        emb, _ = model(embeddings.to(device))
        preds.extend(model.get_logits(emb).argmax(dim=1).cpu().tolist())
        labels.extend(y.tolist())
    return np.array(preds), np.array(labels)


def main(args):
    device = get_device()
    with open(os.path.join(args.metadata_dir, "model_to_label.json")) as f:
        model_to_label = json.load(f)
    names = [None] * len(model_to_label)
    for n, i in model_to_label.items():
        names[i] = n

    ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"),
                      map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    lin = sorted((int(k.split(".")[2]), v.shape)
                 for k, v in sd.items()
                 if k.startswith("projection_net.projection.") and k.endswith(".weight")
                 and v.dim() == 2)
    input_dim = lin[0][1][1]
    hidden_dims = [shape[0] for _, shape in lin[:-1]]
    embedding_dim = lin[-1][1][0]
    if hidden_dims:
        print(f"  inferred architecture: {input_dim} -> {hidden_dims} -> {embedding_dim}")
    model = create_model(input_dim=input_dim, hidden_dims=hidden_dims,
                         embedding_dim=embedding_dim, num_classes=len(names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    ds = RavenEmbeddingDataset(os.path.join(args.metadata_dir, "test_indomain.csv"),
                               args.embeddings_dir, args.modality, l2_branches=args.l2_branches,
                               video_key=args.video_key)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    preds, labels = predict(model, loader, device)

    acc = float((preds == labels).mean() * 100)
    p, r, f1, sup = precision_recall_fscore_support(labels, preds,
                                                    labels=list(range(len(names))), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    macro_f1 = float(np.mean(f1) * 100)
    bal_acc = float(np.mean(r) * 100)

    print(f"\n=== {len(names)}-way closed set: {args.checkpoint_dir} ===")
    print(f"  accuracy          : {acc:.2f}%")
    print(f"  balanced accuracy : {bal_acc:.2f}%")
    print(f"  macro F1          : {macro_f1:.2f}%")
    print(f"\n  {'class':<14}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}")
    for i, n in enumerate(names):
        print(f"  {n:<14}{p[i]*100:>7.2f}%{r[i]*100:>7.2f}%{f1[i]*100:>7.2f}%{sup[i]:>9d}")
    print("\n  confusion matrix (row = true, col = predicted, % of row)")
    print(f"  {'':<14}" + "".join(f"{n[:7]:>9}" for n in names))
    for i, n in enumerate(names):
        row = cm[i] / max(cm[i].sum(), 1) * 100
        print(f"  {n:<14}" + "".join(f"{v:>8.1f}%" for v in row))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
                   "labels": names, "confusion_matrix": cm.tolist(),
                   "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                            "f1": float(f1[i]), "support": int(sup[i])}
                                 for i in range(len(names))}}, f, indent=2)
    print(f"\n  saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_dir", default="metadata_all7")
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--modality", choices=["fusion", "video", "audio"], default="fusion")
    ap.add_argument("--video_key", default="video",
                    choices=["video", "video_hidden", "video_patchmean"])
    ap.add_argument("--l2_branches", action="store_true")
    ap.add_argument("--embedding_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    main(ap.parse_args())
