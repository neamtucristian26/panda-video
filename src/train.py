import argparse
import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data_loader import create_data_loaders, create_frame_level_loaders
from losses import ProxyAnchorLossWithProxies
from model import create_model
from utils import AverageMeter, accuracy, get_device, save_checkpoint, set_seed


def train_epoch(model, loader, criterion, optimizer, device, epoch,
                ce_weight=0.0, ce_source="proxy"):
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for embeddings, labels, _ in pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)

        embeddings_proj, proxies = model(embeddings)
        loss = criterion(embeddings_proj, labels, proxies)
        logits = model.get_logits(embeddings_proj)
        if ce_weight > 0:
            ce_logits = (logits if ce_source == "proxy"
                         else model.projection_net.classifier(embeddings_proj))
            loss = loss + ce_weight * F.cross_entropy(ce_logits, labels)
        acc1 = accuracy(logits, labels, topk=(1,))[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.update(loss.item(), embeddings.size(0))
        top1.update(acc1, embeddings.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"})
    return losses.avg, top1.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for embeddings, labels, _ in tqdm(loader, desc="Validation"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            embeddings_proj, proxies = model(embeddings)
            loss = criterion(embeddings_proj, labels, proxies)
            logits = model.get_logits(embeddings_proj)
            acc1 = accuracy(logits, labels, topk=(1,))[0]
            losses.update(loss.item(), embeddings.size(0))
            top1.update(acc1, embeddings.size(0))
    print(f"Validation - Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%")
    return losses.avg, top1.avg


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}, modality: {args.modality}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(os.path.join(args.metadata_dir, "model_to_label.json")) as f:
        model_to_label = json.load(f)
    num_classes = len(model_to_label)
    print(f"Classes ({num_classes}): {model_to_label}")

    if args.frame_level:
        loaders = create_frame_level_loaders(
            metadata_dir=args.metadata_dir, embeddings_dir=args.embeddings_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        loaders = create_data_loaders(
            metadata_dir=args.metadata_dir,
            embeddings_dir=args.embeddings_dir,
            modality=args.modality,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            l2_branches=args.l2_branches,
            video_key=args.video_key,
        )

    input_dim = loaders["train"].dataset[0][0].shape[0]
    print(f"Inferred input_dim={input_dim} from embeddings on disk (modality={args.modality})")
    model = create_model(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        num_classes=num_classes,
        temperature=args.temperature,
    ).to(device)

    criterion = ProxyAnchorLossWithProxies(margin=args.margin, alpha=args.alpha)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_acc, best_acc_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
        train_epoch(model, loaders["train"], criterion, optimizer, device, epoch,
                    ce_weight=args.ce_weight, ce_source=args.ce_source)
        val_loss, val_acc1 = validate(model, loaders["validation"], criterion, device)
        scheduler.step()

        if val_acc1 > best_acc:
            best_acc, best_acc_epoch = val_acc1, epoch
            save_checkpoint(model, optimizer, epoch, best_acc, args.checkpoint_dir, "best_model.pth")
            print(f"New best Acc@1: {best_acc:.2f}% (epoch {epoch})")

    print(f"\nTraining completed. Best val Acc@1: {best_acc:.2f}% at epoch {best_acc_epoch}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Proxy-Anchor video-generator attribution model")
    parser.add_argument("--metadata_dir", default="metadata_pilot")
    parser.add_argument("--embeddings_dir", default="data/raven_embeddings_pilot")
    parser.add_argument("--checkpoint_dir", default="checkpoints/proxy_anchor_pilot")
    parser.add_argument("--modality", choices=["fusion", "video", "audio"], default="fusion")
    parser.add_argument("--video_key", default="video",
                        choices=["video", "video_hidden", "video_patchmean"],
                        help="which stored vector is the video branch")
    parser.add_argument("--l2_branches", action="store_true",
                        help="L2-normalize video and audio branches separately before concat; needed for mixed-backbone fusion (e.g. DINOv3 video + BRAVEn audio, 119x norm imbalance)")
    parser.add_argument(
        "--frame_level", action="store_true",
        help="Train on individual frames (from --per_frame extraction) instead of pooled "
             "clip embeddings -- each frame is its own sample, clip label repeated per frame. "
             "Always fusion (video+audio concat per frame); --modality is ignored.",
    )

    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[])
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--ce_weight", type=float, default=0.0,
                        help="weight of the auxiliary cross-entropy term (0 = off)")
    parser.add_argument("--ce_source", choices=["proxy", "classifier"], default="proxy")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
