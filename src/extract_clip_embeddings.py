import argparse
import csv
import os
import warnings

import cv2
import numpy as np
import torch

torch.set_num_threads(2)
cv2.setNumThreads(2)
warnings.filterwarnings("ignore")

from extract_dinov3_embeddings import get_crops

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
NUM_FRAMES = 16


@torch.no_grad()
def encode(model, crops, mean, std, device, batch_size=32):
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(crops).float().div_(255.0).permute(0, 3, 1, 2)
    x = (x - mean) / std
    proj, hidden = [], []
    for s in range(0, x.shape[0], batch_size):
        out = model(pixel_values=x[s:s + batch_size].to(device, dtype))
        proj.append(out.image_embeds.float().cpu())
        hidden.append(out.last_hidden_state[:, 0, :].float().cpu())
    return torch.cat(proj).mean(dim=0), torch.cat(hidden).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default="metadata/all_unique.csv")
    ap.add_argument("--output_dir", default="data/clip_embeddings")
    ap.add_argument("--audio_dir", default="data/raven_embeddings_facefix")
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--all_frames", action="store_true",
                    help="use every frame up to 375 (BRAVEn's exact policy) instead of --num_frames uniform samples")
    ap.add_argument("--landmark_cache", default=None,
                    help="dir for cached per-clip landmarks, shared across video backbones")
    ap.add_argument("--fp32", action="store_true", help="force fp32 (default fp16; CLIP is fp16-stable)")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16

    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection
    import face_alignment

    proc = AutoImageProcessor.from_pretrained(args.model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(args.model_id, dtype=dtype).to(device).eval()
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    print(f"[shard {args.shard_id}] {args.model_id} hidden={model.config.hidden_size} "
          f"proj={model.config.projection_dim} dtype={dtype}")

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(device)
    )

    with open(args.metadata_csv) as f:
        rows = [r for i, r in enumerate(csv.DictReader(f)) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}] {len(rows)} clips")

    ok = fail = skip = nan_fail = 0
    for n, row in enumerate(rows):
        gen = row["generative_method"]
        name = os.path.basename(row["video_path"]) + ".pt"
        out_path = os.path.join(args.output_dir, gen, name)
        if os.path.exists(out_path):
            skip += 1
            continue
        try:
            crops = get_crops(row, fa, device, args.all_frames, args.num_frames,
                              args.landmark_cache)
            if crops is None or crops.shape[0] < 2:
                fail += 1
                continue
            video, hidden = encode(model, crops, mean, std, device)
            if not (torch.isfinite(video).all() and torch.isfinite(hidden).all()):
                nan_fail += 1
                fail += 1
                continue

            audio_path = os.path.join(args.audio_dir, gen, name)
            if not os.path.exists(audio_path):
                fail += 1
                continue
            audio = torch.load(audio_path, weights_only=True)["audio"]

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({"video": video, "video_hidden": hidden, "audio": audio}, out_path)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"[shard {args.shard_id}] FAIL {row['video_path']}: {type(e).__name__}: {e}")
        if (n + 1) % 500 == 0:
            print(f"[shard {args.shard_id}] {n+1}/{len(rows)} ok={ok} fail={fail} skip={skip}", flush=True)

    print(f"[shard {args.shard_id}] DONE ok={ok} fail={fail} skip={skip} nan_fail={nan_fail}")


if __name__ == "__main__":
    main()
