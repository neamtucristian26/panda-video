import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_model

BRANCH_ORDER = ["clip", "dinov3", "braven_video", "braven_audio"]
BRANCH_DIMS = {"clip": 768, "dinov3": 1024, "braven_video": 1024, "braven_audio": 1024}
PROXY_ANCHOR_ALPHA = 32.0


def load_attribution_head(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    lin = sorted((int(k.split(".")[2]), v.shape) for k, v in sd.items()
                 if k.startswith("projection_net.projection.") and k.endswith(".weight")
                 and v.dim() == 2)
    input_dim = lin[0][1][1]
    hidden_dims = [shape[0] for _, shape in lin[:-1]]
    embedding_dim = lin[-1][1][0]

    expected = sum(BRANCH_DIMS[b] for b in BRANCH_ORDER)
    if input_dim != expected:
        raise SystemExit(f"checkpoint expects {input_dim}-d input but the branch layout here "
                         f"sums to {expected}-d; they must agree")

    model = create_model(input_dim=input_dim, hidden_dims=hidden_dims,
                         embedding_dim=embedding_dim, num_classes=num_classes)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, input_dim, hidden_dims, embedding_dim


def load_vision_encoders(device, clip_fp32=False):
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
    import face_alignment

    from extract_dinov3_embeddings import DEFAULT_MODEL as DINO_ID
    from extract_clip_embeddings import DEFAULT_MODEL as CLIP_ID

    dproc = AutoImageProcessor.from_pretrained(DINO_ID)
    dino = AutoModel.from_pretrained(DINO_ID, dtype=torch.float32).to(device).eval()

    cdtype = torch.float32 if clip_fp32 else torch.float16
    cproc = AutoImageProcessor.from_pretrained(CLIP_ID)
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, dtype=cdtype).to(device).eval()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                      flip_input=False, device=str(device))
    stats = {
        "dino": (torch.tensor(dproc.image_mean).view(1, 3, 1, 1),
                 torch.tensor(dproc.image_std).view(1, 3, 1, 1)),
        "clip": (torch.tensor(cproc.image_mean).view(1, 3, 1, 1),
                 torch.tensor(cproc.image_std).view(1, 3, 1, 1)),
    }
    return dino, clip, fa, stats


def load_braven(raven_repo, ckpt_dir, video_cfg, audio_cfg, device):
    raven_repo = os.path.abspath(raven_repo)
    if not os.path.isdir(raven_repo):
        raise SystemExit(f"RAVEn repo not found at {raven_repo} -- clone "
                         f"https://github.com/ahaliassos/raven there, or pass --raven_repo")
    sys.path.insert(0, raven_repo)
    try:
        from extract_raven_embeddings import load_encoder
    except ModuleNotFoundError as e:
        raise SystemExit(f"could not import the BRAVEn extractor ({e}). Expected espnet inside "
                         f"{raven_repo}.")

    cfg = lambda p, d: p if p else os.path.join(raven_repo, d)
    vid = load_encoder(cfg(video_cfg, "conf/model/visual_backbone/resnet_transformer_large.yaml"),
                       os.path.join(ckpt_dir, "video.pth"), device)
    aud = load_encoder(cfg(audio_cfg, "conf/model/audio_backbone/resnet_transformer_large.yaml"),
                       os.path.join(ckpt_dir, "audio.pth"), device)
    return vid, aud


def embed_one(path, fa, dino, clip, stats, braven, device, tmp_wav, num_frames, all_frames):
    from extract_dinov3_embeddings import get_crops, encode as dino_encode
    from extract_clip_embeddings import encode as clip_encode
    from extract_raven_embeddings import process_one

    row = {"abs_path": path, "video_path": os.path.basename(path),
           "generative_method": "unknown"}

    crops = get_crops(row, fa, device, all_frames, num_frames, cache_dir=None)
    if crops is None or len(crops) == 0:
        return None, "no face detected in any frame"

    dmean, dstd = stats["dino"]
    cmean, cstd = stats["clip"]
    with torch.no_grad():
        dino_cls, _ = dino_encode(dino, crops, dmean, dstd, device)
        clip_proj, _ = clip_encode(clip, crops, cmean, cstd, device)

    vid_enc, aud_enc = braven
    out = process_one(row, fa, vid_enc, aud_enc, device, tmp_wav)
    if out is None:
        return None, "BRAVEn could not extract a mouth ROI or the audio track was too short"

    parts = {"clip": clip_proj, "dinov3": dino_cls,
             "braven_video": out["video"], "braven_audio": out["audio"]}
    for name, v in parts.items():
        if v.shape[-1] != BRANCH_DIMS[name]:
            return None, f"{name} produced {v.shape[-1]}-d, expected {BRANCH_DIMS[name]}-d"
        if not torch.isfinite(v).all():
            return None, f"{name} produced non-finite values"

    vec = torch.cat([F.normalize(parts[b].float().flatten(), p=2, dim=0) for b in BRANCH_ORDER])
    return vec, None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    videos = list(args.videos)
    if args.list:
        with open(args.list) as f:
            videos += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    missing = [v for v in videos if not os.path.isfile(v)]
    if missing:
        raise SystemExit("not found: " + ", ".join(missing))
    if not videos:
        raise SystemExit("no input videos given (pass paths, or --list FILE)")

    with open(os.path.join(args.metadata_dir, "model_to_label.json")) as f:
        model_to_label = json.load(f)
    names = [None] * len(model_to_label)
    for n, i in model_to_label.items():
        names[i] = n

    head, in_dim, hid, emb = load_attribution_head(args.checkpoint, len(names), device)
    print(f"head: {in_dim} -> {hid} -> {emb}, {len(names)} classes  [{device}]")

    dino, clip, fa, stats = load_vision_encoders(device, clip_fp32=args.clip_fp32)
    braven = load_braven(args.raven_repo, args.braven_ckpt, args.video_cfg, args.audio_cfg, device)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    results, failures = [], []
    try:
        for path in videos:
            vec, err = embed_one(path, fa, dino, clip, stats, braven, device, tmp_wav,
                                 args.num_frames, not args.uniform_frames)
            if vec is None:
                failures.append({"video": path, "error": err})
                print(f"\n{os.path.basename(path)}\n  SKIPPED: {err}")
                continue

            with torch.no_grad():
                proj, _ = head(vec.unsqueeze(0).to(device))
                cos = head.get_logits(proj)[0].cpu() * head.temperature
            prob = torch.softmax(cos * PROXY_ANCHOR_ALPHA, dim=0)
            order = torch.argsort(cos, descending=True)

            top = order[0].item()
            results.append({
                "video": path,
                "prediction": names[top],
                "probability": float(prob[top]),
                "cosine": float(cos[top]),
                "margin": float(cos[order[0]] - cos[order[1]]),
                "scores": {names[i]: {"cosine": float(cos[i]), "probability": float(prob[i])}
                           for i in range(len(names))},
            })
            margin = float(cos[order[0]] - cos[order[1]])
            print(f"\n{os.path.basename(path)}")
            for rank, i in enumerate(order[:args.topk]):
                lead = "  ->" if rank == 0 else "    "
                extra = f"   margin {margin:+.4f}" if rank == 0 else ""
                print(f"{lead} {names[i]:<13} cos={cos[i]:+.4f}{extra}")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    print(f"\n{len(results)} predicted, {len(failures)} skipped")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results, "failures": failures,
                       "checkpoint": args.checkpoint, "classes": names,
                       "branch_order": BRANCH_ORDER}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Predict which generator produced a deepfake video clip.")
    ap.add_argument("videos", nargs="*", help="video file(s)")
    ap.add_argument("--list", help="text file with one video path per line")
    ap.add_argument("--checkpoint", default="model/tri_h2048_1024/best_model.pth")
    ap.add_argument("--metadata_dir", default="metadata_all7",
                    help="only for model_to_label.json, i.e. the class names")
    ap.add_argument("--braven_ckpt", required=True,
                    help="directory holding BRAVEn's video.pth and audio.pth")
    ap.add_argument("--raven_repo", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "raven_repo"))
    ap.add_argument("--video_cfg", default=None, help="override the BRAVEn visual backbone yaml")
    ap.add_argument("--audio_cfg", default=None, help="override the BRAVEn audio backbone yaml")
    ap.add_argument("--num_frames", type=int, default=16,
                    help="frames to sample when --uniform_frames is set")
    ap.add_argument("--uniform_frames", action="store_true",
                    help="sample --num_frames uniformly instead of reading all frames; the "
                         "released model was trained on all-frame features")
    ap.add_argument("--clip_fp32", action="store_true", help="run CLIP in fp32 instead of fp16")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", help="also write predictions to this JSON file")
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
