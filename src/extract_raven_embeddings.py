import argparse
import csv
import os
import subprocess
import sys
import tempfile
import warnings

import cv2
import numpy as np
import torch

torch.set_num_threads(2)
cv2.setNumThreads(2)

RAVEN_REPO = os.path.join(os.path.dirname(__file__), "..", "raven_repo")
sys.path.insert(0, RAVEN_REPO)

from omegaconf import OmegaConf
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

VIDEO_MEAN, VIDEO_STD = 0.421, 0.165
CROP_SIZE, OUT_SIZE = 96, 88
TARGET_SR = 16000
MAX_FRAMES = 375


def load_encoder(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    cfg.ddim, cfg.dheads, cfg.dunits = cfg.adim, cfg.aheads, cfg.eunits
    model = E2E(1003, cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(state, strict=True)
    model.encoder.eval().to(device)
    return model.encoder


def read_frames(path, max_frames=MAX_FRAMES, stride=1):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def mouth_crop(frames, fa, device, crop_size=CROP_SIZE, out_size=OUT_SIZE, batch_size=32, detect_max_dim=160):
    off = (crop_size - out_size) // 2
    half = crop_size // 2

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H0, W0 = grays[0].shape
    scale = detect_max_dim / max(H0, W0) if max(H0, W0) > detect_max_dim else 1.0
    if min(H0, W0) * scale < 64:
        scale = min(1.0, 64.0 / min(H0, W0))

    small_tensors = []
    for f in frames:
        small = cv2.resize(f, (int(W0 * scale), int(H0 * scale))) if scale != 1.0 else f
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        small_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))

    crops = []
    prev_center = None
    for start in range(0, len(small_tensors), batch_size):
        chunk = torch.stack(small_tensors[start:start + batch_size]).to(device)
        with torch.no_grad():
            preds = fa.get_landmarks_from_batch(chunk)
        for i, lm in enumerate(preds):
            if lm is None or len(lm) == 0:
                continue
            arr = np.asarray(lm, dtype=np.float32).reshape(-1, 2)
            faces = arr.reshape(-1, 68, 2) if arr.shape[0] % 68 == 0 else arr[None, :, :]
            if prev_center is None:
                sizes = [(f.max(axis=0) - f.min(axis=0)).max() for f in faces]
                face = faces[int(np.argmax(sizes))]
            else:
                cents = np.stack([f.mean(axis=0) for f in faces])
                face = faces[int(np.argmin(np.linalg.norm(cents - prev_center, axis=1)))]
            prev_center = face.mean(axis=0)

            mouth = face[48:68] / scale
            cx, cy = mouth.mean(axis=0)
            gray = grays[start + i]
            cx = min(max(cx, half), W0 - half)
            cy = min(max(cy, half), H0 - half)
            patch = gray[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
            if patch.shape != (crop_size, crop_size):
                continue
            crops.append(patch[off:off + out_size, off:off + out_size])
    return np.stack(crops) if crops else None


def extract_audio(path, tmp_wav, target_sr=TARGET_SR):
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), tmp_wav],
        check=True, capture_output=True,
    )
    import librosa
    audio, _ = librosa.load(tmp_wav, sr=target_sr)
    return audio


def encode_all_layers(encoder, x, sweep_layers):
    with torch.no_grad():
        all_feats = encoder(x, None, return_feats="all")
    pooled = []
    for layer in sweep_layers:
        feat = all_feats[layer - 1].mean(dim=1).squeeze(0).cpu()
        pooled.append(feat)
    return torch.stack(pooled)


def process_one(row, fa, video_encoder, audio_encoder, device, tmp_wav, stride=1, sweep_layers=None, layer=None, per_frame=False):
    frames = read_frames(row["abs_path"], stride=stride)
    if not frames:
        return None
    crops = mouth_crop(frames, fa, device)
    if crops is None or crops.shape[0] < 2:
        return None

    x = torch.from_numpy(crops).float().unsqueeze(0).to(device)
    x = (x / 255.0 - VIDEO_MEAN) / VIDEO_STD

    audio = extract_audio(row["abs_path"], tmp_wav)
    if len(audio) < TARGET_SR // 25:
        return None
    xa = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)

    if sweep_layers:
        video_emb = encode_all_layers(video_encoder, x, sweep_layers)
        audio_emb = encode_all_layers(audio_encoder, xa, sweep_layers)
        return {"video": video_emb, "audio": audio_emb, "layers": sweep_layers}

    if layer:
        video_emb = encode_all_layers(video_encoder, x, [layer])[0]
        audio_emb = encode_all_layers(audio_encoder, xa, [layer])[0]
        return {"video": video_emb, "audio": audio_emb}

    with torch.no_grad():
        vfeat, _ = video_encoder(x, None)
    with torch.no_grad():
        afeat, _ = audio_encoder(xa, None)

    if per_frame:
        return {"video": vfeat.squeeze(0).cpu(), "audio": afeat.squeeze(0).cpu()}

    video_emb = vfeat.mean(dim=1).squeeze(0).cpu()
    audio_emb = afeat.mean(dim=1).squeeze(0).cpu()
    return {"video": video_emb, "audio": audio_emb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_dir", default="checkpoints/braven_baseplus_lrs3vox2")
    parser.add_argument(
        "--video_cfg",
        default="raven_repo/conf/model/visual_backbone/resnet_transformer_baseplus.yaml",
    )
    parser.add_argument(
        "--audio_cfg",
        default="raven_repo/conf/model/audio_backbone/resnet_transformer_baseplus.yaml",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--frame_stride", type=int, default=1,
        help="Process every Nth frame for landmark/mouth-crop extraction (speed/coverage tradeoff)",
    )
    parser.add_argument(
        "--sweep_layers", type=int, nargs="*", default=None,
        help="1-indexed encoder layers to extract (e.g. 4 8 12 16 20 24) instead of the "
             "final layer. Saves all requested layers' pooled embeddings per clip for a "
             "layer-selection sweep -- one extra forward pass cost, not one per layer.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Extract this single 1-indexed layer instead of the final layer (e.g. the "
             "layer-sweep winner). Flat (D,) output, drop-in compatible with the default "
             "final-layer format -- mutually exclusive with --sweep_layers.",
    )
    parser.add_argument(
        "--per_frame", action="store_true",
        help="Keep the full per-frame (T,D) sequence (final layer) instead of mean-pooling "
             "over time -- for the score-averaged vs. feature-averaged pooling comparison. "
             "Mutually exclusive with --sweep_layers/--layer.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    with open(args.metadata_csv) as f:
        rows = list(csv.DictReader(f))
    rows = rows[args.shard_id::args.num_shards]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(rows)} rows on {device}")

    print("Loading encoders...")
    video_encoder = load_encoder(args.video_cfg, os.path.join(args.ckpt_dir, "video.pth"), device)
    audio_encoder = load_encoder(args.audio_cfg, os.path.join(args.ckpt_dir, "audio.pth"), device)

    import face_alignment
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(device)
        )

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    num_ok, num_fail = 0, 0
    for i, row in enumerate(rows):
        out_path = os.path.join(
            args.output_dir, row["generative_method"], os.path.basename(row["video_path"]) + ".pt"
        )
        if not args.no_resume and os.path.exists(out_path):
            continue
        try:
            result = process_one(row, fa, video_encoder, audio_encoder, device, tmp_wav,
                                  stride=args.frame_stride, sweep_layers=args.sweep_layers, layer=args.layer,
                                  per_frame=args.per_frame)
            if result is None:
                num_fail += 1
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(result, out_path)
            num_ok += 1
        except Exception as e:
            num_fail += 1
            print(f"  FAILED {row['abs_path']}: {e}")

        if (i + 1) % 25 == 0:
            print(f"[shard {args.shard_id}] {i + 1}/{len(rows)}  ok={num_ok} fail={num_fail}")

    os.remove(tmp_wav)
    print(f"[shard {args.shard_id}] DONE. ok={num_ok} fail={num_fail}")


if __name__ == "__main__":
    main()
