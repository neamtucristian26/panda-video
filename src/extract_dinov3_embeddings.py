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

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_SIZE = 224
NUM_FRAMES = 16
DETECT_MAX_DIM = 160


def read_frames_uniform(path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        idxs = np.linspace(0, len(frames) - 1, min(num_frames, len(frames))).astype(int)
        return [frames[i] for i in idxs]

    idxs = np.linspace(0, total - 1, min(num_frames, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


def read_frames_all(path, max_frames=375):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def detect_faces(frames, fa, device, batch_size=16, detect_max_dim=DETECT_MAX_DIM):
    if not frames:
        return None, None
    H0, W0 = frames[0].shape[:2]
    scale = detect_max_dim / max(H0, W0) if max(H0, W0) > detect_max_dim else 1.0
    if min(H0, W0) * scale < 64:
        scale = min(1.0, 64.0 / min(H0, W0))

    small = []
    for f in frames:
        s = cv2.resize(f, (int(W0 * scale), int(H0 * scale))) if scale != 1.0 else f
        small.append(torch.from_numpy(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))

    pts_out = np.zeros((len(frames), 68, 2), dtype=np.float32)
    ok = np.zeros(len(frames), dtype=bool)
    prev_center = None
    for start in range(0, len(small), batch_size):
        chunk = torch.stack(small[start:start + batch_size]).to(device)
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
            pts_out[start + i] = face / scale
            ok[start + i] = True
    return pts_out, ok


def crops_from_landmarks(frames, pts, ok, out_size=IMG_SIZE, expand=1.6, up_shift=0.15):
    H0, W0 = frames[0].shape[:2]
    crops = []
    for i, f in enumerate(frames):
        if i >= len(ok) or not ok[i]:
            continue
        p = pts[i]
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = max(x1 - x0, y1 - y0) * expand
        cy -= side * up_shift
        half = side / 2.0
        xa, ya = int(round(cx - half)), int(round(cy - half))
        xb, yb = int(round(cx + half)), int(round(cy + half))
        pl, pt_ = max(0, -xa), max(0, -ya)
        pr, pb = max(0, xb - W0), max(0, yb - H0)
        xa, ya, xb, yb = max(0, xa), max(0, ya), min(W0, xb), min(H0, yb)
        if xb - xa < 8 or yb - ya < 8:
            continue
        patch = f[ya:yb, xa:xb]
        if pl or pt_ or pr or pb:
            patch = cv2.copyMakeBorder(patch, pt_, pb, pl, pr, cv2.BORDER_REPLICATE)
        patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
        crops.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    return np.stack(crops) if crops else None


def face_crops(frames, fa, device, out_size=IMG_SIZE, batch_size=16,
               detect_max_dim=DETECT_MAX_DIM, expand=1.6, up_shift=0.15):
    if not frames:
        return None
    H0, W0 = frames[0].shape[:2]
    scale = detect_max_dim / max(H0, W0) if max(H0, W0) > detect_max_dim else 1.0
    if min(H0, W0) * scale < 64:
        scale = min(1.0, 64.0 / min(H0, W0))

    small = []
    for f in frames:
        s = cv2.resize(f, (int(W0 * scale), int(H0 * scale))) if scale != 1.0 else f
        small.append(torch.from_numpy(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))

    crops = []
    prev_center = None
    for start in range(0, len(small), batch_size):
        chunk = torch.stack(small[start:start + batch_size]).to(device)
        with torch.no_grad():
            preds = fa.get_landmarks_from_batch(chunk)
        for i, lm in enumerate(preds):
            if lm is None or len(lm) == 0:
                continue
            arr = np.asarray(lm, dtype=np.float32).reshape(-1, 2)
            faces = arr.reshape(-1, 68, 2) if arr.shape[0] % 68 == 0 else arr[None, :, :]
            if prev_center is None:
                sizes = [(f.max(axis=0) - f.min(axis=0)).max() for f in faces]
                pts = faces[int(np.argmax(sizes))]
            else:
                cents = np.stack([f.mean(axis=0) for f in faces])
                pts = faces[int(np.argmin(np.linalg.norm(cents - prev_center, axis=1)))]
            prev_center = pts.mean(axis=0)

            pts = pts / scale
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            side = max(x1 - x0, y1 - y0) * expand
            cy -= side * up_shift
            half = side / 2.0

            xa, ya = int(round(cx - half)), int(round(cy - half))
            xb, yb = int(round(cx + half)), int(round(cy + half))
            pl, pt = max(0, -xa), max(0, -ya)
            pr, pb = max(0, xb - W0), max(0, yb - H0)
            xa, ya, xb, yb = max(0, xa), max(0, ya), min(W0, xb), min(H0, yb)
            if xb - xa < 8 or yb - ya < 8:
                continue
            patch = frames[start + i][ya:yb, xa:xb]
            if pl or pt or pr or pb:
                patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
            patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    return np.stack(crops) if crops else None


def get_crops(row, fa, device, all_frames, num_frames, cache_dir):
    frames = read_frames_all(row["abs_path"]) if all_frames else read_frames_uniform(row["abs_path"], num_frames)
    if not frames:
        return None
    cpath = None
    if cache_dir:
        cpath = os.path.join(cache_dir, row["generative_method"],
                             os.path.basename(row["video_path"]) + ".npz")
    if cpath and os.path.exists(cpath):
        z = np.load(cpath)
        pts, ok = z["pts"].astype(np.float32), z["ok"]
        n = min(len(frames), len(ok))
        if n == 0:
            return None
        return crops_from_landmarks(frames[:n], pts[:n], ok[:n])
    pts, ok = detect_faces(frames, fa, device)
    if pts is None:
        return None
    if cpath:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, pts=pts.astype(np.float32), ok=ok)
    return crops_from_landmarks(frames, pts, ok)

@torch.no_grad()
def encode(model, crops, mean, std, device, batch_size=32, per_frame=False):
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(crops).float().div_(255.0).permute(0, 3, 1, 2)
    x = (x - mean) / std
    n_reg = model.config.num_register_tokens

    cls_all, patch_all = [], []
    for s in range(0, x.shape[0], batch_size):
        out = model(pixel_values=x[s:s + batch_size].to(device, dtype))
        h = out.last_hidden_state.float()
        cls_all.append(h[:, 0, :].cpu())
        patch_all.append(h[:, 1 + n_reg:, :].mean(dim=1).cpu())

    cls = torch.cat(cls_all)
    patch = torch.cat(patch_all)
    if per_frame:
        return cls, patch
    return cls.mean(dim=0), patch.mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default="metadata/all_unique.csv")
    ap.add_argument("--output_dir", default="data/dinov3_embeddings")
    ap.add_argument("--audio_dir", default="data/raven_embeddings",
                    help="existing BRAVEn extraction; its 'audio' vector is copied through")
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--all_frames", action="store_true",
                    help="use every frame up to 375 (BRAVEn's exact policy) instead of "
                         "--num_frames uniform samples")
    ap.add_argument("--landmark_cache", default=None,
                    help="dir for cached per-clip landmarks, shared across video backbones")
    ap.add_argument("--per_frame", action="store_true")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoImageProcessor, AutoModel
    import face_alignment

    proc = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    print(f"[shard {args.shard_id}] {args.model_id} hidden={model.config.hidden_size} "
          f"patch={model.config.patch_size} registers={model.config.num_register_tokens}")

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
            cls, patch = encode(model, crops, mean, std, device, per_frame=args.per_frame)
            if not (torch.isfinite(cls).all() and torch.isfinite(patch).all()):
                nan_fail += 1
                fail += 1
                continue

            audio_path = os.path.join(args.audio_dir, gen, name)
            if not os.path.exists(audio_path):
                fail += 1
                continue
            audio = torch.load(audio_path, weights_only=True)["audio"]

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({"video": cls, "video_patchmean": patch, "audio": audio}, out_path)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"[shard {args.shard_id}] FAIL {row['video_path']}: {type(e).__name__}: {e}")
        if (n + 1) % 200 == 0:
            print(f"[shard {args.shard_id}] {n+1}/{len(rows)} ok={ok} fail={fail} skip={skip}", flush=True)

    print(f"[shard {args.shard_id}] DONE ok={ok} fail={fail} skip={skip} nan_fail={nan_fail}")


if __name__ == "__main__":
    main()
