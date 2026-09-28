import csv
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class RavenEmbeddingDataset(Dataset):

    def __init__(self, metadata_csv: str, embeddings_dir: str, modality: str = "fusion",
                 l2_branches: bool = False, video_key: str = "video"):
        assert modality in ("fusion", "video", "audio")
        self.modality = modality
        self.l2_branches = l2_branches
        self.video_key = video_key
        self.dirs = [d for d in embeddings_dir.split(",") if d]
        self.embeddings_dir = self.dirs[0]

        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))

        self.samples = []
        num_missing = 0
        for r in rows:
            rel = os.path.join(r["generative_method"], os.path.basename(r["video_path"]) + ".pt")
            paths = [os.path.join(d, rel) for d in self.dirs]
            if not all(os.path.exists(pth) for pth in paths):
                num_missing += 1
                continue
            self.samples.append(
                {
                    "path": paths[0],
                    "paths": paths,
                    "label": int(r["label_id"]),
                    "class_name": r["generative_method"],
                    "language": r["language"],
                }
            )

        if num_missing:
            print(f"  [{os.path.basename(metadata_csv)}] {num_missing} rows missing embeddings, skipped")
        print(f"  [{os.path.basename(metadata_csv)}] loaded {len(self.samples)} samples ({modality})")

    def __len__(self):
        return len(self.samples)

    def _norm(self, v):
        return torch.nn.functional.normalize(v, p=2, dim=0) if self.l2_branches else v

    def _load_embedding(self, paths):
        vids, audio = [], None
        for i, pth in enumerate(paths):
            d = torch.load(pth, weights_only=True)
            key = self.video_key if self.video_key in d else "video"
            vids.append(self._norm(d[key]))
            if i == 0:
                audio = self._norm(d["audio"])
        video = torch.cat(vids, dim=0) if len(vids) > 1 else vids[0]
        if self.modality == "video":
            return video
        if self.modality == "audio":
            return audio
        return torch.cat([video, audio], dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        emb = self._load_embedding(sample["paths"])
        return emb, sample["label"], sample["class_name"]


class LayerSweepEmbeddingDataset(Dataset):

    def __init__(self, metadata_csv: str, embeddings_dir: str, layer: int, modality: str = "fusion"):
        assert modality in ("fusion", "video", "audio")
        self.modality = modality
        self.layer = layer
        self._layer_idx_cache = None

        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))

        self.samples = []
        num_missing = 0
        for r in rows:
            emb_path = os.path.join(
                embeddings_dir, r["generative_method"], os.path.basename(r["video_path"]) + ".pt"
            )
            if not os.path.exists(emb_path):
                num_missing += 1
                continue
            self.samples.append(
                {"path": emb_path, "label": int(r["label_id"]), "class_name": r["generative_method"]}
            )

        if num_missing:
            print(f"  [{os.path.basename(metadata_csv)}] {num_missing} rows missing embeddings, skipped")
        print(f"  [{os.path.basename(metadata_csv)}] loaded {len(self.samples)} samples (layer {layer}, {modality})")

    def __len__(self):
        return len(self.samples)

    def _pick_layer(self, tensor_LD, layers_list):
        if self._layer_idx_cache is None:
            self._layer_idx_cache = layers_list.index(self.layer)
        vec = tensor_LD[self._layer_idx_cache]
        return torch.nn.functional.normalize(vec, p=2, dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        d = torch.load(sample["path"], weights_only=True)
        layers_list = d["layers"]
        if self.modality == "video":
            emb = self._pick_layer(d["video"], layers_list)
        elif self.modality == "audio":
            emb = self._pick_layer(d["audio"], layers_list)
        else:
            emb = torch.cat([self._pick_layer(d["video"], layers_list), self._pick_layer(d["audio"], layers_list)])
        return emb, sample["label"], sample["class_name"]


class AllLayersEmbeddingDataset(Dataset):

    def __init__(self, metadata_csv: str, embeddings_dir: str):
        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))

        self.samples = []
        num_missing = 0
        for r in rows:
            emb_path = os.path.join(
                embeddings_dir, r["generative_method"], os.path.basename(r["video_path"]) + ".pt"
            )
            if not os.path.exists(emb_path):
                num_missing += 1
                continue
            self.samples.append(
                {"path": emb_path, "label": int(r["label_id"]), "class_name": r["generative_method"]}
            )

        if num_missing:
            print(f"  [{os.path.basename(metadata_csv)}] {num_missing} rows missing embeddings, skipped")
        print(f"  [{os.path.basename(metadata_csv)}] loaded {len(self.samples)} samples (all layers)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        d = torch.load(sample["path"], weights_only=True)
        video = torch.nn.functional.normalize(d["video"], p=2, dim=1)
        audio = torch.nn.functional.normalize(d["audio"], p=2, dim=1)
        return video, audio, sample["label"], sample["class_name"]


def create_all_layers_loaders(metadata_dir: str, embeddings_dir: str, batch_size: int = 64, num_workers: int = 4):
    buckets = [
        "train", "validation", "test_indomain",
    ]
    loaders = {}
    for name in buckets:
        csv_path = os.path.join(metadata_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        ds = AllLayersEmbeddingDataset(csv_path, embeddings_dir)
        shuffle = name == "train"
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders


def create_layer_sweep_loaders(
    metadata_dir: str, embeddings_dir: str, layer: int, modality: str = "fusion",
    batch_size: int = 64, num_workers: int = 4,
) -> Dict[str, DataLoader]:
    buckets = [
        "train", "validation", "test_indomain",
    ]
    loaders = {}
    for name in buckets:
        csv_path = os.path.join(metadata_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        ds = LayerSweepEmbeddingDataset(csv_path, embeddings_dir, layer, modality)
        shuffle = name == "train"
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders


class FrameLevelDataset(Dataset):

    def __init__(self, metadata_csv: str, embeddings_dir: str):
        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))

        self.frames = []
        num_missing = 0
        for r in rows:
            path = os.path.join(
                embeddings_dir, r["generative_method"], os.path.basename(r["video_path"]) + ".pt"
            )
            if not os.path.exists(path):
                num_missing += 1
                continue
            d = torch.load(path, weights_only=True)
            video, audio = d["video"], d["audio"]
            T = min(video.shape[0], audio.shape[0])
            if T < 1:
                continue
            fused = torch.cat([video[:T], audio[:T]], dim=1)
            label, cls = int(r["label_id"]), r["generative_method"]
            for t in range(T):
                self.frames.append((fused[t], label, cls))

        if num_missing:
            print(f"  [{os.path.basename(metadata_csv)}] {num_missing} rows missing embeddings, skipped")
        print(f"  [{os.path.basename(metadata_csv)}] loaded {len(self.frames)} frame-level "
              f"samples from {len(rows) - num_missing} clips")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


def create_frame_level_loaders(metadata_dir: str, embeddings_dir: str, batch_size: int = 256, num_workers: int = 4):
    loaders = {}
    for name in ("train", "validation"):
        csv_path = os.path.join(metadata_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        ds = FrameLevelDataset(csv_path, embeddings_dir)
        shuffle = name == "train"
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders


def make_loader(metadata_csv, embeddings_dir, modality, batch_size, shuffle, num_workers=4, l2_branches=False, video_key="video"):
    ds = RavenEmbeddingDataset(metadata_csv, embeddings_dir, modality=modality, l2_branches=l2_branches, video_key=video_key)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=num_workers, pin_memory=True,
    )


def create_data_loaders(
    metadata_dir: str,
    embeddings_dir: str,
    modality: str = "fusion",
    batch_size: int = 256,
    num_workers: int = 4,
    l2_branches: bool = False,
    video_key: str = "video",
) -> Dict[str, DataLoader]:
    buckets = [
        "train", "validation", "test_indomain",
    ]
    loaders = {}
    for name in buckets:
        csv_path = os.path.join(metadata_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        shuffle = name == "train"
        loaders[name] = make_loader(
            csv_path, embeddings_dir, modality, batch_size, shuffle, num_workers, l2_branches,
            video_key,
        )
    return loaders
