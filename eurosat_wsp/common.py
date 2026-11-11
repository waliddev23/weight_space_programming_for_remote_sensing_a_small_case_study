"""Shared utilities for the EuroSAT weight-space-programming (WSP) experiments.

Everything the three experiments need lives here so that every model sees exactly
the same data, the same architecture and the same training procedure:

* dataset loading from the official `blanchon/EuroSAT_RGB` splits (no re-splitting),
* a deterministic in-memory cache of the images (uint8, native 64x64),
* the three augmentation policies (none / standard / strong),
* the small CNN used by Experiment 1 (READ) and Experiment 3 (WRITE),
* seeding, one train loop, one evaluation function, checkpoint helpers.

Nothing in this module talks to an LLM; that is deliberately out of scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- paths

WSP_ROOT = Path(__file__).resolve().parent          # .../eurosat_wsp
PROJECT_ROOT = WSP_ROOT.parent                      # repo root
DATA_DIR = WSP_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

EXP1_DIR = WSP_ROOT / "experiment_1_read"
EXP2_DIR = WSP_ROOT / "experiment_2_edit"
EXP3_DIR = WSP_ROOT / "experiment_3_write"

# ----------------------------------------------------------------------- constants

HF_DATASET = "blanchon/EuroSAT_RGB"
IMG_SIZE = 64                 # native EuroSAT resolution; nothing is upscaled
NUM_CLASSES = 10

CLASS_NAMES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "SeaLake",
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ImageNet statistics, matching the notebooks in ../notebooks. Used for both the
# from-scratch CNN and the pretrained ResNet-18 so the split files stay shared.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Small CNN: first conv is fixed at 3 -> 16 for Experiment 3's 432-weight tensor.
CHANNELS = (16, 32, 64, 128)
FIRST_CONV_OUT = CHANNELS[0]

# Single training configuration shared by every Experiment 1 model.
TRAIN_CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR (T_max = epochs)",
    "dropout": 0.2,
    "seeds": [0, 1, 2],
    "num_workers": 0,
    "amp": "torch.autocast(cuda) when CUDA is available",
}

# ------------------------------------------------------------------------ seeding


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------- data


def load_raw():
    """Load the Hugging Face dataset (official train/validation/test splits)."""
    from datasets import load_dataset

    return load_dataset(HF_DATASET)


def preload_split(hf_split) -> tuple[np.ndarray, np.ndarray]:
    """Decode one HF split into uint8 arrays: x (N,64,64,3), y (N,)."""
    n = len(hf_split)
    x = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        item = hf_split[i]
        x[i] = np.asarray(item["image"].convert("RGB"), dtype=np.uint8)
        y[i] = int(item["label"])
    return x, y


def preload_all(raw) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {split: preload_split(raw[split]) for split in ("train", "validation", "test")}


class ArrayDataset(Dataset):
    """(uint8 HWC image, label) dataset with a torchvision transform pipeline."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.x[int(idx)])       # uint8 HWC -> PIL RGB
        return self.transform(image), int(self.y[int(idx)])


# Augmentation policies. Only these change between Experiment 1 conditions.
AUG_POLICIES: dict[str, list] = {
    "none": [],
    "standard": [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
    ],
    "strong": [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0), ratio=(0.7, 1.4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    ],
}


def describe_policy(policy: str) -> list[str]:
    """Human-readable list of the transforms in a policy (for the config files)."""
    return [type(t).__name__ + ("" if not hasattr(t, "p") else f"(p={t.p})")
            for t in AUG_POLICIES[policy]]


def build_train_transform(policy: str):
    if policy not in AUG_POLICIES:
        raise ValueError(f"unknown augmentation policy {policy!r}; "
                         f"choose from {sorted(AUG_POLICIES)}")
    return transforms.Compose(
        AUG_POLICIES[policy]
        + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def build_eval_transform():
    """No augmentation: the patches are already 64x64."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool,
                seed: int = 0, num_workers: int = 0) -> DataLoader:
    """DataLoader with a seeded shuffle order (identical across conditions)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ------------------------------------------------------------------------- model


def conv_block(cin: int, cout: int) -> nn.Sequential:
    """3x3 conv -> BatchNorm -> ReLU -> 2x2 max-pool (stride 2)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four conv stages + global average pooling + dropout + linear head.

    ``conv1`` is deliberately kept separate from ``features`` because Experiment 3
    treats it as a standalone 16x3x3x3 (432-weight, 16-bias) tensor.

    Channels (16, 32, 64, 128) -> 98,954 parameters.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, channels: Sequence[int] = CHANNELS,
                 dropout: float = 0.2):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.dropout = float(dropout)
        c1, c2, c3, c4 = self.channels

        # First conv keeps a bias so its tensor matches the WRITE-experiment spec
        # exactly: 16 x 3 x 3 x 3 weights + 16 biases.
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=True)
        self.features = nn.Sequential(
            conv_block(c1, c2),
            conv_block(c2, c3),
            conv_block(c3, c4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(c4, num_classes),
        )
        self.reset_parameters()

    # -- initialization -----------------------------------------------------
    def reset_parameters(self) -> None:
        """Documented Kaiming initialization (see experiment_3_write/architecture).

        * conv weights: Kaiming normal, fan_out, relu
        * conv biases: zeros
        * BatchNorm: weight 1, bias 0
        * classifier: PyTorch default (Kaiming uniform a=sqrt(5) + uniform bias)
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        head = self.classifier[-1]
        nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(head.in_features)
        nn.init.uniform_(head.bias, -bound, bound)

    # -- forward ------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """128-d penultimate descriptor (global average pooled conv features)."""
        h = self.conv1(x)
        h = self.features(h)
        return self.pool(h).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(self.conv1(x))))


def build_smallcnn(seed: int | None = None, num_classes: int = NUM_CLASSES,
                   channels: Sequence[int] = CHANNELS, dropout: float = 0.2) -> SmallCNN:
    """Build the shared architecture; if ``seed`` is given the init is reproducible."""
    if seed is not None:
        set_seed(seed)
    model = SmallCNN(num_classes=num_classes, channels=channels, dropout=dropout)
    if seed is not None:
        model.reset_parameters()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------- train / evaluate


def make_optimizer(model: nn.Module, lr: float = 1e-3,
                   weight_decay: float = 5e-4) -> torch.optim.AdamW:
    """Decay conv/linear weights; leave BatchNorm weights and biases alone."""
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp: bool) -> tuple[float, float]:
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)

    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = False):
    """Return (loss, accuracy, predictions, targets)."""
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        seen += x.size(0)
        preds.append(pred.cpu())
        targets.append(y.cpu())

    return loss_sum / seen, correct / seen, torch.cat(preds).numpy(), torch.cat(targets).numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy + macro/weighted F1 + per-class P/R/F1 + the confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
        digits=4, zero_division=0, output_dict=True,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


# --------------------------------------------------------------------- checkpoints


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Plain ``{name: tensor}`` on CPU - no metadata, ready for the LLM later."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_state_dict(state: dict[str, torch.Tensor], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_state_dict(path: str | Path, map_location="cpu") -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict):
        data = data["model"]
    return data


def load_smallcnn(path: str | Path, device: torch.device | None = None,
                  channels: Sequence[int] = CHANNELS) -> SmallCNN:
    model = build_smallcnn(seed=None, channels=channels)
    model.load_state_dict(load_state_dict(path))
    model.to(device or get_device()).eval()
    return model


# ------------------------------------------------------------------------ helpers


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(payload, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std()) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
        "abs_mean": float(t.abs().mean()),
        "frac_zero": float((t == 0).float().mean()),
    }


def first_conv_weight(model: nn.Module) -> torch.Tensor:
    return model.conv1.weight.detach().cpu()


def apply_external_first_conv(model: nn.Module, weights) -> None:
    """Initialization interface: drop externally generated weights into ``conv1``.

    Accepts a (16, 3, 3, 3) tensor / ndarray, or a state_dict whose only (or
    matching) key is a first-conv weight tensor. Validates shape before writing.
    """
    if isinstance(weights, dict):
        candidates = [
            v for k, v in weights.items()
            if isinstance(v, torch.Tensor) and tuple(v.shape) == (FIRST_CONV_OUT, 3, 3, 3)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one (16, 3, 3, 3) tensor in the state_dict, "
                f"found {len(candidates)}"
            )
        weights = candidates[0]

    t = torch.as_tensor(weights, dtype=torch.float32)
    expected = tuple(model.conv1.weight.shape)
    if tuple(t.shape) != expected:
        raise ValueError(f"first-conv weights must have shape {expected}, got {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("first-conv weights contain NaN or Inf")

    with torch.no_grad():
        model.conv1.weight.copy_(t.to(model.conv1.weight.device))


def load_external_first_conv(path: str | Path) -> torch.Tensor:
    """Load an externally produced first-conv tensor from .pt/.pth/.npy/.npz."""
    path = Path(path)
    if path.suffix in {".npy"}:
        t = torch.from_numpy(np.load(path))
    elif path.suffix in {".npz"}:
        arrays = np.load(path)
        t = torch.from_numpy(arrays[arrays.files[0]])
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            candidates = [v for v in obj.values()
                          if isinstance(v, torch.Tensor)
                          and tuple(v.shape) == (FIRST_CONV_OUT, 3, 3, 3)]
            if len(candidates) != 1:
                raise ValueError(f"{path}: no unique (16, 3, 3, 3) tensor found")
            t = candidates[0]
        else:
            t = obj
    return torch.as_tensor(t, dtype=torch.float32)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
