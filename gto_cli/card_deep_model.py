from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEEP_MODEL_DIR = PROJECT_ROOT / "pict" / "card_models" / "deep"
RANK_LABELS = tuple("AKQJT98765432")
SUIT_LABELS = tuple("shdc")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_DEEP_MODEL_CACHE: dict[tuple[Path, str], Any | None] = {}


@dataclass(frozen=True)
class GlyphRecord:
    path: Path
    label: str
    label_index: int


def train_deep_card_classifier(
    *,
    glyph_dir: Path,
    extra_glyph_dirs: list[Path] | None = None,
    model_dir: Path = DEFAULT_DEEP_MODEL_DIR,
    kind: str,
    template_dir: Path | None = PROJECT_ROOT / "pict" / "card_templates",
    include_templates: bool = True,
    arch: str = "mobilenet_v3_small",
    pretrained: bool = False,
    epochs: int = 8,
    batch_size: int = 48,
    learning_rate: float = 3e-4,
    val_split: float = 0.18,
    max_images_per_class: int | None = None,
    seed: int = 17,
    image_size: int = 96,
    num_workers: int = 0,
    freeze_backbone: bool = False,
    class_balanced_loss: bool = False,
    weighted_sampler: bool = False,
) -> dict[str, Any]:
    if kind not in ("rank", "suit"):
        raise ValueError("kind must be rank or suit")
    if arch == "vit_b_16" and int(image_size) != 224:
        raise ValueError("torchvision vit_b_16 expects --image-size 224")
    torch, nn, optim, DataLoader = load_torch()
    labels = list(RANK_LABELS if kind == "rank" else SUIT_LABELS)
    records = collect_glyph_records(Path(glyph_dir), kind, labels, max_images_per_class=max_images_per_class)
    for extra_dir in extra_glyph_dirs or []:
        records.extend(collect_glyph_records(Path(extra_dir), kind, labels, max_images_per_class=max_images_per_class))
    template_records = collect_template_records(Path(template_dir), kind, labels) if include_templates and template_dir else []
    records.extend(template_records)
    if len({record.label for record in records}) < 2:
        raise ValueError(f"need at least 2 classes for {kind}, found {len(records)} images")

    train_records, val_records = stratified_split(records, val_split=val_split, seed=seed)
    if not val_records:
        val_records = train_records[:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, class_count=len(labels), pretrained=pretrained)
    if freeze_backbone:
        freeze_model_backbone(model, arch)
    model.to(device)
    train_dataset = GlyphDataset(train_records, image_size=image_size, train=True, pretrained=pretrained)
    val_dataset = GlyphDataset(val_records, image_size=image_size, train=False, pretrained=pretrained)
    train_sampler = build_weighted_sampler(torch, train_records) if weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=max(0, int(num_workers)),
    )
    val_loader = DataLoader(val_dataset, batch_size=max(1, int(batch_size)), shuffle=False, num_workers=max(0, int(num_workers)))
    class_weights = build_class_weights(torch, train_records, len(labels), device=device) if class_balanced_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_params or model.parameters(), lr=float(learning_rate), weight_decay=1e-4)

    best_state = None
    best_metric = -1.0
    history = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        train_loss, train_acc = run_epoch(torch, model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc = run_epoch(torch, model, val_loader, criterion, None, device=device)
        metric = val_acc - val_loss * 0.01
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_acc": round(float(train_acc), 6),
                "val_loss": round(float(val_loss), 6),
                "val_acc": round(float(val_acc), 6),
            }
        )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"deep_{kind}.pt"
    metadata = {
        "kind": kind,
        "arch": arch,
        "pretrained": bool(pretrained),
        "labels": labels,
        "image_size": int(image_size),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "source_count": len(records),
        "glyph_dir": str(glyph_dir),
        "extra_glyph_dirs": [str(path) for path in extra_glyph_dirs or []],
        "template_dir": str(template_dir) if template_dir else "",
        "template_count": len(template_records),
        "include_templates": bool(include_templates),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "val_split": float(val_split),
        "seed": int(seed),
        "num_workers": int(num_workers),
        "freeze_backbone": bool(freeze_backbone),
        "class_balanced_loss": bool(class_balanced_loss),
        "weighted_sampler": bool(weighted_sampler),
        "label_counts": count_labels(records, labels),
        "train_label_counts": count_labels(train_records, labels),
        "val_label_counts": count_labels(val_records, labels),
        "history": history,
    }
    torch.save({"state_dict": best_state or model.state_dict(), "metadata": metadata}, str(model_path))
    _DEEP_MODEL_CACHE.pop((model_dir.resolve(), kind), None)
    return {
        "ok": True,
        "kind": kind,
        "model": str(model_path),
        "metadata": metadata,
        "best_val_acc": max((item["val_acc"] for item in history), default=0.0),
        "last_val_acc": history[-1]["val_acc"] if history else 0.0,
    }


def classify_deep_glyph(
    image: Any,
    kind: str,
    *,
    model_dir: Path | None = None,
    allowed: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if kind not in ("rank", "suit"):
        raise ValueError("kind must be rank or suit")
    resolved_model_dir = resolve_deep_model_dir(model_dir, kind=kind)
    if resolved_model_dir is None:
        return None
    loaded = load_deep_model(resolved_model_dir, kind)
    if loaded is None:
        return None
    torch = loaded["torch"]
    model = loaded["model"]
    metadata = loaded["metadata"]
    labels = list(metadata["labels"])
    image_size = int(metadata.get("image_size") or 96)
    pretrained = bool(metadata.get("pretrained"))
    tensor = image_to_tensor(image, image_size=image_size, train=False, pretrained=pretrained).unsqueeze(0)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    if allowed is not None:
        allowed_set = set(allowed)
        candidates = [(idx, labels[idx], float(probs[idx])) for idx in range(len(labels)) if labels[idx] in allowed_set]
    else:
        candidates = [(idx, labels[idx], float(probs[idx])) for idx in range(len(labels))]
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: item[2], reverse=True)
    _idx, label, score = ordered[0]
    second_score = ordered[1][2] if len(ordered) > 1 else 0.0
    return {
        "label": label,
        "score": float(score),
        "margin": float(score - second_score),
        "second_score": float(second_score),
        "model": str(resolved_model_dir / f"deep_{kind}.pt"),
        "backend": "torch",
        "arch": metadata.get("arch"),
    }


def warm_deep_card_models(
    model_dir: Path | None = None,
    *,
    rank_model_dir: Path | None = None,
    suit_model_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_dirs = {
        "rank": resolve_deep_model_dir(rank_model_dir or model_dir, kind="rank"),
        "suit": resolve_deep_model_dir(suit_model_dir or model_dir, kind="suit"),
    }
    if not any(resolved_dirs.values()):
        return {"ok": True, "enabled": False, "loaded": []}
    loaded = []
    model_dirs = {}
    for kind in ("rank", "suit"):
        resolved_model_dir = resolved_dirs[kind]
        if resolved_model_dir is None:
            continue
        model_dirs[kind] = str(resolved_model_dir)
        if load_deep_model(resolved_model_dir, kind) is not None:
            loaded.append(kind)
    return {"ok": True, "enabled": True, "model_dirs": model_dirs, "loaded": loaded}


def resolve_deep_model_dir(model_dir: Path | None = None, *, kind: str | None = None) -> Path | None:
    if model_dir is not None:
        return Path(model_dir)
    if kind in ("rank", "suit"):
        env_dir = os.environ.get(f"GTO_CARD_DEEP_{kind.upper()}_MODEL_DIR")
        if env_dir:
            return Path(env_dir)
    env_dir = os.environ.get("GTO_CARD_DEEP_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    if os.environ.get("GTO_CARD_DEEP_ENABLE") == "1":
        return DEFAULT_DEEP_MODEL_DIR
    return None


def load_deep_model(model_dir: Path, kind: str) -> dict[str, Any] | None:
    resolved = Path(model_dir).resolve()
    key = (resolved, kind)
    if key in _DEEP_MODEL_CACHE:
        return _DEEP_MODEL_CACHE[key]
    model_path = resolved / f"deep_{kind}.pt"
    if not model_path.exists():
        _DEEP_MODEL_CACHE[key] = None
        return None
    torch, _nn, _optim, _DataLoader = load_torch()
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model = build_model(str(metadata.get("arch") or "mobilenet_v3_small"), class_count=len(metadata["labels"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    payload = {"torch": torch, "model": model, "metadata": metadata}
    _DEEP_MODEL_CACHE[key] = payload
    return payload


def collect_glyph_records(
    glyph_dir: Path,
    kind: str,
    labels: list[str],
    *,
    max_images_per_class: int | None = None,
) -> list[GlyphRecord]:
    root = Path(glyph_dir) / kind
    label_to_index = {label: index for index, label in enumerate(labels)}
    records = []
    for label in labels:
        label_dir = root / label
        if not label_dir.exists():
            continue
        paths = sorted(path for path in label_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if max_images_per_class is not None:
            paths = paths[: max(0, int(max_images_per_class))]
        records.extend(GlyphRecord(path=path, label=label, label_index=label_to_index[label]) for path in paths)
    return records


def collect_template_records(template_dir: Path, kind: str, labels: list[str]) -> list[GlyphRecord]:
    template_dir = Path(template_dir)
    label_to_index = {label: index for index, label in enumerate(labels)}
    prefix = f"{kind}_"
    records = []
    if not template_dir.exists():
        return records
    for path in sorted(template_dir.glob(f"{prefix}*.png")):
        label = path.stem.removeprefix(prefix).split("_", 1)[0]
        if label not in label_to_index:
            continue
        records.append(GlyphRecord(path=path, label=label, label_index=label_to_index[label]))
    return records


def stratified_split(records: list[GlyphRecord], *, val_split: float, seed: int) -> tuple[list[GlyphRecord], list[GlyphRecord]]:
    rng = random.Random(seed)
    by_label: dict[str, list[GlyphRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    train_records = []
    val_records = []
    for label_records in by_label.values():
        shuffled = label_records[:]
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * max(0.0, min(0.8, float(val_split)))))
        if len(shuffled) >= 5:
            val_count = max(1, val_count)
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:] or shuffled[:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


class GlyphDataset:
    def __init__(self, records: list[GlyphRecord], *, image_size: int, train: bool, pretrained: bool):
        self.records = records
        self.image_size = int(image_size)
        self.train = train
        self.pretrained = pretrained

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        image = read_image(record.path)
        return image_to_tensor(image, image_size=self.image_size, train=self.train, pretrained=self.pretrained), int(record.label_index)


class SimpleGlyphCNN:
    def __new__(cls, class_count: int) -> Any:
        torch, nn, _optim, _DataLoader = load_torch()
        del torch
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, class_count),
        )


def build_model(arch: str, *, class_count: int, pretrained: bool) -> Any:
    torch, nn, _optim, _DataLoader = load_torch()
    del torch
    if arch == "simple_cnn":
        return SimpleGlyphCNN(class_count)
    try:
        import torchvision.models as models
    except ImportError as error:
        raise RuntimeError("torchvision is required for pretrained deep card models") from error
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model
    if arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model
    if arch == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, class_count)
        return model
    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, class_count)
        return model
    raise ValueError(f"unsupported arch: {arch}")


def freeze_model_backbone(model: Any, arch: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if arch in ("resnet18", "resnet50"):
        modules = [model.fc]
    elif arch in ("mobilenet_v3_small", "efficientnet_b0", "efficientnet_b2", "convnext_tiny"):
        modules = [model.classifier]
    elif arch == "swin_t":
        modules = [model.head]
    elif arch == "vit_b_16":
        modules = [model.heads]
    else:
        modules = [model]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def count_labels(records: list[GlyphRecord], labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


def build_class_weights(torch: Any, records: list[GlyphRecord], class_count: int, *, device: Any) -> Any:
    counts = [0 for _ in range(class_count)]
    for record in records:
        counts[int(record.label_index)] += 1
    present_counts = [count for count in counts if count > 0]
    if not present_counts:
        return None
    max_count = max(present_counts)
    weights = []
    for count in counts:
        weights.append(math.sqrt(max_count / count) if count > 0 else 0.0)
    present_weight_total = sum(weight for weight, count in zip(weights, counts) if count > 0)
    if present_weight_total > 0:
        scale = len(present_counts) / present_weight_total
        weights = [weight * scale if count > 0 else 0.0 for weight, count in zip(weights, counts)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(torch: Any, records: list[GlyphRecord]) -> Any | None:
    if not records:
        return None
    from torch.utils.data import WeightedRandomSampler

    counts: dict[int, int] = {}
    for record in records:
        counts[int(record.label_index)] = counts.get(int(record.label_index), 0) + 1
    weights = [1.0 / math.sqrt(max(1, counts[int(record.label_index)])) for record in records]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(records), replacement=True)


def run_epoch(torch: Any, model: Any, loader: Any, criterion: Any, optimizer: Any | None, *, device: Any) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total += batch_size
    if total <= 0:
        return math.inf, 0.0
    return total_loss / total, total_correct / total


def read_image(path: Path) -> Any:
    cv2, _np = load_cv()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def image_to_tensor(image: Any, *, image_size: int, train: bool, pretrained: bool) -> Any:
    cv2, np = load_cv()
    torch, _nn, _optim, _DataLoader = load_torch()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if train:
        if random.random() < 0.35:
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
            gray = cv2.warpAffine(gray, matrix, (image_size, image_size), borderValue=0)
        if random.random() < 0.25:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    if pretrained:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std
    return (tensor - 0.5) / 0.5


def format_deep_train_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"train-deep-card-classifier failed: {payload.get('error')}"
    meta = payload["metadata"]
    lines = [
        f"Model: {payload['model']}",
        f"Kind: {payload['kind']} arch={meta['arch']} pretrained={meta['pretrained']}",
        f"Samples: train={meta['train_count']} val={meta['val_count']} source={meta['source_count']}",
        (
            "Options: "
            f"freeze_backbone={meta.get('freeze_backbone', False)} "
            f"class_balanced_loss={meta.get('class_balanced_loss', False)} "
            f"weighted_sampler={meta.get('weighted_sampler', False)}"
        ),
        f"Best val acc: {payload.get('best_val_acc', 0):.3f}",
    ]
    if meta.get("history"):
        last = meta["history"][-1]
        lines.append(
            f"Last epoch: train_acc={last['train_acc']:.3f} val_acc={last['val_acc']:.3f} "
            f"train_loss={last['train_loss']:.4f} val_loss={last['val_loss']:.4f}"
        )
    return "\n".join(lines)


def load_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("PyTorch is required for deep card models: pip install torch torchvision") from error
    return torch, nn, optim, DataLoader


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
