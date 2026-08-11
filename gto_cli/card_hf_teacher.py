from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .card_deep_model import RANK_LABELS, SUIT_LABELS
from .card_teacher_label import (
    collect_crop_records,
    copy_labeled_crop,
    copy_review_crop,
    limit_records_per_kind,
    should_accept_prediction,
    write_predictions_csv,
)


DEFAULT_HF_CLIP_MODEL = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224
_CLIP_CACHE: dict[tuple[str, bool, str], dict[str, Any]] = {}


RANK_PROMPTS = {
    "A": ("a playing card rank letter A", "a capital letter A"),
    "K": ("a playing card rank letter K", "a capital letter K"),
    "Q": ("a playing card rank letter Q", "a capital letter Q"),
    "J": ("a playing card rank letter J", "a capital letter J"),
    "T": ("a playing card rank number 10", "the number 10"),
    "9": ("a playing card rank number 9", "the number 9"),
    "8": ("a playing card rank number 8", "the number 8"),
    "7": ("a playing card rank number 7", "the number 7"),
    "6": ("a playing card rank number 6", "the number 6"),
    "5": ("a playing card rank number 5", "the number 5"),
    "4": ("a playing card rank number 4", "the number 4"),
    "3": ("a playing card rank number 3", "the number 3"),
    "2": ("a playing card rank number 2", "the number 2"),
}

SUIT_PROMPTS = {
    "s": ("a spade playing card suit symbol", "a black spade silhouette"),
    "h": ("a heart playing card suit symbol", "a heart silhouette"),
    "d": ("a diamond playing card suit symbol", "a diamond silhouette"),
    "c": ("a club playing card suit symbol", "a club silhouette"),
}


def label_card_crops_hf(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    kind: str = "both",
    rank_model: str = DEFAULT_HF_CLIP_MODEL,
    suit_model: str = DEFAULT_HF_CLIP_MODEL,
    max_images: int | None = None,
    rank_score_threshold: float = 0.52,
    rank_margin_threshold: float = 0.05,
    suit_score_threshold: float = 0.52,
    suit_margin_threshold: float = 0.05,
    require_current_agreement: bool = False,
    copy_accepted: bool = True,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")

    cv2, _np = load_cv()
    output_dir = Path(output_dir)
    review_dir = output_dir / "review"
    for directory in (output_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    allowed_kinds = ("rank", "suit") if kind == "both" else (kind,)
    records = collect_crop_records([Path(path) for path in input_dirs], allowed_kinds=allowed_kinds)
    if max_images is not None:
        records = limit_records_per_kind(records, max_per_kind=max(0, int(max_images)))

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    accepted = 0
    copied_accepted = 0
    review = 0
    unreadable = 0
    for index, record in enumerate(records):
        image = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            rows.append(build_hf_row(index, record, None, accepted=False, reason="image_read_failed", output_path=""))
            unreadable += 1
            review += 1
            continue
        model_name = rank_model if record.kind == "rank" else suit_model
        prediction = classify_clip_glyph(
            image,
            record.kind,
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
            source_path=record.path,
        )
        accepted_flag, reason = should_accept_prediction(
            record,
            prediction,
            rank_score_threshold=rank_score_threshold,
            rank_margin_threshold=rank_margin_threshold,
            suit_score_threshold=suit_score_threshold,
            suit_margin_threshold=suit_margin_threshold,
            require_current_agreement=require_current_agreement,
        )
        output_path = ""
        if accepted_flag:
            accepted += 1
            if copy_accepted and prediction is not None:
                output_path = str(copy_labeled_crop(record, prediction, output_dir, index))
                copied_accepted += 1
        else:
            review_path = copy_review_crop(record, prediction, review_dir, index)
            output_path = str(review_path) if review_path else ""
            review += 1
        rows.append(build_hf_row(index, record, prediction, accepted=accepted_flag, reason=reason, output_path=output_path))

    predictions_csv = output_dir / "predictions.csv"
    review_csv = output_dir / "review.csv"
    write_predictions_csv(predictions_csv, rows)
    write_predictions_csv(review_csv, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "kind": kind,
        "rank_model": rank_model,
        "suit_model": suit_model,
        "processed": len(rows),
        "accepted": accepted,
        "copied_accepted": copied_accepted,
        "review": review,
        "unreadable": unreadable,
        "thresholds": {
            "rank_score": float(rank_score_threshold),
            "rank_margin": float(rank_margin_threshold),
            "suit_score": float(suit_score_threshold),
            "suit_margin": float(suit_margin_threshold),
            "require_current_agreement": bool(require_current_agreement),
        },
        "counts": count_rows(rows),
        "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        "files": {
            "predictions_csv": str(predictions_csv),
            "review_csv": str(review_csv),
            "accepted_dir": str(output_dir) if copy_accepted else "",
            "review_dir": str(review_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def classify_clip_glyph(
    image: Any,
    kind: str,
    *,
    model_name: str,
    device: str = "auto",
    local_files_only: bool = False,
    source_path: Path | None = None,
) -> dict[str, Any]:
    if kind not in ("rank", "suit"):
        raise ValueError("kind must be rank or suit")
    loaded = load_clip(model_name, device=device, local_files_only=local_files_only)
    torch = loaded["torch"]
    model = loaded["model"]
    processor = loaded["processor"]
    torch_device = loaded["device"]
    prompt_map = RANK_PROMPTS if kind == "rank" else SUIT_PROMPTS
    labels = list(RANK_LABELS if kind == "rank" else SUIT_LABELS)
    prompts: list[str] = []
    prompt_labels: list[str] = []
    for label in labels:
        for prompt in prompt_map[label]:
            prompts.append(prompt)
            prompt_labels.append(label)

    pil_image = glyph_to_pil(image, kind=kind, source_path=source_path)
    inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
    inputs = {key: value.to(torch_device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prompt_probs = torch.softmax(outputs.logits_per_image[0], dim=0).detach().cpu().tolist()

    by_label: dict[str, list[float]] = {label: [] for label in labels}
    for label, probability in zip(prompt_labels, prompt_probs):
        by_label[label].append(float(probability))
    averaged = [(label, sum(values) / max(1, len(values))) for label, values in by_label.items()]
    total = sum(score for _label, score in averaged) or 1.0
    normalized = [(label, score / total) for label, score in averaged]
    ordered = sorted(normalized, key=lambda item: item[1], reverse=True)
    label, score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    return {
        "label": label,
        "score": float(score),
        "margin": float(score - second_score),
        "second_score": float(second_score),
        "model": model_name,
        "backend": "hf_clip_zero_shot",
    }


def load_clip(model_name: str, *, device: str, local_files_only: bool) -> dict[str, Any]:
    torch = load_torch()
    resolved_device = resolve_device(torch, device)
    cache_key = (model_name, bool(local_files_only), str(resolved_device))
    cached = _CLIP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("transformers is required for HuggingFace card teachers: pip install transformers") from error
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=bool(local_files_only))
    model = CLIPModel.from_pretrained(model_name, local_files_only=bool(local_files_only))
    model.eval()
    model.to(resolved_device)
    loaded = {"torch": torch, "processor": processor, "model": model, "device": resolved_device}
    _CLIP_CACHE[cache_key] = loaded
    return loaded


def resolve_device(torch: Any, device: str) -> Any:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "-1"):
        return torch.device("cpu")
    if device.startswith("cuda"):
        return torch.device(device if torch.cuda.is_available() else "cpu")
    try:
        index = int(device)
    except ValueError:
        return torch.device("cpu")
    return torch.device(f"cuda:{index}" if index >= 0 and torch.cuda.is_available() else "cpu")


def glyph_to_pil(image: Any, *, kind: str, source_path: Path | None = None) -> Any:
    cv2, np = load_cv()
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for HuggingFace card teachers: pip install pillow") from error
    if kind == "suit" and source_path is not None:
        card_path = find_matching_card_crop(Path(source_path))
        if card_path is not None:
            card_image = cv2.imread(str(card_path), cv2.IMREAD_COLOR)
            if card_image is not None:
                from_card = suit_from_card_to_pil(card_image)
                if from_card is not None:
                    return from_card
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if int(gray.max()) <= int(gray.min()) + 4:
        mask = np.zeros_like(gray, dtype=np.uint8)
    else:
        _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(mask.mean()) > 160:
            mask = 255 - mask
    mask = clean_glyph_mask(mask, kind=kind)
    ys, xs = np.where(mask > 0)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = max(0, int(xs.min()) - 2), min(mask.shape[1], int(xs.max()) + 3)
        y1, y2 = max(0, int(ys.min()) - 2), min(mask.shape[0], int(ys.max()) + 3)
        piece = mask[y1:y2, x1:x2]
        target_box = 176 if kind == "rank" else 164
        scale = min(target_box / max(1, piece.shape[1]), target_box / max(1, piece.shape[0]))
        resized_w = max(1, int(piece.shape[1] * scale))
        resized_h = max(1, int(piece.shape[0] * scale))
        resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        glyph = 255 - resized
        x_offset = (IMAGE_SIZE - resized_w) // 2
        y_offset = (IMAGE_SIZE - resized_h) // 2
        canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = glyph
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def find_matching_card_crop(path: Path) -> Path | None:
    if not path.stem.endswith("_suit"):
        return None
    target_name = path.stem.removesuffix("_suit") + "_card" + path.suffix
    roots: list[Path] = []
    for parent in path.parents:
        if parent.name.lower() in ("rank", "suit"):
            roots.append(parent.parent)
        if parent.parent == parent:
            break
        if len(roots) >= 3:
            break
    for root in roots:
        direct = root / "cards" / target_name
        if direct.exists():
            return direct
        card_root = root / "card"
        if card_root.exists():
            matches = list(card_root.rglob(target_name))
            if matches:
                return matches[0]
    return None


def suit_from_card_to_pil(card_image: Any) -> Any | None:
    cv2, np = load_cv()
    from PIL import Image

    height, width = card_image.shape[:2]
    if height < 40 or width < 24:
        return None
    x2 = max(18, min(width, int(width * 0.58)))
    y1 = min(height - 1, max(0, int(height * 0.50)))
    y2 = min(height, max(y1 + 12, int(height * 0.80)))
    roi = card_image[y1:y2, 0:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45)
    dark = (hsv[:, :, 2] < 150) & (hsv[:, :, 1] < 210)
    foreground = red if int(red.sum()) >= 40 else dark
    mask = (foreground.astype("uint8")) * 255
    mask = clean_glyph_mask(mask, kind="suit")
    mask = keep_largest_component(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1b, x2b = max(0, int(xs.min()) - 3), min(roi.shape[1], int(xs.max()) + 4)
    y1b, y2b = max(0, int(ys.min()) - 3), min(roi.shape[0], int(ys.max()) + 4)
    piece = roi[y1b:y2b, x1b:x2b].copy()
    piece_mask = mask[y1b:y2b, x1b:x2b] > 0
    if piece.size == 0:
        return None
    piece[~piece_mask] = 255
    piece_h, piece_w = piece.shape[:2]
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    target_box = 164
    scale = min(target_box / max(1, piece_w), target_box / max(1, piece_h))
    resized_w = max(1, int(piece_w * scale))
    resized_h = max(1, int(piece_h * scale))
    resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x_offset = (IMAGE_SIZE - resized_w) // 2
    y_offset = (IMAGE_SIZE - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def keep_largest_component(mask: Any) -> Any:
    cv2, np = load_cv()
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        return mask
    best_component = None
    best_area = 0
    for component in range(1, component_count):
        area = int(stats[component][cv2.CC_STAT_AREA])
        if area > best_area:
            best_component = component
            best_area = area
    if best_component is None or best_area <= 0:
        return mask
    return ((labels == best_component).astype(np.uint8)) * 255


def clean_glyph_mask(mask: Any, *, kind: str) -> Any:
    cv2, np = load_cv()
    height, width = mask.shape[:2]
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for component in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component]]
        if area < 10:
            continue
        if is_border_component(kind, x=x, y=y, w=w, h=h, area=area, width=width, height=height):
            continue
        kept[labels == component] = 255
    if kind == "suit" and int((kept > 0).sum()) >= 20:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    return kept if int((kept > 0).sum()) >= 20 else mask


def is_border_component(kind: str, *, x: int, y: int, w: int, h: int, area: int, width: int, height: int) -> bool:
    del area
    if kind == "rank":
        top_bar = y <= max(6, int(height * 0.10)) and w >= int(width * 0.78) and h <= int(height * 0.32)
        left_edge = x <= 2 and w <= max(5, int(width * 0.14)) and h >= int(height * 0.30)
        right_edge = x + w >= width - 2 and w <= max(5, int(width * 0.14)) and h >= int(height * 0.30)
        return bool(top_bar or left_edge or right_edge)
    left_edge = x <= 3 and w <= max(9, int(width * 0.32)) and h >= int(height * 0.42)
    right_edge = x + w >= width - 2 and w <= max(9, int(width * 0.32)) and h >= int(height * 0.42)
    top_edge = y <= 1 and h <= max(7, int(height * 0.22)) and w >= int(width * 0.42)
    bottom_edge = y + h >= height - 1 and h <= max(7, int(height * 0.25)) and w >= int(width * 0.42)
    return bool(left_edge or right_edge or top_edge or bottom_edge)


def build_hf_row(
    index: int,
    record: Any,
    prediction: dict[str, Any] | None,
    *,
    accepted: bool,
    reason: str,
    output_path: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "kind": record.kind,
        "input_path": str(record.path),
        "current_label": record.current_label,
        "teacher_label": str((prediction or {}).get("label") or ""),
        "teacher_score": format_float((prediction or {}).get("score")),
        "teacher_margin": format_float((prediction or {}).get("margin")),
        "teacher_second_score": format_float((prediction or {}).get("second_score")),
        "teacher_model": str((prediction or {}).get("model") or ""),
        "accepted": bool(accepted),
        "reason": reason,
        "output_path": output_path,
        "rank_path": str(record.path) if record.kind == "rank" else "",
        "suit_path": str(record.path) if record.kind == "suit" else "",
        "card_path": "",
        "final_rank": "",
        "final_suit": "",
        "note": "",
    }


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        reason = str(row.get("reason") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"by_kind": by_kind, "by_reason": by_reason}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: Any) -> str:
    numeric = safe_float(value)
    return "" if numeric is None else f"{numeric:.6f}"


def format_hf_card_crop_label_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"label-card-crops-hf failed: {payload.get('error')}"
    files = payload.get("files") or {}
    counts = payload.get("counts") or {}
    return "\n".join(
        [
            f"Processed: {payload.get('processed', 0)}",
            f"Accepted: {payload.get('accepted', 0)}",
            f"Copied accepted: {payload.get('copied_accepted', payload.get('accepted', 0))}",
            f"Review: {payload.get('review', 0)}",
            f"Counts: {json.dumps(counts, ensure_ascii=False)}",
            f"Rank model: {payload.get('rank_model')}",
            f"Suit model: {payload.get('suit_model')}",
            f"Accepted dir: {files.get('accepted_dir')}",
            f"Review CSV: {files.get('review_csv')}",
            f"Predictions CSV: {files.get('predictions_csv')}",
        ]
    )


def load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for HuggingFace card teachers: pip install torch") from error
    return torch


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
