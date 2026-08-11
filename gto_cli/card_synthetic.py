from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .card_deep_model import RANK_LABELS, SUIT_LABELS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "video_frames" / "card_glyph_synthetic"
RANK_SIZE = (54, 70)
SUIT_SIZE = (42, 42)


def generate_synthetic_card_glyphs(
    *,
    output_dir: Path = DEFAULT_SYNTHETIC_DIR,
    per_class: int = 80,
    seed: int = 20260708,
    include_rank: bool = True,
    include_suit: bool = True,
) -> dict[str, Any]:
    cv2, np = load_cv()
    Image, ImageDraw, ImageFont = load_pil()
    rng = random.Random(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts = discover_fonts(ImageFont)
    if not fonts:
        raise RuntimeError("no usable Windows fonts found for synthetic rank generation")

    records = []
    if include_rank:
        for label in RANK_LABELS:
            label_dir = output_dir / "rank" / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for index in range(max(0, int(per_class))):
                image = render_rank_glyph(
                    label,
                    rng=rng,
                    cv2=cv2,
                    np=np,
                    Image=Image,
                    ImageDraw=ImageDraw,
                    ImageFont=ImageFont,
                    fonts=fonts,
                )
                path = label_dir / f"synthetic_rank_{label}_{index:04d}.png"
                cv2.imwrite(str(path), image)
                records.append({"kind": "rank", "label": label, "path": str(path)})

    if include_suit:
        for label in SUIT_LABELS:
            label_dir = output_dir / "suit" / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for index in range(max(0, int(per_class))):
                image = render_suit_glyph(label, rng=rng, cv2=cv2, np=np)
                path = label_dir / f"synthetic_suit_{label}_{index:04d}.png"
                cv2.imwrite(str(path), image)
                records.append({"kind": "suit", "label": label, "path": str(path)})

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "per_class": int(per_class),
        "seed": int(seed),
        "rank_images": len([record for record in records if record["kind"] == "rank"]),
        "suit_images": len([record for record in records if record["kind"] == "suit"]),
        "fonts": [str(path) for path in fonts],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def discover_fonts(ImageFont: Any) -> list[Path]:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/tahomabd.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/trebucbd.ttf"),
        Path("C:/Windows/Fonts/consolab.ttf"),
        Path("C:/Windows/Fonts/bahnschrift.ttf"),
    ]
    usable = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            ImageFont.truetype(str(path), 32)
        except OSError:
            continue
        usable.append(path)
    return usable


def render_rank_glyph(
    label: str,
    *,
    rng: random.Random,
    cv2: Any,
    np: Any,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
    fonts: list[Path],
) -> Any:
    width, height = RANK_SIZE
    canvas_size = 128
    font_path = rng.choice(fonts)
    font_size = rng.randint(66, 91)
    font = ImageFont.truetype(str(font_path), font_size)
    pil = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = int((canvas_size - text_w) * rng.uniform(0.42, 0.52) - bbox[0] + rng.randint(-5, 5))
    y = int((canvas_size - text_h) * rng.uniform(0.40, 0.50) - bbox[1] + rng.randint(-6, 6))
    draw.text((x, y), label, fill=255, font=font)
    image = np.array(pil, dtype=np.uint8)
    image = random_affine(image, rng=rng, cv2=cv2, np=np, rotate=7.0, shear=0.06, scale=(0.84, 1.08))
    image = tighten_and_resize(image, RANK_SIZE, cv2=cv2, np=np, pad=rng.randint(1, 4))
    return random_mask_noise(image, rng=rng, cv2=cv2, np=np)


def render_suit_glyph(label: str, *, rng: random.Random, cv2: Any, np: Any) -> Any:
    canvas_size = 96
    image = np.zeros((canvas_size, canvas_size), np.uint8)
    cx = canvas_size // 2 + rng.randint(-3, 3)
    cy = canvas_size // 2 + rng.randint(-3, 3)
    scale = rng.uniform(0.82, 1.08)
    if label == "d":
        pts = np.array(
            [
                [cx, cy - int(34 * scale)],
                [cx + int(25 * scale), cy],
                [cx, cy + int(34 * scale)],
                [cx - int(25 * scale), cy],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(image, pts, 255)
    elif label == "h":
        r = int(16 * scale)
        cv2.circle(image, (cx - int(13 * scale), cy - int(10 * scale)), r, 255, -1)
        cv2.circle(image, (cx + int(13 * scale), cy - int(10 * scale)), r, 255, -1)
        pts = np.array(
            [
                [cx - int(31 * scale), cy - int(4 * scale)],
                [cx + int(31 * scale), cy - int(4 * scale)],
                [cx, cy + int(34 * scale)],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(image, pts, 255)
    elif label == "c":
        r = int(15 * scale)
        cv2.circle(image, (cx, cy - int(17 * scale)), r, 255, -1)
        cv2.circle(image, (cx - int(17 * scale), cy + int(4 * scale)), r, 255, -1)
        cv2.circle(image, (cx + int(17 * scale), cy + int(4 * scale)), r, 255, -1)
        cv2.rectangle(
            image,
            (cx - int(5 * scale), cy + int(10 * scale)),
            (cx + int(5 * scale), cy + int(34 * scale)),
            255,
            -1,
        )
        pts = np.array(
            [
                [cx - int(13 * scale), cy + int(34 * scale)],
                [cx + int(13 * scale), cy + int(34 * scale)],
                [cx, cy + int(24 * scale)],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(image, pts, 255)
    elif label == "s":
        r = int(16 * scale)
        cv2.circle(image, (cx - int(13 * scale), cy + int(4 * scale)), r, 255, -1)
        cv2.circle(image, (cx + int(13 * scale), cy + int(4 * scale)), r, 255, -1)
        pts = np.array(
            [
                [cx - int(31 * scale), cy + int(8 * scale)],
                [cx + int(31 * scale), cy + int(8 * scale)],
                [cx, cy - int(34 * scale)],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(image, pts, 255)
        cv2.rectangle(
            image,
            (cx - int(5 * scale), cy + int(12 * scale)),
            (cx + int(5 * scale), cy + int(36 * scale)),
            255,
            -1,
        )
        pts2 = np.array(
            [
                [cx - int(13 * scale), cy + int(36 * scale)],
                [cx + int(13 * scale), cy + int(36 * scale)],
                [cx, cy + int(26 * scale)],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(image, pts2, 255)
    else:
        raise ValueError(f"unsupported suit: {label}")
    image = random_affine(image, rng=rng, cv2=cv2, np=np, rotate=8.0, shear=0.05, scale=(0.88, 1.10))
    image = tighten_and_resize(image, SUIT_SIZE, cv2=cv2, np=np, pad=rng.randint(1, 4))
    return random_mask_noise(image, rng=rng, cv2=cv2, np=np)


def random_affine(
    image: Any,
    *,
    rng: random.Random,
    cv2: Any,
    np: Any,
    rotate: float,
    shear: float,
    scale: tuple[float, float],
) -> Any:
    height, width = image.shape[:2]
    angle = math.radians(rng.uniform(-rotate, rotate))
    sx = rng.uniform(scale[0], scale[1])
    sy = rng.uniform(scale[0], scale[1])
    sh = rng.uniform(-shear, shear)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    matrix = np.array(
        [
            [sx * cos_a + sh * sin_a, -sin_a, rng.uniform(-4, 4)],
            [sin_a, sy * cos_a, rng.uniform(-4, 4)],
        ],
        dtype=np.float32,
    )
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    offset = center - matrix[:, :2] @ center
    matrix[:, 2] += offset
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)


def tighten_and_resize(image: Any, size: tuple[int, int], *, cv2: Any, np: Any, pad: int) -> Any:
    width, height = size
    _, mask = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((height, width), np.uint8)
    x1, x2 = max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)
    y1, y2 = max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)
    piece = mask[y1:y2, x1:x2]
    piece_h, piece_w = piece.shape[:2]
    scale = min(width / max(piece_w, 1), height / max(piece_h, 1))
    resized_w = max(1, int(piece_w * scale))
    resized_h = max(1, int(piece_h * scale))
    resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), np.uint8)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return canvas


def random_mask_noise(image: Any, *, rng: random.Random, cv2: Any, np: Any) -> Any:
    result = image.copy()
    if rng.random() < 0.22:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    if rng.random() < 0.18:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
    if rng.random() < 0.15:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.erode(result, kernel, iterations=1)
    if rng.random() < 0.18:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 5, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    _, result = cv2.threshold(result, 24, 255, cv2.THRESH_BINARY)
    return result


def format_synthetic_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"generate-card-synthetic failed: {payload.get('error')}"
    return "\n".join(
        [
            f"Output: {payload.get('output_dir')}",
            f"Rank images: {payload.get('rank_images', 0)}",
            f"Suit images: {payload.get('suit_images', 0)}",
            f"Per class: {payload.get('per_class')}",
            f"Manifest: {payload.get('manifest')}",
        ]
    )


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np


def load_pil() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Pillow is required for synthetic rank glyphs: pip install pillow") from error
    return Image, ImageDraw, ImageFont
