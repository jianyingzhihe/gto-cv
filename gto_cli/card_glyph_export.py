from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .video_vision import (
    build_layout_profile,
    detect_visible_cards,
    normalized_rank_piece,
    normalized_suit_component,
)
from .card_classifier import extract_corner_glyphs, parse_card_label


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def export_card_glyphs(
    *,
    video_paths: list[Path],
    output_dir: Path,
    every_sec: float = 5.0,
    max_frames: int | None = None,
    lock_layout: bool = True,
    include_board: bool = True,
    min_rank_confidence: float = 0.0,
    min_suit_confidence: float = 0.0,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    videos_summary = []

    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for video_path in video_paths:
            video_path = Path(video_path)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                videos_summary.append({"video": str(video_path), "ok": False, "error": "cannot_open"})
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                cap.release()
                videos_summary.append({"video": str(video_path), "ok": False, "error": "no_frames"})
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, first_frame = cap.read()
            layout_profile = build_layout_profile(first_frame, [], hero_name=None) if ok and lock_layout else None
            step_frames = max(1, int(round(max(0.05, float(every_sec)) * fps)))
            indices = list(range(0, frame_count, step_frames))
            if max_frames is not None:
                indices = indices[: max(0, int(max_frames))]

            exported = 0
            for sample_index, frame_index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                card_result = detect_visible_cards(frame, layout_profile=layout_profile)
                details = list(card_result.get("hero_details") or [])
                if include_board:
                    details.extend(card_result.get("board_details") or [])
                for detail in details:
                    source = str(detail.get("source") or "hero")
                    if source == "board" and not include_board:
                        continue
                    if float(detail.get("rank_confidence") or 0.0) < float(min_rank_confidence):
                        continue
                    if detail.get("suit") not in (None, "?") and float(detail.get("suit_confidence") or 0.0) < float(min_suit_confidence):
                        continue
                    crop = crop_detail(frame, detail.get("roi_box") or {})
                    if crop is None:
                        continue
                    rank_image = export_rank_glyph_image(crop, source)
                    suit_image = normalized_suit_component(crop, (42, 42), source)

                    base = safe_stem(video_path.stem)
                    card = str(detail.get("card") or "_unknown")
                    rank = safe_label(str(detail.get("rank") or "_unknown"))
                    suit = safe_label(str(detail.get("suit") or "_unknown"))
                    prefix = f"{base}_f{frame_index:07d}_{source}{int(detail.get('index') or 0)}_{safe_label(card)}"
                    rank_path = output_dir / "rank" / rank / f"{prefix}_rank.png"
                    suit_path = output_dir / "suit" / suit / f"{prefix}_suit.png"
                    card_path = output_dir / "card" / safe_label(card) / f"{prefix}_card.png"
                    rank_path.parent.mkdir(parents=True, exist_ok=True)
                    suit_path.parent.mkdir(parents=True, exist_ok=True)
                    card_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(rank_path), rank_image)
                    cv2.imwrite(str(suit_path), suit_image)
                    cv2.imwrite(str(card_path), crop)

                    record = {
                        "video": str(video_path),
                        "frame_index": int(frame_index),
                        "timestamp_sec": round(frame_index / fps, 3),
                        "sample_index": int(sample_index),
                        "source": source,
                        "card_index": int(detail.get("index") or 0),
                        "roi_mode": detail.get("roi_mode"),
                        "card": card,
                        "rank": rank,
                        "suit": suit,
                        "rank_confidence": detail.get("rank_confidence"),
                        "rank_margin": detail.get("rank_margin"),
                        "suit_confidence": detail.get("suit_confidence"),
                        "suit_margin": detail.get("suit_margin"),
                        "roi_box": detail.get("roi_box"),
                        "rank_path": str(rank_path),
                        "suit_path": str(suit_path),
                        "card_path": str(card_path),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    exported += 1

            cap.release()
            videos_summary.append(
                {
                    "video": str(video_path),
                    "ok": True,
                    "frame_count": frame_count,
                    "fps": fps,
                    "sampled_frames": len(indices),
                    "exported_cards": exported,
                    "layout_locked": bool(layout_profile),
                    "layout_method": layout_profile.get("method") if layout_profile else None,
                    "hero_search_source": layout_profile.get("hero_search_source") if layout_profile else None,
                }
            )

    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "videos": videos_summary,
        "exported_cards": len(records),
        "rank_images": len(records),
        "suit_images": len(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_rank_glyph_image(crop: Any, source: str) -> Any:
    height = 72 if source == "hero" else 60
    width = 64 if source == "hero" else 55
    return normalized_rank_piece(crop[0 : min(crop.shape[0], height), 0 : min(crop.shape[1], width)], (54, 70))


def ingest_external_card_images(
    *,
    dataset_dirs: list[Path],
    output_dir: Path,
    max_images: int | None = None,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "external_manifest.jsonl"
    records = []
    skipped = []
    seen = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for dataset_dir in dataset_dirs:
            for path in iter_image_files(Path(dataset_dir)):
                if max_images is not None and seen >= max(0, int(max_images)):
                    break
                label = parse_card_label(path)
                if label is None:
                    skipped.append({"path": str(path), "reason": "label_not_found"})
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    skipped.append({"path": str(path), "reason": "image_read_failed"})
                    continue
                extracted = extract_corner_glyphs(image)
                if extracted is None:
                    skipped.append({"path": str(path), "reason": "corner_glyph_extract_failed"})
                    continue
                rank, suit = label
                rank_image, suit_image = extracted
                card = f"{rank}{suit}"
                prefix = f"external_{seen:07d}_{safe_label(card)}_{safe_stem(path.stem)}"
                rank_path = output_dir / "rank" / rank / f"{prefix}_rank.png"
                suit_path = output_dir / "suit" / suit / f"{prefix}_suit.png"
                card_path = output_dir / "card" / card / f"{prefix}_card.png"
                rank_path.parent.mkdir(parents=True, exist_ok=True)
                suit_path.parent.mkdir(parents=True, exist_ok=True)
                card_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rank_path), rank_image)
                cv2.imwrite(str(suit_path), suit_image)
                cv2.imwrite(str(card_path), image)
                record = {
                    "source": "external",
                    "input_path": str(path),
                    "card": card,
                    "rank": rank,
                    "suit": suit,
                    "rank_path": str(rank_path),
                    "suit_path": str(suit_path),
                    "card_path": str(card_path),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                seen += 1
            if max_images is not None and seen >= max(0, int(max_images)):
                break
    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "dataset_dirs": [str(path) for path in dataset_dirs],
        "ingested_cards": len(records),
        "rank_images": len(records),
        "suit_images": len(records),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
    }
    (output_dir / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def crop_detail(frame: Any, box: dict[str, Any]) -> Any | None:
    if not box:
        return None
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, int(box.get("x", 0))))
    y1 = max(0, min(frame_h - 1, int(box.get("y", 0))))
    x2 = max(x1 + 1, min(frame_w, x1 + int(box.get("width", 0))))
    y2 = max(y1 + 1, min(frame_h, y1 + int(box.get("height", 0))))
    return frame[y1:y2, x1:x2]


def safe_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in value)
    return cleaned.strip("_") or "video"


def safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in value)
    return cleaned.strip("_") or "_unknown"


def format_card_glyph_export_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"export-card-glyphs failed: {payload.get('error')}"
    lines = [
        f"Exported cards: {payload.get('exported_cards', 0)}",
        f"Output: {payload.get('output_dir')}",
        f"Manifest: {payload.get('manifest')}",
    ]
    for item in payload.get("videos") or []:
        status = "ok" if item.get("ok") else f"error={item.get('error')}"
        lines.append(
            f"- {Path(item.get('video', '')).name}: {status}, sampled={item.get('sampled_frames', 0)}, "
            f"cards={item.get('exported_cards', 0)}, layout={item.get('hero_search_source') or '-'}"
        )
    return "\n".join(lines)


def format_external_ingest_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"ingest-card-images failed: {payload.get('error')}"
    return "\n".join(
        [
            f"Ingested cards: {payload.get('ingested_cards', 0)}",
            f"Output: {payload.get('output_dir')}",
            f"Manifest: {payload.get('manifest')}",
            f"Skipped: {payload.get('skipped_count', 0)}",
        ]
    )


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
