from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .card_deep_model import RANK_LABELS, SUIT_LABELS, classify_deep_glyph, warm_deep_card_models
from .card_classifier import parse_card_label
from .card_glyph_export import safe_label, safe_stem


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class CropRecord:
    kind: str
    path: Path
    current_label: str


def label_card_crops(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    teacher_model_dir: Path | None = None,
    teacher_rank_model_dir: Path | None = None,
    teacher_suit_model_dir: Path | None = None,
    kind: str = "both",
    max_images: int | None = None,
    rank_score_threshold: float = 0.90,
    rank_margin_threshold: float = 0.20,
    suit_score_threshold: float = 0.88,
    suit_margin_threshold: float = 0.18,
    require_current_agreement: bool = False,
    copy_accepted: bool = True,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    rank_model_dir = teacher_rank_model_dir or teacher_model_dir
    suit_model_dir = teacher_suit_model_dir or teacher_model_dir
    if kind in ("rank", "both") and rank_model_dir is None:
        raise ValueError("rank teacher model is required; pass --teacher-rank-model-dir or --teacher-model-dir")
    if kind in ("suit", "both") and suit_model_dir is None:
        raise ValueError("suit teacher model is required; pass --teacher-suit-model-dir or --teacher-model-dir")

    cv2, _np = load_cv()
    output_dir = Path(output_dir)
    accepted_dir = output_dir
    review_dir = output_dir / "review"
    for directory in (output_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warm_deep_card_models(
        teacher_model_dir,
        rank_model_dir=teacher_rank_model_dir,
        suit_model_dir=teacher_suit_model_dir,
    )

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
        image = cv2.imread(str(record.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            row = build_row(index, record, None, accepted=False, reason="image_read_failed", output_path="")
            rows.append(row)
            unreadable += 1
            review += 1
            continue
        model_dir = rank_model_dir if record.kind == "rank" else suit_model_dir
        prediction = classify_deep_glyph(image, record.kind, model_dir=model_dir)
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
                output_path = str(copy_labeled_crop(record, prediction, accepted_dir, index))
                copied_accepted += 1
        else:
            review_path = copy_review_crop(record, prediction, review_dir, index)
            output_path = str(review_path) if review_path else ""
            review += 1
        rows.append(build_row(index, record, prediction, accepted=accepted_flag, reason=reason, output_path=output_path))

    predictions_csv = output_dir / "predictions.csv"
    review_csv = output_dir / "review.csv"
    write_predictions_csv(predictions_csv, rows)
    write_predictions_csv(review_csv, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "teacher_model_dir": str(teacher_model_dir) if teacher_model_dir else "",
        "teacher_rank_model_dir": str(rank_model_dir) if rank_model_dir else "",
        "teacher_suit_model_dir": str(suit_model_dir) if suit_model_dir else "",
        "kind": kind,
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
            "accepted_dir": str(accepted_dir) if copy_accepted else "",
            "review_dir": str(review_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def organize_card_crops(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    kind: str = "both",
    max_images: int | None = None,
    review_csv: Path | None = None,
    allowed_review_reasons: list[str] | None = None,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_kinds = ("rank", "suit") if kind == "both" else (kind,)
    records = collect_crop_records([Path(path) for path in input_dirs], allowed_kinds=allowed_kinds)
    if max_images is not None:
        records = limit_records_per_kind(records, max_per_kind=max(0, int(max_images)))
    allowed_paths = load_allowed_review_paths(review_csv, allowed_review_reasons or ["ok"]) if review_csv else None

    rows: list[dict[str, Any]] = []
    copied = 0
    skipped = 0
    for index, record in enumerate(records):
        if allowed_paths is not None and record.path.resolve() not in allowed_paths:
            skipped += 1
            rows.append(
                {
                    "index": index,
                    "kind": record.kind,
                    "input_path": str(record.path),
                    "current_label": record.current_label,
                    "accepted": False,
                    "reason": "not_allowed_by_review",
                    "output_path": "",
                }
            )
            continue
        valid_labels = set(RANK_LABELS if record.kind == "rank" else SUIT_LABELS)
        if record.current_label not in valid_labels:
            skipped += 1
            rows.append(
                {
                    "index": index,
                    "kind": record.kind,
                    "input_path": str(record.path),
                    "current_label": record.current_label,
                    "accepted": False,
                    "reason": "label_not_found",
                    "output_path": "",
                }
            )
            continue
        output_path = copy_current_labeled_crop(record, output_dir, index)
        copied += 1
        rows.append(
            {
                "index": index,
                "kind": record.kind,
                "input_path": str(record.path),
                "current_label": record.current_label,
                "teacher_label": record.current_label,
                "teacher_score": "",
                "teacher_margin": "",
                "teacher_second_score": "",
                "teacher_model": "current_filename_label",
                "accepted": True,
                "reason": "accepted_current_label",
                "output_path": str(output_path),
                "rank_path": str(record.path) if record.kind == "rank" else "",
                "suit_path": str(record.path) if record.kind == "suit" else "",
                "card_path": "",
            }
        )

    predictions_csv = output_dir / "predictions.csv"
    output_review_csv = output_dir / "review.csv"
    write_predictions_csv(predictions_csv, rows)
    write_predictions_csv(output_review_csv, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "kind": kind,
        "input_review_csv": str(review_csv) if review_csv else "",
        "allowed_review_reasons": allowed_review_reasons or (["ok"] if review_csv else []),
        "processed": len(rows),
        "copied": copied,
        "skipped": skipped,
        "counts": count_rows(rows),
        "files": {
            "predictions_csv": str(predictions_csv),
            "review_csv": str(output_review_csv),
            "dataset_dir": str(output_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_allowed_review_paths(review_csv: Path | None, allowed_reasons: list[str]) -> set[Path]:
    if review_csv is None:
        return set()
    review_csv = Path(review_csv)
    allowed = {str(reason).strip() for reason in allowed_reasons if str(reason).strip()}
    paths: set[Path] = set()
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            reason = str(row.get("review_reason") or "").strip()
            if allowed and reason not in allowed:
                continue
            for key in ("card0_rank_path", "card1_rank_path", "card0_suit_path", "card1_suit_path"):
                value = str(row.get(key) or "").strip()
                if not value:
                    continue
                path = Path(value)
                if path.exists():
                    paths.add(path.resolve())
    return paths


def collect_crop_records(input_dirs: list[Path], *, allowed_kinds: tuple[str, ...]) -> list[CropRecord]:
    records: list[CropRecord] = []
    seen: set[Path] = set()
    for root in input_dirs:
        root = Path(root)
        for kind in allowed_kinds:
            candidates = candidate_paths(root, kind)
            for path in candidates:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                records.append(CropRecord(kind=kind, path=path, current_label=infer_current_label(path, kind)))
    return records


def limit_records_per_kind(records: list[CropRecord], *, max_per_kind: int) -> list[CropRecord]:
    counts: dict[str, int] = {}
    limited: list[CropRecord] = []
    for record in records:
        count = counts.get(record.kind, 0)
        if count >= max_per_kind:
            continue
        counts[record.kind] = count + 1
        limited.append(record)
    return limited


def candidate_paths(root: Path, kind: str) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS and infer_kind_from_path(root) == kind else []
    if not root.exists():
        return []
    direct = root / kind
    if direct.exists():
        return sorted(path for path in direct.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if root.name.lower() == kind:
        return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def infer_kind_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "rank" in parts:
        return "rank"
    if "suit" in parts:
        return "suit"
    return None


def infer_current_label(path: Path, kind: str) -> str:
    labels = set(RANK_LABELS if kind == "rank" else SUIT_LABELS)
    parent = path.parent.name
    if parent in labels:
        return parent
    parsed_card = parse_card_label(path)
    if parsed_card is not None:
        rank, suit = parsed_card
        return rank if kind == "rank" else suit
    parts = path.stem.split("_")
    for part in parts:
        normalized = part.upper() if kind == "rank" else part.lower()
        if normalized in labels:
            return normalized
    return ""


def should_accept_prediction(
    record: CropRecord,
    prediction: dict[str, Any] | None,
    *,
    rank_score_threshold: float,
    rank_margin_threshold: float,
    suit_score_threshold: float,
    suit_margin_threshold: float,
    require_current_agreement: bool,
) -> tuple[bool, str]:
    if prediction is None:
        return False, "teacher_missing"
    label = str(prediction.get("label") or "")
    valid_labels = set(RANK_LABELS if record.kind == "rank" else SUIT_LABELS)
    if label not in valid_labels:
        return False, "invalid_label"
    score = safe_float(prediction.get("score")) or 0.0
    margin = safe_float(prediction.get("margin")) or 0.0
    score_threshold = rank_score_threshold if record.kind == "rank" else suit_score_threshold
    margin_threshold = rank_margin_threshold if record.kind == "rank" else suit_margin_threshold
    reasons = []
    if score < score_threshold:
        reasons.append("low_score")
    if margin < margin_threshold:
        reasons.append("low_margin")
    if require_current_agreement and record.current_label and record.current_label != label:
        reasons.append("current_disagrees")
    if reasons:
        return False, ",".join(reasons)
    if record.current_label and record.current_label != label:
        return True, "accepted_teacher_over_current"
    return True, "accepted"


def build_row(
    index: int,
    record: CropRecord,
    prediction: dict[str, Any] | None,
    *,
    accepted: bool,
    reason: str,
    output_path: str,
) -> dict[str, Any]:
    label = str((prediction or {}).get("label") or "")
    row = {
        "index": index,
        "kind": record.kind,
        "input_path": str(record.path),
        "current_label": record.current_label,
        "teacher_label": label,
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
    return row


def copy_labeled_crop(record: CropRecord, prediction: dict[str, Any], output_dir: Path, index: int) -> Path:
    label = safe_label(str(prediction.get("label") or "_unknown"))
    destination = output_dir / record.kind / label / f"teacher_{index:06d}_{safe_stem(record.path.stem)}{record.path.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.path, destination)
    return destination


def copy_current_labeled_crop(record: CropRecord, output_dir: Path, index: int) -> Path:
    label = safe_label(record.current_label or "_unknown")
    destination = output_dir / record.kind / label / f"current_{index:06d}_{safe_stem(record.path.stem)}{record.path.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.path, destination)
    return destination


def copy_review_crop(record: CropRecord, prediction: dict[str, Any] | None, review_dir: Path, index: int) -> Path | None:
    if not record.path.exists():
        return None
    label = safe_label(str((prediction or {}).get("label") or "unknown"))
    destination = (
        review_dir
        / record.kind
        / f"review_{index:06d}_{safe_label(record.current_label or 'none')}_to_{label}_{safe_stem(record.path.stem)}{record.path.suffix.lower()}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.path, destination)
    return destination


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], *, include_final_columns: bool = False) -> None:
    fields = [
        "index",
        "kind",
        "input_path",
        "current_label",
        "teacher_label",
        "teacher_score",
        "teacher_margin",
        "teacher_second_score",
        "teacher_model",
        "accepted",
        "reason",
        "output_path",
        "rank_path",
        "suit_path",
        "card_path",
    ]
    if include_final_columns:
        fields.extend(["final_rank", "final_suit", "note"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def format_card_crop_label_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"label-card-crops failed: {payload.get('error')}"
    files = payload.get("files") or {}
    counts = payload.get("counts") or {}
    return "\n".join(
        [
            f"Processed: {payload.get('processed', 0)}",
            f"Accepted: {payload.get('accepted', 0)}",
            f"Copied accepted: {payload.get('copied_accepted', payload.get('accepted', 0))}",
            f"Review: {payload.get('review', 0)}",
            f"Counts: {json.dumps(counts, ensure_ascii=False)}",
            f"Accepted dir: {files.get('accepted_dir')}",
            f"Review CSV: {files.get('review_csv')}",
            f"Predictions CSV: {files.get('predictions_csv')}",
        ]
    )


def format_organize_card_crops_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"organize-card-crops failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Processed: {payload.get('processed', 0)}",
            f"Copied: {payload.get('copied', 0)}",
            f"Skipped: {payload.get('skipped', 0)}",
            f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
            f"Dataset dir: {files.get('dataset_dir')}",
            f"Review CSV: {files.get('review_csv')}",
        ]
    )


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
