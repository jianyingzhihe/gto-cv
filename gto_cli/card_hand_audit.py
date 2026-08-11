from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .card_active_learning import evaluate_glyph, prediction_summary
from .card_classifier import classify_suit_glyph
from .card_deep_model import classify_deep_glyph, warm_deep_card_models


HAND_REVIEW_COLUMNS = [
    "audit_priority",
    "audit_reason",
    "video",
    "timestamp_sec",
    "frame_index",
    "class",
    "raw_hero_cards",
    "stabilized_hero_cards",
    "card0",
    "card0_consensus",
    "card0_rank_eval",
    "card0_suit_eval",
    "card0_open_suit",
    "card0_card_path",
    "card0_rank_path",
    "card0_suit_path",
    "card1",
    "card1_consensus",
    "card1_rank_eval",
    "card1_suit_eval",
    "card1_open_suit",
    "card1_card_path",
    "card1_rank_path",
    "card1_suit_path",
    "table_frame_path",
    "street",
    "dealer",
    "hero_position",
    "hero_turn",
    "final_card0",
    "final_card1",
    "notes",
]


def audit_card_review(
    *,
    review_csv: Path,
    output_dir: Path,
    teacher_model_dir: Path | None = None,
    teacher_rank_model_dir: Path | None = None,
    teacher_suit_model_dir: Path | None = None,
    realtime_model_dir: Path | None = None,
    realtime_rank_model_dir: Path | None = None,
    realtime_suit_model_dir: Path | None = None,
    rank_confidence_threshold: float = 0.82,
    suit_confidence_threshold: float = 0.72,
    open_suit_score_threshold: float = 0.78,
    open_suit_margin_threshold: float = 0.08,
    max_review: int = 240,
    copy_review_assets: bool = True,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_assets_dir = output_dir / "review_assets"
    if copy_review_assets:
        review_assets_dir.mkdir(parents=True, exist_ok=True)

    teacher_rank_dir = teacher_rank_model_dir or teacher_model_dir
    teacher_suit_dir = teacher_suit_model_dir or teacher_model_dir
    realtime_rank_dir = realtime_rank_model_dir or realtime_model_dir
    realtime_suit_dir = realtime_suit_model_dir or realtime_model_dir
    for model_dir in sorted(
        {Path(path) for path in (teacher_rank_dir, teacher_suit_dir, realtime_rank_dir, realtime_suit_dir) if path},
        key=lambda path: str(path),
    ):
        warm_deep_card_models(model_dir)

    input_rows = read_csv_rows(review_csv)
    skipped_no_hand = 0
    audited = []
    for index, row in enumerate(input_rows):
        if str(row.get("class") or "") == "empty_or_no_hand":
            skipped_no_hand += 1
            continue
        audit_row = audit_hand_row(
            cv2=cv2,
            index=index,
            row=row,
            teacher_rank_dir=teacher_rank_dir,
            teacher_suit_dir=teacher_suit_dir,
            realtime_rank_dir=realtime_rank_dir,
            realtime_suit_dir=realtime_suit_dir,
            rank_confidence_threshold=rank_confidence_threshold,
            suit_confidence_threshold=suit_confidence_threshold,
            open_suit_score_threshold=open_suit_score_threshold,
            open_suit_margin_threshold=open_suit_margin_threshold,
        )
        audited.append(audit_row)

    review_rows = sorted(
        (row for row in audited if row.get("needs_review")),
        key=lambda item: float(item.get("audit_priority") or 0.0),
        reverse=True,
    )[: max(0, int(max_review))]
    if copy_review_assets:
        copy_review_files(review_rows, review_assets_dir)

    audit_csv = output_dir / "audit.csv"
    review_out_csv = output_dir / "review.csv"
    review_md = output_dir / "review.md"
    review_sheet = output_dir / "review_sheet.jpg"
    write_audit_csv(audit_csv, audited)
    write_audit_csv(review_out_csv, review_rows, include_final_columns=True)
    write_review_markdown(review_md, review_rows)
    write_review_sheet(review_sheet, review_rows)
    summary = {
        "ok": True,
        "review_csv": str(review_csv),
        "output_dir": str(output_dir),
        "teacher_rank_model_dir": str(teacher_rank_dir) if teacher_rank_dir else "",
        "teacher_suit_model_dir": str(teacher_suit_dir) if teacher_suit_dir else "",
        "realtime_rank_model_dir": str(realtime_rank_dir) if realtime_rank_dir else "",
        "realtime_suit_model_dir": str(realtime_suit_dir) if realtime_suit_dir else "",
        "audited": len(audited),
        "skipped_no_hand": skipped_no_hand,
        "needs_review": len([row for row in audited if row.get("needs_review")]),
        "review_limited_to": len(review_rows),
        "counts": count_reasons(audited),
        "files": {
            "audit_csv": str(audit_csv),
            "review_csv": str(review_out_csv),
            "review_md": str(review_md),
            "review_sheet": str(review_sheet),
            "review_assets_dir": str(review_assets_dir) if copy_review_assets else "",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit_hand_row(
    *,
    cv2: Any,
    index: int,
    row: dict[str, Any],
    teacher_rank_dir: Path | None,
    teacher_suit_dir: Path | None,
    realtime_rank_dir: Path | None,
    realtime_suit_dir: Path | None,
    rank_confidence_threshold: float,
    suit_confidence_threshold: float,
    open_suit_score_threshold: float,
    open_suit_margin_threshold: float,
) -> dict[str, Any]:
    slot_results = []
    reasons = []
    priority = 0.0
    for slot in (0, 1):
        result = audit_card_slot(
            cv2=cv2,
            row=row,
            slot=slot,
            teacher_rank_dir=teacher_rank_dir,
            teacher_suit_dir=teacher_suit_dir,
            realtime_rank_dir=realtime_rank_dir,
            realtime_suit_dir=realtime_suit_dir,
            rank_confidence_threshold=rank_confidence_threshold,
            suit_confidence_threshold=suit_confidence_threshold,
            open_suit_score_threshold=open_suit_score_threshold,
            open_suit_margin_threshold=open_suit_margin_threshold,
        )
        slot_results.append(result)
        priority = max(priority, float(result.get("priority") or 0.0))
        for reason in result.get("reasons") or []:
            reasons.append(f"card{slot}_{reason}")
    if str(row.get("class") or "") != "complete":
        reasons.append(str(row.get("class") or "not_complete"))
        priority += 20.0
    if str(row.get("review_reason") or "") not in ("", "ok"):
        reasons.append(str(row.get("review_reason")))
        priority += 3.0

    flattened = {
        "audit_index": index,
        "audit_priority": round(priority, 4),
        "audit_reason": ";".join(sorted(set(reason for reason in reasons if reason))),
        "needs_review": bool(reasons),
    }
    for key in (
        "video",
        "timestamp_sec",
        "frame_index",
        "class",
        "raw_hero_cards",
        "stabilized_hero_cards",
        "table_frame_path",
        "street",
        "dealer",
        "hero_position",
        "hero_turn",
    ):
        flattened[key] = row.get(key, "")
    for slot, slot_result in enumerate(slot_results):
        flattened.update(
            {
                f"card{slot}": slot_result.get("current_card", ""),
                f"card{slot}_consensus": slot_result.get("consensus_card", ""),
                f"card{slot}_rank_eval": slot_result.get("rank_summary", ""),
                f"card{slot}_suit_eval": slot_result.get("suit_summary", ""),
                f"card{slot}_open_suit": prediction_summary(slot_result.get("open_suit")),
                f"card{slot}_card_path": row.get(f"card{slot}_card_path", ""),
                f"card{slot}_rank_path": row.get(f"card{slot}_rank_path", ""),
                f"card{slot}_suit_path": row.get(f"card{slot}_suit_path", ""),
            }
        )
    flattened["final_card0"] = ""
    flattened["final_card1"] = ""
    flattened["notes"] = ""
    return flattened


def audit_card_slot(
    *,
    cv2: Any,
    row: dict[str, Any],
    slot: int,
    teacher_rank_dir: Path | None,
    teacher_suit_dir: Path | None,
    realtime_rank_dir: Path | None,
    realtime_suit_dir: Path | None,
    rank_confidence_threshold: float,
    suit_confidence_threshold: float,
    open_suit_score_threshold: float,
    open_suit_margin_threshold: float,
) -> dict[str, Any]:
    current_card = clean_card(row.get(f"card{slot}") or "")
    rank_path = Path(row.get(f"card{slot}_rank_path") or "")
    suit_path = Path(row.get(f"card{slot}_suit_path") or "")
    if not current_card and not rank_path.is_file() and not suit_path.is_file():
        return {
            "current_card": "",
            "consensus_card": "",
            "rank": {},
            "suit": {},
            "open_suit": None,
            "rank_summary": "",
            "suit_summary": "",
            "reasons": [],
            "priority": 0.0,
        }
    current_rank = current_card[0] if len(current_card) >= 1 else ""
    current_suit = current_card[1] if len(current_card) >= 2 else ""
    rank_eval = evaluate_glyph(
        cv2=cv2,
        kind="rank",
        image_path=rank_path,
        current_label=current_rank,
        current_confidence=row.get(f"card{slot}_rank_confidence"),
        current_margin=row.get(f"card{slot}_rank_margin"),
        confidence_threshold=rank_confidence_threshold,
        teacher_model_dir=teacher_rank_dir,
        realtime_model_dir=realtime_rank_dir,
    )
    suit_eval = evaluate_glyph(
        cv2=cv2,
        kind="suit",
        image_path=suit_path,
        current_label=current_suit,
        current_confidence=row.get(f"card{slot}_suit_confidence"),
        current_margin=row.get(f"card{slot}_suit_margin"),
        confidence_threshold=suit_confidence_threshold,
        teacher_model_dir=teacher_suit_dir,
        realtime_model_dir=realtime_suit_dir,
    )
    open_suit = classify_open_suit(
        cv2,
        suit_path,
        teacher_model_dir=teacher_suit_dir,
        realtime_model_dir=realtime_suit_dir,
    )
    consensus_rank = str(rank_eval.get("consensus_label") or current_rank or "?")
    consensus_suit = str(suit_eval.get("consensus_label") or current_suit or "?")
    reasons = []
    if rank_eval.get("needs_review"):
        reasons.extend(f"rank_{reason}" for reason in rank_eval.get("reasons") or ["needs_review"])
    if suit_eval.get("needs_review"):
        reasons.extend(f"suit_{reason}" for reason in suit_eval.get("reasons") or ["needs_review"])
    if open_suit:
        open_label = str(open_suit.get("label") or "")
        open_score = safe_float(open_suit.get("score")) or 0.0
        open_margin = safe_float(open_suit.get("margin")) or 0.0
        open_is_confident = open_score >= open_suit_score_threshold and open_margin >= open_suit_margin_threshold
        has_current_suit = bool(current_suit and current_suit != "?")
        open_same_family = same_suit_color(current_suit, open_label) if has_current_suit else True
        if (
            has_current_suit
            and open_label
            and open_label != current_suit
            and open_is_confident
        ):
            if open_same_family:
                reasons.append("open_suit_same_color_alt")
            else:
                reasons.append("open_suit_alt")
        elif (not current_suit or current_suit == "?") and open_label and open_is_confident:
            reasons.append("open_suit_suggests")
    consensus_card = f"{consensus_rank}{consensus_suit}"
    if current_card and "?" not in current_card and consensus_card and "?" not in consensus_card and consensus_card != current_card:
        reasons.append("consensus_card_disagrees")
    priority = max(float(rank_eval.get("priority") or 0.0), float(suit_eval.get("priority") or 0.0))
    if any(reason.endswith("disagrees") or "disagrees" in reason for reason in reasons):
        priority += 18.0
    return {
        "current_card": current_card,
        "consensus_card": consensus_card,
        "rank": rank_eval,
        "suit": suit_eval,
        "open_suit": open_suit,
        "rank_summary": eval_summary(rank_eval),
        "suit_summary": eval_summary(suit_eval),
        "reasons": sorted(set(reasons)),
        "priority": round(priority, 4),
    }


def classify_open_suit(
    cv2: Any,
    image_path: Path,
    *,
    teacher_model_dir: Path | None,
    realtime_model_dir: Path | None,
) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    candidates = []
    knn = classify_suit_glyph(image, allowed=None)
    if knn:
        candidates.append({**knn, "backend": "knn"})
    if teacher_model_dir:
        teacher = classify_deep_glyph(image, "suit", model_dir=teacher_model_dir, allowed=None)
        if teacher:
            candidates.append({**teacher, "backend": "teacher"})
    if realtime_model_dir:
        realtime = classify_deep_glyph(image, "suit", model_dir=realtime_model_dir, allowed=None)
        if realtime:
            candidates.append({**realtime, "backend": "realtime"})
    if not candidates:
        return None
    return max(candidates, key=lambda item: (safe_float(item.get("score")) or 0.0, safe_float(item.get("margin")) or 0.0))


def eval_summary(evaluation: dict[str, Any]) -> str:
    pieces = [
        f"cur={evaluation.get('current_label') or '-'}",
        f"cons={evaluation.get('consensus_label') or '-'}",
        f"conf={format_float(evaluation.get('current_confidence'))}",
        f"margin={format_float(evaluation.get('current_margin'))}",
        f"knn={prediction_summary(evaluation.get('knn')) or '-'}",
        f"teacher={prediction_summary(evaluation.get('teacher')) or '-'}",
        f"rt={prediction_summary(evaluation.get('realtime')) or '-'}",
    ]
    reasons = ",".join(evaluation.get("reasons") or [])
    if reasons:
        pieces.append(f"reasons={reasons}")
    return " ".join(pieces)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_audit_csv(path: Path, rows: list[dict[str, Any]], *, include_final_columns: bool = False) -> None:
    fields = HAND_REVIEW_COLUMNS[:]
    if not include_final_columns:
        fields = [field for field in fields if field not in ("final_card0", "final_card1", "notes")]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_markdown(path: Path, rows: list[dict[str, Any]], limit: int = 240) -> None:
    lines = [
        "# Hand Truth Audit",
        "",
        "| # | Priority | Reason | Time | Cards -> Consensus | Card 0 | Card 1 | Frame |",
        "|---:|---:|---|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        cards = f"{row.get('card0')} {row.get('card1')} -> {row.get('card0_consensus')} {row.get('card1_consensus')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row.get("audit_priority", "")),
                    str(row.get("audit_reason", "")),
                    str(row.get("timestamp_sec", "")),
                    cards,
                    image_md(row.get("card0_card_path"), row.get("card0", "")),
                    image_md(row.get("card1_card_path"), row.get("card1", "")),
                    image_md(row.get("table_frame_path"), "frame"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_sheet(path: Path, rows: list[dict[str, Any]], limit: int = 120) -> None:
    cv2, np = load_cv()
    rows = rows[: max(0, int(limit))]
    if not rows:
        canvas = np.full((120, 900, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No hand audit rows", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
        return
    row_h = 172
    width = 1500
    canvas = np.full((row_h * len(rows), width, 3), 248, dtype=np.uint8)
    for index, row in enumerate(rows):
        y = index * row_h
        cv2.rectangle(canvas, (0, y), (width - 1, y + row_h - 1), (210, 210, 210), 1)
        draw_text(cv2, canvas, f"{index+1:03d} p={row.get('audit_priority')} t={row.get('timestamp_sec')} {row.get('audit_reason')}", 10, y + 24)
        draw_text(
            cv2,
            canvas,
            f"{row.get('card0')} {row.get('card1')} -> {row.get('card0_consensus')} {row.get('card1_consensus')}",
            10,
            y + 52,
        )
        draw_text(cv2, canvas, f"{row.get('street') or '-'} {row.get('hero_position') or '-'} turn={row.get('hero_turn')}", 10, y + 80)
        paste_image(canvas, load_image(cv2, row.get("card0_card_path")), 360, y + 10, 90, 130)
        paste_image(canvas, load_image(cv2, row.get("card1_card_path")), 460, y + 10, 90, 130)
        paste_image(canvas, load_image(cv2, row.get("table_frame_path")), 570, y + 10, 250, 140)
        draw_text(cv2, canvas, f"c0 {shorten(row.get('card0_rank_eval'))}", 840, y + 34)
        draw_text(cv2, canvas, f"c0 suit {shorten(row.get('card0_suit_eval'))} open={row.get('card0_open_suit')}", 840, y + 62)
        draw_text(cv2, canvas, f"c1 {shorten(row.get('card1_rank_eval'))}", 840, y + 98)
        draw_text(cv2, canvas, f"c1 suit {shorten(row.get('card1_suit_eval'))} open={row.get('card1_open_suit')}", 840, y + 126)
    cv2.imwrite(str(path), canvas)


def copy_review_files(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for index, row in enumerate(rows):
        prefix = f"{index:04d}_t{safe_path_text(row.get('timestamp_sec'))}"
        for key in (
            "table_frame_path",
            "card0_card_path",
            "card0_rank_path",
            "card0_suit_path",
            "card1_card_path",
            "card1_rank_path",
            "card1_suit_path",
        ):
            src = Path(str(row.get(key) or ""))
            if not src.is_file():
                continue
            dst = output_dir / f"{prefix}_{key}_{src.name}"
            shutil.copy2(src, dst)


def count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("audit_reason") or "ok").split(";")
        for reason in reasons:
            reason = reason.strip() or "ok"
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def clean_card(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("10", "T")


def same_suit_color(left: str, right: str) -> bool:
    black = {"s", "c"}
    red = {"h", "d"}
    return (left in black and right in black) or (left in red and right in red)


def image_md(path_value: Any, label: Any) -> str:
    if not path_value:
        return str(label or "")
    path = Path(str(path_value)).resolve()
    return f"![{label}]({path.as_posix()})"


def load_image(cv2: Any, path: Any) -> Any | None:
    if not path:
        return None
    file_path = Path(str(path))
    if not file_path.exists():
        return None
    data = file_path.read_bytes()
    _cv2, np = load_cv()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def paste_image(canvas: Any, image: Any | None, x: int, y: int, width: int, height: int) -> None:
    cv2, _np = load_cv()
    if image is None or image.size == 0:
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (180, 180, 180), 1)
        return
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = x + (width - resized_w) // 2
    y0 = y + (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized


def draw_text(cv2: Any, image: Any, text: Any, x: int, y: int) -> None:
    cv2.putText(image, str(text)[:120], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)


def shorten(value: Any, limit: int = 95) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def safe_path_text(value: Any) -> str:
    return "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in str(value or "x"))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: Any) -> str:
    number = safe_float(value)
    return "-" if number is None else f"{number:.3f}"


def format_hand_audit_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"audit-card-review failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Audited hands: {payload.get('audited', 0)}",
            f"Needs review: {payload.get('needs_review', 0)}",
            f"Review rows: {payload.get('review_limited_to', 0)}",
            f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
            f"Review CSV: {files.get('review_csv')}",
            f"Review sheet: {files.get('review_sheet')}",
        ]
    )


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
