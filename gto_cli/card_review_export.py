from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .card_glyph_export import export_rank_glyph_image, safe_label, safe_stem
from .cv_validate import classify_frame
from .live_vision import build_realtime_state, stabilize_hero_cards
from .screen_vision import detect_auto_bbox, poker_table_visibility, should_accept_bbox_refresh
from .screen_vision import bbox_changed
from .video_vision import (
    analyze_video_frame,
    build_layout_profile,
    choose_template,
    layout_profile_is_strong,
    layout_profile_quality,
    load_cv,
    normalized_suit_component_by_label,
    sample_times,
)


REVIEW_COLUMNS = [
    "video",
    "timestamp_sec",
    "frame_index",
    "class",
    "review_reason",
    "raw_hero_cards",
    "stabilized_hero_cards",
    "board",
    "street",
    "dealer",
    "hero_position",
    "hero_turn",
    "card0",
    "card0_rank_confidence",
    "card0_rank_margin",
    "card0_suit_confidence",
    "card0_suit_margin",
    "card0_roi_mode",
    "card0_card_path",
    "card0_rank_path",
    "card0_suit_path",
    "card1",
    "card1_rank_confidence",
    "card1_rank_margin",
    "card1_suit_confidence",
    "card1_suit_margin",
    "card1_roi_mode",
    "card1_card_path",
    "card1_rank_path",
    "card1_suit_path",
    "table_frame_path",
    "final_card0",
    "final_card1",
    "notes",
]


def export_card_review(
    *,
    video_paths: list[Path],
    output_dir: Path,
    template_path: Path | None = None,
    seat_count: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 10.0,
    max_frames: int | None = None,
    min_confidence: float = 0.35,
    auto_bbox_refresh_sec: float = 300.0,
    lock_layout: bool = True,
    only_suspicious: bool = False,
    max_sheet_rows: int = 160,
) -> dict[str, Any]:
    cv2, np = load_cv()
    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    cards_dir = output_dir / "cards"
    ranks_dir = output_dir / "rank"
    suits_dir = output_dir / "suit"
    for directory in (output_dir, frames_dir, cards_dir, ranks_dir, suits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    review_csv_path = output_dir / "review.csv"
    review_md_path = output_dir / "review.md"
    sheet_path = output_dir / "review_sheet.jpg"

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    videos_summary: list[dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_stream:
        for video_path in [Path(path) for path in video_paths]:
            video_summary, video_rows = export_video_card_review(
                cv2=cv2,
                np=np,
                video_path=video_path,
                output_dir=output_dir,
                frames_dir=frames_dir,
                cards_dir=cards_dir,
                ranks_dir=ranks_dir,
                suits_dir=suits_dir,
                template=template,
                template_path=template_path,
                seat_count=seat_count,
                start_sec=start_sec,
                end_sec=end_sec,
                every_sec=every_sec,
                max_frames=max_frames,
                min_confidence=min_confidence,
                auto_bbox_refresh_sec=auto_bbox_refresh_sec,
                lock_layout=lock_layout,
                only_suspicious=only_suspicious,
                manifest_stream=manifest_stream,
            )
            videos_summary.append(video_summary)
            rows.extend(video_rows)

    write_review_csv(review_csv_path, rows)
    write_review_markdown(review_md_path, rows)
    write_review_sheet(sheet_path, rows[: max(0, int(max_sheet_rows))])

    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "template": str(template_path),
        "video_count": len(videos_summary),
        "sample": {
            "every_sec": float(every_sec),
            "max_frames_per_video": max_frames,
            "rows": len(rows),
            "only_suspicious": bool(only_suspicious),
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "counts": count_rows(rows),
        "files": {
            "manifest": str(manifest_path),
            "review_csv": str(review_csv_path),
            "review_md": str(review_md_path),
            "review_sheet": str(sheet_path),
            "frames_dir": str(frames_dir),
            "cards_dir": str(cards_dir),
        },
        "videos": videos_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_video_card_review(
    *,
    cv2: Any,
    np: Any,
    video_path: Path,
    output_dir: Path,
    frames_dir: Path,
    cards_dir: Path,
    ranks_dir: Path,
    suits_dir: Path,
    template: Any,
    template_path: Path,
    seat_count: int,
    start_sec: float | None,
    end_sec: float | None,
    every_sec: float,
    max_frames: int | None,
    min_confidence: float,
    auto_bbox_refresh_sec: float,
    lock_layout: bool,
    only_suspicious: bool,
    manifest_stream: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "ok": False, "error": "cannot_open"}, []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps else 0.0
    sample_start = 0.0 if start_sec is None else float(start_sec)
    sample_end = duration_sec if end_sec is None else float(end_sec)
    times = sample_times(sample_start, sample_end, every_sec, max_frames)

    region: dict[str, int] | None = None
    last_auto_bbox_sec = float("-inf")
    layout_profile: dict[str, Any] | None = None
    layout_locked = False
    hero_card_cache: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    sampled = 0
    skipped = 0
    slug = safe_stem(video_path.stem)

    for sample_index, timestamp in enumerate(times):
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            skipped += 1
            continue
        sampled += 1

        search_region = {"left": 0, "top": 0, "width": int(frame.shape[1]), "height": int(frame.shape[0])}
        warmup_bbox_refresh = bool(lock_layout and layout_locked and sample_index <= 6)
        should_refresh_bbox = (
            region is None
            or (lock_layout and not layout_locked)
            or warmup_bbox_refresh
            or (float(auto_bbox_refresh_sec) > 0 and timestamp - last_auto_bbox_sec >= max(1.0, float(auto_bbox_refresh_sec)))
        )
        detection = None
        if should_refresh_bbox:
            detection = detect_auto_bbox(
                cv2,
                np,
                frame,
                search_region,
                template,
                min_confidence,
                allow_native_window=False,
            )
            last_auto_bbox_sec = float(timestamp)
            if detection is not None:
                if region is None:
                    region = detection["region"]
                    if not layout_locked:
                        layout_profile = None
                else:
                    accepted, _reason = should_accept_bbox_refresh(region, detection["region"], search_region, detection)
                    if accepted:
                        changed = bbox_changed(region, detection["region"])
                        region = detection["region"]
                        if changed:
                            layout_profile = None
                            if lock_layout:
                                layout_locked = False
                        elif not layout_locked:
                            layout_profile = None
        if region is None:
            skipped += 1
            continue

        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        crop = frame[y : y + h, x : x + w]
        table_visible, _visibility = poker_table_visibility(cv2, crop)
        if not table_visible:
            skipped += 1
            continue

        if lock_layout and not layout_locked:
            candidate_profile = build_layout_profile(crop, [], hero_name=None)
            if layout_profile_is_strong(candidate_profile):
                layout_profile = candidate_profile
                layout_locked = True
        active_layout_profile = layout_profile if layout_locked else None

        try:
            frame_result = analyze_video_frame(
                crop,
                template,
                seat_count=seat_count,
                min_confidence=min_confidence,
                ocr=None,
                layout_profile=active_layout_profile,
            )
        except Exception:
            skipped += 1
            continue

        state = build_realtime_state(
            frame_result,
            video_path=video_path,
            timestamp_sec=round(float(timestamp), 3),
            frame_index=frame_index,
            sample_index=sample_index,
        )
        raw_hero_details = list((frame_result.get("cards") or {}).get("hero_details") or [])
        raw_hero_cards = list((frame_result.get("cards") or {}).get("hero") or [])
        hero_card_cache = stabilize_hero_cards(state, hero_card_cache)
        stabilized_cards = list((state.get("hero") or {}).get("cards") or [])
        row_class = classify_frame(crop, stabilized_cards)
        reason = review_reason(row_class, raw_hero_details, raw_hero_cards, stabilized_cards)
        if only_suspicious and reason == "ok":
            continue

        prefix = f"{slug}_f{frame_index:07d}_t{timestamp:08.3f}".replace(".", "p")
        table_frame_path = frames_dir / f"{prefix}_table.png"
        save_png(cv2, table_frame_path, crop)

        card_outputs = []
        for card_slot in range(2):
            detail = raw_hero_details[card_slot] if card_slot < len(raw_hero_details) else None
            card_outputs.append(
                export_card_slot(
                    cv2=cv2,
                    crop=crop,
                    detail=detail,
                    slot=card_slot,
                    prefix=prefix,
                    cards_dir=cards_dir,
                    ranks_dir=ranks_dir,
                    suits_dir=suits_dir,
                )
            )

        row = build_review_row(
            video_path=video_path,
            timestamp=timestamp,
            frame_index=frame_index,
            row_class=row_class,
            reason=reason,
            raw_hero_cards=raw_hero_cards,
            stabilized_cards=stabilized_cards,
            state=state,
            raw_hero_details=raw_hero_details,
            table_frame_path=table_frame_path,
            card_outputs=card_outputs,
            region=region,
            layout_locked=layout_locked,
            layout_quality=layout_profile_quality(layout_profile),
            detection=detection,
            template_path=template_path,
        )
        manifest_stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        rows.append(row)

    cap.release()
    return (
        {
            "video": str(video_path),
            "ok": True,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": round(float(duration_sec), 3),
            "sampled_frames": sampled,
            "exported_rows": len(rows),
            "skipped_frames": skipped,
        },
        rows,
    )


def export_card_slot(
    *,
    cv2: Any,
    crop: Any,
    detail: dict[str, Any] | None,
    slot: int,
    prefix: str,
    cards_dir: Path,
    ranks_dir: Path,
    suits_dir: Path,
) -> dict[str, str]:
    if not detail:
        return {"card_path": "", "rank_path": "", "suit_path": ""}
    card_crop = crop_detail(crop, detail.get("roi_box") or {})
    if card_crop is None or card_crop.size == 0:
        return {"card_path": "", "rank_path": "", "suit_path": ""}
    card = safe_label(str(detail.get("card") or "unknown"))
    card_path = cards_dir / f"{prefix}_slot{slot}_{card}_card.png"
    rank_path = ranks_dir / f"{prefix}_slot{slot}_{card}_rank.png"
    suit_path = suits_dir / f"{prefix}_slot{slot}_{card}_suit.png"
    save_png(cv2, card_path, card_crop)
    save_png(cv2, rank_path, export_rank_glyph_image(card_crop, str(detail.get("source") or "hero")))
    source = str(detail.get("source") or "hero")
    save_png(cv2, suit_path, normalized_suit_component_by_label(card_crop, (42, 42), source=source))
    return {"card_path": str(card_path), "rank_path": str(rank_path), "suit_path": str(suit_path)}


def build_review_row(
    *,
    video_path: Path,
    timestamp: float,
    frame_index: int,
    row_class: str,
    reason: str,
    raw_hero_cards: list[Any],
    stabilized_cards: list[Any],
    state: dict[str, Any],
    raw_hero_details: list[dict[str, Any]],
    table_frame_path: Path,
    card_outputs: list[dict[str, str]],
    region: dict[str, Any],
    layout_locked: bool,
    layout_quality: int,
    detection: dict[str, Any] | None,
    template_path: Path,
) -> dict[str, Any]:
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    hero_turn = state.get("hero_turn") or {}
    row: dict[str, Any] = {
        "video": str(video_path),
        "timestamp_sec": round(float(timestamp), 3),
        "frame_index": int(frame_index),
        "class": row_class,
        "review_reason": reason,
        "raw_hero_cards": " ".join(str(card) for card in raw_hero_cards),
        "stabilized_hero_cards": " ".join(str(card) for card in stabilized_cards),
        "board": " ".join(str(card) for card in (table.get("board") or [])),
        "street": table.get("street"),
        "dealer": table.get("dealer_seat"),
        "hero_position": hero.get("position"),
        "hero_turn": "yes" if hero_turn.get("is_turn") else "no",
        "table_frame_path": str(table_frame_path),
        "final_card0": "",
        "final_card1": "",
        "notes": "",
        "region": dict(region),
        "layout_locked": bool(layout_locked),
        "layout_quality": int(layout_quality),
        "auto_bbox": {
            "method": detection.get("method") if detection else None,
            "score": detection.get("score") if detection else None,
            "dealer_confidence": detection.get("dealer_confidence") if detection else None,
        },
        "template": str(template_path),
    }
    for slot in range(2):
        detail = raw_hero_details[slot] if slot < len(raw_hero_details) else {}
        outputs = card_outputs[slot] if slot < len(card_outputs) else {}
        row.update(
            {
                f"card{slot}": detail.get("card", ""),
                f"card{slot}_rank_confidence": detail.get("rank_confidence", ""),
                f"card{slot}_rank_margin": detail.get("rank_margin", ""),
                f"card{slot}_suit_confidence": detail.get("suit_confidence", ""),
                f"card{slot}_suit_margin": detail.get("suit_margin", ""),
                f"card{slot}_roi_mode": detail.get("roi_mode", ""),
                f"card{slot}_card_path": outputs.get("card_path", ""),
                f"card{slot}_rank_path": outputs.get("rank_path", ""),
                f"card{slot}_suit_path": outputs.get("suit_path", ""),
            }
        )
    return row


def review_reason(
    row_class: str,
    details: list[dict[str, Any]],
    raw_cards: list[Any],
    stabilized_cards: list[Any],
) -> str:
    if row_class != "complete":
        return row_class
    if raw_cards != stabilized_cards:
        return "stabilized"
    if len(details) < 2:
        return "missing_detail"
    reasons = []
    for detail in details[:2]:
        card = str(detail.get("card") or "")
        if "?" in card:
            reasons.append("unknown")
        rank_conf = float(detail.get("rank_confidence") or 0.0)
        rank_margin = float(detail.get("rank_margin") or 0.0)
        suit_conf = float(detail.get("suit_confidence") or 0.0)
        suit_margin = float(detail.get("suit_margin") or 0.0)
        if rank_conf < 0.62 or rank_margin < 0.10:
            reasons.append("rank_low")
        if suit_conf < 0.78 or suit_margin < 0.04:
            reasons.append("suit_low")
    return ",".join(sorted(set(reasons))) if reasons else "ok"


def crop_detail(frame: Any, box: dict[str, Any]) -> Any | None:
    if not box:
        return None
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, int(box.get("x", 0))))
    y1 = max(0, min(frame_h - 1, int(box.get("y", 0))))
    x2 = max(x1 + 1, min(frame_w, x1 + int(box.get("width", 0))))
    y2 = max(y1 + 1, min(frame_h, y1 + int(box.get("height", 0))))
    return frame[y1:y2, x1:x2]


def save_png(cv2: Any, path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        path.write_bytes(encoded.tobytes())


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_markdown(path: Path, rows: list[dict[str, Any]], limit: int = 240) -> None:
    lines = [
        "# Hero Card Review",
        "",
        "| # | Time | Reason | Raw | Stabilized | Board | Card 0 | Card 1 | Frame |",
        "|---:|---:|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        frame_link = row.get("table_frame_path") or ""
        card0 = row.get("card0_card_path") or ""
        card1 = row.get("card1_card_path") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row.get("timestamp_sec", "")),
                    str(row.get("review_reason", "")),
                    str(row.get("raw_hero_cards", "")),
                    str(row.get("stabilized_hero_cards", "")),
                    str(row.get("board", "")),
                    image_cell(card0, row.get("card0", "")),
                    image_cell(card1, row.get("card1", "")),
                    image_cell(frame_link, "frame"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def image_cell(path: str, label: Any) -> str:
    if not path:
        return str(label or "")
    return f"![{label}]({Path(path).resolve().as_posix()})"


def write_review_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    cv2, np = load_cv()
    if not rows:
        canvas = np.full((120, 900, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No review rows", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
        return
    row_h = 132
    width = 1280
    canvas = np.full((row_h * len(rows), width, 3), 250, dtype=np.uint8)
    for index, row in enumerate(rows):
        y = index * row_h
        cv2.rectangle(canvas, (0, y), (width - 1, y + row_h - 1), (210, 210, 210), 1)
        draw_text(cv2, canvas, f"{index+1:03d} t={row.get('timestamp_sec')} {row.get('review_reason')}", 10, y + 24)
        draw_text(cv2, canvas, f"raw {row.get('raw_hero_cards') or '-'}  stable {row.get('stabilized_hero_cards') or '-'}", 10, y + 52)
        draw_text(cv2, canvas, f"{row.get('street') or '-'} {row.get('hero_position') or '-'} turn={row.get('hero_turn')}", 10, y + 80)
        paste_image(canvas, load_image(cv2, row.get("card0_card_path")), 370, y + 8, 90, 116)
        paste_image(canvas, load_image(cv2, row.get("card1_card_path")), 470, y + 8, 90, 116)
        paste_image(canvas, load_image(cv2, row.get("table_frame_path")), 590, y + 8, 240, 116)
        draw_text(cv2, canvas, detail_text(row, 0), 850, y + 34)
        draw_text(cv2, canvas, detail_text(row, 1), 850, y + 70)
    save_png(cv2, path, canvas)


def detail_text(row: dict[str, Any], slot: int) -> str:
    return (
        f"c{slot} {row.get(f'card{slot}') or '-'} "
        f"r={row.get(f'card{slot}_rank_confidence') or '-'} "
        f"rm={row.get(f'card{slot}_rank_margin') or '-'} "
        f"s={row.get(f'card{slot}_suit_confidence') or '-'} "
        f"sm={row.get(f'card{slot}_suit_margin') or '-'}"
    )


def draw_text(cv2: Any, image: Any, text: str, x: int, y: int) -> None:
    cv2.putText(image, str(text)[:80], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)


def load_image(cv2: Any, path: Any) -> Any | None:
    if not path:
        return None
    file_path = Path(str(path))
    if not file_path.exists():
        return None
    data = file_path.read_bytes()
    cv2_module, np = load_cv()
    del cv2_module
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image


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


def count_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("review_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def format_card_review_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"export-card-review failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Review rows: {payload.get('sample', {}).get('rows')}",
            f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
            f"Review CSV: {files.get('review_csv')}",
            f"Review sheet: {files.get('review_sheet')}",
            f"Manifest: {files.get('manifest')}",
        ]
    )
