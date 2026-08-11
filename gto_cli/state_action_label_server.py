from __future__ import annotations

import csv
import html
import json
import mimetypes
import shutil
import threading
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

from .state_review import resolve_source_image


ACTION_LABEL_QUEUE_COLUMNS = (
    "label_id",
    "event_index",
    "line_number",
    "timestamp_sec",
    "source_kind",
    "frame_scope",
    "street",
    "hero_position",
    "hero_gto_position",
    "hero_preflop_order",
    "hero_postflop_order",
    "hero_cards",
    "board",
    "pot_bb",
    "to_call_bb",
    "visible_bets",
    "cv_hero_turn",
    "cv_turn_confidence",
    "cv_turn_reason",
    "cv_advice_reason",
    "cv_advice_summary",
    "cv_preflop_tracker_reason",
    "cv_preflop_history",
    "cv_analysis",
    "cv_actions",
    "cv_call_amount_bb",
    "cv_raise_amount_bb",
    "cv_red_button_regions",
    "frame_path",
    "panel_crop_path",
    "final_hero_turn",
    "final_fast_fold_state",
    "final_panel_template",
    "final_actions",
    "final_disabled_actions",
    "final_call_amount_bb",
    "final_raise_to_bb",
    "notes",
    "ignored",
    "updated_at",
)

HERO_TURN_VALUES = ("yes", "no", "uncertain")
FAST_FOLD_VALUES = ("available", "not_visible", "uncertain")
PANEL_TEMPLATES = (
    "no_hero_action",
    "preflop_fold_call_raise",
    "postflop_check_bet",
    "postflop_fold_call_raise",
    "allin_or_special",
    "action_panel_cropped",
    "no_action_panel",
    "other_or_occluded",
)
VISIBLE_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")

TEMPLATE_LABELS = {
    "no_hero_action": "未轮到 Hero（无常规可点按钮）",
    "preflop_fold_call_raise": "翻前 FOLD / CALL / RAISE",
    "postflop_check_bet": "翻后 CHECK / BET",
    "postflop_fold_call_raise": "翻后 FOLD / CALL / RAISE",
    "allin_or_special": "全下 / 特殊按钮",
    "action_panel_cropped": "操作区被截图裁掉",
    "no_action_panel": "无操作条 / 未轮到 Hero",
    "other_or_occluded": "其他 / 被遮挡",
}

TEMPLATE_ACTIONS = {
    "no_hero_action": (),
    "preflop_fold_call_raise": ("fold", "call", "raise"),
    "postflop_check_bet": ("check", "bet"),
    "postflop_fold_call_raise": ("fold", "call", "raise"),
    "allin_or_special": ("all_in",),
    "action_panel_cropped": (),
    "no_action_panel": (),
    "other_or_occluded": (),
}


def prepare_state_action_label_queue(
    *,
    events_path: Path,
    output_dir: Path,
    max_items: int = 240,
    extra_events: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Build a local, screenshot-backed queue for validating action panels.

    This deliberately stores human observations separately from CV output.  A
    CV signal such as ``hero_turn=true`` is only metadata in this queue, not a
    label or a decision input.
    """

    events_path = Path(events_path)
    if not events_path.is_file():
        raise FileNotFoundError(f"events JSONL not found: {events_path}")
    if max_items < 1:
        raise ValueError("max_items must be at least 1")

    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    queue_csv = output_dir / "state_action_label_queue.csv"
    previous = load_action_queue_csv(queue_csv)[0] if queue_csv.exists() else []
    previous_by_id = {str(row.get("label_id") or ""): row for row in previous}

    event_paths = [Path(events_path), *(Path(path) for path in extra_events)]
    candidates: list[dict[str, Any]] = []
    stats = {
        "lines": 0,
        "ok_events": 0,
        "image_backed": 0,
        "missing_visual_evidence": 0,
        "hero_turn": 0,
        "control_visible": 0,
    }
    for source_index, path in enumerate(event_paths):
        source_candidates, source_stats = load_action_candidates(path, source_index=source_index)
        candidates.extend(source_candidates)
        for key, value in source_stats.items():
            stats[key] = stats.get(key, 0) + value
    selected = select_action_candidates(candidates, limit=max_items)
    rows: list[dict[str, str]] = []
    for event in selected:
        label_id = event["label_id"]
        frame_path = materialize_candidate_frame(event, assets_dir, f"{label_id}_frame")
        panel_path = write_panel_crop(
            frame_path,
            assets_dir / f"{label_id}_panel.png",
            event=as_dict(event.get("event")),
        )
        row = event_to_queue_row(event, frame_path=frame_path, panel_path=panel_path)
        existing = previous_by_id.get(label_id)
        if existing:
            for column in (
                "final_hero_turn",
                "final_fast_fold_state",
                "final_panel_template",
                "final_actions",
                "final_disabled_actions",
                "final_call_amount_bb",
                "final_raise_to_bb",
                "notes",
                "ignored",
                "updated_at",
            ):
                row[column] = str(existing.get(column) or "")
            migrate_legacy_action_row(row)
        rows.append(row)

    write_action_queue_csv(queue_csv, rows, list(ACTION_LABEL_QUEUE_COLUMNS))
    manifest_path = output_dir / "state_action_label_manifest.json"
    payload = {
        "ok": True,
        "events_path": str(events_path),
        "extra_events": [str(path) for path in extra_events],
        "output_dir": str(output_dir),
        "queue_csv": str(queue_csv),
        "manifest_path": str(manifest_path),
        "stats": {**stats, "selected": len(rows), "not_selected": len(candidates) - len(rows)},
        "progress": action_progress(rows),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def serve_state_action_label_queue(
    *,
    queue_csv: Path,
    host: str = "127.0.0.1",
    port: int = 8771,
    open_browser: bool = False,
) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    rows, _fieldnames = load_action_queue_csv(queue_csv)
    if not rows:
        raise ValueError(f"queue csv has no rows: {queue_csv}")

    server = ThreadingHTTPServer((host, int(port)), make_action_handler(queue_csv))
    url = f"http://{host}:{int(port)}/"
    print(f"Action-panel audit: {url}", flush=True)
    print(f"Queue CSV: {queue_csv}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"ok": True, "url": url, "queue_csv": str(queue_csv), "row_count": len(rows)}


def load_action_candidates(events_path: Path, *, source_index: int = 0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "lines": 0,
        "ok_events": 0,
        "image_backed": 0,
        "missing_visual_evidence": 0,
        "hero_turn": 0,
        "control_visible": 0,
    }
    candidates: list[dict[str, Any]] = []
    with Path(events_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not event.get("ok"):
                continue
            stats["ok_events"] += 1
            source_image = resolve_source_image(event, events_path)
            source_video = resolve_source_video(event, events_path)
            if source_image is None and source_video is None:
                stats["missing_visual_evidence"] += 1
                continue
            stats["image_backed"] += 1
            hero_turn = as_dict(event.get("hero_turn"))
            controls = as_dict(event.get("action_controls"))
            if hero_turn.get("is_turn"):
                stats["hero_turn"] += 1
            if controls.get("visible"):
                stats["control_visible"] += 1
            source = as_dict(event.get("source"))
            event_index = as_dict(event.get("event")).get("index", line_number)
            candidates.append(
                {
                    "event": event,
                    "source_image": source_image,
                    "source_video": source_video,
                    "source_key": str(source_index),
                    "line_number": line_number,
                    "timestamp_sec": as_float(source.get("timestamp_sec")),
                    "label_id": f"A{source_index:02d}_{line_number:05d}_{event_index}",
                }
            )
    return candidates, stats


def select_action_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Avoid a queue full of identical adjacent states while keeping time order."""

    if len(candidates) <= limit:
        return sorted(candidates, key=action_candidate_priority, reverse=True)

    selected: list[dict[str, Any]] = []
    signatures: set[tuple[str, str, str, str, int]] = set()
    for candidate in candidates:
        event = as_dict(candidate.get("event"))
        table = as_dict(event.get("table"))
        controls = as_dict(event.get("action_controls"))
        hero_turn = as_dict(event.get("hero_turn"))
        signature = (
            str(candidate.get("source_key") or ""),
            str(table.get("street") or ""),
            ",".join(str(item) for item in list(controls.get("actions") or [])),
            "turn" if hero_turn.get("is_turn") else "watch",
            int(float(candidate.get("timestamp_sec") or 0.0) // 3),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return sorted(selected, key=action_candidate_priority, reverse=True)


def action_candidate_priority(candidate: dict[str, Any]) -> tuple[float, ...]:
    event = as_dict(candidate.get("event"))
    controls = as_dict(event.get("action_controls"))
    regions = [as_dict(item) for item in list(controls.get("red_button_regions") or [])]
    largest_width = max((as_float(region.get("width")) for region in regions), default=0.0)
    largest_height = max((as_float(region.get("height")) for region in regions), default=0.0)
    return (
        1.0 if isinstance(candidate.get("source_video"), Path) else 0.0,
        1.0 if largest_width >= 80.0 and largest_height >= 35.0 else 0.0,
        largest_width * largest_height,
        float(len(list(controls.get("actions") or []))),
        -float(candidate.get("line_number") or 0),
    )


def event_to_queue_row(event_data: dict[str, Any], *, frame_path: Path, panel_path: Path) -> dict[str, str]:
    event = as_dict(event_data.get("event"))
    table = as_dict(event.get("table"))
    hero = as_dict(event.get("hero"))
    turn = as_dict(event.get("hero_turn"))
    controls = as_dict(event.get("action_controls"))
    advice = as_dict(event.get("gto_advice"))
    tracker = as_dict(event.get("preflop_tracker"))
    preflop = as_dict(event.get("preflop"))
    source = as_dict(event.get("source"))
    sample = as_dict(source.get("card_sample"))
    bets = list(event.get("bets") or [])
    return {
        "label_id": str(event_data["label_id"]),
        "event_index": str(as_dict(event.get("event")).get("index", "")),
        "line_number": str(event_data["line_number"]),
        "timestamp_sec": format_number(event_data.get("timestamp_sec")),
        "source_kind": str(source.get("kind") or ""),
        "frame_scope": frame_scope(source, sample),
        "street": str(table.get("street") or ""),
        "hero_position": str(hero.get("position") or ""),
        "hero_gto_position": str(hero.get("gto_position") or ""),
        "hero_preflop_order": str(hero.get("preflop_action_order") or ""),
        "hero_postflop_order": str(hero.get("postflop_action_order") or ""),
        "hero_cards": " ".join(str(card) for card in list(hero.get("cards") or [])),
        "board": " ".join(str(card) for card in list(table.get("board") or [])),
        "pot_bb": format_number(table.get("pot_bb")),
        "to_call_bb": format_number(table.get("to_call_bb")),
        "visible_bets": ", ".join(
            f"{as_dict(bet).get('seat', '?')}:{format_number(as_dict(bet).get('amount_bb'))}BB" for bet in bets
        ),
        "cv_hero_turn": "yes" if turn.get("is_turn") else "no",
        "cv_turn_confidence": format_number(turn.get("confidence")),
        "cv_turn_reason": str(turn.get("reason") or ""),
        "cv_advice_reason": str(advice.get("reason") or ""),
        "cv_advice_summary": str(advice.get("summary") or ""),
        "cv_preflop_tracker_reason": str(tracker.get("reason") or ""),
        "cv_preflop_history": ", ".join(
            f"{as_dict(item).get('position', '?')}:{as_dict(item).get('action', '?')}"
            for item in list(preflop.get("action_history") or [])
        ),
        "cv_analysis": build_action_analysis(event),
        "cv_actions": ", ".join(str(item) for item in list(controls.get("actions") or [])),
        "cv_call_amount_bb": format_number(controls.get("call_amount_bb")),
        "cv_raise_amount_bb": format_number(controls.get("raise_amount_bb")),
        "cv_red_button_regions": json.dumps(list(controls.get("red_button_regions") or []), ensure_ascii=False),
        "frame_path": str(frame_path),
        "panel_crop_path": str(panel_path),
        "final_hero_turn": "",
        "final_fast_fold_state": "",
        "final_panel_template": "",
        "final_actions": "",
        "final_disabled_actions": "",
        "final_call_amount_bb": "",
        "final_raise_to_bb": "",
        "notes": "",
        "ignored": "",
        "updated_at": "",
    }


def build_action_analysis(event: dict[str, Any]) -> str:
    """Explain the evidence chain shown beside a manually reviewed frame."""

    table = as_dict(event.get("table"))
    hero = as_dict(event.get("hero"))
    turn = as_dict(event.get("hero_turn"))
    controls = as_dict(event.get("action_controls"))
    advice = as_dict(event.get("gto_advice"))
    tracker = as_dict(event.get("preflop_tracker"))
    preflop = as_dict(event.get("preflop"))
    context = as_dict(advice.get("preflop_context"))

    street = str(table.get("street") or "未识别街道")
    hero_position = str(hero.get("position") or "未识别位置")
    dealer = str(table.get("dealer_position") or "未识别庄家")
    actions = ", ".join(str(action) for action in list(controls.get("actions") or [])) or "未看到常规操作按钮"
    lines = [
        f"视觉判断：{street}；Hero 位置 {hero_position}；庄家位置 {dealer}。",
        (
            f"Hero 回合：{'是' if turn.get('is_turn') else '否'}；"
            f"依据：{plain_turn_reason(turn.get('reason'))}；可见按钮：{actions}。"
        ),
    ]

    if advice.get("ready"):
        lines.append(f"博弈论最优（GTO）建议：{advice.get('summary') or '已生成，但摘要为空'}。")
    else:
        lines.append(f"建议暂停：{plain_advice_reason(advice.get('reason'))}。")

    if street == "preflop":
        history = list(preflop.get("action_history") or [])
        history_text = ", ".join(
            f"{as_dict(item).get('position', '?')}:{as_dict(item).get('action', '?')}"
            for item in history
        ) or "尚未获得可信的前序行动"
        lines.append(
            "翻前证据："
            f"{history_text}；追踪状态：{preflop_tracker_text(tracker.get('reason'), history)}。"
        )
        if context:
            scenario = str(context.get("scenario") or "未确定")
            needs = ", ".join(str(item) for item in list(context.get("needs") or [])) or "无"
            lines.append(f"翻前局面：{scenario}；仍需证据：{needs}。")
    return "\n".join(lines)


def plain_turn_reason(reason: Any) -> str:
    value = str(reason or "").strip()
    mapping = {
        "red_buttons_and_action_text": "检测到红色可点按钮和操作文字",
        "hero_action_controls_not_visible": "未检测到可用的 Hero 操作区",
        "hero_turn_not_confirmed": "没有足够按钮或文字证据确认轮到 Hero",
    }
    return mapping.get(value, value or "没有提供依据")


def plain_advice_reason(reason: Any) -> str:
    value = str(reason or "").strip()
    mapping = {
        "hero_action_controls_visible": "证据充分，已生成建议",
        "hero_action_controls_not_visible": "当前未看到可用的 Hero 操作按钮，可能未轮到 Hero 或操作区不可见",
        "hero_turn_not_confirmed": "尚未确认轮到 Hero，不能输出操作建议",
        "preflop_context_incomplete": "翻前早期行动没有被完整、可信地记录，不能把单个下注数字直接猜成加注",
        "preflop_scenario_not_supported": "翻前行动顺序已经识别，但当前策略只覆盖开局、面对开局和面对三次下注，尚未覆盖这一类局面",
        "board_cards_incomplete": "公共牌仍在发牌动画或尚未完整识别，等待牌面稳定后再计算建议",
        "pot_amount_unavailable": "没有可靠识别到底池金额，翻后下注比例无法安全计算",
        "hero_cards_incomplete": "两张手牌尚未完整识别",
    }
    return mapping.get(value, value or "未启用建议或没有返回原因")


def plain_tracker_reason(reason: Any) -> str:
    value = str(reason or "").strip()
    if value == "blind_posts_unconfirmed":
        return "仅确认到盲注，尚未确认首个加注者"
    if value == "hero_already_invested_before_sync":
        return "开始跟踪时 Hero 已经投入筹码，缺少此前行动"
    if value.startswith("prior_seat_unresolved:"):
        return f"Hero 之前的座位尚未确认：{value.split(':', 1)[1]}"
    return value or "尚未建立翻前行动记录"


def preflop_tracker_text(reason: Any, history: list[dict[str, Any]]) -> str:
    if reason:
        return plain_tracker_reason(reason)
    if history:
        return "\u5df2\u5f62\u6210\u53ef\u4fe1\u7684\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55"
    return "\u5c1a\u672a\u5efa\u7acb\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55"


def copy_action_asset(source: Path, assets_dir: Path, stem: str) -> Path:
    destination = assets_dir / f"{stem}{source.suffix.lower() or '.png'}"
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)
    return destination.resolve()


def materialize_candidate_frame(candidate: dict[str, Any], assets_dir: Path, stem: str) -> Path:
    source_image = candidate.get("source_image")
    if isinstance(source_image, Path):
        return copy_action_asset(source_image, assets_dir, stem)
    source_video = candidate.get("source_video")
    if not isinstance(source_video, Path):
        raise ValueError(f"candidate has no readable source: {candidate.get('label_id')}")
    destination = assets_dir / f"{stem}.png"
    if not destination.exists():
        capture = cv2.VideoCapture(str(source_video))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(candidate.get("timestamp_sec") or 0.0) * 1000.0)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"could not extract frame from {source_video}")
        if not cv2.imwrite(str(destination), frame):
            raise ValueError(f"could not write extracted frame: {destination}")
    return destination.resolve()


def resolve_source_video(event: dict[str, Any], events_path: Path) -> Path | None:
    source = as_dict(event.get("source"))
    if str(source.get("kind") or "").lower() != "video":
        return None
    raw_text = str(source.get("path") or "").strip()
    if not raw_text:
        return None
    raw_path = Path(raw_text)
    candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path, events_path.parent / raw_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def write_panel_crop(frame_path: Path, destination: Path, *, event: dict[str, Any] | None = None) -> Path:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not load copied frame: {frame_path}")
    height, width = image.shape[:2]
    crop = monitor_context_action_crop(image, as_dict(event))
    if crop is None:
        y0 = max(0, int(round(height * 0.66)))
        x0 = max(0, int(round(width * 0.04)))
        x1 = min(width, int(round(width * 0.96)))
        crop = image[y0:height, x0:x1]
    if crop.size == 0:
        crop = image
    cv2.imwrite(str(destination), crop)
    return destination.resolve()


def monitor_context_action_crop(image: Any, event: dict[str, Any]) -> Any | None:
    """Crop the lower part of the analysis table from a full-monitor sample."""

    source = as_dict(event.get("source"))
    sample = as_dict(source.get("card_sample"))
    if str(sample.get("screen_context_scope") or "").lower() != "monitor_full":
        return None
    analysis = as_dict(source.get("screen_region"))
    monitor = as_dict(source.get("monitor_region"))
    if not all(key in analysis for key in ("left", "top", "width", "height")):
        return None
    if not all(key in monitor for key in ("left", "top", "width", "height")):
        return None
    image_h, image_w = image.shape[:2]
    analysis_x = int(analysis["left"]) - int(monitor["left"])
    analysis_y = int(analysis["top"]) - int(monitor["top"])
    analysis_w = max(1, int(analysis["width"]))
    analysis_h = max(1, int(analysis["height"]))
    pad_x = max(20, int(round(analysis_w * 0.05)))
    x0 = max(0, analysis_x - pad_x)
    x1 = min(image_w, analysis_x + analysis_w + pad_x)
    y0 = max(0, analysis_y + int(round(analysis_h * 0.54)))
    # Include the entire remaining display below the table. Bottom buttons may
    # lie just outside the analysis ROI selected during calibration.
    y1 = image_h
    if x1 - x0 < max(80, int(image_w * 0.15)) or y1 - y0 < max(50, int(image_h * 0.08)):
        return None
    return image[y0:y1, x0:x1]


def load_action_queue_csv(queue_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    queue_csv = Path(queue_csv)
    if not queue_csv.exists():
        return [], list(ACTION_LABEL_QUEUE_COLUMNS)
    with queue_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for column in ACTION_LABEL_QUEUE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for row in rows:
        for column in fieldnames:
            row.setdefault(column, "")
    return rows, fieldnames


def write_action_queue_csv(queue_csv: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with Path(queue_csv).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_action_queue_csv(queue_csv: Path, payload: dict[str, Any]) -> dict[str, Any]:
    label_id = str(payload.get("label_id") or "").strip()
    if not label_id:
        raise ValueError("label_id is required")
    rows, fieldnames = load_action_queue_csv(queue_csv)
    target = next((row for row in rows if str(row.get("label_id") or "") == label_id), None)
    if target is None:
        raise ValueError(f"label_id not found: {label_id}")

    if payload.get("ignored"):
        target["ignored"] = "1"
        target["final_hero_turn"] = ""
        target["final_fast_fold_state"] = ""
        target["final_panel_template"] = ""
        target["final_actions"] = ""
        target["final_disabled_actions"] = ""
    else:
        hero_turn = normalize_choice(payload.get("final_hero_turn"), HERO_TURN_VALUES, "hero turn")
        # A separate fast-fold control is not present in every normal Hero-turn
        # panel. Keep missing observations explicit instead of blocking a valid
        # FOLD/CALL/RAISE annotation.
        fast_fold = normalize_choice(payload.get("final_fast_fold_state"), FAST_FOLD_VALUES, "fast fold state") or "uncertain"
        template = normalize_choice(payload.get("final_panel_template"), PANEL_TEMPLATES, "panel template")
        actions = normalize_actions(payload.get("final_actions"))
        disabled_actions = normalize_actions(payload.get("final_disabled_actions"))
        if template in {"no_hero_action", "no_action_panel", "action_panel_cropped"} and actions:
            raise ValueError(f"{template} cannot have active Hero actions")
        normal_hero_templates = {
            "preflop_fold_call_raise",
            "postflop_check_bet",
            "postflop_fold_call_raise",
            "allin_or_special",
        }
        if template in normal_hero_templates and hero_turn != "yes":
            raise ValueError(f"{template} requires final_hero_turn=yes")
        if template == "no_hero_action" and hero_turn == "yes":
            raise ValueError("no_hero_action cannot have final_hero_turn=yes")
        if hero_turn != "yes" and actions:
            raise ValueError("clickable Hero actions require final_hero_turn=yes")
        overlap = set(actions).intersection(disabled_actions)
        if overlap:
            raise ValueError(f"an action cannot be both clickable and disabled: {', '.join(sorted(overlap))}")
        target["ignored"] = ""
        target["final_hero_turn"] = hero_turn
        target["final_fast_fold_state"] = fast_fold
        target["final_panel_template"] = template
        target["final_actions"] = ",".join(actions)
        target["final_disabled_actions"] = ",".join(disabled_actions)
        target["final_call_amount_bb"] = format_optional_number(payload.get("final_call_amount_bb"), "call amount")
        target["final_raise_to_bb"] = format_optional_number(payload.get("final_raise_to_bb"), "raise-to amount")
    if "notes" in payload:
        target["notes"] = str(payload.get("notes") or "").strip()
    target["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_action_queue_csv(queue_csv, rows, fieldnames)
    return {"label_id": label_id, "row": public_action_row(target), "progress": action_progress(rows)}


def normalize_choice(value: Any, allowed: tuple[str, ...], name: str) -> str:
    text = str(value or "").strip().lower()
    if text and text not in allowed:
        raise ValueError(f"invalid {name}: {text}")
    return text


def normalize_actions(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").split(",")
    selected = {str(item).strip().lower() for item in raw if str(item).strip()}
    invalid = selected.difference(VISIBLE_ACTIONS)
    if invalid:
        raise ValueError(f"invalid actions: {', '.join(sorted(invalid))}")
    return [action for action in VISIBLE_ACTIONS if action in selected]


def format_optional_number(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"invalid {name}: {text}") from error
    if number < 0:
        raise ValueError(f"invalid {name}: {text}")
    return format_number(number)


def action_progress(rows: list[dict[str, Any]]) -> dict[str, int]:
    active = [row for row in rows if not is_ignored(row)]
    completed = [
        row
        for row in active
        if (
            str(row.get("final_hero_turn") or "")
            and str(row.get("final_fast_fold_state") or "")
            and str(row.get("final_panel_template") or "")
        )
    ]
    return {
        "rows": len(active),
        "ignored": len(rows) - len(active),
        "completed": len(completed),
        "hero_turn_yes": sum(1 for row in completed if row.get("final_hero_turn") == "yes"),
        "fast_fold_available": sum(1 for row in completed if row.get("final_fast_fold_state") == "available"),
        "quick_fold": sum(
            1
            for row in completed
            if row.get("final_fast_fold_state") == "available"
        ),
        "preflop": sum(1 for row in completed if row.get("final_panel_template") == "preflop_fold_call_raise"),
        "postflop_check_bet": sum(1 for row in completed if row.get("final_panel_template") == "postflop_check_bet"),
        "postflop_facing": sum(1 for row in completed if row.get("final_panel_template") == "postflop_fold_call_raise"),
    }


def is_ignored(row: dict[str, Any]) -> bool:
    return str(row.get("ignored") or "").strip().lower() in {"1", "true", "yes", "ignored"}


def frame_scope(source: dict[str, Any], sample: dict[str, Any]) -> str:
    state_audit = as_dict(source.get("state_audit"))
    if state_audit.get("frame"):
        return "manual_outer_bbox"
    if str(source.get("kind") or "").lower() == "video":
        return "video_full_frame"
    if sample.get("screen_context"):
        if str(sample.get("screen_context_scope") or "").lower() == "monitor_full":
            return "monitor_full_context"
        return "expanded_action_context"
    if str(source.get("kind") or "").lower() == "screen":
        return "analysis_roi_only"
    return "unknown"


def load_action_audit_coverage(queue_csv: Path) -> dict[str, int]:
    manifest_path = Path(queue_csv).with_name("state_action_label_manifest.json")
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    stats = as_dict(payload.get("stats"))
    return {
        "ok_events": int(stats.get("ok_events") or 0),
        "image_backed": int(stats.get("image_backed") or 0),
        "missing_visual_evidence": int(stats.get("missing_visual_evidence") or 0),
        "selected": int(stats.get("selected") or 0),
        "not_selected": int(stats.get("not_selected") or 0),
    }


def migrate_legacy_action_row(row: dict[str, str]) -> None:
    """Keep prior work after splitting quick-fold from normal Hero actions."""

    template = str(row.get("final_panel_template") or "")
    if template not in {"quick_fold_only", "quick_fold_check"}:
        return
    row["final_fast_fold_state"] = row.get("final_fast_fold_state") or "available"
    if str(row.get("final_hero_turn") or "") == "yes" and template == "quick_fold_check":
        row["final_panel_template"] = "postflop_check_bet"
        row["final_actions"] = "check"
    else:
        row["final_panel_template"] = "no_hero_action"
        row["final_actions"] = ""
        if template == "quick_fold_check":
            row["final_disabled_actions"] = row.get("final_disabled_actions") or "check"


def public_action_row(row: dict[str, Any]) -> dict[str, str]:
    return {column: str(row.get(column) or "") for column in ACTION_LABEL_QUEUE_COLUMNS}


def make_action_handler(queue_csv: Path) -> type[BaseHTTPRequestHandler]:
    queue_root = queue_csv.parent.resolve()

    class StateActionLabelHandler(BaseHTTPRequestHandler):
        server_version = "StateActionAudit/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self.send_text(render_action_audit_html(queue_csv), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/rows":
                rows, _ = load_action_queue_csv(queue_csv)
                self.send_json(
                    {
                        "ok": True,
                        "queue_csv": str(queue_csv),
                        "rows": [public_action_row(row) for row in rows],
                        "progress": action_progress(rows),
                        "coverage": load_action_audit_coverage(queue_csv),
                    }
                )
                return
            if parsed.path == "/file":
                query = urllib.parse.parse_qs(parsed.query)
                self.send_file(query.get("path", [""])[0])
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/update":
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                self.send_json({"ok": True, **update_action_queue_csv(queue_csv, payload)})
            except ValueError as error:
                self.send_json({"ok": False, "error": str(error)}, status=400)
            except Exception as error:
                self.send_json({"ok": False, "error": str(error)}, status=500)

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path_text: str) -> None:
            try:
                path = Path(urllib.parse.unquote(path_text)).resolve()
                path.relative_to(queue_root)
            except (OSError, ValueError):
                self.send_error(403, "outside queue assets")
                return
            if not path.is_file():
                self.send_error(404, "file not found")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return StateActionLabelHandler


def render_action_audit_html(queue_csv: Path) -> str:
    escaped_queue = html.escape(str(queue_csv))
    return ACTION_AUDIT_HTML.replace("__QUEUE_CSV__", escaped_queue)


def format_state_action_label_queue_summary(payload: dict[str, Any]) -> str:
    progress = as_dict(payload.get("progress"))
    return "\n".join(
        [
            f"Action queue: {payload.get('queue_csv')}",
            f"Rows: {progress.get('rows', 0)} | ignored: {progress.get('ignored', 0)}",
            f"Selected: {as_dict(payload.get('stats')).get('selected', 0)}",
        ]
    )


def format_state_action_label_server_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Action-panel audit UI: {payload.get('url')}",
            f"Queue CSV: {payload.get('queue_csv')}",
            f"Rows: {payload.get('row_count')}",
        ]
    )


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}".rstrip("0").rstrip(".")


ACTION_AUDIT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>操作面板人工校对</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111;color:#eee;font-family:Arial,'Microsoft YaHei',sans-serif}header{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#202020;border-bottom:1px solid #4a4a4a}button,input,textarea,select{font:inherit}button{padding:8px 11px;color:#eee;background:#333;border:1px solid #666;border-radius:4px;cursor:pointer}button:hover,button.selected{background:#6f2430;border-color:#e35d6a}.primary{background:#992b35;border-color:#d2525f}.muted{color:#a9a9a9;font-size:12px;max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.wrap{padding:14px}.layout{display:grid;grid-template-columns:minmax(530px,1.4fr) minmax(380px,.8fr);gap:14px}.panel{background:#1b1b1b;border:1px solid #3c3c3c;padding:12px}.image-grid{display:grid;grid-template-columns:minmax(340px,1fr) minmax(250px,.58fr);gap:12px}.image-box{min-width:0}.image-box h3{margin:0 0 8px;font-size:15px}.frame-scope{min-height:19px;margin:-3px 0 7px;color:#ffcf70;font-size:12px}.frame{width:100%;max-height:630px;object-fit:contain;background:#090909;border:1px solid #555}.crop{width:100%;max-height:300px;object-fit:contain;background:#090909;border:1px solid #555}.meta{line-height:1.55;color:#d0d0d0;font-size:14px;border-bottom:1px solid #444;padding-bottom:10px;margin-bottom:12px}.cv{color:#ffcf70}.warn{color:#ff9898;font-weight:bold}.grid-title{font-weight:bold;margin:13px 0 7px}.choice-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.action-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.choice-grid button,.action-grid button{min-height:42px}.inputs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.inputs input,textarea{width:100%;background:#101010;color:#fff;border:1px solid #555;border-radius:3px;padding:8px}textarea{height:70px;resize:vertical;margin-top:8px}.bottom{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:9px}.status{min-height:20px;color:#8ee28e}.danger{color:#ff9999}.empty{padding:20px}.pill{display:inline-block;border:1px solid #555;border-radius:12px;padding:2px 8px;margin:2px 2px 2px 0;color:#bbb}@media(max-width:1000px){.layout{grid-template-columns:1fr}.image-grid{grid-template-columns:1fr}.crop{max-height:none}.muted{max-width:250px}}
</style>
</head>
<body>
<header>
  <strong>操作面板人工校对</strong><span id="progress">加载中...</span>
  <select id="filter" onchange="refreshFilter()"><option value="todo">待校对</option><option value="all">全部</option><option value="turn">CV 判定轮到 Hero</option><option value="watch">CV 判定未轮到 Hero</option><option value="done">已校对</option></select>
  <select id="streetFilter" onchange="refreshFilter()"><option value="all">所有街道</option><option value="preflop">翻前</option><option value="flop">翻牌</option><option value="turn">转牌</option><option value="river">河牌</option></select>
  <button onclick="move(-1)" title="上一张">上一张</button><button onclick="move(1)" title="下一张">下一张</button>
  <span class="muted">__QUEUE_CSV__</span>
</header><span id="coverage" class="muted"></span>
<main class="wrap"><div id="empty" class="panel empty" hidden>当前筛选没有待校对样本。</div><div id="content" class="layout">
  <section class="panel"><div class="image-grid"><div class="image-box"><h3>原始画面</h3><div id="frameScope" class="frame-scope"></div><img id="frame" class="frame"></div><div class="image-box"><h3>底部操作区</h3><img id="crop" class="crop"></div></div></section>
  <section class="panel"><div id="meta" class="meta"></div>
    <div class="grid-title">是否真的轮到 Hero</div><div id="turnButtons" class="choice-grid"></div>
    <div class="grid-title">快速弃牌（预操作，未轮到 Hero 时也可能可点）</div><div id="fastFoldButtons" class="choice-grid"></div>
    <div class="grid-title">Hero 当前常规操作面板</div><div id="templateButtons" class="choice-grid"></div>
    <div class="grid-title">当前可点击的常规按钮</div><div id="actionButtons" class="action-grid"></div>
    <div class="grid-title">可见但黑色 / 不可点击的标签</div><div id="disabledButtons" class="action-grid"></div>
    <div class="inputs"><input id="callAmount" placeholder="CALL 金额 BB（可空）"><input id="raiseAmount" placeholder="RAISE TO 金额 BB（可空）"></div>
    <textarea id="notes" placeholder="备注（可空）"></textarea>
    <div class="bottom"><button class="primary" onclick="saveCurrent()">保存并下一张</button><button onclick="markIgnored()">画面不可用</button><span id="status" class="status"></span></div>
  </section>
</div></main>
<script>
const heroTurns=[['yes','是，轮到 Hero'],['no','否，未轮到 Hero'],['uncertain','不确定']];
const fastFolds=[['available','快速弃牌可点'],['not_visible','未显示快速弃牌'],['uncertain','不确定']];
const templates=[['no_hero_action','未轮到 Hero（无常规可点按钮）'],['preflop_fold_call_raise','翻前 FOLD / CALL / RAISE'],['postflop_check_bet','翻后 CHECK / BET'],['postflop_fold_call_raise','翻后 FOLD / CALL / RAISE'],['allin_or_special','全下 / 特殊按钮'],['action_panel_cropped','操作区被截图裁掉'],['other_or_occluded','其他 / 被遮挡']];
const templateActions={no_hero_action:[],preflop_fold_call_raise:['fold','call','raise'],postflop_check_bet:['check','bet'],postflop_fold_call_raise:['fold','call','raise'],allin_or_special:['all_in'],action_panel_cropped:[],other_or_occluded:[]};
const actionLabels={fold:'FOLD',check:'CHECK',call:'CALL',bet:'BET',raise:'RAISE',all_in:'ALL-IN'};
let rows=[],filtered=[],index=0,draft={turn:'',fastFold:'uncertain',template:'',actions:new Set(),disabledActions:new Set()};
const fileUrl=p=>p?'/file?path='+encodeURIComponent(p):'';
function esc(v){return String(v||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function isDone(r){return !!(r.final_hero_turn&&r.final_fast_fold_state&&r.final_panel_template);}
function isIgnored(r){return ['1','true','yes','ignored'].includes(String(r.ignored||'').toLowerCase());}
async function loadRows(){const data=await (await fetch('/api/rows')).json();rows=data.rows||[];updateProgress(data.progress||{});updateCoverage(data.coverage||{});refreshFilter();}
function updateProgress(p){document.getElementById('progress').textContent=`已校对 ${p.completed||0}/${p.rows||0} | Hero回合 ${p.hero_turn_yes||0} | 快弃可点 ${p.fast_fold_available||0} | 翻前 ${p.preflop||0} | 翻后 ${Number(p.postflop_check_bet||0)+Number(p.postflop_facing||0)}`;}
function refreshFilter(){const filter=document.getElementById('filter').value,street=document.getElementById('streetFilter').value,current=filtered[index]?.label_id;filtered=rows.filter(r=>{if(street!=='all'&&r.street!==street)return false;if(filter==='todo')return !isIgnored(r)&&!isDone(r);if(filter==='done')return !isIgnored(r)&&isDone(r);if(filter==='turn')return !isIgnored(r)&&r.cv_hero_turn==='yes';if(filter==='watch')return !isIgnored(r)&&r.cv_hero_turn!=='yes';return true;});index=Math.max(0,filtered.findIndex(r=>r.label_id===current));if(index<0)index=0;render();}
function move(delta){if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render();}
function renderButtons(rootId,items,selected,click){const root=document.getElementById(rootId);root.innerHTML='';for(const [key,label] of items){const b=document.createElement('button');b.textContent=label;if(selected===key||(selected instanceof Set&&selected.has(key)))b.classList.add('selected');b.onclick=()=>click(key);root.appendChild(b);}}
function render(){const empty=!filtered.length;document.getElementById('empty').hidden=!empty;document.getElementById('content').hidden=empty;if(empty)return;const r=filtered[index];draft={turn:r.final_hero_turn||'',fastFold:r.final_fast_fold_state||'uncertain',template:r.final_panel_template||'',actions:new Set((r.final_actions||'').split(',').filter(Boolean)),disabledActions:new Set((r.final_disabled_actions||'').split(',').filter(Boolean))};document.getElementById('frame').src=fileUrl(r.frame_path);document.getElementById('crop').src=fileUrl(r.panel_crop_path);const frameScope=r.frame_scope==='manual_outer_bbox'?'原图范围：这就是当时手动拖出的完整大框。':r.frame_scope==='monitor_full_context'?'历史整屏截图：画面中的绿框才是当时的大框，不能把整张图当作操作区截图。':r.frame_scope==='analysis_roi_only'?'历史内框截图：底部操作区可能已被裁掉。':'原图范围：已保留操作区上下文。';document.getElementById('frameScope').textContent=frameScope;document.getElementById('callAmount').value=r.final_call_amount_bb||'';document.getElementById('raiseAmount').value=r.final_raise_to_bb||'';document.getElementById('notes').value=r.notes||'';const board=r.board||'-',cards=r.hero_cards||'-',bets=r.visible_bets||'-';const scope=r.frame_scope==='analysis_roi_only'?'<br><span class="warn">此图只保存了分析 ROI，底部操作区可能被裁掉；请选“操作区被截图裁掉”，不要据此判断无按钮。</span>':r.frame_scope==='monitor_full_context'?'<br><span class="cv">截图范围：完整显示器画面；黄色框是 CV 分析 ROI。</span>':r.frame_scope==='expanded_action_context'?'<br><span class="cv">截图范围：已向下外扩，包含操作区上下文。</span>':'';const pause=r.cv_advice_reason||'-',tracker=r.cv_preflop_tracker_reason||'-',history=r.cv_preflop_history||'-';document.getElementById('meta').innerHTML=`<b>${esc(r.label_id)} | ${index+1}/${filtered.length}</b><br>时间 ${esc(r.timestamp_sec)}s | 街道 <b>${esc(r.street||'-')}</b> | Hero ${esc(r.hero_position||'-')} / GTO ${esc(r.hero_gto_position||'-')}<br>手牌 <b>${esc(cards)}</b> | 公共牌 <b>${esc(board)}</b><br>底池 ${esc(r.pot_bb||'-')}BB | 跟注 ${esc(r.to_call_bb||'-')}BB${scope}<br><span class="cv">CV 原判：hero_turn=${esc(r.cv_hero_turn)} (${esc(r.cv_turn_confidence||'-')})；动作=${esc(r.cv_actions||'-')}；CALL=${esc(r.cv_call_amount_bb||'-')}BB；RAISE=${esc(r.cv_raise_amount_bb||'-')}BB</span><br><span class="cv">轮到 Hero 的依据：${esc(r.cv_turn_reason||'-')}</span><br><span class="warn">未给建议的原因：${esc(pause)}</span><br><span class="cv">翻前行动记录状态：${esc(tracker)}；已记录行动：${esc(history)}</span><br><span class="cv">可见下注：${esc(bets)}</span>`;renderButtonGroups();document.getElementById('status').textContent='';}
function renderButtonGroups(){renderButtons('turnButtons',heroTurns,draft.turn,key=>{draft.turn=key;renderButtonGroups();});renderButtons('fastFoldButtons',fastFolds,draft.fastFold,key=>{draft.fastFold=key;renderButtonGroups();});renderButtons('templateButtons',templates,draft.template,key=>{draft.template=key;draft.actions=new Set(templateActions[key]||[]);renderButtonGroups();});renderButtons('actionButtons',Object.entries(actionLabels),draft.actions,key=>{if(draft.actions.has(key))draft.actions.delete(key);else draft.actions.add(key);draft.disabledActions.delete(key);renderButtonGroups();});renderButtons('disabledButtons',Object.entries(actionLabels),draft.disabledActions,key=>{if(draft.disabledActions.has(key))draft.disabledActions.delete(key);else draft.disabledActions.add(key);draft.actions.delete(key);renderButtonGroups();});}
async function saveCurrent(){if(!filtered.length)return;const r=filtered[index],status=document.getElementById('status');if(!draft.turn||!draft.template){status.textContent='请确认是否轮到 Hero 和常规面板。快速弃牌未单独确认时会记为“不确定”。';status.className='status danger';return;}status.className='status';status.textContent='保存中...';const payload={label_id:r.label_id,final_hero_turn:draft.turn,final_fast_fold_state:draft.fastFold||'uncertain',final_panel_template:draft.template,final_actions:[...draft.actions],final_disabled_actions:[...draft.disabledActions],final_call_amount_bb:document.getElementById('callAmount').value,final_raise_to_bb:document.getElementById('raiseAmount').value,notes:document.getElementById('notes').value};const data=await (await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();if(!data.ok){status.textContent=data.error||'保存失败';status.className='status danger';return;}Object.assign(r,data.row);const original=rows.find(x=>x.label_id===r.label_id);if(original)Object.assign(original,data.row);updateProgress(data.progress||{});if(document.getElementById('filter').value==='todo')refreshFilter();else move(1);}
async function markIgnored(){if(!filtered.length)return;const r=filtered[index],status=document.getElementById('status');status.textContent='保存中...';const data=await (await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label_id:r.label_id,ignored:true,notes:document.getElementById('notes').value})})).json();if(!data.ok){status.textContent=data.error||'保存失败';status.className='status danger';return;}Object.assign(r,data.row);const original=rows.find(x=>x.label_id===r.label_id);if(original)Object.assign(original,data.row);updateProgress(data.progress||{});refreshFilter();}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key==='Enter')saveCurrent();});
function updateCoverage(c){document.getElementById('coverage').textContent=`可审核画面 ${c.image_backed||0}/${c.ok_events||0}；历史无图 ${c.missing_visual_evidence||0}；未选入 ${c.not_selected||0}`;}
const renderBeforeAuditAnalysis=render;
render=()=>{renderBeforeAuditAnalysis();const r=filtered[index];if(!r)return;let box=document.getElementById('analysis');if(!box){box=document.createElement('div');box.id='analysis';box.style.cssText='white-space:pre-wrap;line-height:1.55;color:#c6e3ff;font-size:14px;background:#101820;border:1px solid #395166;padding:10px;margin:0 0 12px';document.getElementById('meta').insertAdjacentElement('afterend',box);}const scope=r.frame_scope==='manual_outer_bbox'?'审核原图：第一次人工拖出的完整牌桌框。\n':'';box.textContent=`系统分析\n${scope}${r.cv_analysis||'此历史样本没有保存分析文本。'}`;};
const renderWithCorrectedStatus=render;
render=()=>{renderWithCorrectedStatus();const r=filtered[index];if(!r)return;const box=document.getElementById('analysis');if(!box)return;const scope=r.frame_scope==='manual_outer_bbox'?'\u5ba1\u6838\u539f\u56fe\uff1a\u7b2c\u4e00\u6b21\u4eba\u5de5\u62d6\u51fa\u7684\u5b8c\u6574\u724c\u684c\u6846\u3002\n':'';const adviceReady=r.cv_advice_reason==='hero_action_controls_visible'&&!!r.cv_advice_summary;const adviceText=adviceReady?`\u5efa\u8bae\u72b6\u6001\uff1a\u5df2\u751f\u6210\u5efa\u8bae\u3002\u4f9d\u636e\uff1a${r.cv_advice_reason}`:`\u5efa\u8bae\u72b6\u6001\uff1a\u6682\u4e0d\u8f93\u51fa\u3002\u539f\u56e0\uff1a${r.cv_advice_reason||'-'}`;const hasHistory=!!r.cv_preflop_history&&r.cv_preflop_history!=='-';const trackerText=r.cv_preflop_tracker_reason&&r.cv_preflop_tracker_reason!=='-'?r.cv_preflop_tracker_reason:(hasHistory?'\u5df2\u5f62\u6210\u53ef\u4fe1\u7684\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55':'\u5c1a\u672a\u5efa\u7acb\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55');const meta=document.getElementById('meta');if(adviceReady)meta.innerHTML=meta.innerHTML.replace('<span class="warn">\u672a\u7ed9\u5efa\u8bae\u7684\u539f\u56e0\uff1a','<span class="cv">\u5efa\u8bae\u5df2\u751f\u6210\u7684\u4f9d\u636e\uff1a');if(hasHistory&&!r.cv_preflop_tracker_reason)meta.innerHTML=meta.innerHTML.replace('\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55\u72b6\u6001\uff1a-\uff1b','\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55\u72b6\u6001\uff1a\u5df2\u5f62\u6210\u53ef\u4fe1\u7684\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55\uff1b');box.textContent=`\u7cfb\u7edf\u5206\u6790\n${scope}${adviceText}\n\u7ffb\u524d\u884c\u52a8\u8bb0\u5f55\uff1a${trackerText}\n${r.cv_analysis||'\u6b64\u5386\u53f2\u6837\u672c\u6ca1\u6709\u4fdd\u5b58\u5206\u6790\u6587\u672c\u3002'}`;};
loadRows();
</script>
</body></html>"""
