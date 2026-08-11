from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .video_vision import BET_ANCHORS_8, BOARD_CARD_ROIS, HERO_CARD_ROIS, load_cv
from .vision import build_seats


CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
VIDEO_X = 24
VIDEO_Y = 90
VIDEO_W = 960
VIDEO_H = 672
PANEL_X = 1010
PANEL_Y = 24
PANEL_W = 566
PANEL_H = 852

BG = (18, 21, 27)
PANEL = (27, 32, 41)
PANEL_2 = (35, 42, 53)
LINE = (72, 84, 100)
TEXT = (232, 238, 244)
MUTED = (150, 160, 174)
GREEN = (90, 220, 142)
GRAY = (110, 118, 130)
CYAN = (72, 213, 235)
ORANGE = (255, 178, 72)
YELLOW = (78, 218, 245)
RED = (82, 94, 238)
BLUE = (245, 170, 78)
ACCOUNTING_TOLERANCE_BB = 0.05


def render_dashboard_video(
    video_path: Path,
    states_jsonl: Path,
    output_path: Path,
    ocr_events_jsonl: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
    output_fps: float | None = None,
    ocr_hold_sec: float = 1.25,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> dict[str, Any]:
    cv2, np = load_cv()
    started = time.perf_counter()
    video_path = Path(video_path)
    states_jsonl = Path(states_jsonl)
    output_path = Path(output_path)
    ocr_events_jsonl = Path(ocr_events_jsonl) if ocr_events_jsonl else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 19.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps else 0.0
    start_sec = 0.0 if start_sec is None else max(0.0, float(start_sec))
    end_sec = duration_sec if end_sec is None else min(duration_sec, float(end_sec))
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(frame_count - 1, int(round(end_sec * fps)))
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames - 1)
    if end_frame < start_frame:
        raise ValueError("end frame is before start frame")

    states = load_states_by_frame(states_jsonl, start_frame, end_frame)
    ocr_events = load_ocr_events(ocr_events_jsonl) if ocr_events_jsonl else []
    ocr_events = [event for event in ocr_events if event_timestamp(event) <= end_sec + ocr_hold_sec]
    ocr_events.sort(key=event_timestamp)

    actual_output_fps = float(output_fps or fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, actual_output_fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"cannot create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    rendered = 0
    last_state: dict[str, Any] | None = None
    ocr_i = 0
    latest_ocr: dict[str, Any] | None = None
    recent_changes: list[str] = []
    previous_signature: str | None = None
    contribution_tracker = ContributionTracker()

    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        state = states.get(frame_index) or last_state
        if state is None:
            state = fallback_state(frame_index, frame_index / fps)
        else:
            last_state = state

        timestamp = frame_index / fps
        while ocr_i < len(ocr_events) and event_timestamp(ocr_events[ocr_i]) <= timestamp + 1e-6:
            latest_ocr = ocr_events[ocr_i]
            contribution_tracker.update(latest_ocr)
            ocr_i += 1
        merged_state = merge_ocr_state(state, latest_ocr, timestamp, ocr_hold_sec)
        merged_state["contribution"] = contribution_tracker.snapshot(merged_state)
        signature = display_signature(merged_state)
        if signature != previous_signature:
            recent_changes.insert(0, describe_change(merged_state, timestamp))
            recent_changes = recent_changes[:5]
            previous_signature = signature

        canvas = np.full((height, width, 3), BG, dtype=np.uint8)
        draw_header(cv2, canvas, video_path, timestamp, frame_index, fps, end_frame)
        draw_video_region(cv2, canvas, frame, merged_state, source_w, source_h)
        draw_panel(cv2, canvas, merged_state, recent_changes, timestamp)
        draw_progress(cv2, canvas, timestamp, start_sec, end_sec)
        writer.write(canvas)

        rendered += 1
        frame_index += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - started
    summary = {
        "ok": True,
        "video": str(video_path),
        "states_jsonl": str(states_jsonl),
        "ocr_events_jsonl": str(ocr_events_jsonl) if ocr_events_jsonl else "",
        "output_video": str(output_path),
        "source": {
            "width": source_w,
            "height": source_h,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
        },
        "render": {
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": rendered,
            "output_width": width,
            "output_height": height,
            "output_fps": actual_output_fps,
            "wall_time_sec": round(elapsed, 3),
            "render_fps": round(rendered / elapsed, 3) if elapsed else None,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def load_states_by_frame(path: Path, start_frame: int, end_frame: int) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            state = json.loads(line)
            frame_index = int(state.get("frame_index") or state.get("source", {}).get("frame_index") or -1)
            if frame_index < start_frame:
                continue
            if frame_index > end_frame:
                break
            states[frame_index] = normalize_state_shape(state)
    return states


def load_ocr_events(path: Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                events.append(normalize_state_shape(json.loads(line)))
    return events


def normalize_state_shape(state: dict[str, Any]) -> dict[str, Any]:
    if "source" in state:
        return state
    return {
        "ok": state.get("ok"),
        "source": {
            "timestamp_sec": state.get("timestamp_sec"),
            "frame_index": state.get("frame_index"),
            "sample_index": state.get("sample_index"),
            "visual_diff": state.get("visual_diff"),
        },
        "table": state.get("table") or {},
        "hero": state.get("hero") or {},
        "seats": state.get("seats") or [],
        "bets": state.get("bets") or derive_bets(state.get("seats") or []),
        "confidence": state.get("confidence") or {},
    }


def merge_ocr_state(
    base: dict[str, Any],
    ocr_state: dict[str, Any] | None,
    timestamp: float,
    hold_sec: float,
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    source = merged.setdefault("source", {})
    source["timestamp_sec"] = round(timestamp, 3)
    source["frame_index"] = int(round(timestamp * 19.0)) if source.get("frame_index") is None else source.get("frame_index")
    if not ocr_state or not ocr_state.get("ok"):
        source["ocr_status"] = "none"
        return merged

    age = timestamp - event_timestamp(ocr_state)
    if age < -1e-6 or age > hold_sec:
        source["ocr_status"] = "stale"
        source["ocr_age_sec"] = round(age, 3)
        return merged

    for key in ("table", "hero", "seats", "bets", "action_controls", "gto_advice", "confidence"):
        if key in ocr_state:
            merged[key] = copy.deepcopy(ocr_state[key])
    source["ocr_status"] = "recent"
    source["ocr_age_sec"] = round(age, 3)
    return merged


def event_timestamp(state: dict[str, Any]) -> float:
    return float((state.get("source") or {}).get("timestamp_sec") or state.get("timestamp_sec") or 0.0)


def fallback_state(frame_index: int, timestamp: float) -> dict[str, Any]:
    return {
        "ok": False,
        "source": {"timestamp_sec": round(timestamp, 3), "frame_index": frame_index},
        "table": {},
        "hero": {},
        "seats": [],
        "bets": [],
        "confidence": {},
    }


def draw_header(cv2: Any, canvas: Any, video_path: Path, timestamp: float, frame_index: int, fps: float, end_frame: int) -> None:
    draw_text(cv2, canvas, "Poker CV realtime dashboard", VIDEO_X, 42, 0.82, TEXT, 2)
    draw_text(cv2, canvas, str(video_path.name), VIDEO_X, 68, 0.45, MUTED, 1)
    right = f"{timestamp:07.3f}s  frame {frame_index}/{end_frame}  source {fps:.2f}fps"
    draw_text(cv2, canvas, right, PANEL_X, 42, 0.56, TEXT, 1)


def draw_video_region(cv2: Any, canvas: Any, frame: Any, state: dict[str, Any], source_w: int, source_h: int) -> None:
    resized = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
    canvas[VIDEO_Y : VIDEO_Y + VIDEO_H, VIDEO_X : VIDEO_X + VIDEO_W] = resized
    overlay = canvas.copy()
    cv2.rectangle(overlay, (VIDEO_X, VIDEO_Y), (VIDEO_X + VIDEO_W, VIDEO_Y + VIDEO_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.04, canvas, 0.96, 0, canvas)
    cv2.rectangle(canvas, (VIDEO_X, VIDEO_Y), (VIDEO_X + VIDEO_W, VIDEO_Y + VIDEO_H), LINE, 1)

    draw_card_rois(cv2, canvas, state, source_w, source_h)
    draw_seat_overlay(cv2, canvas, state, source_w, source_h)
    draw_bet_anchors(cv2, canvas, state, source_w, source_h)
    draw_table_labels(cv2, canvas, state)


def draw_card_rois(cv2: Any, canvas: Any, state: dict[str, Any], source_w: int, source_h: int) -> None:
    hero_cards = cards_of(state.get("hero", {}))
    board_cards = list((state.get("table") or {}).get("board") or [])
    for index, roi in enumerate(HERO_CARD_ROIS):
        x1, y1, x2, y2 = map_roi(roi, source_w, source_h)
        card = hero_cards[index] if index < len(hero_cards) else ""
        draw_roi(cv2, canvas, x1, y1, x2, y2, ORANGE, card)
    for index, roi in enumerate(BOARD_CARD_ROIS):
        x1, y1, x2, y2 = map_roi(roi, source_w, source_h)
        card = board_cards[index] if index < len(board_cards) else ""
        draw_roi(cv2, canvas, x1, y1, x2, y2, CYAN, card)


def draw_roi(cv2: Any, canvas: Any, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], label: str) -> None:
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        draw_badge(cv2, canvas, label, x1 + 3, max(VIDEO_Y + 18, y1 - 8), color)


def draw_seat_overlay(cv2: Any, canvas: Any, state: dict[str, Any], source_w: int, source_h: int) -> None:
    seat_lookup = {seat.get("seat_index"): seat for seat in state.get("seats") or []}
    dealer_index = (state.get("table") or {}).get("dealer_seat_index")
    seats = build_seats(source_w, source_h, 8)
    for seat in seats:
        seat_state = seat_lookup.get(seat["index"], {})
        sx = VIDEO_X + int(round(seat["screen"]["x"] * VIDEO_W / source_w))
        sy = VIDEO_Y + int(round(seat["screen"]["y"] * VIDEO_H / source_h))
        status = str(seat_state.get("status") or "")
        active = status.startswith("active")
        color = GREEN if active else GRAY
        if seat["index"] == 0:
            color = ORANGE
        if seat["index"] == dealer_index:
            color = CYAN
        cv2.circle(canvas, (sx, sy), 17, color, 2)
        if seat["index"] == dealer_index:
            cv2.circle(canvas, (sx - 22, sy - 18), 13, YELLOW, -1)
            draw_text(cv2, canvas, "D", sx - 28, sy - 12, 0.47, (20, 26, 32), 2)
        pos = seat_state.get("position") or "-"
        order = seat_state.get("preflop_action_order") if (state.get("table") or {}).get("street") == "preflop" else seat_state.get("postflop_action_order")
        state_text = "IN" if active else "OUT"
        label = f"{seat['index']} {pos} #{order or '-'} {state_text}"
        draw_label(cv2, canvas, label, sx + 20, sy + 5, color)


def draw_bet_anchors(cv2: Any, canvas: Any, state: dict[str, Any], source_w: int, source_h: int) -> None:
    seat_bets = {bet.get("seat_index"): bet.get("amount_bb") for bet in derive_bets(state.get("seats") or [])}
    for bet in state.get("bets") or []:
        seat_bets[bet.get("seat_index")] = bet.get("amount_bb")
    for seat_index, amount in seat_bets.items():
        if amount is None or seat_index not in BET_ANCHORS_8:
            continue
        ax, ay = BET_ANCHORS_8[seat_index]
        x = VIDEO_X + int(round(ax * VIDEO_W))
        y = VIDEO_Y + int(round(ay * VIDEO_H))
        cv2.circle(canvas, (x, y), 11, YELLOW, -1)
        draw_label(cv2, canvas, f"{float(amount):g}BB", x + 14, y + 5, YELLOW)


def draw_table_labels(cv2: Any, canvas: Any, state: dict[str, Any]) -> None:
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    pot = fmt_bb(table.get("pot_bb"))
    street = str(table.get("street") or "-").upper()
    hero_cards = fmt_cards(cards_of(hero))
    board = fmt_cards(table.get("board") or [])
    draw_badge(cv2, canvas, f"{street}  POT {pot}", VIDEO_X + 360, VIDEO_Y + 28, CYAN)
    draw_badge(cv2, canvas, f"HERO {hero.get('position') or '-'}  {hero_cards}", VIDEO_X + 350, VIDEO_Y + VIDEO_H - 48, ORANGE)
    draw_badge(cv2, canvas, f"BOARD {board}", VIDEO_X + 350, VIDEO_Y + VIDEO_H - 18, CYAN)


def draw_panel(cv2: Any, canvas: Any, state: dict[str, Any], recent_changes: list[str], timestamp: float) -> None:
    cv2.rectangle(canvas, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H), PANEL, -1)
    cv2.rectangle(canvas, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H), LINE, 1)
    x = PANEL_X + 24
    y = PANEL_Y + 36
    draw_text(cv2, canvas, "Current state", x, y, 0.78, TEXT, 2)
    y += 34
    source = state.get("source") or {}
    ocr_status = source.get("ocr_status") or "none"
    ocr_age = source.get("ocr_age_sec")
    status_line = f"time {timestamp:07.3f}s  OCR {ocr_status}"
    if ocr_age is not None:
        status_line += f" ({ocr_age:.2f}s)"
    draw_text(cv2, canvas, status_line, x, y, 0.48, MUTED, 1)
    y += 26

    table = state.get("table") or {}
    hero = state.get("hero") or {}
    confidence = state.get("confidence") or {}
    contribution = state.get("contribution") or {}
    seat_totals = {int(k): v for k, v in (contribution.get("seat_totals") or {}).items()}
    y = draw_section(cv2, canvas, "Table", x, y)
    y = draw_kv(cv2, canvas, x, y, "Street", str(table.get("street") or "-").upper())
    y = draw_kv(cv2, canvas, x, y, "Dealer", f"{table.get('dealer_seat') or '-'} / {table.get('dealer_position') or '-'}")
    y = draw_kv(cv2, canvas, x, y, "Pot", fmt_bb(table.get("pot_bb")))
    y = draw_kv(cv2, canvas, x, y, "To call", fmt_bb(table.get("to_call_bb")))
    y = draw_kv(cv2, canvas, x, y, "Board", fmt_cards(table.get("board") or []))
    y += 8

    y = draw_section(cv2, canvas, "Hero", x, y)
    action_order = hero.get("preflop_action_order") if table.get("street") == "preflop" else hero.get("postflop_action_order")
    y = draw_kv(cv2, canvas, x, y, "Cards", fmt_cards(cards_of(hero)))
    y = draw_kv(cv2, canvas, x, y, "Position", str(hero.get("position") or "-"))
    y = draw_kv(cv2, canvas, x, y, "Action", f"#{action_order or '-'}")
    y = draw_kv(cv2, canvas, x, y, "Status", status_label(hero.get("status")))
    y += 8

    advice = state.get("gto_advice") or {}
    if advice:
        y = draw_section(cv2, canvas, "GTO Advice", x, y)
        if advice.get("ready"):
            draw_text(cv2, canvas, str(advice.get("summary") or "-")[:48], x + 10, y, 0.5, GREEN, 1)
            y += 22
            mix = advice.get("mix") or {}
            mix_text = " / ".join(f"{key} {value}%" for key, value in mix.items())
            draw_text(cv2, canvas, mix_text[:58] or "-", x + 10, y, 0.42, MUTED, 1)
            y += 22
            size_text = ((advice.get("size_mix") or {}).get("summary") or "")[:58]
            if size_text:
                draw_text(cv2, canvas, size_text, x + 10, y, 0.40, YELLOW, 1)
                y += 22
        else:
            draw_text(cv2, canvas, f"not ready: {advice.get('reason') or '-'}"[:58], x + 10, y, 0.43, MUTED, 1)
            y += 22
        y += 8

    y = draw_section(cv2, canvas, "Visible Bets", x, y)
    bets = state.get("bets") or derive_bets(state.get("seats") or [])
    if bets:
        for bet in bets[:5]:
            seat = short_seat(bet.get("seat") or seat_name_from_index(bet.get("seat_index")))
            amount = fmt_bb(bet.get("amount_bb"))
            draw_text(cv2, canvas, f"{seat}: {amount}", x + 10, y, 0.5, YELLOW, 1)
            y += 22
    else:
        draw_text(cv2, canvas, "none detected on current OCR frame", x + 10, y, 0.46, MUTED, 1)
        y += 24
    audit_text = (
        f"Accounted {fmt_bb(contribution.get('total_invested'))} / "
        f"Pot {fmt_bb(contribution.get('pot_bb'))} / "
        f"Unseen {fmt_bb(contribution.get('unseen_bb'))} / "
        f"Overread {fmt_bb(contribution.get('overread_bb'))}"
    )
    draw_text(cv2, canvas, audit_text, x + 10, y, 0.43, MUTED, 1)
    y += 24
    y += 8

    y = draw_section(cv2, canvas, "Seats", x, y)
    draw_text(cv2, canvas, "Seat        Pos   Ord  Status       Invest", x + 10, y, 0.42, MUTED, 1)
    y += 18
    dealer_index = table.get("dealer_seat_index")
    for seat in state.get("seats") or []:
        seat_index = seat.get("seat_index")
        row_color = TEXT
        if seat_index == 0:
            row_color = ORANGE
        if seat_index == dealer_index:
            row_color = CYAN
        order = seat.get("preflop_action_order") if table.get("street") == "preflop" else seat.get("postflop_action_order")
        prefix = "D " if seat_index == dealer_index else "  "
        text = (
            f"{prefix}{short_seat(seat.get('seat')):<9} "
            f"{str(seat.get('position') or '-'):<5} "
            f"{str(order or '-'):<3} "
            f"{status_label(seat.get('status')):<11} "
            f"{fmt_bb(seat_totals.get(int(seat_index or 0), seat.get('bet_bb')))}"
        )
        draw_text(cv2, canvas, text, x + 8, y, 0.43, row_color, 1)
        y += 24
    if contribution.get("unseen_bb"):
        draw_text(cv2, canvas, f"  UNSEEN    --    --   IN POT       {fmt_bb(contribution.get('unseen_bb'))}", x + 8, y, 0.43, YELLOW, 1)
        y += 22
    if contribution.get("overread_bb"):
        draw_text(cv2, canvas, f"  OVERREAD  --    --   OCR EXTRA   -{fmt_bb(contribution.get('overread_bb'))}", x + 8, y, 0.43, RED, 1)
        y += 22
    y += 6

    y = draw_section(cv2, canvas, "Confidence", x, y)
    y = draw_kv(cv2, canvas, x, y, "Dealer D", fmt_conf(confidence.get("dealer_button")))
    y = draw_kv(cv2, canvas, x, y, "Pot OCR", fmt_conf(confidence.get("pot_ocr")))

    bottom = PANEL_Y + PANEL_H - 112
    if recent_changes and y < bottom - 8:
        y = draw_section(cv2, canvas, "Recent changes", x, bottom)
        for line in recent_changes[:4]:
            draw_text(cv2, canvas, line[:58], x + 8, y, 0.4, MUTED, 1)
            y += 20


def draw_progress(cv2: Any, canvas: Any, timestamp: float, start_sec: float, end_sec: float) -> None:
    x1 = VIDEO_X
    y = VIDEO_Y + VIDEO_H + 32
    x2 = VIDEO_X + VIDEO_W
    cv2.line(canvas, (x1, y), (x2, y), LINE, 6)
    total = max(0.001, end_sec - start_sec)
    ratio = min(1.0, max(0.0, (timestamp - start_sec) / total))
    cv2.line(canvas, (x1, y), (x1 + int((x2 - x1) * ratio), y), CYAN, 6)
    draw_text(cv2, canvas, f"{timestamp - start_sec:05.1f}s / {total:05.1f}s", x1, y + 26, 0.48, MUTED, 1)


def draw_section(cv2: Any, canvas: Any, title: str, x: int, y: int) -> int:
    cv2.rectangle(canvas, (x, y - 18), (PANEL_X + PANEL_W - 24, y + 8), PANEL_2, -1)
    draw_text(cv2, canvas, title, x + 10, y, 0.53, TEXT, 1)
    return y + 34


def draw_kv(cv2: Any, canvas: Any, x: int, y: int, key: str, value: str) -> int:
    draw_text(cv2, canvas, f"{key}:", x + 10, y, 0.48, MUTED, 1)
    draw_text(cv2, canvas, value, x + 120, y, 0.5, TEXT, 1)
    return y + 24


def draw_text(
    cv2: Any,
    image: Any,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_label(cv2: Any, image: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    cv2.rectangle(image, (x - 3, y - th - 5), (x + tw + 5, y + 4), (12, 15, 19), -1)
    draw_text(cv2, image, text, x, y, 0.43, color, 1)


def draw_badge(cv2: Any, image: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(image, (x - 7, y - th - 7), (x + tw + 8, y + 7), (15, 20, 26), -1)
    cv2.rectangle(image, (x - 7, y - th - 7), (x + tw + 8, y + 7), color, 1)
    draw_text(cv2, image, text, x, y, 0.48, color, 1)


def map_roi(roi: tuple[float, float, float, float], source_w: int, source_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        VIDEO_X + int(round(x1 * VIDEO_W)),
        VIDEO_Y + int(round(y1 * VIDEO_H)),
        VIDEO_X + int(round(x2 * VIDEO_W)),
        VIDEO_Y + int(round(y2 * VIDEO_H)),
    )


def fmt_bb(value: Any) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):g} BB"
    except (TypeError, ValueError):
        return "--"


def fmt_conf(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def fmt_cards(cards: list[Any]) -> str:
    return " ".join(str(card) for card in cards if card) or "-"


def cards_of(hero: dict[str, Any]) -> list[Any]:
    return list(hero.get("cards") or [])


def status_label(status: Any) -> str:
    status_text = str(status or "-")
    if status_text.startswith("active"):
        return "ACTIVE"
    if status_text.startswith("folded") or status_text == "empty":
        return "FOLDED"
    return status_text.upper()


def short_seat(seat: Any) -> str:
    mapping = {
        "bottom_hero": "HERO",
        "bottom_left": "B_LEFT",
        "left": "LEFT",
        "top_left": "T_LEFT",
        "top": "TOP",
        "top_right": "T_RIGHT",
        "right": "RIGHT",
        "bottom_right": "B_RIGHT",
    }
    return mapping.get(str(seat), str(seat or "-"))


def seat_name_from_index(index: Any) -> str:
    names = {
        0: "bottom_hero",
        1: "bottom_left",
        2: "left",
        3: "top_left",
        4: "top",
        5: "top_right",
        6: "right",
        7: "bottom_right",
    }
    try:
        return names.get(int(index), "-")
    except (TypeError, ValueError):
        return "-"


def derive_bets(seats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bets = []
    for seat in seats:
        amount = seat.get("bet_bb")
        if amount is None:
            continue
        bets.append(
            {
                "seat_index": seat.get("seat_index"),
                "seat": seat.get("seat"),
                "amount_bb": amount,
            }
        )
    return bets


def display_signature(state: dict[str, Any]) -> str:
    payload = {
        "table": state.get("table"),
        "hero": state.get("hero"),
        "bets": state.get("bets") or derive_bets(state.get("seats") or []),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def describe_change(state: dict[str, Any], timestamp: float) -> str:
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    return (
        f"{timestamp:07.3f}s {str(table.get('street') or '-').upper()} "
        f"D={table.get('dealer_seat') or '-'} "
        f"H={fmt_cards(cards_of(hero))} "
        f"B={fmt_cards(table.get('board') or [])} "
        f"POT={fmt_bb(table.get('pot_bb'))}"
    )


class ContributionTracker:
    def __init__(self) -> None:
        self.hand_index = 0
        self.dealer_seat: str | None = None
        self.street: str | None = None
        self.last_pot: float | None = None
        self.seat_totals: dict[int, float] = {}
        self.street_bets: dict[int, float] = {}
        self.unseen_bb = 0.0
        self.overread_bb = 0.0

    def update(self, state: dict[str, Any]) -> None:
        if not state.get("ok"):
            return
        table = state.get("table") or {}
        street = table.get("street")
        dealer = table.get("dealer_seat")
        pot = as_float(table.get("pot_bb"))
        if pot is None:
            return
        if self.should_reset(table):
            self.hand_index += 1
            self.seat_totals = {}
            self.street_bets = {}
            self.unseen_bb = 0.0
            self.overread_bb = 0.0
        elif street != self.street:
            self.street_bets = {}

        self.dealer_seat = dealer
        self.street = street
        self.last_pot = pot if pot is not None else self.last_pot

        for bet in state.get("bets") or derive_bets(state.get("seats") or []):
            seat_index = safe_int(bet.get("seat_index"))
            amount = as_float(bet.get("amount_bb"))
            if seat_index is None or amount is None:
                continue
            previous_street_bet = self.street_bets.get(seat_index, 0.0)
            increment = max(0.0, amount - previous_street_bet)
            if increment > 0:
                self.seat_totals[seat_index] = round(self.seat_totals.get(seat_index, 0.0) + increment, 2)
            self.street_bets[seat_index] = max(previous_street_bet, amount)

        self.reconcile_pot(pot, state.get("bets") or derive_bets(state.get("seats") or []))

    def should_reset(self, table: dict[str, Any]) -> bool:
        street = table.get("street")
        board = table.get("board") or []
        dealer = table.get("dealer_seat")
        pot = as_float(table.get("pot_bb"))
        if self.dealer_seat is None:
            return True
        if street == "preflop" and not board:
            if self.street in {"flop", "turn", "river"}:
                return True
            if self.last_pot is not None and pot is not None and pot + 0.5 < self.last_pot:
                return True
            if dealer != self.dealer_seat and (pot is None or pot <= 6.0):
                return True
            if pot is not None and sum(self.seat_totals.values()) > pot + max(2.0, pot * 0.2):
                return True
        return False

    def reconcile_pot(self, pot: float, bets: list[dict[str, Any]]) -> None:
        assigned = round(sum(self.seat_totals.values()), 2)
        current_seats = {safe_int(bet.get("seat_index")) for bet in bets}
        current_seats.discard(None)
        if assigned > pot + 0.5:
            excess = assigned - pot
            stale_seats = [
                (seat_index, amount)
                for seat_index, amount in self.seat_totals.items()
                if seat_index not in current_seats and amount > 0
            ]
            for seat_index, amount in sorted(stale_seats, key=lambda item: item[1], reverse=True):
                if excess <= 0.5:
                    break
                reduction = min(amount, excess)
                remaining = round(amount - reduction, 2)
                if remaining > 0.25:
                    self.seat_totals[seat_index] = remaining
                else:
                    self.seat_totals.pop(seat_index, None)
                    self.street_bets.pop(seat_index, None)
                excess = round(excess - reduction, 2)

        assigned = round(sum(self.seat_totals.values()), 2)
        if assigned > pot + 0.5:
            # Keep the dashboard accounting exact when OCR over-reads a visible bet.
            scale = pot / assigned if assigned else 0.0
            for seat_index in list(self.seat_totals):
                adjusted = round(self.seat_totals[seat_index] * scale, 2)
                if adjusted > 0.25:
                    self.seat_totals[seat_index] = adjusted
                    self.street_bets[seat_index] = min(self.street_bets.get(seat_index, adjusted), adjusted)
                else:
                    self.seat_totals.pop(seat_index, None)
                    self.street_bets.pop(seat_index, None)

        assigned = round(sum(self.seat_totals.values()), 2)
        residual = round(pot - assigned, 2)
        self.unseen_bb = residual if residual > ACCOUNTING_TOLERANCE_BB else 0.0
        self.overread_bb = round(-residual, 2) if residual < -ACCOUNTING_TOLERANCE_BB else 0.0

    def snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        table = state.get("table") or {}
        pot = as_float(table.get("pot_bb"))
        assigned = round(sum(self.seat_totals.values()), 2)
        unseen = self.unseen_bb
        overread = self.overread_bb
        if pot is not None:
            residual = round(pot - assigned, 2)
            unseen = residual if residual > ACCOUNTING_TOLERANCE_BB else 0.0
            overread = round(-residual, 2) if residual < -ACCOUNTING_TOLERANCE_BB else 0.0
        return {
            "hand_index": self.hand_index,
            "seat_totals": {str(k): round(v, 2) for k, v in sorted(self.seat_totals.items()) if v > 0},
            "total_assigned": assigned,
            "unseen_bb": round(unseen, 2),
            "overread_bb": round(overread, 2),
            "total_invested": round(assigned + max(0.0, unseen) - max(0.0, overread), 2),
            "pot_bb": pot,
            "audit_diff_bb": round((pot or 0.0) - assigned - max(0.0, unseen) + max(0.0, overread), 2) if pot is not None else None,
            "street": self.street,
        }


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_dashboard_summary(payload: dict[str, Any]) -> str:
    render = payload["render"]
    return "\n".join(
        [
            f"Dashboard video: {payload['output_video']}",
            f"Frames: {render['frames']} ({render['start_sec']}s - {render['end_sec']}s)",
            f"Output: {render['output_width']}x{render['output_height']} @ {render['output_fps']} fps",
            f"Render speed: {render.get('render_fps')} fps",
            f"Summary JSON: {payload.get('summary_json', '')}",
        ]
    )
