from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .vision import (
    POSITION_ORDER_BY_SEATS,
    action_order_number,
    build_seats,
    find_dealer_button,
    nearest_seat_index,
    to_gto_position,
)
from .card_classifier import classify_rank_glyph, classify_suit_glyph
from .card_deep_model import classify_deep_glyph


DEFAULT_VIDEO_TEMPLATE = Path(__file__).resolve().parent.parent / "pict" / "D_purple.png"
FALLBACK_TEMPLATE = Path(__file__).resolve().parent.parent / "pict" / "D.png"
CARD_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "pict" / "card_templates"
BOARD_RANK_MODEL_PATH = Path(__file__).resolve().parent.parent / "pict" / "card_models" / "card_glyph_board_knn.npz"
HERO_RANK_MODEL_PATH = Path(__file__).resolve().parent.parent / "pict" / "card_models" / "card_glyph_hero_rank_knn.npz"
HERO_SUIT_MODEL_PATH = Path(__file__).resolve().parent.parent / "pict" / "card_models" / "card_glyph_suit_knn.npz"
HERO_BLACK_SUIT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "pict" / "card_models" / "card_glyph_hero_black_suit_knn.npz"
)
BOARD_BLACK_SUIT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "pict" / "card_models" / "card_glyph_board_black_suit_knn.npz"
)
_CARD_TEMPLATES: dict[str, dict[str, Any]] | None = None
_TEMPLATE_GROUP_CACHE: dict[int, dict[str, Any]] = {}
_CARD_RANK_RECOGNITION_CACHE: dict[tuple[Any, ...], tuple[float, str, float]] = {}
_CARD_SUIT_RECOGNITION_CACHE: dict[tuple[Any, ...], tuple[str | None, float | None, float | None, str]] = {}
_CARD_RECOGNITION_CACHE_LIMIT = 4096

BET_ANCHORS_8 = {
    0: (0.50, 0.76),
    1: (0.34, 0.70),
    2: (0.22, 0.52),
    3: (0.34, 0.39),
    4: (0.50, 0.34),
    5: (0.72, 0.39),
    6: (0.78, 0.53),
    7: (0.64, 0.67),
}

# Amount text sits farther out than the chip icon on the current WPT layout.
# Keep this separate from BET_ANCHORS_8 so a successful chip match preserves
# its existing, stricter seat assignment.
BET_TEXT_ANCHORS_8 = {
    0: (0.53, 0.70),
    1: (0.34, 0.66),
    2: (0.15, 0.48),
    3: (0.21, 0.30),
    4: (0.53, 0.25),
    5: (0.81, 0.29),
    6: (0.87, 0.48),
    7: (0.69, 0.65),
}

CARD_ROIS_8 = {
    0: (0.43, 0.74, 0.58, 0.92),
    1: (0.13, 0.63, 0.29, 0.82),
    2: (0.02, 0.39, 0.17, 0.56),
    3: (0.09, 0.10, 0.26, 0.27),
    4: (0.42, 0.08, 0.57, 0.25),
    5: (0.76, 0.10, 0.91, 0.27),
    6: (0.83, 0.39, 0.98, 0.57),
    7: (0.68, 0.63, 0.84, 0.82),
}

HERO_CARD_ROIS = (
    (0.445, 0.755, 0.490, 0.895),
    (0.490, 0.755, 0.565, 0.895),
)

HERO_CARD_OVERLAP_ROIS = (
    (0.390, 0.735, 0.475, 0.895),
    (0.440, 0.735, 0.555, 0.895),
)

HERO_CARD_ROI_VARIANTS = (
    (0.00, 0.00, 1.00, 1.00),
    (0.02, 0.01, 0.92, 0.82),
    (0.03, 0.01, 0.88, 0.80),
    (0.04, 0.02, 0.86, 0.78),
    (0.04, 0.02, 0.72, 0.72),
    (0.06, 0.02, 0.65, 0.70),
    (-0.03, 0.01, 0.96, 0.82),
    (0.00, 0.03, 0.82, 0.72),
)

HERO_CARD_ROI_VARIANTS_SLOT0 = tuple(variant for variant in HERO_CARD_ROI_VARIANTS if variant[0] <= 0.001)
HERO_CARD_ROI_VARIANTS_SLOT1 = tuple(variant for variant in HERO_CARD_ROI_VARIANTS if variant[0] >= -0.001)

BOARD_CARD_ROIS = (
    (0.297, 0.407, 0.375, 0.574),
    (0.380, 0.407, 0.458, 0.574),
    (0.464, 0.407, 0.542, 0.574),
    (0.547, 0.407, 0.625, 0.574),
    (0.630, 0.407, 0.708, 0.574),
)

THREE_BLIND_POSITIONS_8 = ("BTN", "SB", "BB", "THIRD_BLIND", "UTG", "LJ", "HJ", "CO")

BOARD_RANK_WINDOWS = (
    (0, 0, 55, 60),
    (0, 0, 64, 72),
    (4, 0, 55, 70),
    (0, 8, 55, 68),
    (4, 8, 55, 68),
    (8, 10, 50, 70),
    (10, 14, 46, 70),
    (0, 16, 55, 72),
    (5, 18, 50, 72),
    (12, 20, 42, 72),
)

STACK_ROIS_8 = {
    0: (0.43, 0.84, 0.57, 0.96),
    1: (0.16, 0.73, 0.30, 0.85),
    2: (0.03, 0.50, 0.17, 0.62),
    3: (0.10, 0.24, 0.25, 0.35),
    4: (0.43, 0.18, 0.58, 0.28),
    5: (0.77, 0.24, 0.91, 0.35),
    6: (0.83, 0.50, 0.98, 0.62),
    7: (0.69, 0.73, 0.84, 0.85),
}


def analyze_video(
    video_path: Path,
    output_dir: Path,
    template_path: Path | None = None,
    seat_count: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 5.0,
    middle: bool = False,
    max_frames: int | None = None,
    min_confidence: float = 0.45,
    save_frames: bool = True,
    save_annotated: bool = True,
) -> dict[str, Any]:
    cv2, np = load_cv()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    annotated_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 19.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps else 0.0
    if middle:
        start_sec = duration_sec * 0.25 if start_sec is None else start_sec
        end_sec = duration_sec * 0.75 if end_sec is None else end_sec
    else:
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = duration_sec if end_sec is None else end_sec

    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    ocr = load_ocr()
    results: list[dict[str, Any]] = []
    blind_structure_cache: dict[str, Any] | None = None
    times = sample_times(float(start_sec), float(end_sec), every_sec, max_frames)
    for index, timestamp in enumerate(times):
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        basename = f"frame_{index:04d}_{int(timestamp):06d}s"
        frame_path = frames_dir / f"{basename}.png"
        if save_frames:
            cv2.imwrite(str(frame_path), frame)

        try:
            frame_result = analyze_video_frame(
                frame,
                template,
                seat_count=seat_count,
                min_confidence=min_confidence,
                ocr=ocr,
                blind_structure_hint=blind_structure_cache,
            )
            if (frame_result.get("blind_structure") or {}).get("kind") == "three_blind":
                blind_structure_cache = frame_result["blind_structure"]
            frame_result["ok"] = True
        except Exception as error:
            frame_result = {"ok": False, "error": str(error)}

        frame_result.update(
            {
                "index": index,
                "timestamp_sec": round(float(timestamp), 3),
                "frame_index": frame_index,
                "frame_path": str(frame_path) if save_frames else "",
            }
        )

        if save_annotated and frame_result.get("ok"):
            annotated_path = annotated_dir / f"{basename}.png"
            annotate_video_frame(frame, frame_result, annotated_path)
            frame_result["annotated_path"] = str(annotated_path)
        results.append(frame_result)

    cap.release()

    payload = {
        "ok": True,
        "video": str(video_path),
        "template": str(template_path),
        "output_dir": str(output_dir),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
        },
        "sample": {
            "start_sec": round(float(start_sec), 3),
            "end_sec": round(float(end_sec), 3),
            "every_sec": every_sec,
            "frame_count": len(results),
        },
        "frames": results,
    }
    (output_dir / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", payload)
    return payload


def analyze_video_frame(
    frame: Any,
    template: Any,
    seat_count: int = 8,
    min_confidence: float = 0.45,
    ocr: Any | None = None,
    ocr_result_hint: list[Any] | None = None,
    return_ocr_result: bool = False,
    dealer_button_hint: dict[str, Any] | None = None,
    cards_hint: dict[str, Any] | None = None,
    ocr_scale: float = 1.0,
    layout_profile: dict[str, Any] | None = None,
    blind_structure_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    timing: dict[str, float] = {}
    height, width = frame.shape[:2]
    step_started = time.perf_counter()
    button = dealer_button_hint or find_dealer_button(
        frame,
        template,
        min_confidence=min_confidence,
        min_scale=0.75,
        max_scale=1.65,
    )
    timing["dealer_ms"] = elapsed_ms(step_started)
    step_started = time.perf_counter()
    seats = build_seats(width, height, seat_count)
    dealer_index = nearest_seat_index(button["center"], seats)
    timing["seats_ms"] = elapsed_ms(step_started)
    step_started = time.perf_counter()
    if ocr_result_hint is None:
        ocr_result = run_ocr(frame, ocr, scale=ocr_scale)
        timing["ocr_ms"] = elapsed_ms(step_started)
    else:
        # 调用方只有在确认牌桌像素未变化后才能复用光学字符识别（OCR）结果；
        # 底池和下注仍经过同一套坐标解析，避免改变识别语义。
        ocr_result = ocr_result_hint
        timing["ocr_ms"] = 0.0
        timing["ocr_cached"] = 1.0
    blind_structure = detect_blind_structure(ocr_result, frame_height=height)
    if (
        blind_structure.get("kind") == "two_blind"
        and (blind_structure_hint or {}).get("kind") == "three_blind"
    ):
        blind_structure = {**blind_structure_hint, "source": "cached_table_title"}
    step_started = time.perf_counter()
    pot = detect_pot(frame, ocr_result)
    bets = detect_bets(frame, seats, ocr_result=ocr_result, pot=pot)
    action_controls = detect_action_controls(frame, ocr_result)
    timing["money_action_ms"] = elapsed_ms(step_started)
    step_started = time.perf_counter()
    cards = cards_hint if cards_hint is not None else detect_visible_cards(frame, layout_profile=layout_profile)
    timing["cards_ms"] = elapsed_ms(step_started)
    if should_retry_missing_top_preflop_bet(cards, pot, bets, seats):
        step_started = time.perf_counter()
        retry_ocr = run_ocr_in_roi(frame, ocr, (0.40, 0.18, 0.65, 0.44), scale=1.0)
        retry_bets = detect_bets(frame, seats, ocr_result=[*ocr_result, *retry_ocr], pot=pot)
        timing["bet_retry_ocr_ms"] = elapsed_ms(step_started)
        if bets_reconcile_pot_better(pot, bets, retry_bets):
            ocr_result = [*ocr_result, *retry_ocr]
            bets = retry_bets
            timing["bet_retry_used"] = 1.0
    step_started = time.perf_counter()
    card_statuses = detect_card_statuses(frame, seat_count)
    timing["card_status_ms"] = elapsed_ms(step_started)
    step_started = time.perf_counter()

    position_order = (
        THREE_BLIND_POSITIONS_8
        if seat_count == 8 and blind_structure.get("kind") == "three_blind"
        else POSITION_ORDER_BY_SEATS[seat_count]
    )
    for seat in seats:
        offset = (seat["index"] - dealer_index) % seat_count
        position = position_order[offset]
        seat["distance_from_dealer_clockwise"] = offset
        seat["preflop_action_order"] = (
            three_blind_action_order_number(offset, seat_count)
            if blind_structure.get("kind") == "three_blind"
            else action_order_number(offset, seat_count, "preflop")
        )
        seat["postflop_action_order"] = action_order_number(offset, seat_count, "postflop")
        seat["position"] = position
        seat["gto_position"] = to_gto_position(position)
        seat["bet_bb"] = bets.get(seat["index"], {}).get("amount_bb")
        seat["bet_text"] = bets.get(seat["index"], {}).get("text", "")
        seat["has_cards"] = card_statuses.get(seat["index"], {}).get("has_cards", False)
        seat["status"] = "active_or_showdown" if seat["has_cards"] else "folded_or_empty"
        seat["card_metrics"] = card_statuses.get(seat["index"], {})
    timing["seat_enrich_ms"] = elapsed_ms(step_started)
    timing["total_ms"] = elapsed_ms(started_at)

    hero = seats[0]
    dealer = seats[dealer_index]
    result = {
        "dealer_button": button,
        "dealer": {
            "seat_index": dealer_index,
            "seat": dealer["name"],
            "position": dealer["position"],
        },
        "hero": {
            "seat_index": 0,
            "seat": hero["name"],
            "distance_from_dealer_clockwise": hero["distance_from_dealer_clockwise"],
            "preflop_action_order": hero["preflop_action_order"],
            "postflop_action_order": hero["postflop_action_order"],
            "position": hero["position"],
            "gto_position": hero["gto_position"],
            "bet_bb": hero["bet_bb"],
            "has_cards": hero["has_cards"],
            "status": hero["status"],
            "cards": cards["hero"],
        },
        "pot": pot,
        "cards": cards,
        "action_controls": action_controls,
        "blind_structure": blind_structure,
        "seats": seats,
        "detected_bets": list(bets.values()),
        "timing_ms": timing,
        "ocr_item_count": len(ocr_result),
        "cards_hint_used": cards_hint is not None,
    }
    if return_ocr_result:
        result["_ocr_result"] = ocr_result
    return result


def detect_blind_structure(ocr_result: list[Any], frame_height: int) -> dict[str, Any]:
    """Read a three-value table title such as 0.20/0.50/1 as three forced blinds."""

    pattern = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])")
    for item in ocr_result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = item[0]
        text = str(item[1] or "")
        try:
            top = min(float(point[1]) for point in box)
        except (TypeError, ValueError, IndexError):
            continue
        if top > max(40.0, frame_height * 0.10):
            continue
        match = pattern.search(text)
        if match is None:
            continue
        values = [float(value) for value in match.groups()]
        if not (0 < values[0] < values[1] < values[2]):
            continue
        posts_bb = [round(value / values[1], 2) for value in values]
        if not (0.15 <= posts_bb[0] <= 0.75 and 1.4 <= posts_bb[2] <= 3.0):
            continue
        return {
            "kind": "three_blind",
            "source": "table_title_ocr",
            "title_text": text,
            "display_values": values,
            "posts_bb": {
                "SB": posts_bb[0],
                "BB": posts_bb[1],
                "THIRD_BLIND": posts_bb[2],
            },
        }
    return {"kind": "two_blind", "source": "default"}


def run_blind_structure_title_ocr(frame: Any, ocr: Any | None) -> list[Any]:
    """Read the tiny client title from a tight, enlarged crop."""

    if ocr is None:
        return []
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    title_height = max(18, min(32, int(round(height * 0.035))))
    title_width = max(1, int(round(width * 0.56)))
    crop = frame[:title_height, :title_width]
    if crop.size == 0:
        return []
    scale = 4.0
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return scale_ocr_result(run_ocr(enlarged, ocr), 1.0 / scale)


def three_blind_action_order_number(distance_from_dealer: int, seat_count: int) -> int:
    """Act after the third forced blind, leaving that blind with the final option."""

    offsets = [*range(4, seat_count), 0, 1, 2, 3]
    return offsets.index(distance_from_dealer) + 1


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def run_ocr(frame: Any, ocr: Any | None, scale: float = 1.0) -> list[Any]:
    if ocr is None:
        return []
    scale = float(scale or 1.0)
    if 0.1 < scale < 0.999:
        cv2, _np = load_cv()
        resized = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        result, _elapsed = ocr(resized)
        return scale_ocr_result(result or [], 1.0 / scale)
    result, _elapsed = ocr(frame)
    return result or []


def run_ocr_in_roi(
    frame: Any,
    ocr: Any | None,
    roi: tuple[float, float, float, float],
    scale: float = 1.0,
) -> list[Any]:
    """Run OCR on a normalized subregion and return boxes in full-frame coordinates."""

    if ocr is None:
        return []
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = scale_roi(roi, width, height)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    return offset_ocr_result(run_ocr(crop, ocr, scale=scale), x1, y1)


def should_retry_missing_top_preflop_bet(
    cards: dict[str, Any],
    pot: dict[str, Any] | None,
    bets: dict[int, dict[str, Any]],
    seats: list[dict[str, Any]],
) -> bool:
    """Retry only the top contribution when a common preflop amount is missing."""

    if len(seats) != 8 or 4 in bets:
        return False
    if list(cards.get("board") or []) or cards.get("board_pending"):
        return False
    pot_amount = float(pot.get("amount_bb")) if pot and pot.get("amount_bb") is not None else None
    if pot_amount is None or pot_amount <= 0:
        return False
    visible_total = sum(float(item.get("amount_bb") or 0.0) for item in bets.values())
    missing = pot_amount - visible_total
    return any(abs(missing - amount) <= 0.15 for amount in (0.4, 1.0, 2.0))


def bets_reconcile_pot_better(
    pot: dict[str, Any] | None,
    current: dict[int, dict[str, Any]],
    retry: dict[int, dict[str, Any]],
) -> bool:
    """Accept a retry only when it moves visible contributions toward the pot."""

    pot_amount = float(pot.get("amount_bb")) if pot and pot.get("amount_bb") is not None else None
    if pot_amount is None:
        return False
    current_total = sum(float(item.get("amount_bb") or 0.0) for item in current.values())
    retry_total = sum(float(item.get("amount_bb") or 0.0) for item in retry.values())
    current_gap = abs(pot_amount - current_total)
    retry_gap = abs(pot_amount - retry_total)
    return retry_gap + 0.15 < current_gap and retry_total <= pot_amount + 0.15


def scale_ocr_result(result: list[Any], factor: float) -> list[Any]:
    scaled = []
    for item in result:
        if len(item) < 3:
            continue
        box, text, confidence = item[0], item[1], item[2]
        scaled_box = [[float(point[0]) * factor, float(point[1]) * factor] for point in box]
        scaled.append((scaled_box, text, confidence))
    return scaled


def offset_ocr_result(result: list[Any], offset_x: int, offset_y: int) -> list[Any]:
    """Translate OCR boxes from a crop back into the original frame."""

    shifted = []
    for item in result:
        if len(item) < 3:
            continue
        box, text, confidence = item[0], item[1], item[2]
        shifted_box = [
            [float(point[0]) + float(offset_x), float(point[1]) + float(offset_y)] for point in box
        ]
        shifted.append((shifted_box, text, confidence))
    return shifted


def detect_pot(frame: Any, ocr_result: list[Any]) -> dict[str, Any] | None:
    if not ocr_result:
        return None
    height, width = frame.shape[:2]
    candidates = []
    for box, raw_text, raw_conf in ocr_result:
        text = normalize_ocr_text(str(raw_text))
        text_box = ocr_box_bounds(box)
        center_x = text_box["x"] + text_box["width"] / 2
        center_y = text_box["y"] + text_box["height"] / 2
        # The WPT pot label sits immediately above the board, around 30% of
        # the reviewed table height. Keep the search clear of top player stacks
        # while accepting the actual pot-label band used by live captures.
        in_pot_area = 0.28 * width <= center_x <= 0.72 * width and 0.23 * height <= center_y <= 0.43 * height
        if not in_pot_area:
            continue
        amount = parse_bb_amount(text)
        if amount is None:
            continue
        explicit_label = is_explicit_pot_text(text)
        if not explicit_label and is_player_stack_region(text_box, frame.shape, padding_x=0.0, padding_y=0.0):
            continue
        distance = math.hypot(center_x - width * 0.5, center_y - height * 0.30)
        candidates.append(
            {
                "amount_bb": amount,
                "text": text,
                "confidence": round(float(raw_conf), 4),
                "box": text_box,
                "_rank": (0 if explicit_label else 1, distance - float(raw_conf) * 20),
            }
        )
    if not candidates:
        return None
    best = min(candidates, key=lambda item: item["_rank"])
    best.pop("_rank", None)
    return best


def detect_bets(frame: Any, seats: list[dict[str, Any]], ocr_result: list[Any], pot: dict[str, Any] | None = None) -> dict[int, dict[str, Any]]:
    if not ocr_result:
        return {}

    chips = detect_red_chips(frame)
    bets: dict[int, dict[str, Any]] = {}
    for box, raw_text, raw_conf in ocr_result:
        text = normalize_ocr_text(str(raw_text))
        amount = parse_bb_amount(text)
        if amount is None:
            continue
        text_box = ocr_box_bounds(box)
        if is_ignored_bet_text(text, text_box, frame.shape, pot):
            continue
        chip = nearby_chip(text_box, chips)
        text_seat_index, text_anchor_distance = nearest_bet_text_seat(text_box, frame.shape, len(seats))
        pot_amount = float(pot.get("amount_bb")) if pot and pot.get("amount_bb") is not None else None
        raw_amount = amount
        amount = repair_bet_amount(amount, text, pot_amount)
        close_blind_text = (
            chip is None
            and text_seat_index is not None
            and raw_amount <= 2.05
            and text_anchor_distance <= 84.0
        )
        if (
            chip is None
            and is_player_stack_region(text_box, frame.shape, padding_x=0.0, padding_y=0.0)
            and not close_blind_text
        ):
            continue
        stack_sized = amount >= 20.0 if pot_amount is None else amount >= 20.0 and amount > max(20.0, pot_amount * 1.35)
        # 玩家筹码余量与真实红筹码可能处在同一横向区域。只要大额数字
        # 位于玩家面板，就不是桌面下注；小盲注仍需保留。
        if stack_sized and is_player_stack_region(text_box, frame.shape):
            continue
        # 金额文字固定在各座位的下注区域；红色筹码会在动画中向内移动。
        # 因此筹码只能确认“这是下注”，不能覆盖金额文字已经给出的归属。
        if text_seat_index is not None:
            seat_index = text_seat_index
        elif chip is None:
            seat_index, anchor_distance = text_seat_index, text_anchor_distance
            if seat_index is None:
                continue
            chip = {
                "x": text_box["x"] + text_box["width"] / 2,
                "y": text_box["y"] + text_box["height"] / 2,
                "area": 0.0,
                "circularity": 0.0,
                "source": "text_anchor",
                "anchor_distance": round(anchor_distance, 2),
                "box": text_box,
            }
        else:
            seat_index = nearest_bet_seat(chip, frame.shape, len(seats))
        current = bets.get(seat_index)
        item = {
            "seat_index": seat_index,
            "seat": seats[seat_index]["name"],
            "text": text,
            "amount_bb": amount,
            "confidence": round(float(raw_conf), 4),
            "box": text_box,
            "chip": chip,
        }
        if current is None or item["confidence"] > current["confidence"]:
            bets[seat_index] = item
    return bets


def detect_action_controls(frame: Any, ocr_result: list[Any]) -> dict[str, Any]:
    height, width = frame.shape[:2]
    red_buttons = detect_bottom_action_buttons(frame)
    action_button_row = dominant_action_button_row(red_buttons)
    bottom_texts = []
    call_amounts = []
    raise_amounts = []
    truncated_call_prefixes = []
    label_boxes: list[tuple[str, dict[str, int]]] = []
    for box, raw_text, raw_conf in ocr_result:
        text = normalize_ocr_text(str(raw_text))
        text_box = ocr_box_bounds(box)
        center_x = text_box["x"] + text_box["width"] / 2
        center_y = text_box["y"] + text_box["height"] / 2
        if center_y < 0.84 * height:
            continue
        bottom_texts.append({"text": text, "confidence": round(float(raw_conf), 4), "box": text_box})
        label = action_label_from_text(text)
        if label:
            label_boxes.append((label, text_box))
        amount = parse_bb_amount(text)
        # Player stacks also live near the lower edge of the table.  Amounts
        # above the button row must never be treated as CALL/RAISE amounts.
        if center_y < 0.90 * height:
            continue
        if amount is None:
            if action_amount_role(center_x, center_y, action_button_row) == "call":
                truncated_prefix = parse_truncated_bb_prefix(text)
                if truncated_prefix is not None:
                    truncated_call_prefixes.append((float(raw_conf), truncated_prefix))
            continue
        amount_role = action_amount_role(center_x, center_y, action_button_row)
        if (
            amount_role is None
            and label in {"call", "raise"}
            and label_box_overlaps_button(text_box, action_button_row)
        ):
            # Some themes expose one large action button rather than a full
            # three-button row. Its explicit label is still a safe amount
            # anchor; a nearby bare blind amount is not.
            amount_role = label
        if amount_role == "call":
            call_amounts.append((float(raw_conf), amount))
        elif amount_role == "raise":
            raise_amounts.append((float(raw_conf), amount))
        elif not action_button_row:
            # Text-only themes have no colored surfaces to anchor against.
            if 0.66 * width <= center_x <= 0.84 * width:
                call_amounts.append((float(raw_conf), amount))
            elif center_x > 0.84 * width:
                raise_amounts.append((float(raw_conf), amount))

    call_amount = best_amount(call_amounts)
    raise_amount = best_amount(raise_amounts)
    detected_labels = {label for label, _box in label_boxes}
    if call_amount is None:
        call_amount = amount_inside_labeled_button("call", label_boxes, bottom_texts, action_button_row)
    if raise_amount is None:
        raise_amount = amount_inside_labeled_button("raise", label_boxes, bottom_texts, action_button_row)
    if red_buttons:
        # The client leaves pre-action labels such as "call 2BB" visible in
        # gray while only quick-fold is clickable.  Once a red surface was
        # found, a label counts as an action only when it lies on that surface.
        actions = sorted(
            {
                label
                for label, text_box in label_boxes
                if label_box_overlaps_button(text_box, red_buttons)
            }
        )
        disabled_actions = sorted(detected_labels.difference(actions))
    else:
        # Some themes expose only label text.  Preserve that fallback when no
        # button surface is visible rather than hiding an otherwise usable row.
        actions = sorted(detected_labels)
        disabled_actions = []
    if looks_like_three_action_button_panel(red_buttons):
        # This geometry is specific to the standard FOLD/CALL/RAISE panel.
        # OCR can miss any one of its tiny labels, especially the middle CALL.
        # The two-button CHECK/BET panel never enters this branch.
        actions = sorted({*actions, "fold", "call", "raise"})
        disabled_actions = sorted(detected_labels.difference(actions))
    if "call" in actions and call_amount is None and raise_amount is not None:
        call_amount = infer_truncated_call_amount(truncated_call_prefixes, raise_amount)
    # A bare number is not proof that Hero can act: it can be a stack, a
    # slider preset, or a cropped control from a neighbouring area.  Require
    # a detected button surface or an actual action label before reporting a
    # Hero action panel.
    visible = bool(red_buttons or actions)
    if not visible:
        call_amount = None
        raise_amount = None
    if "call" not in actions:
        call_amount = None
    if "raise" not in actions:
        raise_amount = None
    if visible and call_amount is not None and "call" not in actions:
        actions.append("call")
    if visible and raise_amount is not None and "raise" not in actions:
        actions.append("raise")
    return {
        "visible": visible,
        "actions": actions,
        "call_amount_bb": call_amount,
        "call_amount_evidence": "button_row_ocr" if call_amount is not None and action_button_row else None,
        "raise_amount_bb": raise_amount,
        "red_button_regions": red_buttons,
        "bottom_texts": bottom_texts,
        "disabled_actions": disabled_actions,
    }


def label_box_overlaps_button(text_box: dict[str, int], buttons: list[dict[str, Any]]) -> bool:
    """Return whether an action label is inside a detected red clickable surface."""

    center_x = float(text_box["x"]) + float(text_box["width"]) / 2.0
    center_y = float(text_box["y"]) + float(text_box["height"]) / 2.0
    for button in buttons:
        left = float(button.get("x") or 0)
        top = float(button.get("y") or 0)
        right = left + float(button.get("width") or 0)
        bottom = top + float(button.get("height") or 0)
        if left <= center_x <= right and top <= center_y <= bottom:
            return True
    return False


def amount_inside_labeled_button(
    label: str,
    label_boxes: list[tuple[str, dict[str, int]]],
    texts: list[dict[str, Any]],
    buttons: list[dict[str, Any]],
) -> float | None:
    """Join a split action label and amount when both occupy one button."""

    target_buttons = [
        button
        for button in buttons
        if any(item_label == label and label_box_overlaps_button(box, [button]) for item_label, box in label_boxes)
    ]
    candidates = []
    for item in texts:
        amount = parse_bb_amount(str(item.get("text") or ""))
        if amount is None or not label_box_overlaps_button(item.get("box") or {}, target_buttons):
            continue
        candidates.append((float(item.get("confidence") or 0.0), amount))
    return best_amount(candidates)


def looks_like_three_action_button_panel(buttons: list[dict[str, Any]]) -> bool:
    ordered = dominant_action_button_row(buttons)
    if len(ordered) != 3:
        return False
    widths = [float(item.get("width") or 0.0) for item in ordered]
    heights = [float(item.get("height") or 0.0) for item in ordered]
    centers_y = [
        float(item.get("y") or 0.0) + float(item.get("height") or 0.0) / 2.0
        for item in ordered
    ]
    if min(widths, default=0.0) <= 0.0 or min(heights, default=0.0) <= 0.0:
        return False
    if max(widths) / min(widths) > 1.65 or max(heights) / min(heights) > 1.65:
        return False
    if max(centers_y) - min(centers_y) > max(heights) * 0.55:
        return False
    for left, right in zip(ordered, ordered[1:]):
        left_edge = float(left.get("x") or 0.0) + float(left.get("width") or 0.0)
        gap = float(right.get("x") or 0.0) - left_edge
        if gap < -min(widths) * 0.15 or gap > max(widths) * 0.75:
            return False
    return True


def dominant_action_button_row(buttons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the large action buttons and discard the small slider controls."""

    if not buttons:
        return []
    areas = [
        float(item.get("area") or 0.0)
        or float(item.get("width") or 0.0) * float(item.get("height") or 0.0)
        for item in buttons
    ]
    widths = [float(item.get("width") or 0.0) for item in buttons]
    heights = [float(item.get("height") or 0.0) for item in buttons]
    max_area = max(areas, default=0.0)
    max_width = max(widths, default=0.0)
    max_height = max(heights, default=0.0)
    dominant = [
        item
        for item, area, button_width, button_height in zip(buttons, areas, widths, heights)
        if area >= max_area * 0.42
        and button_width >= max_width * 0.55
        and button_height >= max_height * 0.55
    ]
    return sorted(dominant, key=lambda item: float(item.get("x") or 0.0))


def action_amount_role(
    center_x: float,
    center_y: float,
    buttons: list[dict[str, Any]],
) -> str | None:
    """Map an amount to the middle CALL or right RAISE button."""

    if len(buttons) != 3:
        return None
    for index, button in enumerate(buttons):
        left = float(button.get("x") or 0.0)
        top = float(button.get("y") or 0.0)
        right = left + float(button.get("width") or 0.0)
        bottom = top + float(button.get("height") or 0.0)
        if left <= center_x <= right and top <= center_y <= bottom:
            if index == 1:
                return "call"
            if index == 2:
                return "raise"
    return None


def infer_truncated_call_amount(
    candidates: list[tuple[float, int]],
    raise_amount: float,
) -> float | None:
    if not candidates or raise_amount <= 0.0:
        return None
    expected = float(raise_amount) / 2.0
    _confidence, prefix = max(candidates, key=lambda item: item[0])
    if int(math.floor(expected)) != int(prefix) or not 0.0 <= expected - float(prefix) < 1.0:
        return None
    return round(expected, 2)


def action_label_from_text(text: str) -> str | None:
    compact = text.replace(" ", "")
    if "\u5f03\u724c" in compact:
        return "fold"
    if "\u5168\u4e0b" in compact or "allin" in compact.lower() or "all-in" in compact.lower():
        return "all_in"
    if "\u8ddf\u6ce8" in compact:
        return "call"
    if "\u52a0\u6ce8" in compact:
        return "raise"
    if "\u8fc7\u724c" in compact or "\u770b\u724c" in compact:
        return "check"
    if "\u4e0b\u6ce8" in compact:
        return "bet"
    return None


def is_explicit_pot_text(text: str) -> bool:
    """Return whether OCR text explicitly names the pot rather than a bare amount."""

    compact = text.replace(" ", "").lower()
    return any(token in compact for token in ("\u5e95\u6c60", "\u6c60", "pot", "袝蟹袚懈", "袚懈"))


def best_amount(candidates: list[tuple[float, float]]) -> float | None:
    if not candidates:
        return None
    return round(max(candidates, key=lambda item: item[0])[1], 2)


def detect_bottom_action_buttons(frame: Any) -> list[dict[str, Any]]:
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    y1 = int(height * 0.84)
    x1 = int(width * 0.42)
    crop = frame[y1:height, x1:width]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 75, 65), (12, 255, 255)) | cv2.inRange(hsv, (170, 75, 65), (180, 255, 255))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1200:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 20:
            continue
        regions.append({"x": x + x1, "y": y + y1, "width": w, "height": h, "area": round(float(area), 1)})
    return regions


def detect_visible_cards(frame: Any, layout_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    height, width = frame.shape[:2]
    hero_cards = []
    if layout_profile is not None:
        return detect_visible_cards_from_layout(frame, layout_profile)
    auto_hero_cards = detect_auto_hero_cards(frame)
    if auto_hero_cards:
        # In live screen captures the auto bbox can land on the inner felt, which
        # shifts the fixed hero ROI upward into the board. When bottom-card
        # components are visible, trust those crops first and avoid board leakage.
        hero_cards = auto_hero_cards
    else:
        for index, roi in enumerate(HERO_CARD_ROIS):
            detail = recognize_best_hero_card_roi(frame, roi, index)
            if detail:
                hero_cards.append(detail)
    board_cards = detect_board_cards_default(frame)
    return {
        "hero": [item["card"] for item in hero_cards],
        "board": [item["card"] for item in board_cards],
        "hero_details": hero_cards,
        "board_details": board_cards,
    }


def detect_visible_cards_from_layout(frame: Any, layout_profile: dict[str, Any]) -> dict[str, Any]:
    height, width = frame.shape[:2]
    hero_cards = detect_locked_hero_cards(frame, layout_profile)
    board_cards = detect_board_cards_default(frame)
    return {
        "hero": [item["card"] for item in hero_cards],
        "board": [item["card"] for item in board_cards],
        "hero_details": hero_cards,
        "board_details": board_cards,
        "layout": {
            "id": layout_profile.get("id"),
            "method": layout_profile.get("method"),
            "frame_size": {"width": width, "height": height},
        },
    }


def detect_board_cards_default(frame: Any) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    board_cards = []
    for index, roi in enumerate(BOARD_CARD_ROIS):
        x1, y1, x2, y2 = scale_roi(roi, width, height)
        detail = recognize_card_crop(frame[y1:y2, x1:x2], source="board", index=index)
        if detail:
            board_cards.append(detail)
    auto_board_cards = detect_auto_board_cards(frame)
    return merge_card_details_by_index(board_cards, auto_board_cards)


def detect_locked_hero_cards(frame: Any, layout_profile: dict[str, Any]) -> list[dict[str, Any]]:
    if layout_profile.get("hero_card_source") == "manual_hero_cards":
        return recognize_cards_from_relative_boxes(
            frame,
            layout_profile.get("hero_card_boxes") or [],
            source="hero",
            roi_mode="manual_hero_card",
        )

    locked_cards = recognize_cards_from_relative_boxes(
        frame,
        locked_profile_hero_read_boxes(layout_profile),
        source="hero",
        roi_mode="locked_profile_anchor",
    )
    raw_locked_cards = recognize_cards_from_relative_boxes(
        frame,
        profile_hero_card_boxes(layout_profile),
        source="hero",
        roi_mode="locked_profile_raw",
    )
    locked_cards = select_locked_hero_variants(locked_cards, raw_locked_cards)
    if not card_set_is_incomplete(locked_cards) and not card_set_needs_roi_repair(locked_cards):
        return locked_cards

    search_box = layout_profile.get("hero_search_box") or {"x": 0.30, "y": 0.55, "width": 0.40, "height": 0.41}
    crops = detect_auto_hero_card_crops_in_relative_box(
        frame,
        search_box,
    )
    cards = []
    for index, crop_info in enumerate(crops[:2]):
        detail = recognize_card_crop(crop_info["crop"], source="hero", index=index)
        if detail:
            detail["roi_mode"] = "locked_layout_search"
            detail["roi_box"] = crop_info["box"]
            cards.append(detail)
    candidate_sets = [cards for cards in (locked_cards, raw_locked_cards) if cards]
    if cards:
        candidate_sets.append(cards)
    auto_cards = filter_cards_to_relative_box(detect_auto_hero_cards(frame), search_box, frame.shape)
    if auto_cards:
        for detail in auto_cards:
            detail["roi_mode"] = "locked_layout_auto_validated"
        candidate_sets.append(auto_cards)
    primary_candidate_sets = list(candidate_sets)
    # A profile learned from two visible cards already has table-relative read
    # windows. Generic overlap ROIs can jump into chips/table text when H2 is
    # partly covered, so they are only retained for weaker fallback profiles.
    static_backup_allowed = layout_profile.get("hero_card_source") != "auto_visible_cards"
    if (
        static_backup_allowed
        and layout_profile.get("hero_card_source") in {"auto_visible_cards", "fixed_visible_cards", "overlap_visible_cards", "default_roi"}
        and card_sets_need_backup(candidate_sets)
    ):
        overlap_cards = recognize_cards_from_rois(
            frame,
            HERO_CARD_OVERLAP_ROIS,
            source="hero",
            roi_mode="locked_layout_overlap",
            allow_partial=True,
        )
        if overlap_cards:
            candidate_sets.append(overlap_cards)
    if (
        static_backup_allowed
        and layout_profile.get("hero_card_source") in {"auto_visible_cards", "fixed_visible_cards", "overlap_visible_cards", "default_roi"}
        and card_sets_need_backup(candidate_sets)
    ):
        fixed_cards = []
        for index, roi in enumerate(HERO_CARD_ROIS):
            detail = recognize_best_hero_card_roi(frame, roi, index)
            if detail:
                detail["roi_mode"] = "locked_layout_fixed_refresh"
                fixed_cards.append(detail)
        if fixed_cards:
            candidate_sets.append(fixed_cards)
    if candidate_sets:
        merged_cards = merge_best_hero_cards_by_index(candidate_sets)
        if merged_cards:
            candidate_sets.append(merged_cards)
        if primary_candidate_sets:
            primary_best = max(primary_candidate_sets, key=locked_hero_card_set_score)
            if not card_set_is_incomplete(primary_best) and not card_set_needs_roi_repair(primary_best):
                return repair_low_margin_suits(primary_best, candidate_sets)
        if merged_cards and not card_set_is_incomplete(merged_cards) and not card_set_needs_roi_repair(merged_cards):
            return repair_low_margin_suits(merged_cards, candidate_sets)
        repaired = repair_low_margin_suits(max(candidate_sets, key=locked_hero_card_set_score), candidate_sets)
        return contiguous_valid_hero_cards(repaired)
    return cards


def locked_profile_hero_read_boxes(layout_profile: dict[str, Any]) -> list[dict[str, float]]:
    boxes = profile_hero_card_boxes(layout_profile)
    if len(boxes) < 2 or layout_profile.get("hero_card_source") == "manual_hero_cards":
        return boxes

    # The auto component for the overlapped right card starts inside H1. Move the
    # fixed read window to H2's visible rank corner and keep it there for the run.
    right = boxes[1]
    right["x"] += right["width"] * 0.12
    right["y"] += right["height"] * 0.03
    right["height"] *= 0.90
    right["x"] = min(max(0.0, right["x"]), 1.0 - right["width"])
    right["y"] = min(max(0.0, right["y"]), 1.0 - right["height"])
    return boxes


def profile_hero_card_boxes(layout_profile: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "x": float(box["x"]),
            "y": float(box["y"]),
            "width": float(box["width"]),
            "height": float(box["height"]),
        }
        for box in (layout_profile.get("hero_card_boxes") or [])[:2]
    ]


def contiguous_valid_hero_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {
        int(detail.get("index") or 0): detail
        for detail in cards
        if not detail_needs_roi_repair(detail)
    }
    output = []
    for index in range(2):
        detail = by_index.get(index)
        if detail is None:
            break
        output.append(detail)
    return output


def card_sets_need_backup(candidate_sets: list[list[dict[str, Any]]]) -> bool:
    if not candidate_sets:
        return True
    best = max(candidate_sets, key=locked_hero_card_set_score)
    if len(best) < 2:
        return True
    return any(
        "?" in str(detail.get("card") or "")
        or detail_needs_suit_repair(detail)
        or detail_needs_roi_repair(detail)
        for detail in best[:2]
    )


def card_set_is_incomplete(cards: list[dict[str, Any]]) -> bool:
    if len(cards) < 2:
        return True
    return any("?" in str(detail.get("card") or "") for detail in cards[:2])


def card_set_needs_roi_repair(cards: list[dict[str, Any]]) -> bool:
    return any(detail_needs_roi_repair(detail) for detail in cards[:2])


def merge_best_hero_cards_by_index(candidate_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for candidate_set in candidate_sets:
        for detail in candidate_set:
            by_index.setdefault(int(detail.get("index") or 0), []).append(detail)
    merged = []
    for index in sorted(by_index):
        best = max(by_index[index], key=hero_card_merge_score)
        best = dict(best)
        best["roi_mode"] = f"{best.get('roi_mode') or 'hero'}_merged_best"
        merged.append(best)
    return remove_duplicate_hero_card_reads(merged[:2])


def select_locked_hero_variants(
    shifted_details: list[dict[str, Any]],
    raw_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shifted = {int(item.get("index") or 0): item for item in shifted_details}
    raw = {int(item.get("index") or 0): item for item in raw_details}
    selected = []
    for slot in sorted(set(shifted) | set(raw)):
        shifted_item = shifted.get(slot)
        raw_item = raw.get(slot)
        shifted_complete = card_detail_is_complete(shifted_item)
        raw_complete = card_detail_is_complete(raw_item)
        shifted_clean = str((shifted_item or {}).get("rank_source") or "") == "clean_corner"
        raw_clean = str((raw_item or {}).get("rank_source") or "") == "clean_corner"
        if raw_complete and raw_clean and not shifted_clean:
            selected.append(raw_item)
        elif shifted_complete and shifted_clean and not raw_clean:
            selected.append(shifted_item)
        elif shifted_complete and raw_complete and shifted_clean and raw_clean:
            selected.append(
                max(
                    (shifted_item, raw_item),
                    key=lambda item: (
                        float(item.get("rank_margin") or 0.0),
                        float(item.get("rank_confidence") or 0.0),
                    ),
                )
            )
        elif shifted_complete and raw_complete:
            selected.append(
                max(
                    (shifted_item, raw_item),
                    key=lambda item: (
                        float(item.get("rank_confidence") or 0.0),
                        min(0.20, float(item.get("rank_margin") or 0.0)),
                    ),
                )
            )
        elif shifted_complete:
            selected.append(shifted_item)
        elif raw_complete:
            selected.append(raw_item)
        elif shifted_item is not None and raw_item is not None:
            # When both reads only lack a suit, do not blindly prefer the
            # shifted H2 crop. A shifted crop can remove the top-left of a 7
            # and trigger the black-K hint with an artificially large margin.
            selected.append(
                max(
                    (shifted_item, raw_item),
                    key=lambda item: (
                        float(item.get("rank_confidence") or 0.0),
                        min(0.20, float(item.get("rank_margin") or 0.0)),
                    ),
                )
            )
        elif shifted_item is not None:
            selected.append(shifted_item)
        elif raw_item is not None:
            selected.append(raw_item)
    return selected


def card_detail_is_complete(detail: dict[str, Any] | None) -> bool:
    card = str((detail or {}).get("card") or "")
    return bool(card) and "?" not in card


def remove_duplicate_hero_card_reads(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for detail in sorted(cards, key=hero_card_merge_score, reverse=True):
        card = str(detail.get("card") or "")
        if card and "?" not in card and any(str(item.get("card") or "") == card for item in deduped):
            continue
        deduped.append(detail)
    return sorted(deduped, key=lambda item: int(item.get("index") or 0))[:2]


def hero_card_merge_score(detail: dict[str, Any]) -> float:
    score = hero_card_detail_score(detail)
    card = str(detail.get("card") or "")
    roi_mode = str(detail.get("roi_mode") or "")
    complete = bool(card) and "?" not in card
    if complete and ("locked_layout_search" in roi_mode or "auto_validated" in roi_mode):
        score += 0.55
    if complete and roi_mode == "locked_layout":
        score += 0.25
    if "fixed_refresh" in roi_mode or "overlap" in roi_mode:
        score -= 0.12
    return score


def detail_needs_suit_repair(detail: dict[str, Any]) -> bool:
    if str(detail.get("color") or "") != "black":
        return False
    suit = str(detail.get("suit") or "?")
    margin = float(detail.get("suit_margin") or 0.0)
    score = float(detail.get("suit_confidence") or 0.0)
    return suit == "?" or (score >= 0.78 and margin < 0.04)


def detail_needs_roi_repair(detail: dict[str, Any]) -> bool:
    face_cover = float(detail.get("face_cover") or 0.0)
    face_fill = float(detail.get("face_fill") or 0.0)
    aspect = float(detail.get("face_aspect") or 0.0)
    white_ratio = float(detail.get("white_ratio") or 0.0)
    roi_mode = str(detail.get("roi_mode") or "")
    if face_cover <= 0.0:
        return False
    if face_cover < 0.30:
        return True
    if face_cover < 0.36 and aspect and aspect < 0.80:
        return True
    if ("overlap" in roi_mode or "partial" in roi_mode) and face_cover < 0.40 and aspect and aspect < 0.90:
        return True
    if white_ratio < 0.32 and face_cover < 0.38:
        return True
    if face_fill and face_fill < 0.58 and face_cover < 0.42:
        return True
    return False


def repair_low_margin_suits(
    cards: list[dict[str, Any]],
    candidate_sets: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    repaired = [dict(detail) for detail in cards]
    by_index: dict[int, list[dict[str, Any]]] = {}
    for candidate_set in candidate_sets:
        for detail in candidate_set:
            by_index.setdefault(int(detail.get("index") or 0), []).append(detail)
    for position, detail in enumerate(repaired):
        if not detail_needs_suit_repair(detail):
            continue
        index = int(detail.get("index") or position)
        current_margin = float(detail.get("suit_margin") or 0.0)
        alternatives = [
            item
            for item in by_index.get(index, [])
            if str(item.get("suit") or "?") in {"s", "c"}
            and float(item.get("suit_confidence") or 0.0) >= 0.78
            and float(item.get("suit_margin") or 0.0) >= current_margin + 0.018
        ]
        if not alternatives:
            continue
        best = max(
            alternatives,
            key=lambda item: (float(item.get("suit_margin") or 0.0), float(item.get("suit_confidence") or 0.0)),
        )
        new_suit = str(best.get("suit"))
        if new_suit == str(detail.get("suit") or "?"):
            continue
        detail["suit"] = new_suit
        detail["card"] = f"{detail.get('rank')}{new_suit}"
        detail["suit_confidence"] = best.get("suit_confidence")
        detail["suit_margin"] = best.get("suit_margin")
        detail["roi_mode"] = f"{detail.get('roi_mode') or 'hero'}_suit_repaired"
        detail["suit_repair_from"] = best.get("roi_mode")
    return repaired


def filter_cards_to_relative_box(
    cards: list[dict[str, Any]],
    rel_box: dict[str, Any],
    shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    height, width = shape[:2]
    absolute = absolute_box_from_relative(rel_box, width, height)
    filtered = []
    for detail in cards:
        box = detail.get("roi_box") or {}
        if box_center_inside(box, absolute) or boxes_overlap(box, absolute, padding=8):
            filtered.append(dict(detail))
    return filtered


def box_center_inside(box: dict[str, Any], container: dict[str, Any]) -> bool:
    if not box or not container:
        return False
    center_x = float(box.get("x", 0)) + float(box.get("width", 0)) / 2
    center_y = float(box.get("y", 0)) + float(box.get("height", 0)) / 2
    return (
        float(container.get("x", 0)) <= center_x <= float(container.get("x", 0)) + float(container.get("width", 0))
        and float(container.get("y", 0)) <= center_y <= float(container.get("y", 0)) + float(container.get("height", 0))
    )


def locked_hero_card_set_score(cards: list[dict[str, Any]]) -> float:
    complete_cards = sum(1 for detail in cards if "?" not in str(detail.get("card") or ""))
    return card_read_score(cards) + len(cards) * 0.35 + complete_cards * 0.30


def recognize_cards_from_relative_boxes(
    frame: Any,
    boxes: list[dict[str, Any]],
    source: str,
    *,
    roi_mode: str = "locked_layout",
) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    cards = []
    for index, rel_box in enumerate(boxes):
        box = absolute_box_from_relative(rel_box, width, height)
        crop_info = make_crop_info(frame, box["x"], box["y"], box["width"], box["height"])
        detail = recognize_card_crop(crop_info["crop"], source=source, index=index)
        if not detail:
            continue
        detail["roi_mode"] = roi_mode
        detail["roi_box"] = crop_info["box"]
        cards.append(detail)
    return cards


def recognize_cards_from_rois(
    frame: Any,
    rois: tuple[tuple[float, float, float, float], ...],
    *,
    source: str,
    roi_mode: str,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    cards = []
    for index, roi in enumerate(rois):
        detail = recognize_best_hero_card_roi(frame, roi, index, allow_partial=allow_partial)
        if not detail:
            continue
        detail["roi_mode"] = roi_mode
        cards.append(detail)
    return cards


def build_layout_profile(
    frame: Any,
    ocr_result: list[Any] | None = None,
    hero_name: str | None = None,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    hero_details = detect_auto_hero_cards(frame)
    hero_name_anchor = find_hero_name_anchor(ocr_result or [], hero_name)
    hero_card_source = "auto_visible_cards"
    if len(hero_details) < 2:
        fixed_details = []
        for index, roi in enumerate(HERO_CARD_ROIS):
            detail = recognize_best_hero_card_roi(frame, roi, index)
            if detail:
                fixed_details.append(detail)
        if len(fixed_details) >= 2:
            hero_details = fixed_details
            hero_card_source = "fixed_visible_cards"
    if len(hero_details) >= 2:
        hero_card_boxes = [relative_box_from_absolute(detail["roi_box"], width, height) for detail in hero_details[:2]]
        hero_search_box = expanded_relative_box_from_absolute_boxes(
            [detail["roi_box"] for detail in hero_details[:2]],
            width,
            height,
        )
        hero_search_source = "auto_visible_cards"
    else:
        hero_card_boxes = [
            relative_box_from_roi(roi)
            for roi in HERO_CARD_ROIS
        ]
        hero_card_source = "default_roi"
        hero_search_box = hero_search_box_from_name_anchor(hero_name_anchor, width, height)
        hero_search_source = "hero_name_anchor" if hero_search_box else "default"
        if hero_search_box is None:
            hero_search_box = {"x": 0.30, "y": 0.55, "width": 0.40, "height": 0.41}
    board_card_boxes = [relative_box_from_roi(roi) for roi in BOARD_CARD_ROIS]
    profile = {
        "id": f"locked-{width}x{height}",
        "method": "hero-name-anchor" if hero_name_anchor else hero_card_source,
        "strict": True,
        "frame_size": {"width": width, "height": height},
        "hero_name": hero_name,
        "hero_name_anchor": hero_name_anchor,
        "hero_card_source": hero_card_source,
        "hero_search_source": hero_search_source,
        "hero_search_box": hero_search_box,
        "hero_card_boxes": hero_card_boxes,
        "board_card_boxes": board_card_boxes,
        "seat_card_boxes": {
            str(index): relative_box_from_roi(roi)
            for index, roi in CARD_ROIS_8.items()
        },
        "seat_stack_boxes": {
            str(index): relative_box_from_roi(roi)
            for index, roi in STACK_ROIS_8.items()
        },
        "bet_anchors": {
            str(index): {"x": float(anchor[0]), "y": float(anchor[1])}
            for index, anchor in BET_ANCHORS_8.items()
        },
        "pot_search_box": {"x": 0.28, "y": 0.23, "width": 0.44, "height": 0.20},
        "action_controls_search_box": {"x": 0.42, "y": 0.84, "width": 0.58, "height": 0.16},
        "created_from": {
            "hero_cards": [detail.get("card") for detail in hero_details],
            "hero_card_count": len(hero_details),
        },
    }
    return profile


def layout_profile_quality(profile: dict[str, Any] | None) -> int:
    if not profile:
        return 0
    if profile.get("hero_card_source") == "manual_hero_cards" and len(profile.get("hero_card_boxes") or []) == 2:
        return 4
    created_from = profile.get("created_from") or {}
    hero_card_count = int(created_from.get("hero_card_count") or 0)
    if profile.get("hero_card_source") in {"auto_visible_cards", "fixed_visible_cards"} and hero_card_count >= 2:
        return 3
    if profile.get("hero_search_source") == "hero_name_anchor" and profile.get("hero_name_anchor"):
        return 2
    if profile.get("hero_card_source") == "auto_visible_cards" or hero_card_count > 0:
        return 1
    return 0


def layout_profile_is_strong(profile: dict[str, Any] | None) -> bool:
    return layout_profile_quality(profile) >= 2


def expanded_relative_box_from_absolute_boxes(boxes: list[dict[str, Any]], width: int, height: int) -> dict[str, float]:
    x1 = min(int(box["x"]) for box in boxes)
    y1 = min(int(box["y"]) for box in boxes)
    x2 = max(int(box["x"]) + int(box["width"]) for box in boxes)
    y2 = max(int(box["y"]) + int(box["height"]) for box in boxes)
    union_w = max(1, x2 - x1)
    union_h = max(1, y2 - y1)
    pad_x = max(24, int(round(union_w * 0.45)))
    pad_top = max(18, int(round(union_h * 0.22)))
    pad_bottom = max(22, int(round(union_h * 0.28)))
    expanded = {
        "x": max(0, x1 - pad_x),
        "y": max(0, y1 - pad_top),
        "width": min(width, x2 + pad_x) - max(0, x1 - pad_x),
        "height": min(height, y2 + pad_bottom) - max(0, y1 - pad_top),
    }
    return relative_box_from_absolute(expanded, width, height)


def hero_search_box_from_name_anchor(anchor: dict[str, Any] | None, width: int, height: int) -> dict[str, float] | None:
    if not anchor:
        return None
    box = anchor.get("box") or {}
    try:
        anchor_x = int(box["x"])
        anchor_y = int(box["y"])
        anchor_w = int(box["width"])
        anchor_h = int(box["height"])
    except (KeyError, TypeError, ValueError):
        return None
    center_x = anchor_x + anchor_w / 2
    search_w = max(int(width * 0.18), anchor_w * 3)
    search_h = max(int(height * 0.18), anchor_h * 5)
    search_x = int(round(center_x - search_w / 2))
    search_y = int(round(anchor_y - search_h * 0.95))
    absolute = {
        "x": max(0, search_x),
        "y": max(0, search_y),
        "width": min(width, search_x + search_w) - max(0, search_x),
        "height": min(height, search_y + search_h) - max(0, search_y),
    }
    if absolute["width"] <= 0 or absolute["height"] <= 0:
        return None
    return relative_box_from_absolute(absolute, width, height)


def find_hero_name_anchor(ocr_result: list[Any], hero_name: str | None) -> dict[str, Any] | None:
    if not ocr_result or not hero_name:
        return None
    target = compact_anchor_text(hero_name)
    if not target:
        return None
    candidates = []
    for box, raw_text, raw_conf in ocr_result:
        text = normalize_ocr_text(str(raw_text))
        compact = compact_anchor_text(text)
        if not compact:
            continue
        matched = target in compact or compact in target
        if not matched:
            continue
        text_box = ocr_box_bounds(box)
        candidates.append(
            {
                "text": text,
                "confidence": round(float(raw_conf), 4),
                "box": text_box,
                "_score": float(raw_conf) + (0.25 if target == compact else 0.0),
            }
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["_score"])
    best.pop("_score", None)
    return best


def compact_anchor_text(text: str) -> str:
    return re.sub(r"[\s:：·.,，。_\-]+", "", text).lower()


def relative_box_from_roi(roi: tuple[float, float, float, float]) -> dict[str, float]:
    x1, y1, x2, y2 = roi
    return {
        "x": float(x1),
        "y": float(y1),
        "width": float(x2 - x1),
        "height": float(y2 - y1),
    }


def relative_box_from_absolute(box: dict[str, Any], width: int, height: int) -> dict[str, float]:
    return {
        "x": float(box["x"]) / max(width, 1),
        "y": float(box["y"]) / max(height, 1),
        "width": float(box["width"]) / max(width, 1),
        "height": float(box["height"]) / max(height, 1),
    }


def absolute_box_from_relative(box: dict[str, Any], width: int, height: int) -> dict[str, int]:
    return {
        "x": int(round(float(box["x"]) * width)),
        "y": int(round(float(box["y"]) * height)),
        "width": max(1, int(round(float(box["width"]) * width))),
        "height": max(1, int(round(float(box["height"]) * height))),
    }


def recognize_best_hero_card_roi(
    frame: Any,
    roi: tuple[float, float, float, float],
    index: int,
    *,
    allow_partial: bool = False,
) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = scale_roi(roi, width, height)
    base_width = max(1, x2 - x1)
    base_height = max(1, y2 - y1)
    candidates: list[dict[str, Any]] = []
    seen_boxes = set()
    for dx, dy, width_scale, height_scale in hero_card_roi_variants_for_index(index):
        crop_x = int(round(x1 + base_width * dx))
        crop_y = int(round(y1 + base_height * dy))
        crop_width = int(round(base_width * width_scale))
        crop_height = int(round(base_height * height_scale))
        crop_info = make_crop_info(frame, crop_x, crop_y, crop_width, crop_height)
        box = crop_info["box"]
        key = (box["x"], box["y"], box["width"], box["height"])
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        detail = recognize_card_crop(
            crop_info["crop"],
            source="hero",
            index=index,
            allow_partial_hero=allow_partial,
        )
        if not detail:
            continue
        if allow_partial:
            detail["roi_mode"] = "partial_hero_variant" if key != (x1, y1, base_width, base_height) else "partial_hero"
        else:
            detail["roi_mode"] = "fixed_hero_variant" if key != (x1, y1, base_width, base_height) else "fixed_hero"
        detail["roi_box"] = box
        if weak_hero_roi_candidate(detail):
            continue
        candidates.append(detail)
    if not candidates:
        return None
    return select_consensus_hero_card_candidate(candidates)


def hero_card_roi_variants_for_index(index: int) -> tuple[tuple[float, float, float, float], ...]:
    if int(index) == 0:
        return HERO_CARD_ROI_VARIANTS_SLOT0
    if int(index) == 1:
        return HERO_CARD_ROI_VARIANTS_SLOT1
    return HERO_CARD_ROI_VARIANTS


def select_consensus_hero_card_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detail in candidates:
        grouped.setdefault(str(detail.get("card") or ""), []).append(detail)

    def group_score(items: list[dict[str, Any]]) -> float:
        best_score = max(hero_card_detail_score(item) for item in items)
        support = min(len(items), 4)
        complete_bonus = 0.08 if all("?" not in str(item.get("card") or "") for item in items) else 0.0
        return best_score + max(0, support - 1) * 0.14 + complete_bonus

    best_group = max(grouped.values(), key=group_score)
    best = max(best_group, key=hero_card_detail_score)
    if len(best_group) > 1:
        best = dict(best)
        best["variant_support"] = len(best_group)
    return best


def weak_hero_roi_candidate(detail: dict[str, Any]) -> bool:
    white_ratio = float(detail.get("white_ratio") or 0.0)
    rank_confidence = float(detail.get("rank_confidence") or 0.0)
    suit_confidence = float(detail.get("suit_confidence") or 0.0)
    roi_mode = str(detail.get("roi_mode") or "")
    if roi_mode.startswith("fixed_hero") and white_ratio < 0.43:
        return True
    if white_ratio < 0.43 and rank_confidence < 0.60 and suit_confidence < 0.70:
        return True
    if white_ratio < 0.30 and rank_confidence < 0.60:
        return True
    if "?" in str(detail.get("card") or "") and white_ratio < 0.34 and suit_confidence < 0.60:
        return True
    return False


def merge_card_details_by_index(base_cards: list[dict[str, Any]], extra_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for detail in [*base_cards, *extra_cards]:
        index = int(detail.get("index") or 0)
        current = merged.get(index)
        if current is None or board_card_merge_score(detail) > board_card_merge_score(current):
            merged[index] = detail
    return [merged[index] for index in sorted(merged)]


def board_card_merge_score(detail: dict[str, Any]) -> float:
    score = card_read_score([detail])
    score += min(float(detail.get("rank_margin") or 0.0), 1.0) * 0.18
    score += min(float(detail.get("suit_margin") or 0.0), 1.0) * 0.28
    if "?" in str(detail.get("card") or ""):
        score -= 0.50
    roi_mode = str(detail.get("roi_mode") or "")
    if "auto_board_component" in roi_mode:
        score += 0.10
    rank_margin = float(detail.get("rank_margin") or 0.0)
    suit_margin = float(detail.get("suit_margin") or 0.0)
    if rank_margin < 0.018:
        score -= 0.28
    if suit_margin < 0.045:
        score -= 0.10
    elif suit_margin < 0.080:
        score -= 0.03
    return score


def detect_auto_board_cards(frame: Any) -> list[dict[str, Any]]:
    crops = detect_auto_board_card_crops(frame)
    cards = []
    for index, crop_info in enumerate(crops[:5]):
        detail = recognize_card_crop(crop_info["crop"], source="board", index=index)
        if detail:
            detail["roi_mode"] = "auto_board_component"
            detail["roi_box"] = crop_info["box"]
            cards.append(detail)
    return cards


def detect_auto_board_card_crops(frame: Any) -> list[dict[str, Any]]:
    cv2, np = load_cv()
    height, width = frame.shape[:2]
    search_x1 = int(width * 0.20)
    search_x2 = int(width * 0.80)
    search_y1 = int(height * 0.26)
    search_y2 = int(height * 0.64)
    search = frame[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return []
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 1] < 78) & (hsv[:, :, 2] > 150)).astype("uint8")) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(1200, width * height * 0.00065):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        abs_x = x + search_x1
        abs_y = y + search_y1
        if w < width * 0.040 or w > width * 0.105 or h < height * 0.085 or h > height * 0.20:
            continue
        center_y = abs_y + h / 2
        if center_y < height * 0.32 or center_y > height * 0.60:
            continue
        candidates.append((abs_x, abs_y, w, h, area))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    merged: list[tuple[int, int, int, int, float]] = []
    for candidate in candidates:
        x, y, w, h, area = candidate
        if merged and abs(x - merged[-1][0]) < width * 0.035:
            if area > merged[-1][4]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return [
        make_crop_info(frame, int(x - 5), int(y - 5), int(w + 10), int(h + 10))
        for x, y, w, h, _area in merged[:5]
    ]


def detect_auto_hero_cards(frame: Any) -> list[dict[str, Any]]:
    crops = detect_auto_hero_card_crops(frame)
    cards = []
    for index, crop_info in enumerate(crops[:2]):
        detail = recognize_card_crop(crop_info["crop"], source="hero", index=index)
        if detail:
            detail["roi_mode"] = "auto_hero_component"
            detail["roi_box"] = crop_info["box"]
            if weak_hero_roi_candidate(detail):
                continue
            cards.append(detail)
    return cards


def detect_auto_hero_card_crops(frame: Any) -> list[dict[str, Any]]:
    cv2, np = load_cv()
    height, width = frame.shape[:2]
    search_x1 = int(width * 0.25)
    search_x2 = int(width * 0.75)
    search_y1 = int(height * 0.54)
    search_y2 = int(height * 0.93)
    search = frame[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return []

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright_cards = (sat < 78) & (val > 145)

    def collect_candidates(mask_bool: Any) -> list[tuple[float, int, int, int, int, float]]:
        mask = (mask_bool.astype("uint8")) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(1800, width * height * 0.0012):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            abs_x = x + search_x1
            abs_y = y + search_y1
            center_x = abs_x + w / 2
            center_y = abs_y + h / 2
            if center_x < width * 0.34 or center_x > width * 0.66:
                continue
            if center_y < height * 0.60 or center_y > height * 0.90:
                continue
            if w < width * 0.07 or w > width * 0.22 or h < height * 0.055 or h > height * 0.24:
                continue
            center_penalty = abs(center_x - width * 0.50) / max(width, 1)
            y_bonus = max(0.0, center_y - height * 0.62) / max(height, 1)
            score = area / max(width * height, 1) * 100 - center_penalty * 8 + y_bonus * 2
            found.append((score, abs_x, abs_y, w, h, area))
        return found

    candidates = collect_candidates(bright_cards)
    if not candidates:
        adaptive_floor = max(85.0, float(np.percentile(val, 70)) - 8.0)
        adaptive_cards = (sat < 115) & (val > adaptive_floor) & (val > float(np.percentile(val, 50)) + 10.0)
        candidates = collect_candidates(adaptive_cards)
    if not candidates:
        return []

    _score, x, y, w, h, _area = max(candidates, key=lambda item: item[0])
    crop_h = int(max(105, min(height * 0.17, h * 1.25)))
    crop_w0 = int(max(58, min(width * 0.065, w * 0.43)))
    crop_w1 = int(max(84, min(width * 0.095, w * 0.60)))
    slot0_x = int(x)
    slot1_x = int(x + w * 0.30)
    crop_y = int(max(0, y - h * 0.02))
    return [
        make_crop_info(frame, slot0_x, crop_y, crop_w0, crop_h),
        make_crop_info(frame, slot1_x, crop_y, crop_w1, crop_h),
    ]


def detect_auto_hero_card_crops_in_relative_box(frame: Any, rel_box: dict[str, Any]) -> list[dict[str, Any]]:
    cv2, np = load_cv()
    height, width = frame.shape[:2]
    box = absolute_box_from_relative(rel_box, width, height)
    search_x1 = max(0, min(width - 1, int(box["x"])))
    search_y1 = max(0, min(height - 1, int(box["y"])))
    search_x2 = max(search_x1 + 1, min(width, search_x1 + int(box["width"])))
    search_y2 = max(search_y1 + 1, min(height, search_y1 + int(box["height"])))
    search = frame[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return []

    search_h, search_w = search.shape[:2]
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask_candidates = [
        (sat < 78) & (val > 145),
        (sat < 120) & (val > 110),
    ]
    for mask_bool in mask_candidates:
        best: tuple[float, int, int, int, int, float] | None = None
        mask = (mask_bool.astype("uint8")) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(900, search_w * search_h * 0.015):
                continue
            x, y, crop_w, crop_h = cv2.boundingRect(contour)
            center_x = x + crop_w / 2
            center_y = y + crop_h / 2
            if crop_w < search_w * 0.12 or crop_w > search_w * 0.55:
                continue
            if crop_h < search_h * 0.18 or crop_h > search_h * 0.75:
                continue
            score = area - abs(center_x - search_w * 0.5) * 10 - abs(center_y - search_h * 0.55) * 4
            candidate = (float(score), int(x), int(y), int(crop_w), int(crop_h), float(area))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            break
    else:
        return []

    _score, x, y, crop_w, crop_h, _area = best
    abs_x = search_x1 + x
    abs_y = search_y1 + y
    card_h = int(max(105, min(height * 0.17, crop_h * 1.25)))
    card_w0 = int(max(58, min(width * 0.065, crop_w * 0.43)))
    card_w1 = int(max(84, min(width * 0.095, crop_w * 0.60)))
    card_y = int(max(0, abs_y - crop_h * 0.02))
    return [
        make_crop_info(frame, abs_x, card_y, card_w0, card_h),
        make_crop_info(frame, int(abs_x + crop_w * 0.30), card_y, card_w1, card_h),
    ]


def make_crop_info(frame: Any, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, int(x)))
    y1 = max(0, min(frame_h - 1, int(y)))
    x2 = max(x1 + 1, min(frame_w, x1 + int(width)))
    y2 = max(y1 + 1, min(frame_h, y1 + int(height)))
    return {
        "box": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
        "crop": frame[y1:y2, x1:x2],
    }


def card_read_score(details: list[dict[str, Any]]) -> float:
    score = 0.0
    for detail in details:
        card = str(detail.get("card") or "")
        if len(card) >= 1:
            score += 1.0
        if len(card) >= 2 and "?" not in card:
            score += 1.0
        score += min(float(detail.get("rank_confidence") or 0.0), 1.0) * 0.2
        score += min(float(detail.get("suit_confidence") or 0.0), 1.0) * 0.1
    return score


def hero_card_detail_score(detail: dict[str, Any]) -> float:
    score = card_read_score([detail])
    if "?" in str(detail.get("card") or ""):
        score -= 0.45
    score += min(float(detail.get("rank_margin") or 0.0), 1.0) * 0.25
    score += min(float(detail.get("suit_margin") or 0.0), 1.0) * 0.20
    score += min(float(detail.get("white_ratio") or 0.0), 1.0) * 0.05
    score += hero_card_roi_quality_score(detail)
    return score


def hero_card_roi_quality_score(detail: dict[str, Any]) -> float:
    face_cover = float(detail.get("face_cover") or 0.0)
    face_fill = float(detail.get("face_fill") or 0.0)
    aspect = float(detail.get("face_aspect") or 0.0)
    white_ratio = float(detail.get("white_ratio") or 0.0)
    score = 0.0
    if face_cover >= 0.48:
        score += min(0.30, (face_cover - 0.48) * 0.70 + 0.16)
    elif face_cover > 0.0 and face_cover < 0.30:
        score -= 0.55
    elif face_cover < 0.38:
        score -= 0.25
    if 0.85 <= aspect <= 2.25:
        score += 0.18
    elif 0.0 < aspect < 0.75:
        score -= 0.35
    if face_fill and face_fill < 0.60:
        score -= 0.20
    if white_ratio < 0.30 and face_cover < 0.38:
        score -= 0.20
    return score


def recognize_card_crop(
    crop: Any,
    source: str,
    index: int,
    *,
    allow_partial_hero: bool = False,
    return_rejected: bool = False,
) -> dict[str, Any] | None:
    white_ratio = card_white_ratio(crop)
    prepared_crop = crop
    if white_ratio < 0.22:
        enhanced_crop = enhance_dim_card_crop(crop)
        enhanced_white_ratio = card_white_ratio(enhanced_crop)
        if enhanced_white_ratio >= max(0.20, white_ratio + 0.10):
            prepared_crop = enhanced_crop
            white_ratio = enhanced_white_ratio
    if white_ratio < 0.22:
        return None
    if source == "board" and white_ratio < 0.48:
        return None
    face_metrics = card_face_rect_metrics(prepared_crop)
    if source == "hero" and not hero_card_face_is_plausible(face_metrics):
        if not allow_partial_hero or not partial_hero_card_face_is_plausible(face_metrics):
            return None
        if white_ratio < 0.24:
            tightened_crop = tighten_hero_card_face_crop(prepared_crop)
            tightened_metrics = card_face_rect_metrics(tightened_crop)
            if hero_card_face_is_plausible(tightened_metrics) or partial_hero_card_face_is_plausible(tightened_metrics):
                prepared_crop = tightened_crop
                white_ratio = card_white_ratio(prepared_crop)
                face_metrics = tightened_metrics
    if source == "hero" and allow_partial_hero and white_ratio < 0.24:
        return None
    templates = load_card_templates()
    if not templates["ranks"]:
        return None
    rank_score, rank, next_rank_score = recognize_card_rank(prepared_crop, templates, source, index)
    rank_margin = rank_score - next_rank_score
    if allow_partial_hero:
        min_hero_rank_margin = 0.045 if rank_score >= 0.56 else 0.07
    else:
        min_hero_rank_margin = 0.055 if rank_score >= 0.56 else 0.08
    # Clear red kings in the second hero slot consistently land just below the
    # generic ambiguity gate. The K shape still wins across several windows,
    # so keep that narrow case instead of erasing it to an unknown card.
    if (
        source == "hero"
        and rank == "K"
        and rank_score >= 0.63
        and card_glyph_color(prepared_crop) == "red"
    ):
        min_hero_rank_margin = min(min_hero_rank_margin, 0.03)
    suit, suit_score, suit_margin, color = recognize_card_suit(prepared_crop, templates, source)
    code_suit = suit if suit else "?"
    rank_code = "T" if rank == "T" else rank
    rank_rejected = rank_score < 0.36 or (source == "hero" and rank_margin < min_hero_rank_margin)
    if rank_rejected and not return_rejected:
        return None
    detail = {
        "card": "??" if rank_rejected else f"{rank_code}{code_suit}",
        "rank": rank_code,
        "suit": code_suit,
        "source": source,
        "index": index,
        "white_ratio": round(float(white_ratio), 4),
        "face_fill": round(float(face_metrics.get("fill") or 0.0), 4),
        "face_cover": round(float(face_metrics.get("cover") or 0.0), 4),
        "face_aspect": round(float(face_metrics.get("aspect") or 0.0), 4),
        "rank_confidence": round(float(rank_score), 4),
        "rank_margin": round(float(rank_score - next_rank_score), 4),
        "suit_confidence": round(float(suit_score), 4) if suit_score is not None else None,
        "suit_margin": round(float(suit_margin), 4) if suit_margin is not None else None,
        "color": color,
    }
    clean_prediction = clean_rank_prediction(prepared_crop, source)
    if (
        clean_prediction_is_decisive(clean_prediction, source=source)
        and str((clean_prediction or {}).get("label") or "") == rank_code
    ):
        detail["rank_source"] = "clean_corner"
        detail["rank_clean_confidence"] = round(float(clean_prediction["score"]), 4)
        detail["rank_clean_margin"] = round(float(clean_prediction["margin"]), 4)
    if rank_rejected:
        detail["rejected"] = True
        detail["reject_reason"] = "rank_score" if rank_score < 0.36 else "rank_margin"
    return detail


def recognize_card_rank(
    crop: Any,
    templates: dict[str, dict[str, Any]],
    source: str,
    index: int,
) -> tuple[float, str, float]:
    cache_key = card_recognition_cache_key(crop, kind="rank", source=source, index=index)
    cached = _CARD_RANK_RECOGNITION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    best: tuple[float, str, float, float] | None = None
    seen_rank_images: set[bytes] = set()
    deep_votes: list[dict[str, Any]] = []
    alternatives: list[tuple[str, float, float]] = []
    aligned_classifier_votes: list[tuple[str, float, float]] = []
    glyph_color = card_glyph_color(crop)
    for x_offset, y_offset, rank_width, rank_height in rank_candidate_windows(source, index=index, crop_width=crop.shape[1]):
        if y_offset >= crop.shape[0] or x_offset >= crop.shape[1]:
            continue
        rank_roi = crop[
            y_offset : min(crop.shape[0], y_offset + rank_height),
            x_offset : min(crop.shape[1], x_offset + rank_width),
        ]
        if rank_roi.size == 0 or rank_roi.shape[0] < 24 or rank_roi.shape[1] < 20:
            continue
        # Keep the promoted runtime model on its original normalization until
        # the clean-rank dataset is labeled, retrained, and regression-gated.
        rank_image = normalized_card_piece(rank_roi, (54, 70))
        rank_image_key = rank_image.tobytes()
        if rank_image_key in seen_rank_images:
            continue
        seen_rank_images.add(rank_image_key)
        rank_scores = sorted(
            (
                (best_template_score(rank_image, images), rank)
                for rank, images in templates["ranks"].items()
            ),
            reverse=True,
        )
        if not rank_scores:
            continue
        rank_score, rank = rank_scores[0]
        next_rank_score = rank_scores[1][0] if len(rank_scores) > 1 else -1.0
        template_scores = {label: score for score, label in rank_scores}
        k_hint = (
            source == "hero"
            and index == 1
            and glyph_color == "black"
            and float(template_scores.get("K") or -1.0) >= rank_score - 0.015
            and float(template_scores.get("K") or -1.0) >= 0.27
            and rank_score < 0.40
        )
        classifier = classify_rank_glyph(rank_image)
        if classifier is not None:
            classifier_rank = str(classifier["label"])
            classifier_score = float(classifier["score"])
            classifier_margin = float(classifier["margin"])
            classifier_second_score = float(classifier.get("second_score", -1.0))
            alternatives.append((classifier_rank, classifier_score, float(classifier.get("second_score", -1.0))))
            if source == "hero" and y_offset == 0:
                aligned_classifier_votes.append(
                    (classifier_rank, classifier_score, classifier_second_score)
                )
            if classifier_rank == rank:
                previous_score = rank_score
                if classifier_margin > 1e-6:
                    rank_score = max(rank_score, classifier_score)
                    if classifier_score > previous_score:
                        next_rank_score = max(next_rank_score, min(rank_score, classifier_second_score))
                    selector_bonus = max(0.0, classifier_margin) * 0.20 + 0.04
                else:
                    selector_bonus = 0.0
            elif (
                classifier_score >= 0.42
                and classifier_margin >= 0.045
                and (rank_score < 0.58 or rank_score - next_rank_score < 0.12)
            ):
                next_rank_score = max(classifier_second_score, min(rank_score, classifier_score - 0.01))
                rank_score = max(classifier_score, rank_score * 0.92)
                rank = classifier_rank
                selector_bonus = max(0.0, classifier_margin) * 0.18
            elif (
                source == "hero"
                and glyph_color == "black"
                and classifier_rank == "4"
                and classifier_score >= 0.84
                and float(template_scores.get("4") or 0.0) >= 0.40
            ):
                rank = "4"
                rank_score = max(classifier_score, rank_score)
                next_rank_score = min(rank_score - 0.08, max(0.42, next_rank_score))
                selector_bonus = 0.08
            else:
                selector_bonus = 0.0
        else:
            selector_bonus = 0.0
        if rank_score < 0.78 or rank_score - next_rank_score < 0.20:
            deep_classifier = classify_deep_glyph(rank_image, "rank")
            if deep_classifier is not None:
                deep_votes.append(
                    {
                        "rank": str(deep_classifier.get("label") or ""),
                        "score": float(deep_classifier.get("score") or 0.0),
                        "margin": float(deep_classifier.get("margin") or 0.0),
                        "width": int(rank_width),
                    }
                )
                deep_rank = str(deep_classifier["label"])
                deep_score = float(deep_classifier["score"])
                deep_margin = float(deep_classifier["margin"])
                if deep_rank == rank:
                    rank_score = max(rank_score, deep_score)
                    selector_bonus += max(0.0, deep_margin) * 0.12 + 0.02
                elif (
                    deep_score >= 0.62
                    and deep_margin >= 0.12
                    and (rank_score < 0.62 or rank_score - next_rank_score < 0.12)
                ):
                    next_rank_score = max(next_rank_score, min(rank_score, deep_score - 0.01))
                    rank_score = max(deep_score, rank_score * 0.90)
                    rank = deep_rank
                    selector_bonus += max(0.0, deep_margin) * 0.10
        if k_hint and rank_score < 0.62:
            next_rank_score = max(next_rank_score, min(rank_score, float(template_scores.get("4") or -1.0)))
            rank_score = max(0.66, float(template_scores.get("K") or 0.0) + 0.34)
            rank = "K"
            selector_bonus += 0.18
        if weak_hero_rank_window(
            source=source,
            rank_width=rank_width,
            rank_score=rank_score,
            next_rank_score=next_rank_score,
        ):
            continue
        if wide_hero_rank_conflicts_with_existing(
            source=source,
            rank_width=rank_width,
            rank=rank,
            rank_score=rank_score,
            next_rank_score=next_rank_score,
            best=best,
        ):
            continue
        if source == "hero" and rank_width >= 72:
            strong_ten = rank == "T" and rank_score >= 0.70
            strong_wide_rank = rank_score >= 0.82 and rank_score - next_rank_score >= 0.30
            if not (strong_ten or strong_wide_rank):
                continue
        selector = rank_score + max(0.0, rank_score - next_rank_score) * 0.25
        selector += selector_bonus
        alternatives.append((rank, rank_score, next_rank_score))
        if best is None or selector > best[0]:
            best = (selector, rank, rank_score, next_rank_score)
        if (
            source == "hero"
            and rank_width >= 48
            and rank_score >= 0.97
            and rank_score - next_rank_score >= 0.25
            and not (glyph_color == "red" and rank == "8")
        ):
            break
        if source == "hero" and (x_offset, y_offset, rank_width, rank_height) == (0, 0, 64, 72) and best is not None:
            _best_selector, best_rank, best_score, best_next_score = best
            if not should_expand_rank_windows(
                glyph_color=glyph_color,
                rank=best_rank,
                rank_score=best_score,
                next_rank_score=best_next_score,
            ):
                break
    if best is None:
        return store_rank_recognition_cache(cache_key, (-1.0, "", -1.0))
    _selector, rank, rank_score, next_rank_score = best
    clean_rank_override = choose_clean_rank_override(
        crop,
        source=source,
        rank=rank,
        rank_score=rank_score,
        next_rank_score=next_rank_score,
    )
    if clean_rank_override is not None:
        return store_rank_recognition_cache(cache_key, clean_rank_override)
    aligned_king_override = choose_aligned_king_rank_override(
        crop,
        source=source,
        rank=rank,
        rank_score=rank_score,
        next_rank_score=next_rank_score,
        aligned_votes=aligned_classifier_votes,
    )
    if aligned_king_override is not None:
        return store_rank_recognition_cache(cache_key, aligned_king_override)
    red_five_override = choose_red_five_rank_override(crop, rank, rank_score, next_rank_score, deep_votes)
    if red_five_override is not None:
        return store_rank_recognition_cache(cache_key, red_five_override)
    red_nine_override = choose_red_nine_rank_override(crop, rank, rank_score, next_rank_score, alternatives)
    if red_nine_override is not None:
        return store_rank_recognition_cache(cache_key, red_nine_override)
    red_eight_override = choose_red_eight_rank_override(crop, rank, rank_score, next_rank_score, alternatives)
    if red_eight_override is not None:
        return store_rank_recognition_cache(cache_key, red_eight_override)
    red_four_override = choose_red_four_rank_override(crop, rank, rank_score, next_rank_score, alternatives)
    if red_four_override is not None:
        return store_rank_recognition_cache(cache_key, red_four_override)
    return store_rank_recognition_cache(cache_key, (rank_score, rank, next_rank_score))


def choose_clean_rank_override(
    crop: Any,
    *,
    source: str,
    rank: str,
    rank_score: float,
    next_rank_score: float,
) -> tuple[float, str, float] | None:
    prediction = clean_rank_prediction(crop, source)
    if not clean_prediction_is_decisive(prediction, source=source):
        return None
    clean_score = float(prediction.get("score") or 0.0)
    clean_margin = float(prediction.get("margin") or 0.0)
    clean_rank = str(prediction.get("label") or "")
    if not clean_rank:
        return None
    clean_second = float(prediction.get("second_score") or (clean_score - clean_margin))
    if clean_rank == rank and clean_score <= float(rank_score):
        return None
    return clean_score, clean_rank, clean_second


def clean_rank_prediction(crop: Any, source: str) -> dict[str, Any] | None:
    if source == "hero":
        width, height = 64, 72
        x_offsets = (0,)
        model_path = None
    elif source == "board":
        width, height = 55, 60
        # Some ranks need the original corner, while others touch the magenta
        # left card border. Score both the aligned crop and a small inset crop
        # so border removal cannot lower a previously clean prediction.
        x_offsets = (0, min(6, max(0, crop.shape[1] - 1)))
        model_path = Path(os.environ.get("GTO_CARD_BOARD_KNN_MODEL") or BOARD_RANK_MODEL_PATH)
        if not model_path.exists():
            return None
    else:
        return None
    predictions = []
    if source == "hero":
        fixed_glyph = normalized_hero_rank_window(crop, (54, 70))
        hero_rank_model_path = Path(
            os.environ.get("GTO_CARD_HERO_RANK_MODEL") or HERO_RANK_MODEL_PATH
        )
        specialist_prediction = (
            classify_rank_glyph(fixed_glyph, model_path=hero_rank_model_path)
            if fixed_glyph is not None and hero_rank_model_path.exists()
            else None
        )
        if specialist_prediction is not None:
            predictions.append({**specialist_prediction, "hero_rank_specialist": True})
        fixed_prediction = classify_rank_glyph(fixed_glyph) if fixed_glyph is not None else None
        if fixed_prediction is not None:
            predictions.append({**fixed_prediction, "fixed_rank_window": True})
    for x_offset in dict.fromkeys(x_offsets):
        corner = crop[
            0 : min(crop.shape[0], height),
            x_offset : min(crop.shape[1], x_offset + width),
        ]
        prediction = classify_rank_glyph(normalized_rank_piece(corner, (54, 70)), model_path=model_path)
        if prediction is not None:
            predictions.append(prediction)
    if not predictions:
        return None
    # For board cards, the inset corner can remove the magenta card edge.
    # Prefer any already-decisive clean read before comparing raw similarity,
    # otherwise a slightly higher but ambiguous edge-contaminated crop can
    # suppress the valid inset read and let a shifted fragment win later.
    candidates = [
        item for item in predictions if clean_prediction_is_decisive(item, source=source)
    ] or predictions
    prediction = max(
        candidates,
        key=lambda item: (float(item.get("score") or 0.0), float(item.get("margin") or 0.0)),
    )
    return {**prediction, "source": source} if prediction is not None else None


def clean_prediction_is_decisive(
    prediction: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    if prediction is None:
        return False
    score = float(prediction.get("score") or 0.0)
    margin = float(prediction.get("margin") or 0.0)
    if bool(prediction.get("hero_rank_specialist")):
        # This model is trained only on manually confirmed, fixed hero-rank
        # windows. Queens are a specific generic-model failure mode: their
        # open tail is often scored as an 8 by the template path. On the
        # confirmed hero set, the specialist's Q votes are unambiguous, so
        # retain that well-separated Q read below the generic 0.92 gate.
        if str(prediction.get("label") or "") == "Q":
            return score >= 0.84 and margin >= 0.15
        # Keep the specialist as a high-confidence override for all other
        # labels, where we have less coverage than for Q versus 8.
        return score >= 0.92 and margin >= 0.10
    if bool(prediction.get("fixed_rank_window")):
        return score >= 0.92 and margin >= 0.05
    if (source or str(prediction.get("source") or "")) == "board":
        if str(prediction.get("label") or "") == "Q":
            # The WPT board-font Queen has a long thin tail. It is a strong
            # shape match but often has a tiny Q/8 similarity margin after
            # normalization. Manually checked board cards found no true 8
            # with this clean Q read; retain a score floor to avoid accepting
            # low-quality fragments as Queens.
            return score >= 0.90 and margin >= 0.01
        if str(prediction.get("label") or "") == "3":
            # The board-specific 3 and the generic 8 have near-identical
            # embedding scores, but the fixed board crop preserves the open
            # left edge of a 3. Confirmed board 8s did not trigger this
            # specialist read; allow the stable, clear 3 even with its small
            # template margin.
            return score >= 0.86 and margin >= 0.006
        # The board-card inset corner removes the magenta left edge. On the
        # manually checked board set, 0.89 / 0.10 adds the clean Ace read
        # (including a clean Ace) without accepting a wrong read.
        return (score >= 0.92 and margin >= 0.07) or (score >= 0.89 and margin >= 0.10)
    # The fixed WPT font's 3 has a close 8 neighbor after normalization.  A
    # clean aligned 3 at this score is more reliable than shifted-window
    # 3/8 correction votes, which may only contain the right half of the glyph.
    if str(prediction.get("label") or "") == "3":
        return score >= 0.86 and margin >= 0.07
    return (score >= 0.82 and margin >= 0.10) or (score >= 0.78 and margin >= 0.20)


def choose_aligned_king_rank_override(
    crop: Any,
    *,
    source: str,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    aligned_votes: list[tuple[str, float, float]],
) -> tuple[float, str, float] | None:
    if source != "hero" or rank != "J" or card_glyph_color(crop) != "red":
        return None
    king_votes = [
        (float(score), float(score) - float(second))
        for label, score, second in aligned_votes
        if label == "K" and float(score) >= 0.62
    ]
    if len(king_votes) < 2:
        return None
    best_score, best_margin = max(king_votes, key=lambda item: (item[0], item[1]))
    if float(rank_score) - best_score > 0.03:
        return None
    corrected_score = max(0.64, best_score)
    corrected_margin = max(0.04, min(0.12, best_margin))
    corrected_next = min(corrected_score - corrected_margin, float(next_rank_score))
    return corrected_score, "K", corrected_next


def weak_hero_rank_window(
    *,
    source: str,
    rank_width: int,
    rank_score: float,
    next_rank_score: float,
) -> bool:
    if source != "hero":
        return False
    margin = float(rank_score) - float(next_rank_score)
    if margin < 0.018:
        return True
    # A wide aligned crop can contain the complete glyph while its nearest
    # neighbor remains visually close (notably Q versus 8).  Preserve a
    # genuinely strong full-glyph vote instead of falling back to a shifted
    # fragment with a cleaner but much weaker margin.
    if rank_width >= 64 and float(rank_score) >= 0.90 and margin >= 0.04:
        return False
    if rank_width >= 64 and float(rank_score) < 0.97 and margin < 0.055:
        return True
    return False


def rank_candidate_windows(
    source: str,
    *,
    index: int = -1,
    crop_width: int | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    if source == "board":
        return BOARD_RANK_WINDOWS
    if source != "hero":
        return ((0, 0, 55, 60),)
    return (
        (0, 0, 42, 72),
        (0, 0, 48, 72),
        (0, 0, 55, 72),
        (8, 18, 42, 74),
        (12, 20, 38, 70),
        (10, 22, 34, 62),
        (0, 0, 64, 72),
        (2, 0, 42, 66),
        (4, 0, 34, 66),
        (10, 10, 42, 72),
    )


def wide_hero_rank_conflicts_with_existing(
    *,
    source: str,
    rank_width: int,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    best: tuple[float, str, float, float] | None,
) -> bool:
    if source != "hero" or rank_width < 64 or best is None:
        return False
    _best_selector, best_rank, best_score, best_next_score = best
    if not best_rank or best_rank == rank:
        return False
    best_margin = float(best_score) - float(best_next_score)
    current_margin = float(rank_score) - float(next_rank_score)
    if float(best_score) < 0.82 or best_margin < 0.14:
        return False
    if float(rank_score) - float(best_score) > 0.16:
        return False
    if current_margin >= 0.30 and float(rank_score) >= 0.98:
        return False
    return True


def should_expand_rank_windows(
    *,
    glyph_color: str,
    rank: str,
    rank_score: float,
    next_rank_score: float,
) -> bool:
    margin = float(rank_score) - float(next_rank_score)
    if glyph_color == "red" and rank == "8":
        return True
    if glyph_color == "red" and rank == "3":
        return True
    if glyph_color == "red" and rank == "6" and rank_score < 0.985:
        return True
    if rank_score < 0.78 or margin < 0.12:
        return True
    if glyph_color == "black" and rank in {"4", "J"} and rank_score < 0.95:
        return True
    return False


def choose_red_nine_rank_override(
    crop: Any,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    alternatives: list[tuple[str, float, float]],
) -> tuple[float, str, float] | None:
    if card_glyph_color(crop) != "red" or rank != "8":
        return None
    close_nine_votes = [
        (float(score), float(score) - float(second))
        for label, score, second in alternatives
        if (
            label == "9"
            and float(score) >= 0.94
            and float(score) - float(second) >= 0.12
            and float(rank_score) - float(score) <= 0.025
        )
    ]
    if not close_nine_votes:
        return None
    nine_score, nine_margin = max(close_nine_votes, key=lambda item: (item[0], item[1]))
    corrected_score = max(0.70, min(0.985, nine_score))
    corrected_next = min(corrected_score - 0.12, max(0.42, corrected_score - max(0.12, nine_margin)))
    return corrected_score, "9", corrected_next


def choose_red_four_rank_override(
    crop: Any,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    alternatives: list[tuple[str, float, float]],
) -> tuple[float, str, float] | None:
    if card_glyph_color(crop) != "red" or rank not in {"J", "6"}:
        return None
    if rank == "J" and rank_score >= 0.82:
        return None
    if rank == "6" and rank_score >= 0.985:
        return None
    four_votes = [
        (float(score), float(score) - float(second))
        for label, score, second in alternatives
        if label == "4" and float(score) >= 0.68 and float(score) - float(second) >= 0.018
    ]
    if rank == "6":
        four_votes = [(score, margin) for score, margin in four_votes if score >= 0.80 and margin >= 0.08]
        if len(four_votes) < 2:
            return None
    tied_high_four_votes = [
        float(score)
        for label, score, second in alternatives
        if label == "4" and float(score) >= 0.82 and abs(float(score) - float(second)) <= 1e-6
    ]
    if not four_votes and len(tied_high_four_votes) < 3:
        return None
    if four_votes:
        four_score, four_margin = max(four_votes, key=lambda item: (item[0], item[1]))
    else:
        four_score = max(tied_high_four_votes)
        four_margin = 0.10
    corrected_score = max(0.70, min(0.82, four_score + 0.06))
    corrected_next = min(corrected_score - 0.10, max(0.42, corrected_score - max(0.10, four_margin + 0.05)))
    return corrected_score, "4", corrected_next


def choose_red_eight_rank_override(
    crop: Any,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    alternatives: list[tuple[str, float, float]],
) -> tuple[float, str, float] | None:
    if card_glyph_color(crop) != "red" or rank == "8":
        return None
    margin = float(rank_score) - float(next_rank_score)
    tied_high_three_votes = [
        float(score)
        for label, score, second in alternatives
        if label == "3" and float(score) >= 0.98 and abs(float(score) - float(second)) <= 1e-6
    ]
    eight_votes = [
        (float(score), float(score) - float(second))
        for label, score, second in alternatives
        if label == "8" and float(score) >= 0.60 and float(score) - float(second) >= 0.018
    ]
    if not tied_high_three_votes or not eight_votes:
        if margin >= 0.095:
            return None
        if not eight_votes:
            return None
    eight_score, eight_margin = max(eight_votes, key=lambda item: (item[0], item[1]))
    if tied_high_three_votes:
        if len(tied_high_three_votes) < 2 and float(rank_score) < 0.94:
            if eight_score < 0.84 or eight_margin < 0.02:
                return None
    else:
        if rank not in {"Q", "9", "K", "7", "J", "5", "6", "3"}:
            return None
        if eight_score + 0.09 < float(rank_score) and margin >= 0.055:
            return None
    corrected_score = max(0.70, min(0.82, eight_score + 0.04))
    corrected_next = min(corrected_score - 0.10, max(0.42, corrected_score - max(0.10, eight_margin + 0.06)))
    return corrected_score, "8", corrected_next


def choose_red_five_rank_override(
    crop: Any,
    rank: str,
    rank_score: float,
    next_rank_score: float,
    deep_votes: list[dict[str, Any]],
) -> tuple[float, str, float] | None:
    if card_glyph_color(crop) != "red":
        return None
    if rank not in {"8", "9", "3", "4", "Q", "J"}:
        return None
    if rank_score - next_rank_score >= 0.10:
        return None
    five_votes = [
        vote
        for vote in deep_votes
        if vote.get("rank") == "5" and float(vote.get("score") or 0.0) >= 0.18 and float(vote.get("margin") or 0.0) >= 0.06
    ]
    distinct_widths = {int(vote.get("width") or 0) for vote in five_votes}
    total_margin = sum(float(vote.get("margin") or 0.0) for vote in five_votes)
    if len(distinct_widths) < 4 or total_margin < 0.70:
        return None
    corrected_score = max(0.68, min(0.78, float(rank_score) * 0.82))
    corrected_next = min(corrected_score - 0.16, max(0.36, float(next_rank_score) * 0.60))
    return corrected_score, "5", corrected_next


def recognize_card_suit(
    crop: Any,
    templates: dict[str, dict[str, Any]],
    source: str = "hero",
) -> tuple[str | None, float | None, float | None, str]:
    cache_key = card_recognition_cache_key(crop, kind="suit", source=source, index=-1)
    cached = _CARD_SUIT_RECOGNITION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    color = card_glyph_color(crop)
    allowed = ("h", "d") if color == "red" else ("s", "c")
    board_clean_prediction = clean_board_suit_prediction(crop, source=source, allowed=allowed)
    if clean_board_suit_prediction_is_decisive(board_clean_prediction):
        return store_suit_recognition_cache(
            cache_key,
            (
                str(board_clean_prediction["label"]),
                float(board_clean_prediction["score"]),
                float(board_clean_prediction["margin"]),
                color,
            ),
        )
    clean_prediction = clean_hero_suit_prediction(crop, source=source, allowed=allowed)
    if clean_hero_suit_prediction_is_decisive(clean_prediction, allowed=allowed):
        return store_suit_recognition_cache(
            cache_key,
            (
                str(clean_prediction["label"]),
                float(clean_prediction["score"]),
                float(clean_prediction["margin"]),
                color,
            ),
        )
    suit_templates = {key: templates["suits"].get(key) for key in allowed if key in templates["suits"]}
    if not suit_templates:
        return store_suit_recognition_cache(cache_key, (None, None, None, color))
    ranked_candidates = []
    seen_suit_images: set[bytes] = set()
    for suit_image in normalized_suit_candidates(crop, (42, 42), source):
        add_suit_candidate(
            suit_image=suit_image,
            suit_templates=suit_templates,
            ranked_candidates=ranked_candidates,
            seen_suit_images=seen_suit_images,
            allowed=allowed,
        )
    if not ranked_candidates:
        return store_suit_recognition_cache(cache_key, (None, None, None, color))
    margin, best_score, suit, second_score = choose_best_suit_candidate(ranked_candidates)
    detail_candidates: list[tuple[float, float, str, float]] = []
    if should_try_hero_suit_detail_windows(source=source, color=color, suit=suit, best_score=best_score, margin=margin):
        for suit_image in normalized_hero_suit_detail_windows(crop, (42, 42)):
            add_suit_candidate(
                suit_image=suit_image,
                suit_templates=suit_templates,
                ranked_candidates=detail_candidates,
                seen_suit_images=seen_suit_images,
                allowed=allowed,
            )
        ranked_candidates.extend(detail_candidates)
        margin, best_score, suit, second_score = choose_best_suit_candidate(ranked_candidates)
        diamond_override = choose_red_diamond_suit_override(
            suit=suit,
            best_score=best_score,
            detail_candidates=detail_candidates,
        )
        if diamond_override is not None:
            margin, best_score, suit, second_score = diamond_override
    if (
        source == "hero"
        and color == "black"
        and clean_prediction is not None
        and str(clean_prediction.get("label") or "") in allowed
        and float(clean_prediction.get("score") or 0.0) >= 0.65
        and float(clean_prediction.get("margin") or 0.0) >= 0.02
        and margin < 0.03
    ):
        return store_suit_recognition_cache(
            cache_key,
            (
                str(clean_prediction["label"]),
                float(clean_prediction["score"]),
                float(clean_prediction["margin"]),
                color,
            ),
        )
    min_score = 0.15 if source == "board" else 0.22
    if source == "board":
        min_margin = 0.0 if best_score >= 0.42 else 0.03
    else:
        min_margin = 0.05 if best_score >= 0.72 else 0.08
    if best_score < min_score or margin < min_margin:
        relaxed_red = source == "hero" and color == "red" and best_score >= 0.40 and margin >= 0.030
        relaxed_black = source == "hero" and color == "black" and best_score >= 0.80 and margin >= 0.020
        if not (relaxed_red or relaxed_black):
            return store_suit_recognition_cache(cache_key, (None, best_score, margin, color))
    return store_suit_recognition_cache(cache_key, (suit, best_score, margin, color))


def clean_hero_suit_prediction(
    crop: Any,
    *,
    source: str,
    allowed: tuple[str, ...],
) -> dict[str, Any] | None:
    if source != "hero":
        return None
    is_black_pair = set(allowed) == {"s", "c"}
    configured_model = (
        os.environ.get("GTO_CARD_HERO_BLACK_SUIT_MODEL")
        if is_black_pair
        else os.environ.get("GTO_CARD_HERO_SUIT_MODEL")
    )
    default_model = HERO_BLACK_SUIT_MODEL_PATH if is_black_pair and HERO_BLACK_SUIT_MODEL_PATH.exists() else HERO_SUIT_MODEL_PATH
    model_path = Path(configured_model or default_model)
    if not model_path.exists():
        return None
    # Keep the live model's input identical to the review/training asset.  The
    # short fixed window can cut off the lower stem of a club or spade on the
    # current card layout, while the component extractor keeps the whole suit
    # and falls back to that clean window only when the card edge is merged.
    glyph = (
        normalized_hero_black_suit_component(crop, (42, 42))
        if is_black_pair
        else normalized_hero_red_suit_component(crop, (42, 42))
    )
    used_red_component = not is_black_pair and glyph is not None
    if glyph is None:
        glyph = normalized_suit_component_by_label(crop, (42, 42), source="hero")
    prediction = classify_suit_glyph(glyph, allowed=allowed, model_path=model_path)
    if prediction is not None and is_black_pair:
        prediction = {**prediction, "hero_black_component": True}
    elif prediction is not None and used_red_component:
        prediction = {**prediction, "hero_red_component": True}
    return prediction


def clean_board_suit_prediction(
    crop: Any,
    *,
    source: str,
    allowed: tuple[str, ...],
) -> dict[str, Any] | None:
    if source != "board" or set(allowed) != {"s", "c"}:
        return None
    glyph = normalized_board_suit_window(crop, (42, 42))
    if glyph is None:
        return None
    model_path = Path(os.environ.get("GTO_CARD_BOARD_BLACK_SUIT_MODEL") or BOARD_BLACK_SUIT_MODEL_PATH)
    if not model_path.exists():
        return classify_suit_glyph(glyph, allowed=allowed)
    return classify_suit_glyph(glyph, allowed=allowed, model_path=model_path)


def clean_board_suit_prediction_is_decisive(prediction: dict[str, Any] | None) -> bool:
    if prediction is None or str(prediction.get("label") or "") not in {"s", "c"}:
        return False
    model_name = Path(str(prediction.get("model") or "")).name
    if model_name == BOARD_BLACK_SUIT_MODEL_PATH.name:
        return float(prediction.get("score") or 0.0) >= 0.90 and float(prediction.get("margin") or 0.0) >= 0.05
    # Board cards use a single isolated suit component. For black suits the
    # classifier can be nearly tied between spade and club while still giving
    # the correct clean component a reliably high similarity. Do not let a
    # low-margin template fragment override that cleaner read.
    return float(prediction.get("score") or 0.0) >= 0.88


def clean_hero_suit_prediction_is_decisive(
    prediction: dict[str, Any] | None,
    *,
    allowed: tuple[str, ...] | None = None,
) -> bool:
    if prediction is None:
        return False
    if set(allowed or ()) == {"s", "c"} and prediction.get("hero_black_component"):
        return float(prediction.get("score") or 0.0) >= 0.90 and float(prediction.get("margin") or 0.0) >= 0.02
    if set(allowed or ()) == {"h", "d"} and prediction.get("hero_red_component"):
        return float(prediction.get("score") or 0.0) >= 0.84 and float(prediction.get("margin") or 0.0) >= 0.04
    min_margin = 0.04 if set(allowed or ()) == {"s", "c"} else 0.06
    return float(prediction.get("score") or 0.0) >= 0.92 and float(prediction.get("margin") or 0.0) >= min_margin


def add_suit_candidate(
    *,
    suit_image: Any,
    suit_templates: dict[str, Any],
    ranked_candidates: list[tuple[float, float, str, float]],
    seen_suit_images: set[bytes],
    allowed: tuple[str, ...],
) -> None:
    suit_image_key = suit_image.tobytes()
    if suit_image_key in seen_suit_images:
        return
    seen_suit_images.add(suit_image_key)
    scores = sorted(
        ((best_template_score(suit_image, images), suit) for suit, images in suit_templates.items()),
        reverse=True,
    )
    if not scores:
        return
    best_score, candidate_suit = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    ranked_candidates.append((best_score - second_score, best_score, candidate_suit, second_score))
    classifier = classify_suit_glyph(suit_image, allowed=allowed)
    if classifier is not None:
        classifier_score = float(classifier["score"])
        classifier_margin = float(classifier["margin"])
        if classifier_score >= 0.20 and classifier_margin >= 0.02:
            ranked_candidates.append(
                (
                    classifier_margin,
                    classifier_score,
                    str(classifier["label"]),
                    float(classifier.get("second_score", -1.0)),
                )
            )
    if best_score < 0.58 or best_score - second_score < 0.12:
        deep_classifier = classify_deep_glyph(suit_image, "suit", allowed=allowed)
        if deep_classifier is not None:
            deep_score = float(deep_classifier["score"])
            deep_margin = float(deep_classifier["margin"])
            if deep_score >= 0.38 and deep_margin >= 0.08:
                ranked_candidates.append(
                    (
                        deep_margin,
                        deep_score,
                        str(deep_classifier["label"]),
                        float(deep_classifier.get("second_score", -1.0)),
                    )
                )


def choose_best_suit_candidate(ranked_candidates: list[tuple[float, float, str, float]]) -> tuple[float, float, str, float]:
    def selection_key(item: tuple[float, float, str, float]) -> tuple[float, float, float]:
        candidate_margin, score, _label, _second = item
        # A narrowly-cropped hero card can produce a high raw template score from
        # the rank glyph and a slightly lower but much cleaner classifier vote
        # from the actual suit blob. Let strong top-vs-second separation break
        # those near ties without overwhelming genuinely high scores.
        robust_score = float(score) + min(max(float(candidate_margin), 0.0) * 0.25, 0.07)
        return robust_score, float(candidate_margin), float(score)

    _candidate_margin, best_score, suit, second_score = max(ranked_candidates, key=selection_key)
    margin = best_score - second_score
    return margin, best_score, suit, second_score


def should_try_hero_suit_detail_windows(
    *,
    source: str,
    color: str,
    suit: str,
    best_score: float,
    margin: float,
) -> bool:
    return source == "hero" and color == "red" and suit == "h" and (best_score < 0.93 or margin < 0.12)


def choose_red_diamond_suit_override(
    *,
    suit: str,
    best_score: float,
    detail_candidates: list[tuple[float, float, str, float]],
) -> tuple[float, float, str, float] | None:
    if suit != "h" or best_score >= 0.985:
        return None
    diamond_votes = [
        (float(candidate_margin), float(score), float(second))
        for candidate_margin, score, label, second in detail_candidates
        if label == "d" and float(score) >= 0.78 and float(candidate_margin) >= 0.22
    ]
    if len(diamond_votes) < 2:
        return None
    candidate_margin, score, second = max(diamond_votes, key=lambda item: (item[1], item[0]))
    return score - second, score, "d", second


def card_recognition_cache_key(crop: Any, *, kind: str, source: str, index: int) -> tuple[Any, ...]:
    height, width = crop.shape[:2]
    roi = crop[0 : min(height, 132), 0 : min(width, 96)]
    digest = hashlib.blake2b(roi.tobytes(), digest_size=16).hexdigest()
    return (
        kind,
        source,
        int(index),
        int(height),
        int(width),
        digest,
        os.environ.get("GTO_CARD_KNN_MODEL") or "",
        os.environ.get("GTO_CARD_BOARD_KNN_MODEL") or "",
        os.environ.get("GTO_CARD_HERO_RANK_MODEL") or "",
        os.environ.get("GTO_CARD_HERO_SUIT_MODEL") or "",
        os.environ.get("GTO_CARD_HERO_BLACK_SUIT_MODEL") or "",
        os.environ.get("GTO_CARD_BOARD_BLACK_SUIT_MODEL") or "",
        os.environ.get("GTO_CARD_DEEP_MODEL_DIR") or "",
    )


def store_rank_recognition_cache(
    key: tuple[Any, ...],
    value: tuple[float, str, float],
) -> tuple[float, str, float]:
    if len(_CARD_RANK_RECOGNITION_CACHE) >= _CARD_RECOGNITION_CACHE_LIMIT:
        _CARD_RANK_RECOGNITION_CACHE.clear()
    _CARD_RANK_RECOGNITION_CACHE[key] = value
    return value


def store_suit_recognition_cache(
    key: tuple[Any, ...],
    value: tuple[str | None, float | None, float | None, str],
) -> tuple[str | None, float | None, float | None, str]:
    if len(_CARD_SUIT_RECOGNITION_CACHE) >= _CARD_RECOGNITION_CACHE_LIMIT:
        _CARD_SUIT_RECOGNITION_CACHE.clear()
    _CARD_SUIT_RECOGNITION_CACHE[key] = value
    return value


def load_card_templates() -> dict[str, dict[str, Any]]:
    global _CARD_TEMPLATES
    if _CARD_TEMPLATES is not None:
        return _CARD_TEMPLATES
    cv2, _np = load_cv()
    ranks: dict[str, list[Any]] = {}
    suits: dict[str, list[Any]] = {}
    for path in CARD_TEMPLATE_DIR.glob("rank_*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            label = path.stem.removeprefix("rank_").split("_", 1)[0]
            ranks.setdefault(label, []).append(image)
    for path in CARD_TEMPLATE_DIR.glob("suit_*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            label = path.stem.removeprefix("suit_").split("_", 1)[0]
            suits.setdefault(label, []).append(image)
    _CARD_TEMPLATES = {"ranks": ranks, "suits": suits}
    return _CARD_TEMPLATES


def card_white_ratio(crop: Any) -> float:
    cv2, _np = load_cv()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)).mean())


def card_face_rect_metrics(crop: Any) -> dict[str, float]:
    cv2, _np = load_cv()
    if crop.size == 0:
        return {"fill": 0.0, "cover": 0.0, "aspect": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)).astype("uint8")) * 255
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"fill": 0.0, "cover": 0.0, "aspect": 0.0}
    height, width = mask.shape[:2]
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    _x, _y, rect_w, rect_h = cv2.boundingRect(best)
    rect_area = max(1.0, float(rect_w * rect_h))
    return {
        "fill": area / rect_area,
        "cover": area / max(1.0, float(width * height)),
        "aspect": float(rect_h) / max(1.0, float(rect_w)),
    }


def tighten_hero_card_face_crop(crop: Any) -> Any:
    cv2, np = load_cv()
    if crop.size == 0:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)).astype("uint8")) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop
    height, width = crop.shape[:2]
    best = max(contours, key=cv2.contourArea)
    x, y, rect_w, rect_h = cv2.boundingRect(best)
    if rect_w < width * 0.25 or rect_h < height * 0.25:
        return crop
    pad_x = max(2, int(round(rect_w * 0.06)))
    pad_y = max(2, int(round(rect_h * 0.06)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + rect_w + pad_x)
    y2 = min(height, y + rect_h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return crop
    return crop[y1:y2, x1:x2]


def hero_card_face_is_plausible(metrics: dict[str, float]) -> bool:
    fill = float(metrics.get("fill") or 0.0)
    cover = float(metrics.get("cover") or 0.0)
    aspect = float(metrics.get("aspect") or 0.0)
    if cover < 0.48:
        return False
    if fill < 0.72:
        return False
    if aspect < 0.85 or aspect > 2.20:
        return False
    return True


def partial_hero_card_face_is_plausible(metrics: dict[str, float]) -> bool:
    fill = float(metrics.get("fill") or 0.0)
    cover = float(metrics.get("cover") or 0.0)
    aspect = float(metrics.get("aspect") or 0.0)
    if cover < 0.20:
        return False
    if fill < 0.60:
        return False
    if aspect < 0.35 or aspect > 1.35:
        return False
    return True


def enhance_dim_card_crop(crop: Any) -> Any:
    cv2, np = load_cv()
    if crop.size == 0:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    p90 = float(np.percentile(val, 90))
    if p90 >= 155:
        return crop
    alpha = min(3.2, max(1.0, 210.0 / max(p90, 1.0)))
    return cv2.convertScaleAbs(crop, alpha=alpha, beta=0)


def card_glyph_color(crop: Any) -> str:
    cv2, _np = load_cv()
    roi = crop[0:92, 0:55]
    mask = card_foreground_mask(roi) > 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = ((((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)) & mask).sum()
    dark = (((hsv[:, :, 2] < 135) & (hsv[:, :, 1] < 190)) & mask).sum()
    return "red" if red > dark * 0.65 else "black"


def normalized_card_piece(crop: Any, size: tuple[int, int]) -> Any:
    cv2, np = load_cv()
    width, height = size
    if crop.size == 0:
        return np.zeros((height, width), np.uint8)
    mask = card_foreground_mask(crop)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((height, width), np.uint8)
    x1, x2 = max(0, int(xs.min()) - 1), min(mask.shape[1], int(xs.max()) + 2)
    y1, y2 = max(0, int(ys.min()) - 1), min(mask.shape[0], int(ys.max()) + 2)
    piece = mask[y1:y2, x1:x2]
    piece_h, piece_w = piece.shape
    scale = min(width / piece_w, height / piece_h)
    resized_w = max(1, int(piece_w * scale))
    resized_h = max(1, int(piece_h * scale))
    resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), np.uint8)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return canvas


def normalized_rank_piece(crop: Any, size: tuple[int, int] = (54, 70)) -> Any:
    cv2, np = load_cv()
    if crop.size == 0:
        return normalized_card_piece(crop, size)
    mask = rank_foreground_mask(crop)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    image_height, image_width = mask.shape[:2]
    kept = 0
    for component in range(1, component_count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        if area < 20 or width < 2 or height < 5:
            continue
        wide_border = width >= image_width * 0.72 and height <= image_height * 0.30
        edge_vertical = (
            height >= image_height * 0.42
            and width <= max(6, int(round(image_width * 0.12)))
            and (x <= 2 or x + width >= image_width - 2)
        )
        far_right_fragment = x >= image_width * 0.86 and area < image_width * image_height * 0.08
        if wide_border or edge_vertical or far_right_fragment:
            continue
        cleaned[labels == component] = 255
        kept += 1
    if kept == 0 or int((cleaned > 0).sum()) < 30:
        return normalized_card_piece(crop, size)
    return normalized_mask_piece(cleaned, size)


def normalized_hero_rank_window(crop: Any, size: tuple[int, int] = (54, 70)) -> Any | None:
    """Return the fixed top-left rank window for a locked hero card."""
    height, width = crop.shape[:2]
    if height < 70 or width < 40:
        return None
    x1, x2 = int(round(width * 0.12)), int(round(width * 0.60))
    y1, y2 = int(round(height * 0.06)), int(round(height * 0.51))
    if x2 - x1 < 20 or y2 - y1 < 32:
        return None
    return normalized_rank_piece(crop[y1:y2, x1:x2], size)


def rank_foreground_mask(crop: Any) -> Any:
    cv2, np = load_cv()
    if crop.size == 0:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    red = ((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    dark = (hsv[:, :, 2] < 125) & (hsv[:, :, 1] < 180)
    gray_dark = gray < 155
    # Do not run MORPH_CLOSE here: the WPT "10" glyph has a one-pixel gap,
    # and closing merges the 1 and 0 into the K/8-like shape seen in live play.
    mask = (red | dark | gray_dark).astype("uint8") * 255
    return cv2.medianBlur(mask, 3)


def normalized_suit_component(crop: Any, size: tuple[int, int] = (42, 42), source: str = "hero") -> Any:
    return normalized_suit_candidates(crop, size, source)[0]


def normalized_suit_candidates(crop: Any, size: tuple[int, int] = (42, 42), source: str = "hero") -> list[Any]:
    candidates = []
    if source == "board":
        board_piece = normalized_board_suit_window(crop, size)
        if board_piece is not None:
            candidates.append(board_piece)
    if source == "hero":
        hero_piece = normalized_hero_suit_window(crop, size)
        if hero_piece is not None:
            candidates.append(hero_piece)
    candidates.append(normalized_suit_component_by_label(crop, size, source))
    return candidates


def normalized_hero_suit_window(crop: Any, size: tuple[int, int]) -> Any | None:
    cv2, _np = load_cv()
    y1 = min(crop.shape[0], 64)
    y2 = min(crop.shape[0], 96)
    x1 = min(crop.shape[1], 4)
    x2 = min(crop.shape[1], 46)
    if y2 - y1 < 25 or x2 - x1 < 22:
        return None
    piece = crop[y1:y2, x1:x2]
    mask = card_foreground_mask(piece)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, component_count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        cx, cy = centroids[component]
        if area < 70 or area > 850:
            continue
        if width < 8 or height < 8 or width > 38 or height > 31:
            continue
        edge_line_penalty = 8.0 if x <= 1 and width <= 5 else 0.0
        distance = abs(float(cx) - 21.0) + abs(float(cy) - 18.0) * 0.7 + edge_line_penalty
        candidates.append((distance, component))
    if not candidates:
        return None
    _distance, component = min(candidates, key=lambda item: item[0])
    component_mask = (labels == component).astype("uint8") * 255
    return normalized_mask_piece(component_mask, size)


def normalized_hero_red_suit_component(crop: Any, size: tuple[int, int]) -> Any | None:
    if card_glyph_color(crop) != "red":
        return None
    cv2, np = load_cv()
    y1 = min(crop.shape[0], 52)
    y2 = min(crop.shape[0], 106)
    x1 = min(crop.shape[1], 4)
    x2 = min(crop.shape[1], 56)
    if y2 - y1 < 30 or x2 - x1 < 28:
        return None
    piece = crop[y1:y2, x1:x2]
    hsv = cv2.cvtColor(piece, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (0, 70, 55), (15, 255, 255))
        | cv2.inRange(hsv, (165, 70, 55), (180, 255, 255))
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, component_count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        cx, cy = centroids[component]
        if area < 55 or area > 520:
            continue
        if width < 7 or height < 8 or width > 32 or height > 34:
            continue
        if x <= 1 or x + width >= piece.shape[1] - 1:
            continue
        distance = abs(float(cx) - 24.0) + abs(float(cy) - 24.0) * 0.75
        candidates.append((distance, -area, component))
    if not candidates:
        return None
    _distance, _negative_area, component = min(candidates)
    component_mask = (labels == component).astype(np.uint8) * 255
    return normalized_mask_piece(component_mask, size)


def normalized_hero_suit_detail_windows(crop: Any, size: tuple[int, int]) -> list[Any]:
    if card_glyph_color(crop) != "red":
        return []
    candidates = []
    seen: set[bytes] = set()
    for x, y, width, height in (
        (16, 55, 28, 48),
        (12, 55, 32, 48),
        (8, 55, 36, 48),
        (12, 60, 24, 36),
        (16, 85, 42, 36),
        (12, 90, 42, 32),
    ):
        if y >= crop.shape[0] or x >= crop.shape[1]:
            continue
        piece = crop[y : min(crop.shape[0], y + height), x : min(crop.shape[1], x + width)]
        if piece.size == 0 or piece.shape[0] < 18 or piece.shape[1] < 18:
            continue
        mask = card_foreground_mask(piece)
        if int((mask > 0).sum()) < 30:
            continue
        normalized = normalized_card_piece(piece, size)
        key = normalized.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)
    return candidates


def normalized_board_suit_window(crop: Any, size: tuple[int, int]) -> Any | None:
    cv2, np = load_cv()
    # The board card itself is locked. Its large lower-right pip occupies the
    # same relative rectangle across the recorded layouts, so crop that fixed
    # rectangle first. Connected components below only remove border noise;
    # they no longer decide where to look for the suit.
    height, width = crop.shape[:2]
    if height < 80 or width < 60:
        return None
    x1, x2 = int(round(width * 0.30)), int(round(width * 0.98))
    y1, y2 = int(round(height * 0.53)), int(round(height * 0.98))
    roi = crop[y1:y2, x1:x2]
    if roi.shape[0] < 35 or roi.shape[1] < 35:
        return None
    mask = card_foreground_mask(roi)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, component_count):
        x, y, component_width, component_height, area = [int(value) for value in stats[component]]
        cx, cy = centroids[component]
        if area < 100 or component_width < 12 or component_height < 12:
            continue
        if x <= 0 or y <= 0 or x + component_width >= mask.shape[1] - 1 or y + component_height >= mask.shape[0] - 1:
            continue
        candidates.append((float(area), component))
    if candidates:
        _area, component = max(candidates, key=lambda item: item[0])
        return normalized_mask_piece((labels == component).astype(np.uint8) * 255, size)

    # Some transition frames reveal only the corner pip. Keep a narrow fixed
    # fallback for those frames rather than scanning the whole board card.
    small_mask = card_foreground_mask(crop[0 : min(crop.shape[0], 100), 0 : min(crop.shape[1], 64)])
    if small_mask.shape[0] < 70 or small_mask.shape[1] < 24:
        return None
    piece = small_mask[56 : min(84, small_mask.shape[0]), 6 : min(34, small_mask.shape[1])]
    if int((piece > 0).sum()) < 30:
        return None
    return normalized_mask_piece(piece, size)


def normalized_hero_black_suit_component(crop: Any, size: tuple[int, int]) -> Any | None:
    """Extract the lower-left black suit without merging it into the card rim."""
    cv2, np = load_cv()
    if crop.size == 0:
        return None
    height, width = crop.shape[:2]
    roi = crop[0 : min(height, 130), 0 : min(width, 64)]
    if roi.shape[0] < 78 or roi.shape[1] < 36:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Do not use card_foreground_mask here: its closing step can connect a
    # black pip to the dark card rim, which makes the pip disappear as a
    # selectable component on the narrower left hero card.
    mask = ((gray < 145) & (hsv[:, :, 1] < 190)).astype(np.uint8) * 255
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, component_count):
        x, y, component_width, component_height, area = [int(value) for value in stats[component]]
        cx, cy = centroids[component]
        if area < 70 or area > 800:
            continue
        if x < 5 or x > 42 or y < 52 or cy > 102:
            continue
        if component_width < 8 or component_height < 8 or component_width > 34 or component_height > 34:
            continue
        distance = abs(float(cx) - 24.0) * 1.2 + abs(float(cy) - 79.0)
        candidates.append((distance, component))
    if not candidates:
        return None
    _distance, component = min(candidates, key=lambda item: item[0])
    return normalized_mask_piece((labels == component).astype(np.uint8) * 255, size)


def normalized_suit_component_by_label(crop: Any, size: tuple[int, int], source: str = "hero") -> Any:
    cv2, np = load_cv()
    roi_height = 130 if source == "hero" else 140
    roi_width = 64 if source == "hero" else 74
    roi = crop[0 : min(crop.shape[0], roi_height), 0 : min(crop.shape[1], roi_width)]
    mask = card_foreground_mask(roi)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for component in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component]]
        cx, cy = centroids[component]
        if source == "hero":
            if area < 45 or area > 920:
                continue
            if x > 42 or y < 52 or cy < 55 or cy > 112:
                continue
            if w < 8 or h < 8 or w > 42 or h > 40:
                continue
            distance = abs(float(cx) - 22.0) * 1.3 + abs(float(cy) - 94.0) + max(0, w - 34) * 1.1
        else:
            if area < 45 or area > 900:
                continue
            if x > 42 or y < 54 or cy < 45 or cy > 118:
                continue
            if w < 8 or h < 8 or w > 42 or h > 42:
                continue
            target_cy = 104.0 if cy >= 84 else 70.0
            distance = abs(float(cx) - 30.0) * 1.2 + abs(float(cy) - target_cy) + max(0, w - 32) * 0.8
        candidates.append((distance, component, x, y, w, h))
    if not candidates:
        if source == "hero":
            # Some compact hero crops connect the card edge to the lower card
            # area. Returning a raw bottom rectangle then leaks rank/card-edge
            # pixels into the suit dataset. The local suit window still
            # isolates the symbol in this layout, so use it as the fallback.
            hero_piece = normalized_hero_suit_window(crop, size)
            if hero_piece is not None:
                return hero_piece
        return normalized_card_piece(crop[48:92, 0:48], size)

    _distance, component, x, y, w, h = min(candidates, key=lambda item: item[0])
    x1 = max(0, x - 2)
    y1 = max(0, y - 2)
    x2 = min(mask.shape[1], x + w + 2)
    y2 = min(mask.shape[0], y + h + 2)
    piece = ((labels[y1:y2, x1:x2] == component).astype("uint8")) * 255
    return normalized_mask_piece(piece, size)


def normalized_mask_piece(mask: Any, size: tuple[int, int]) -> Any:
    cv2, np = load_cv()
    ys, xs = np.where(mask > 0)
    width, height = size
    if len(xs) == 0:
        return np.zeros((height, width), np.uint8)
    x1, x2 = max(0, int(xs.min()) - 1), min(mask.shape[1], int(xs.max()) + 2)
    y1, y2 = max(0, int(ys.min()) - 1), min(mask.shape[0], int(ys.max()) + 2)
    piece = mask[y1:y2, x1:x2]
    piece_h, piece_w = piece.shape
    scale = min(width / piece_w, height / piece_h)
    resized_w = max(1, int(piece_w * scale))
    resized_h = max(1, int(piece_h * scale))
    resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), np.uint8)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return canvas


def card_foreground_mask(crop: Any) -> Any:
    cv2, np = load_cv()
    if crop.size == 0:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    red = ((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    dark = (hsv[:, :, 2] < 125) & (hsv[:, :, 1] < 180)
    gray_dark = gray < 155
    mask = (red | dark | gray_dark).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.medianBlur(mask, 3)


def template_score(candidate: Any, template: Any) -> float:
    cv2, np = load_cv()
    if candidate.shape != template.shape:
        template = cv2.resize(template, (candidate.shape[1], candidate.shape[0]), interpolation=cv2.INTER_AREA)
    candidate_f = candidate.astype(np.float32)
    template_f = template.astype(np.float32)
    if candidate_f.std() < 1 or template_f.std() < 1:
        return -1.0
    normalized = ((candidate_f - candidate_f.mean()) * (template_f - template_f.mean())).mean()
    return float(normalized / (candidate_f.std() * template_f.std()))


def best_template_score(candidate: Any, templates: list[Any]) -> float:
    cv2, np = load_cv()
    if not templates:
        return -1.0
    if len(templates) == 1:
        return template_score(candidate, templates[0])
    stats = template_group_stats(templates)
    if stats is None or stats["shape"] != candidate.shape:
        return max(template_score(candidate, template) for template in templates)
    candidate_f = candidate.astype(np.float32)
    candidate_std = float(candidate_f.std())
    if candidate_std < 1:
        return -1.0
    centered_candidate = candidate_f - float(candidate_f.mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (stats["centered"] * centered_candidate).mean(axis=(1, 2)) / (stats["stds"] * candidate_std)
    if scores.size == 0:
        return -1.0
    return float(np.nanmax(scores))


def template_group_stats(templates: list[Any]) -> dict[str, Any] | None:
    cv2, np = load_cv()
    cache_key = id(templates)
    cached = _TEMPLATE_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shapes = {template.shape for template in templates}
    if len(shapes) != 1:
        return None
    stack = np.stack([template.astype(np.float32) for template in templates], axis=0)
    means = stack.mean(axis=(1, 2), keepdims=True)
    stds = stack.std(axis=(1, 2))
    stats = {
        "shape": templates[0].shape,
        "centered": stack - means,
        "stds": np.where(stds < 1, np.inf, stds),
    }
    _TEMPLATE_GROUP_CACHE[cache_key] = stats
    return stats


def detect_red_chips(frame: Any) -> list[dict[str, Any]]:
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 60), (12, 255, 255)) | cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
    mask[: int(height * 0.20), :] = 0
    mask[int(height * 0.86) :, :] = 0
    mask[:, : int(width * 0.06)] = 0
    mask[:, int(width * 0.95) :] = 0
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chips = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30 or area > 800:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 6 or w > 36 or h > 36:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.74 or aspect > 1.36:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / max(perimeter * perimeter, 1.0)
        if circularity < 0.42:
            continue
        chips.append(
            {
                "x": round(x + w / 2, 1),
                "y": round(y + h / 2, 1),
                "area": round(float(area), 1),
                "circularity": round(float(circularity), 3),
                "box": {"x": x, "y": y, "width": w, "height": h},
            }
        )
    return chips


def detect_card_statuses(frame: Any, seat_count: int) -> dict[int, dict[str, Any]]:
    if seat_count != 8:
        return {}
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    statuses = {}
    for seat_index, roi in CARD_ROIS_8.items():
        x1, y1, x2, y2 = scale_roi(roi, width, height)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_ratio = float(
            (
                ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
                & (hsv[:, :, 1] > 70)
                & (hsv[:, :, 2] > 90)
            ).mean()
        )
        white_ratio = float(((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 170)).mean())
        has_cards = red_ratio > 0.045 or white_ratio > 0.14
        statuses[seat_index] = {
            "has_cards": bool(has_cards),
            "red_card_ratio": round(red_ratio, 4),
            "white_card_ratio": round(white_ratio, 4),
        }
    return statuses


def scale_roi(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def nearby_chip(text_box: dict[str, int], chips: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for chip in chips:
        if (
            text_box["x"] - 75 <= chip["x"] <= text_box["x"] + text_box["width"] + 12
            and text_box["y"] - 24 <= chip["y"] <= text_box["y"] + text_box["height"] + 24
        ):
            distance = math.hypot(float(chip["x"]) - text_box["x"], float(chip["y"]) - (text_box["y"] + text_box["height"] / 2))
            candidates.append((distance, chip))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def nearest_bet_seat(chip: dict[str, Any], shape: tuple[int, ...], seat_count: int) -> int:
    if seat_count != 8:
        return 0
    height, width = shape[:2]
    return min(
        BET_ANCHORS_8,
        key=lambda index: (chip["x"] - BET_ANCHORS_8[index][0] * width) ** 2
        + (chip["y"] - BET_ANCHORS_8[index][1] * height) ** 2,
    )


def nearest_bet_text_seat(text_box: dict[str, int], shape: tuple[int, ...], seat_count: int) -> tuple[int | None, float]:
    if seat_count != 8:
        return None, float("inf")
    height, width = shape[:2]
    center_x = text_box["x"] + text_box["width"] / 2
    center_y = text_box["y"] + text_box["height"] / 2
    seat_index = min(
        BET_TEXT_ANCHORS_8,
        key=lambda index: (center_x - BET_TEXT_ANCHORS_8[index][0] * width) ** 2
        + (center_y - BET_TEXT_ANCHORS_8[index][1] * height) ** 2,
    )
    anchor_x = BET_TEXT_ANCHORS_8[seat_index][0] * width
    anchor_y = BET_TEXT_ANCHORS_8[seat_index][1] * height
    distance = math.hypot(center_x - anchor_x, center_y - anchor_y)
    if distance > max(84.0, 0.15 * min(width, height)):
        return None, distance
    return seat_index, distance


def normalize_ocr_text(text: str) -> str:
    return text.replace(" ", " ").strip().replace("Ｂ", "B").replace("８", "B")


def parse_bb_amount(text: str) -> float | None:
    compact = text.replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*B{1,2}", compact, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_truncated_bb_prefix(text: str) -> int | None:
    compact = text.replace(" ", "")
    match = re.search(r"(\d+)\.B{1,2}", compact, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def repair_bet_amount(amount: float, text: str, pot_amount: float | None) -> float:
    """Remove a chip glyph that OCR merged into the front of a bet amount."""

    if pot_amount is None or amount <= pot_amount + 0.15:
        return amount
    compact = text.replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*B{1,2}", compact, flags=re.IGNORECASE)
    if not match:
        return amount
    token = match.group(1)
    candidates: list[tuple[int, int, float]] = []
    for start in range(1, len(token)):
        suffix = token[start:]
        if suffix.startswith("."):
            suffix = f"0{suffix}"
        try:
            candidate = float(suffix)
        except ValueError:
            continue
        if 0.0 < candidate <= pot_amount + 0.15:
            candidates.append((1 if "." in suffix else 0, len(suffix), candidate))
    if not candidates:
        return amount
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def is_ignored_bb_text(text: str, box: dict[str, int], shape: tuple[int, ...]) -> bool:
    height, width = shape[:2]
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    compact = text.replace(" ", "")
    if "底池" in text or "池" in text or (0.36 * width <= center_x <= 0.62 * width and 0.32 * height <= center_y <= 0.44 * height):
        return True
    if center_y > 0.84 * height:
        return True
    if compact.endswith("BB") and box["width"] > 72 and center_y < 0.84 * height:
        # Player stack labels are usually wide and have no chip next to them; this is only a soft pre-filter.
        return False
    return False


def is_ignored_bet_text(text: str, box: dict[str, int], shape: tuple[int, ...], pot: dict[str, Any] | None = None) -> bool:
    height = shape[0]
    center_y = box["y"] + box["height"] / 2
    compact = text.replace(" ", "")
    if "\u5e95\u6c60" in text or "\u6c60" in text or "ЕзГи" in text or "Ги" in text:
        return True
    if pot and boxes_overlap(box, pot.get("box") or {}, padding=4):
        return True
    if center_y > 0.84 * height:
        return True
    if compact.endswith("BB") and box["width"] > 72 and center_y < 0.84 * height:
        return False
    return False


def boxes_overlap(a: dict[str, int], b: dict[str, Any], padding: int = 0) -> bool:
    if not a or not b:
        return False
    ax1 = int(a.get("x", 0)) - padding
    ay1 = int(a.get("y", 0)) - padding
    ax2 = int(a.get("x", 0)) + int(a.get("width", 0)) + padding
    ay2 = int(a.get("y", 0)) + int(a.get("height", 0)) + padding
    bx1 = int(b.get("x", 0))
    by1 = int(b.get("y", 0))
    bx2 = int(b.get("x", 0)) + int(b.get("width", 0))
    by2 = int(b.get("y", 0)) + int(b.get("height", 0))
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def is_player_stack_region(box: dict[str, int], shape: tuple[int, ...], padding_x: float = 0.02, padding_y: float = 0.055) -> bool:
    height, width = shape[:2]
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    for roi in STACK_ROIS_8.values():
        padded = (
            max(0.0, roi[0] - padding_x),
            max(0.0, roi[1] - padding_y),
            min(1.0, roi[2] + padding_x),
            min(1.0, roi[3] + padding_y),
        )
        x1, y1, x2, y2 = scale_roi(padded, width, height)
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            return True
    return False


def ocr_box_bounds(box: list[list[float]]) -> dict[str, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    x = int(min(xs))
    y = int(min(ys))
    return {
        "x": x,
        "y": y,
        "width": int(max(xs) - x),
        "height": int(max(ys) - y),
    }


def annotate_video_frame(frame: Any, result: dict[str, Any], output_path: Path) -> None:
    cv2, _np = load_cv()
    annotated = frame.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for seat in result.get("seats", []):
        x = int(seat["screen"]["x"])
        y = int(seat["screen"]["y"])
        color = (90, 220, 90) if seat["has_cards"] else (120, 120, 120)
        if seat["index"] == 0:
            color = (255, 180, 60)
        if seat["index"] == result.get("dealer", {}).get("seat_index"):
            color = (80, 240, 255)
        label = f"{seat['index']} {seat['position']} P{seat['preflop_action_order']}"
        if seat.get("bet_bb") is not None:
            label += f" bet {seat['bet_bb']:g}"
        cv2.circle(annotated, (x, y), 12, color, 2)
        cv2.putText(annotated, label, (x + 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    button = result.get("dealer_button")
    if button:
        box = button["box"]
        cv2.rectangle(
            annotated,
            (box["x"], box["y"]),
            (box["x"] + box["width"], box["y"] + box["height"]),
            (255, 255, 255),
            2,
        )
    cards = result.get("cards", {})
    hero_cards = " ".join(cards.get("hero", [])) or "?"
    board_cards = " ".join(cards.get("board", [])) or "-"
    cv2.putText(annotated, f"H: {hero_cards}", (360, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"B: {board_cards}", (360, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 120), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), annotated)


def write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "timestamp_sec",
        "frame_index",
        "ok",
        "dealer_seat",
        "hero_distance_from_dealer",
        "hero_preflop_order",
        "hero_postflop_order",
        "hero_status",
        "pot_bb",
        "hero_cards",
        "board_cards",
        "bets",
        "folded_or_empty_seats",
        "frame_path",
        "annotated_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in payload["frames"]:
            if not frame.get("ok"):
                writer.writerow(
                    {
                        "timestamp_sec": frame.get("timestamp_sec"),
                        "frame_index": frame.get("frame_index"),
                        "ok": False,
                        "frame_path": frame.get("frame_path", ""),
                        "annotated_path": frame.get("annotated_path", ""),
                    }
                )
                continue
            bets = "; ".join(
                f"{seat['name']}={seat['bet_bb']:g}BB"
                for seat in frame["seats"]
                if seat.get("bet_bb") is not None
            )
            folded = "; ".join(
                seat["name"]
                for seat in frame["seats"]
                if seat.get("status") == "folded_or_empty"
            )
            hero = frame["hero"]
            cards = frame.get("cards", {})
            writer.writerow(
                {
                    "timestamp_sec": frame["timestamp_sec"],
                    "frame_index": frame["frame_index"],
                    "ok": True,
                    "dealer_seat": frame["dealer"]["seat"],
                    "hero_distance_from_dealer": hero["distance_from_dealer_clockwise"],
                    "hero_preflop_order": hero["preflop_action_order"],
                    "hero_postflop_order": hero["postflop_action_order"],
                    "hero_status": hero["status"],
                    "pot_bb": frame.get("pot", {}).get("amount_bb") if frame.get("pot") else "",
                    "hero_cards": " ".join(cards.get("hero", [])),
                    "board_cards": " ".join(cards.get("board", [])),
                    "bets": bets,
                    "folded_or_empty_seats": folded,
                    "frame_path": frame.get("frame_path", ""),
                    "annotated_path": frame.get("annotated_path", ""),
                }
            )


def sample_times(start_sec: float, end_sec: float, every_sec: float, max_frames: int | None) -> list[float]:
    every_sec = max(0.2, float(every_sec))
    times = []
    current = max(0.0, start_sec)
    end_sec = max(current, end_sec)
    while current <= end_sec + 1e-6:
        times.append(current)
        if max_frames and len(times) >= max_frames:
            break
        current += every_sec
    return times


def choose_template(template_path: Path | None) -> Path:
    if template_path:
        return Path(template_path)
    return DEFAULT_VIDEO_TEMPLATE if DEFAULT_VIDEO_TEMPLATE.exists() else FALLBACK_TEMPLATE


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np


def load_ocr() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR(
        use_angle_cls=False,
        det_limit_type="max",
        det_limit_side_len=736,
        det_model_path=None,
    )


def format_video_summary(payload: dict[str, Any], limit: int = 12) -> str:
    lines = [
        f"视频：{payload['video']}",
        f"输出目录：{payload['output_dir']}",
        f"抽帧：{payload['sample']['frame_count']} 帧，每 {payload['sample']['every_sec']} 秒一帧",
        f"JSON：{Path(payload['output_dir']) / 'analysis.json'}",
        f"CSV：{Path(payload['output_dir']) / 'summary.csv'}",
    ]
    for frame in payload["frames"][:limit]:
        if not frame.get("ok"):
            lines.append(f"{frame['timestamp_sec']:>7.1f}s 识别失败：{frame.get('error')}")
            continue
        hero = frame["hero"]
        bets = ", ".join(
            f"{seat['name']} {seat['bet_bb']:g}BB"
            for seat in frame["seats"]
            if seat.get("bet_bb") is not None
        ) or "无"
        pot = frame.get("pot") or {}
        pot_text = f"{pot['amount_bb']:g}BB" if pot.get("amount_bb") is not None else "未知"
        cards = frame.get("cards", {})
        hero_cards = " ".join(cards.get("hero", [])) or "?"
        board_cards = " ".join(cards.get("board", [])) or "-"
        lines.append(
            f"{frame['timestamp_sec']:>7.1f}s D={frame['dealer']['seat']} "
            f"你离D {hero['distance_from_dealer_clockwise']}格 "
            f"翻前第{hero['preflop_action_order']} 翻后第{hero['postflop_action_order']} "
            f"手牌：{hero_cards} 公牌：{board_cards} 底池：{pot_text} 下注：{bets}"
        )
    if len(payload["frames"]) > limit:
        lines.append(f"... 还有 {len(payload['frames']) - limit} 帧在 CSV/JSON 里")
    return "\n".join(lines)
