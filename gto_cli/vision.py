from __future__ import annotations

import math
from pathlib import Path
from typing import Any


POSITION_ORDER_BY_SEATS = {
    2: ("BTN/SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "CO"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
    7: ("BTN", "SB", "BB", "UTG", "LJ", "HJ", "CO"),
    8: ("BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO"),
    9: ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"),
    10: ("BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "LJ", "HJ", "CO"),
}

GTO_POSITIONS = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
GTO_POSITION_ALIASES = {
    "BTN/SB": "SB",
    "THIRD_BLIND": "BB",
    "UTG+1": "UTG",
    "UTG+2": "UTG",
    "MP": "UTG",
    "LJ": "HJ",
}

NAMED_8_SEATS = (
    "bottom_hero",
    "bottom_left",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
)


def analyze_table_image(
    image_path: Path,
    template_path: Path,
    seat_count: int = 8,
    min_confidence: float = 0.45,
    min_scale: float = 0.55,
    max_scale: float = 1.6,
    annotate_path: Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    template_path = Path(template_path)
    if seat_count not in POSITION_ORDER_BY_SEATS:
        raise ValueError("seat_count must be between 2 and 10")

    cv2, np = load_cv()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    button = find_dealer_button(
        image,
        template,
        min_confidence=min_confidence,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    height, width = image.shape[:2]
    seats = build_seats(width, height, seat_count)
    dealer_index = nearest_seat_index(button["center"], seats)
    positions = POSITION_ORDER_BY_SEATS[seat_count]

    for seat in seats:
        offset = (seat["index"] - dealer_index) % seat_count
        position = positions[offset]
        seat["offset_from_dealer_clockwise"] = offset
        seat["distance_from_dealer_clockwise"] = offset
        seat["preflop_action_order"] = action_order_number(offset, seat_count, street="preflop")
        seat["postflop_action_order"] = action_order_number(offset, seat_count, street="postflop")
        seat["position"] = position
        seat["gto_position"] = to_gto_position(position)

    hero = seats[0]
    dealer = seats[dealer_index]
    result = {
        "ok": True,
        "image": str(image_path),
        "template": str(template_path),
        "table": {
            "seat_count": seat_count,
            "rotation": "clockwise",
            "hero_seat_index": 0,
            "hero_seat": hero["name"],
        },
        "dealer_button": button,
        "dealer": {
            "seat_index": dealer["index"],
            "seat": dealer["name"],
            "position": dealer["position"],
        },
        "hero": {
            "seat_index": hero["index"],
            "seat": hero["name"],
            "distance_from_dealer_clockwise": hero["distance_from_dealer_clockwise"],
            "preflop_action_order": hero["preflop_action_order"],
            "postflop_action_order": hero["postflop_action_order"],
            "position": hero["position"],
            "gto_position": hero["gto_position"],
            "offset_from_dealer_clockwise": hero["offset_from_dealer_clockwise"],
            "action_order_note": "Assumes every seat is still active; skip folded seats in real play.",
        },
        "seats": seats,
    }

    if annotate_path:
        annotate_table_image(image, result, Path(annotate_path))
        result["annotation"] = str(annotate_path)

    return result


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np


def find_dealer_button(
    image: Any,
    template: Any,
    min_confidence: float = 0.45,
    min_scale: float = 0.55,
    max_scale: float = 1.6,
) -> dict[str, Any]:
    cv2, np = load_cv()
    fallback = find_dealer_button_component(image)
    if fallback is not None and float(fallback.get("adjusted_score", 0.0)) >= 0.95:
        if float(fallback.get("confidence", 0.0)) >= min_confidence:
            return fallback
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best: dict[str, Any] | None = None
    scale_low = min(float(min_scale), float(max_scale))
    scale_high = max(float(min_scale), float(max_scale))
    for scale in np.linspace(scale_low, scale_high, 32):
        width = max(8, int(template_gray.shape[1] * float(scale)))
        height = max(8, int(template_gray.shape[0] * float(scale)))
        if width >= image_gray.shape[1] or height >= image_gray.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        scaled = cv2.resize(template_gray, (width, height), interpolation=interpolation)
        scores = cv2.matchTemplate(image_gray, scaled, cv2.TM_CCOEFF_NORMED)
        flat_scores = scores.ravel()
        top_count = min(8, flat_scores.size)
        top_indexes = np.argpartition(flat_scores, -top_count)[-top_count:]
        for flat_index in top_indexes:
            y, x = np.unravel_index(flat_index, scores.shape)
            score = float(flat_scores[flat_index])
            center_y = y + height / 2
            if center_y < image_gray.shape[0] * 0.07 or center_y > image_gray.shape[0] * 0.88:
                continue
            metrics = button_visual_metrics(image, int(x), int(y), width, height)
            if metrics["white_ratio"] < 0.08 or metrics["red_ratio"] > 0.12:
                continue
            adjusted_score = score + metrics["white_ratio"] * 0.2 - metrics["red_ratio"] * 0.3
            if best is None or adjusted_score > best["adjusted_score"]:
                best = {
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                    "center": {
                        "x": round(x + width / 2, 1),
                        "y": round(y + height / 2, 1),
                    },
                    "confidence": round(score, 4),
                    "adjusted_score": round(adjusted_score, 4),
                    "scale": round(float(scale), 3),
                    "white_ratio": round(metrics["white_ratio"], 4),
                    "red_ratio": round(metrics["red_ratio"], 4),
                }

    if best is None:
        best = fallback
    elif fallback is not None:
        fallback_score = float(fallback.get("adjusted_score", fallback.get("confidence", 0.0)))
        template_score = float(best.get("adjusted_score", best.get("confidence", 0.0)))
        if best["center"]["y"] > image_gray.shape[0] * 0.84 or fallback_score > template_score + 0.15:
            best = fallback
    if best is None:
        raise ValueError("dealer button template could not be matched")
    if best["confidence"] < min_confidence:
        raise ValueError(
            f"dealer button confidence {best['confidence']:.3f} is below threshold {min_confidence:.3f}"
        )
    return best


def button_visual_metrics(image: Any, x: int, y: int, width: int, height: int) -> dict[str, float]:
    cv2, _np = load_cv()
    patch = image[y : y + height, x : x + width]
    if patch.size == 0:
        return {"white_ratio": 0.0, "red_ratio": 1.0}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white_like = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 145)
    red_like = ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)
    return {
        "white_ratio": float(white_like.mean()),
        "red_ratio": float(red_like.mean()),
    }


def find_dealer_button_component(image: Any) -> dict[str, Any] | None:
    cv2, np = load_cv()
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 135)).astype("uint8") * 255
    mask[: int(height * 0.12), :] = 0
    mask[int(height * 0.88) :, :] = 0
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    table_center = {"x": width * 0.5, "y": height * 0.51}
    target_radius = min(width, height) * 0.30
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 280 or area > 1700:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 22 or w > 58 or h > 58:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.72 or aspect > 1.42:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        if circularity < 0.62:
            continue
        patch = image[y : y + h, x : x + w]
        if patch.size == 0:
            continue
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        dark_ratio = float(((patch_hsv[:, :, 2] < 125) & (patch_hsv[:, :, 1] < 180)).mean())
        if dark_ratio < 0.12 or dark_ratio > 0.58:
            continue
        center_x = x + w / 2
        center_y = y + h / 2
        distance_to_table = math.hypot(center_x - table_center["x"], center_y - table_center["y"])
        radius_penalty = abs(distance_to_table - target_radius) / max(target_radius, 1.0)
        score = circularity + dark_ratio * 0.6 - radius_penalty * 0.35 + min(area / 900.0, 1.0) * 0.2
        candidates.append(
            {
                "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
                "confidence": round(float(max(0.35, min(0.95, score))), 4),
                "adjusted_score": round(float(score), 4),
                "scale": 0.0,
                "white_ratio": round(float(area / max(w * h, 1)), 4),
                "red_ratio": 0.0,
                "_score": score,
                "method": "component",
            }
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["_score"])
    best.pop("_score", None)
    return best


def build_seats(width: int, height: int, seat_count: int) -> list[dict[str, Any]]:
    center_x = width * 0.5
    center_y = height * 0.51
    radius_x = width * 0.39
    radius_y = height * 0.35
    seats = []
    for index in range(seat_count):
        angle = math.radians(90 + index * (360 / seat_count))
        x = center_x + math.cos(angle) * radius_x
        y = center_y + math.sin(angle) * radius_y
        seats.append(
            {
                "index": index,
                "name": seat_name(index, seat_count),
                "screen": {"x": round(x, 1), "y": round(y, 1)},
            }
        )
    return seats


def seat_name(index: int, seat_count: int) -> str:
    if seat_count == 8:
        return NAMED_8_SEATS[index]
    if index == 0:
        return "bottom_hero"
    return f"seat_{index}_clockwise_from_hero"


def nearest_seat_index(point: dict[str, float], seats: list[dict[str, Any]]) -> int:
    point_x = float(point["x"])
    point_y = float(point["y"])
    return min(
        (seat["index"] for seat in seats),
        key=lambda index: squared_distance(point_x, point_y, seats[index]["screen"]),
    )


def squared_distance(x: float, y: float, point: dict[str, float]) -> float:
    return (x - float(point["x"])) ** 2 + (y - float(point["y"])) ** 2


def action_order_number(distance_from_dealer: int, seat_count: int, street: str) -> int:
    offsets = action_offsets(seat_count, street)
    return offsets.index(distance_from_dealer) + 1


def action_offsets(seat_count: int, street: str) -> list[int]:
    if seat_count == 2:
        if street == "preflop":
            return [0, 1]
        return [1, 0]
    if street == "preflop":
        return [*range(3, seat_count), 0, 1, 2]
    return [*range(1, seat_count), 0]


def to_gto_position(position: str) -> str:
    if position in GTO_POSITIONS:
        return position
    return GTO_POSITION_ALIASES.get(position, "UTG")


def annotate_table_image(image: Any, result: dict[str, Any], output_path: Path) -> None:
    cv2, _np = load_cv()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()

    for seat in result["seats"]:
        x = int(round(seat["screen"]["x"]))
        y = int(round(seat["screen"]["y"]))
        color = (80, 210, 255) if seat["index"] == 0 else (120, 180, 255)
        if seat["index"] == result["dealer"]["seat_index"]:
            color = (60, 220, 90)
        cv2.circle(annotated, (x, y), 12, color, 2)
        label = f"{seat['index']} {seat['position']}"
        cv2.putText(annotated, label, (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    box = result["dealer_button"]["box"]
    start = (box["x"], box["y"])
    end = (box["x"] + box["width"], box["y"] + box["height"])
    cv2.rectangle(annotated, start, end, (255, 255, 255), 2)
    center = result["dealer_button"]["center"]
    cv2.putText(
        annotated,
        f"D {result['dealer_button']['confidence']:.2f}",
        (int(center["x"]) + 10, int(center["y"]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(output_path), annotated)


def format_vision_text(result: dict[str, Any]) -> str:
    button = result["dealer_button"]
    hero = result["hero"]
    dealer = result["dealer"]
    lines = [
        f"D 按钮：({button['center']['x']}, {button['center']['y']})，置信度 {button['confidence']:.3f}",
        f"庄家：{dealer['seat']}，位置 {dealer['position']}",
        (
            f"你：{hero['seat']}，离庄家顺时针 {hero['distance_from_dealer_clockwise']} 格，"
            f"翻前第 {hero['preflop_action_order']} 个行动，翻后第 {hero['postflop_action_order']} 个行动"
        ),
        "顺时针座位：" + " -> ".join(
            f"{seat['name']}:{seat['position']}" for seat in result["seats"]
        ),
    ]
    if result.get("annotation"):
        lines.append(f"标注图：{result['annotation']}")
    return "\n".join(lines)
