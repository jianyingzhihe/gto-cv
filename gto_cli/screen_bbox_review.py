from __future__ import annotations

from typing import Any, Callable


ACCEPT_KEYS = {10, 13, 32, ord("a"), ord("A")}
REDRAW_KEYS = {ord("r"), ord("R")}
CANCEL_KEYS = {27, ord("c"), ord("C")}


def review_bbox_interactively(
    cv2: Any,
    outer_frame: Any,
    outer_region: dict[str, int],
    proposed_region: dict[str, int] | None,
    *,
    selector: Callable[[], tuple[int, int, int, int]],
    key_reader: Callable[[Any, str], int] | None = None,
) -> dict[str, Any]:
    """Review an automatic inner bbox and allow repeated manual redraws."""
    current = dict(proposed_region) if proposed_region else None
    source = "auto_accepted" if current else "manual_adjustment"
    adjustments = 0
    read_key = key_reader or (lambda image, title: default_key_reader(cv2, image, title))

    while True:
        preview = draw_bbox_review(cv2, outer_frame, outer_region, current, source=source)
        key = int(read_key(preview, "Auto inner bbox: ENTER accept | R redraw | C cancel")) & 0xFF
        action = review_key_action(key, has_proposal=current is not None)
        if action == "accept":
            if current is None:
                continue
            return {
                "region": dict(current),
                "source": source,
                "adjustments": adjustments,
                "preview": preview,
            }
        if action == "redraw":
            left, top, width, height = selector()
            current = {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height),
            }
            source = "manual_adjustment"
            adjustments += 1
            continue
        if action == "wait":
            continue
        raise ValueError("inner bbox review canceled")


def review_key_action(key: int, *, has_proposal: bool) -> str:
    key = int(key) & 0xFF
    if key in ACCEPT_KEYS and has_proposal:
        return "accept"
    if key in REDRAW_KEYS or (key in ACCEPT_KEYS and not has_proposal):
        return "redraw"
    if key in CANCEL_KEYS:
        return "cancel"
    return "wait"


def default_key_reader(cv2: Any, image: Any, title: str) -> int:
    height, width = image.shape[:2]
    scale = min(1600 / max(width, 1), 950 / max(height, 1), 1.0)
    if scale < 1.0:
        display = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = image
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, display)
    key = cv2.waitKey(0)
    cv2.destroyWindow(title)
    return int(key)


def draw_bbox_review(
    cv2: Any,
    outer_frame: Any,
    outer_region: dict[str, int],
    inner_region: dict[str, int] | None,
    *,
    source: str,
) -> Any:
    preview = outer_frame.copy()
    height, width = preview.shape[:2]
    cv2.rectangle(preview, (2, 2), (max(2, width - 3), max(2, height - 3)), (255, 210, 30), 3)
    put_label(cv2, preview, "1 OUTER WINDOW (your large box)", 12, 30, (255, 210, 30))
    if inner_region is not None:
        box = region_inside_outer(inner_region, outer_region, width, height)
        x1, y1 = box["x"], box["y"]
        x2, y2 = x1 + box["width"], y1 + box["height"]
        color = (60, 235, 80) if source == "manual_adjustment" else (0, 255, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 4)
        label = "3 MANUAL INNER (reviewed)" if source == "manual_adjustment" else "2 AUTO INNER (proposal)"
        put_label(cv2, preview, label, x1 + 8, max(58, y1 + 30), color)
        touch_flags = inner_border_flags(box, width, height)
        if touch_flags:
            put_label(cv2, preview, f"TOUCHES OUTER: {','.join(touch_flags)}", x1 + 8, max(88, y1 + 60), (40, 40, 255))
    else:
        put_label(cv2, preview, "AUTO INNER NOT FOUND - press R and drag it", 12, 66, (40, 40, 255))
    put_label(cv2, preview, "ENTER/SPACE accept    R redraw inner box    C/ESC cancel", 12, max(28, height - 24), (245, 245, 245))
    return preview


def region_inside_outer(
    inner_region: dict[str, int],
    outer_region: dict[str, int],
    outer_width: int,
    outer_height: int,
) -> dict[str, int]:
    x1 = max(0, min(outer_width - 1, int(inner_region["left"] - outer_region["left"])))
    y1 = max(0, min(outer_height - 1, int(inner_region["top"] - outer_region["top"])))
    x2 = max(x1 + 1, min(outer_width, x1 + int(inner_region["width"])))
    y2 = max(y1 + 1, min(outer_height, y1 + int(inner_region["height"])))
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def crop_inner_from_outer(
    outer_frame: Any,
    outer_region: dict[str, int],
    inner_region: dict[str, int],
) -> Any:
    height, width = outer_frame.shape[:2]
    box = region_inside_outer(inner_region, outer_region, width, height)
    x1, y1 = box["x"], box["y"]
    return outer_frame[y1 : y1 + box["height"], x1 : x1 + box["width"]].copy()


def inner_border_flags(box: dict[str, int], width: int, height: int, tolerance: int = 3) -> list[str]:
    flags: list[str] = []
    if int(box["x"]) <= tolerance:
        flags.append("LEFT")
    if int(box["y"]) <= tolerance:
        flags.append("TOP")
    if int(box["x"]) + int(box["width"]) >= width - tolerance:
        flags.append("RIGHT")
    if int(box["y"]) + int(box["height"]) >= height - tolerance:
        flags.append("BOTTOM")
    return flags


def put_label(
    cv2: Any,
    image: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)
