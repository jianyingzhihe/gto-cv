from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .video_vision import locked_profile_hero_read_boxes


MANUAL_HERO_PROFILE_VERSION = 1
TRANSPARENT_COLOR = "#010203"


def select_manual_hero_profile(
    cv2: Any,
    frame: Any,
    region: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    """Let the user select the two visible hero-card faces on one capture."""
    height, width = frame.shape[:2]
    scale = min(1600 / max(width, 1), 950 / max(height, 1), 1.0)
    if scale < 1.0:
        display_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        base_display = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    else:
        base_display = frame.copy()

    selected: list[dict[str, int]] = []
    for slot in range(2):
        display = base_display.copy()
        for previous_slot, box in enumerate(selected):
            x1 = int(round(box["x"] * scale))
            y1 = int(round(box["y"] * scale))
            x2 = int(round((box["x"] + box["width"]) * scale))
            y2 = int(round((box["y"] + box["height"]) * scale))
            cv2.rectangle(display, (x1, y1), (x2, y2), (70, 230, 70), 2)
            cv2.putText(
                display,
                f"H{previous_slot + 1}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (70, 230, 70),
                2,
                cv2.LINE_AA,
            )
        hint = "start at H2 rank corner; exclude H1 overlap" if slot == 1 else "include readable rank and suit"
        window_name = f"Select HERO card {slot + 1}/2 ({hint}), then Enter or Space"
        roi = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        x, y, selected_width, selected_height = (int(value) for value in roi)
        if selected_width <= 0 or selected_height <= 0:
            raise ValueError(f"hero card {slot + 1} selection canceled")
        selected.append(
            clamp_box(
                {
                    "x": int(round(x / scale)),
                    "y": int(round(y / scale)),
                    "width": int(round(selected_width / scale)),
                    "height": int(round(selected_height / scale)),
                },
                width,
                height,
            )
        )

    normalized_boxes = [relative_box(box, width, height) for box in selected]
    warnings = [
        f"H{slot + 1} touches the analyzed-frame border; widen/reselect the table bbox before live use."
        for slot, box in enumerate(selected)
        if box_touches_border(box, width, height, tolerance=3)
    ]
    profile = {
        "version": MANUAL_HERO_PROFILE_VERSION,
        "coordinate_space": "analysis_frame_normalized",
        "reference_region": {
            "left": int(region["left"]),
            "top": int(region["top"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        },
        "frame_size": {"width": width, "height": height},
        "hero_card_boxes": normalized_boxes,
        "hero_search_box": expanded_union_box(normalized_boxes),
        "warnings": warnings,
        "instructions": (
            "H1 and H2 are normalized to the analyzed poker-table frame. "
            "H2 starts at its own rank corner and excludes the overlapping H1 area."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "hero_card_rois.json"
    preview_path = output_dir / "hero_card_rois_preview.png"
    crop_paths: list[str] = []
    preview = frame.copy()
    colors = ((50, 240, 80), (40, 210, 255))
    for slot, box in enumerate(selected):
        color = colors[slot]
        x1, y1 = box["x"], box["y"]
        x2, y2 = x1 + box["width"], y1 + box["height"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        cv2.putText(preview, f"H{slot + 1}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        crop_path = output_dir / f"hero_card_{slot + 1}_selected.png"
        write_png(cv2, crop_path, frame[y1:y2, x1:x2])
        crop_paths.append(str(crop_path))
    write_png(cv2, preview_path, preview)
    profile["files"] = {
        "profile": str(profile_path),
        "preview": str(preview_path),
        "crops": crop_paths,
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def load_manual_hero_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"hero cards file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    boxes = payload.get("hero_card_boxes") or []
    if int(payload.get("version") or 0) != MANUAL_HERO_PROFILE_VERSION:
        raise ValueError(f"unsupported hero cards file version: {payload.get('version')}")
    if payload.get("coordinate_space") != "analysis_frame_normalized":
        raise ValueError("hero cards file must use analysis_frame_normalized coordinates")
    if len(boxes) != 2:
        raise ValueError("hero cards file must contain exactly two hero_card_boxes")
    for index, box in enumerate(boxes):
        validate_relative_box(box, f"hero_card_boxes[{index}]")
    return payload


def apply_manual_hero_profile(
    base_profile: dict[str, Any],
    manual_profile: dict[str, Any],
    frame_shape: tuple[int, ...],
) -> dict[str, Any]:
    height, width = frame_shape[:2]
    profile = copy.deepcopy(base_profile)
    boxes = [dict(box) for box in manual_profile["hero_card_boxes"]]
    profile.update(
        {
            "id": f"manual-hero-{width}x{height}",
            "method": "manual_hero_cards",
            "strict": True,
            "frame_size": {"width": width, "height": height},
            "hero_card_source": "manual_hero_cards",
            "hero_search_source": "manual_hero_cards",
            "hero_card_boxes": boxes,
            "hero_search_box": dict(manual_profile.get("hero_search_box") or expanded_union_box(boxes)),
            "manual_profile": {
                "version": manual_profile.get("version"),
                "reference_region": manual_profile.get("reference_region"),
                "files": manual_profile.get("files"),
            },
            "created_from": {
                "hero_cards": [],
                "hero_card_count": 2,
                "source": "manual_selection",
            },
        }
    )
    return profile


def render_diagnostic_frame(
    cv2: Any,
    frame: Any,
    frame_result: dict[str, Any] | None,
    state: dict[str, Any] | None,
    layout_profile: dict[str, Any] | None,
) -> Any:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    result = frame_result or {}
    cards = result.get("cards") or {}

    locked_boxes = locked_profile_hero_read_boxes(layout_profile or {})
    if locked_boxes:
        for slot, rel_box in enumerate(locked_boxes[:2]):
            box = absolute_box(rel_box, width, height)
            source = "MANUAL" if (layout_profile or {}).get("hero_card_source") == "manual_hero_cards" else "LOCK"
            draw_box(
                cv2,
                annotated,
                box,
                (70, 230, 70),
                f"{source} H{slot + 1}",
                dashed=True,
                label_row=slot + 2,
            )

    for slot, detail in enumerate(cards.get("hero_details") or []):
        box = detail.get("roi_box") or {}
        flags = card_detail_flags(detail, box, width, height)
        color = (50, 70, 255) if flags else (50, 220, 255)
        label = card_box_label("H", slot, detail, flags)
        draw_box(cv2, annotated, box, color, label, label_row=slot)

    for slot, detail in enumerate(cards.get("board_details") or []):
        box = detail.get("roi_box") or {}
        flags = card_detail_flags(detail, box, width, height)
        color = (255, 100, 40) if flags else (220, 170, 60)
        draw_box(cv2, annotated, box, color, card_box_label("B", slot, detail, flags), label_row=slot % 2)

    dealer_box = ((result.get("dealer_button") or {}).get("box") or {})
    if dealer_box:
        draw_box(cv2, annotated, dealer_box, (240, 240, 240), "DEALER D")

    lines = diagnostic_status_lines(state, result, layout_profile, frame.shape)
    draw_status_panel(cv2, annotated, lines)
    draw_seat_state_badges(cv2, annotated, seat_state_entries(state, result, frame.shape))
    return annotated


def render_full_window_diagnostic_frame(
    cv2: Any,
    outer_frame: Any,
    capture_region: dict[str, int],
    analysis_region: dict[str, int],
    diagnostic_frame: Any,
) -> Any:
    """Compose the capture-safe equivalent of the live transparent overlay."""
    composed = outer_frame.copy()
    outer_height, outer_width = composed.shape[:2]
    x = int(analysis_region["left"] - capture_region["left"])
    y = int(analysis_region["top"] - capture_region["top"])
    x = max(0, min(outer_width - 1, x))
    y = max(0, min(outer_height - 1, y))
    crop_height = max(1, min(outer_height - y, diagnostic_frame.shape[0]))
    crop_width = max(1, min(outer_width - x, diagnostic_frame.shape[1]))
    composed[y : y + crop_height, x : x + crop_width] = diagnostic_frame[:crop_height, :crop_width]
    cv2.rectangle(composed, (2, 2), (outer_width - 3, outer_height - 3), (255, 215, 0), 2)
    cv2.putText(composed, "FULL CLIENT (ACTIONS)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 215, 0), 2, cv2.LINE_AA)
    cv2.rectangle(
        composed,
        (x, y),
        (min(outer_width - 2, x + crop_width), min(outer_height - 2, y + crop_height)),
        (255, 255, 0),
        3,
    )
    cv2.putText(
        composed,
        "INNER TABLE (CARDS)",
        (x + 6, max(22, y + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return composed


def render_layout_region_preview(cv2: Any, frame: Any, layout_profile: dict[str, Any]) -> Any:
    """Draw every fixed relative region after the reviewed inner bbox is locked."""
    preview = frame.copy()
    height, width = preview.shape[:2]
    for slot, box in enumerate(layout_profile.get("hero_card_boxes") or []):
        draw_box(cv2, preview, absolute_box(box, width, height), (70, 230, 70), f"H{slot + 1}", label_row=slot)
    for slot, box in enumerate(layout_profile.get("board_card_boxes") or []):
        draw_box(cv2, preview, absolute_box(box, width, height), (230, 170, 50), f"B{slot + 1}", label_row=slot % 2)
    pot_box = layout_profile.get("pot_search_box") or {}
    if pot_box:
        draw_box(cv2, preview, absolute_box(pot_box, width, height), (0, 220, 255), "POT OCR")
    action_box = layout_profile.get("action_controls_search_box") or {}
    if action_box:
        draw_box(cv2, preview, absolute_box(action_box, width, height), (60, 70, 255), "ACTION CONTROLS")
    for index, box in (layout_profile.get("seat_stack_boxes") or {}).items():
        draw_box(cv2, preview, absolute_box(box, width, height), (160, 160, 160), f"S{index}")
    for index, anchor in (layout_profile.get("bet_anchors") or {}).items():
        x = int(round(float(anchor.get("x", 0.0)) * width))
        y = int(round(float(anchor.get("y", 0.0)) * height))
        cv2.circle(preview, (x, y), 10, (0, 145, 255), 2)
        cv2.putText(preview, f"BET{index}", (x + 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 145, 255), 1, cv2.LINE_AA)
    draw_status_panel(
        cv2,
        preview,
        [
            "LOCKED INNER BBOX: all fixed ROIs are relative to this frame",
            "H=hero  B=board  POT/ACTION=OCR regions  S=seat stack  BET=bet anchor",
            "If only H1/H2 are off, run the generated hero-card picker command.",
        ],
    )
    return preview


class LivePokerOverlay:
    """Transparent, click-through Windows overlay for live CV diagnostics."""

    def __init__(self, search_region: dict[str, int]):
        enable_windows_dpi_awareness()
        import tkinter as tk

        self._tk = tk
        self.search_region = dict(search_region)
        self.root = tk.Tk()
        self.root.title("Poker CV diagnostic overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            self.root.attributes("-alpha", 0.82)
        left = int(search_region["left"])
        top = int(search_region["top"])
        width = int(search_region["width"])
        height = int(search_region["height"])
        geometry = f"{width}x{height}{left:+d}{top:+d}"
        self.root.geometry(geometry)
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            background=TRANSPARENT_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self.root.update()
        self._make_click_through()

    @classmethod
    def create(cls, search_region: dict[str, int]) -> tuple[LivePokerOverlay | None, str | None]:
        try:
            return cls(search_region), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    def _make_click_through(self) -> None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            tk_hwnd = int(self.root.winfo_id())
            hwnd = int(user32.GetAncestor(tk_hwnd, 2)) or tk_hwnd
            gwl_exstyle = -20
            style = int(user32.GetWindowLongW(hwnd, gwl_exstyle))
            ws_ex_layered = 0x00080000
            ws_ex_transparent = 0x00000020
            ws_ex_noactivate = 0x08000000
            ws_ex_toolwindow = 0x00000080
            user32.SetWindowLongW(
                hwnd,
                gwl_exstyle,
                style | ws_ex_layered | ws_ex_transparent | ws_ex_noactivate | ws_ex_toolwindow,
            )
            # Keep the diagnostic overlay out of MSS/meeting capture when Windows supports it.
            user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except Exception:
            pass

    def update(
        self,
        *,
        analysis_region: dict[str, int],
        frame_shape: tuple[int, ...],
        frame_result: dict[str, Any] | None,
        state: dict[str, Any] | None,
        layout_profile: dict[str, Any] | None,
    ) -> None:
        self.canvas.delete("all")
        search = self.search_region
        offset_x = int(analysis_region["left"] - search["left"])
        offset_y = int(analysis_region["top"] - search["top"])
        frame_h, frame_w = frame_shape[:2]
        self._rectangle(2, 2, int(search["width"]) - 3, int(search["height"]) - 3, "#00d7ff", 2, "SEARCH")
        self._rectangle(
            offset_x,
            offset_y,
            offset_x + int(analysis_region["width"]),
            offset_y + int(analysis_region["height"]),
            "#00ffff",
            3,
            "ANALYSIS TABLE",
        )

        if (layout_profile or {}).get("hero_card_source") == "manual_hero_cards":
            for slot, rel_box in enumerate((layout_profile or {}).get("hero_card_boxes") or []):
                box = absolute_box(rel_box, frame_w, frame_h)
                self._local_box(
                    box,
                    offset_x,
                    offset_y,
                    "#36e66b",
                    f"MANUAL H{slot + 1}",
                    dash=(7, 4),
                    label_row=slot,
                )

        result = frame_result or {}
        cards = result.get("cards") or {}
        for slot, detail in enumerate(cards.get("hero_details") or []):
            box = detail.get("roi_box") or {}
            flags = card_detail_flags(detail, box, frame_w, frame_h)
            color = "#ff4040" if flags else "#ffd43b"
            self._local_box(box, offset_x, offset_y, color, card_box_label("H", slot, detail, flags), label_row=slot)
        for slot, detail in enumerate(cards.get("board_details") or []):
            box = detail.get("roi_box") or {}
            flags = card_detail_flags(detail, box, frame_w, frame_h)
            color = "#ff5c5c" if flags else "#44aaff"
            self._local_box(
                box,
                offset_x,
                offset_y,
                color,
                card_box_label("B", slot, detail, flags),
                label_row=slot % 2,
            )
        dealer_box = ((result.get("dealer_button") or {}).get("box") or {})
        if dealer_box:
            self._local_box(dealer_box, offset_x, offset_y, "#ffffff", "DEALER D")

        lines = diagnostic_status_lines(state, result, layout_profile, frame_shape)
        y = 22
        for line in lines:
            self._text(14, y, line, "#ffffff", anchor="nw", font=("Consolas", 12, "bold"))
            y += 22
        for entry in seat_state_entries(state, result, frame_shape):
            self._text(
                offset_x + entry["x"],
                offset_y + entry["y"],
                entry["text"],
                entry["color"],
                anchor="ne" if entry["anchor"] == "right" else "nw",
                justify=entry["justify"],
                font=("Consolas", 10, "bold"),
            )
        self.root.update_idletasks()
        self.root.update()

    def _local_box(
        self,
        box: dict[str, Any],
        offset_x: int,
        offset_y: int,
        color: str,
        label: str,
        dash: tuple[int, int] | None = None,
        label_row: int = 0,
    ) -> None:
        if not box:
            return
        x1 = offset_x + int(box.get("x", 0))
        y1 = offset_y + int(box.get("y", 0))
        x2 = x1 + int(box.get("width", 0))
        y2 = y1 + int(box.get("height", 0))
        self._rectangle(x1, y1, x2, y2, color, 3, label, dash=dash, label_row=label_row)

    def _rectangle(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: str,
        width: int,
        label: str,
        dash: tuple[int, int] | None = None,
        label_row: int = 0,
    ) -> None:
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash)
        label_y = max(4, y1 - 20 - max(0, label_row) * 20)
        self._text(x1 + 4, label_y, label, color, anchor="nw", font=("Consolas", 11, "bold"))

    def _text(self, x: int, y: int, value: str, color: str, **kwargs: Any) -> None:
        shadow_kwargs = dict(kwargs)
        self.canvas.create_text(x + 2, y + 2, text=value, fill="#000000", **shadow_kwargs)
        self.canvas.create_text(x, y, text=value, fill=color, **kwargs)

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def diagnostic_status_lines(
    state: dict[str, Any] | None,
    frame_result: dict[str, Any],
    layout_profile: dict[str, Any] | None,
    frame_shape: tuple[int, ...] | None = None,
) -> list[str]:
    state = state or {}
    cards = frame_result.get("cards") or {}
    source = state.get("source") or {}
    auto_bbox = source.get("auto_bbox") or {}
    timing = source.get("cv_timing_ms") or frame_result.get("timing_ms") or {}
    hero = " ".join(str(card) for card in cards.get("hero") or []) or "-"
    board = " ".join(str(card) for card in cards.get("board") or []) or "-"
    profile_source = (layout_profile or {}).get("hero_card_source") or "auto"
    if frame_shape is not None:
        frame_height, frame_width = frame_shape[:2]
    else:
        profile_frame = (layout_profile or {}).get("frame_size") or {}
        frame_width = int(profile_frame.get("width") or 0)
        frame_height = int(profile_frame.get("height") or 0)
    lines = [
        f"CV OVERLAY  hero={hero}  board={board}",
        f"hero_roi={profile_source}  bbox={auto_bbox.get('method') or 'manual/search'}  total={timing.get('total_ms') or '-'}ms",
    ]
    if not state.get("ok", True):
        lines.append(f"ERROR: {state.get('error') or 'analysis failed'}")
    else:
        table = state.get("table") or {}
        hero_state = state.get("hero") or {}
        turn = state.get("hero_turn") or {}
        confidence = state.get("confidence") or {}
        pot = table.get("pot_bb")
        pot_text = "-" if pot is None else f"{format_amount(pot)}BB"
        pot_confidence = confidence.get("pot_ocr")
        pot_confidence_text = "-" if pot_confidence is None else f"{float(pot_confidence):.2f}"
        lines.append(
            "TABLE "
            f"street={table.get('street') or '-'} pot={pot_text} pot_ocr={pot_confidence_text} "
            f"dealer={table.get('dealer_position') or '-'} hero={hero_state.get('position') or '-'} "
            f"turn={'YES' if turn.get('is_turn') else 'NO'}"
        )
        history = preflop_history_text(state)
        if history:
            lines.append(f"PREFLOP {history}")
        visible_bets = visible_bets_text(state)
        if visible_bets:
            lines.append(f"VISIBLE BETS {visible_bets}")
        controls = state.get("action_controls") or {}
        clickable = "/".join(str(action).upper() for action in controls.get("actions") or []) or "-"
        disabled = "/".join(str(action).upper() for action in controls.get("disabled_actions") or []) or "-"
        lines.append(f"CONTROLS clickable={clickable} disabled={disabled}")
    for slot, detail in enumerate(cards.get("hero_details") or []):
        box = detail.get("roi_box") or {}
        flags = card_detail_flags(detail, box, frame_width, frame_height)
        lines.append(card_detail_label("H", slot, detail, flags))
    return lines


def seat_state_entries(
    state: dict[str, Any] | None,
    frame_result: dict[str, Any] | None,
    frame_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Return one compact, screen-anchored audit label for each detected seat."""
    state = state or {}
    result = frame_result or {}
    frame_height, frame_width = frame_shape[:2]
    state_seats = {
        int(seat.get("seat_index")): seat
        for seat in state.get("seats") or []
        if seat.get("seat_index") is not None
    }
    hero_index = (state.get("hero") or {}).get("seat_index")
    hero_turn = bool((state.get("hero_turn") or {}).get("is_turn"))
    actions_by_seat = latest_actions_by_seat(state)
    entries: list[dict[str, Any]] = []

    for raw_seat in result.get("seats") or []:
        index = raw_seat.get("index")
        if index is None:
            continue
        seat = state_seats.get(int(index), raw_seat)
        point = raw_seat.get("screen") or {}
        if point.get("x") is None or point.get("y") is None:
            continue
        is_hero = int(index) == hero_index
        position = str(seat.get("position") or raw_seat.get("position") or "?")
        order = seat.get("preflop_action_order") or raw_seat.get("preflop_action_order")
        action = actions_by_seat.get(str(seat.get("seat") or raw_seat.get("name") or ""))
        status = seat_status_text(seat, action, is_hero and hero_turn)
        bet = seat.get("bet_bb")
        if bet is None:
            bet = raw_seat.get("bet_bb")
        bet_text = "BET -" if bet is None else f"BET {format_amount(bet)}BB"
        actor = f"HERO {position}" if is_hero else position
        order_text = "?" if order is None else str(order)
        action_text = action_text_for_overlay(action)
        label_parts = [f"{actor} #{order_text}", status, bet_text]
        if action_text:
            label_parts.append(action_text)
        text = " ".join(label_parts)
        x, y, anchor, justify = seat_label_anchor(
            int(round(float(point["x"]))),
            int(round(float(point["y"]))),
            frame_width,
            frame_height,
        )
        entries.append(
            {
                "x": x,
                "y": y,
                "text": text,
                "color": seat_overlay_color(is_hero, status, action),
                "anchor": anchor,
                "justify": justify,
                "point": {"x": int(round(float(point["x"]))), "y": int(round(float(point["y"])))} ,
            }
        )
    return entries


def draw_seat_state_badges(cv2: Any, image: Any, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        point = entry["point"]
        color = hex_to_bgr(str(entry["color"]))
        cv2.circle(image, (point["x"], point["y"]), 5, color, -1, cv2.LINE_AA)
        draw_text_badge(cv2, image, int(entry["x"]), int(entry["y"]), entry["text"], color, entry["anchor"])


def draw_text_badge(
    cv2: Any,
    image: Any,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    anchor: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if anchor == "right":
        x1 = max(2, x - text_width - 8)
        x2 = min(image.shape[1] - 2, x + 2)
        text_x = x1 + 4
    else:
        x1 = max(2, x - 2)
        x2 = min(image.shape[1] - 2, x + text_width + 8)
        text_x = x1 + 4
    y1 = max(2, y - text_height - baseline - 6)
    y2 = min(image.shape[0] - 2, y + baseline + 5)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)
    cv2.putText(image, text, (text_x, y - baseline), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (text_x, y - baseline), font, font_scale, color, thickness, cv2.LINE_AA)


def latest_actions_by_seat(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    history = ((state.get("preflop") or {}).get("action_history") or [])
    latest: dict[str, dict[str, Any]] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        seat = str(item.get("seat") or "")
        if seat:
            latest[seat] = item
    return latest


def preflop_history_text(state: dict[str, Any]) -> str:
    history = ((state.get("preflop") or {}).get("action_history") or [])
    parts: list[str] = []
    for item in history[-10:]:
        if not isinstance(item, dict):
            continue
        action = action_text_for_overlay(item)
        actor = str(item.get("position") or item.get("seat") or "?")
        if action:
            parts.append(f"{actor} {action}")
    if parts:
        return " -> ".join(parts)
    tracker = state.get("preflop_tracker") or {}
    reason = tracker.get("reason") if isinstance(tracker, dict) else None
    return f"unconfirmed ({reason})" if reason else "unconfirmed"


def visible_bets_text(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for seat in state.get("seats") or []:
        amount = seat.get("bet_bb")
        if amount is None:
            continue
        position = str(seat.get("position") or seat.get("seat") or "?")
        parts.append(f"{position}:{format_amount(amount)}BB")
    return ", ".join(parts) if parts else "-"


def action_text_for_overlay(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    raw = str(action.get("action") or "").lower()
    labels = {
        "fold": "FOLD",
        "call": "CALL",
        "limp": "LIMP",
        "raise": "RAISE",
        "3bet": "3BET",
        "4bet": "4BET",
        "5bet": "5BET",
        "all_in": "ALL-IN",
        "hero_to_act": "TURN",
    }
    label = labels.get(raw, raw.upper())
    if not label:
        return ""
    amount = action.get("amount_bb")
    return label if amount is None else f"{label} {format_amount(amount)}BB"


def seat_status_text(seat: dict[str, Any], action: dict[str, Any] | None, hero_turn: bool) -> str:
    if hero_turn:
        return "TURN"
    if action and str(action.get("action") or "").lower() == "fold":
        return "FOLDED"
    if str(seat.get("status") or "") == "active_or_showdown":
        return "ACTIVE"
    return "NO-CARD"


def seat_overlay_color(is_hero: bool, status: str, action: dict[str, Any] | None) -> str:
    if is_hero:
        return "#ffd43b"
    if status in {"FOLDED", "NO-CARD"}:
        return "#a8a8a8"
    raw_action = str((action or {}).get("action") or "").lower()
    if raw_action in {"raise", "3bet", "4bet", "5bet", "all_in"}:
        return "#ff7657"
    return "#69e890"


def seat_label_anchor(x: int, y: int, width: int, height: int) -> tuple[int, int, str, str]:
    anchor = "right" if x > width * 0.56 else "left"
    justify = "right" if anchor == "right" else "left"
    if y >= height * 0.66:
        return x, max(26, y - 58), anchor, justify
    if y <= height * 0.46:
        # The diagnostic panel occupies the top-left of the inner table. Put
        # top-seat labels below it so every seat remains independently legible.
        return x, min(height - 8, y + 80), anchor, justify
    return x, max(26, min(height - 8, y - 12)), anchor, justify


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)


def format_amount(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def enable_windows_dpi_awareness() -> None:
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def card_detail_flags(detail: dict[str, Any], box: dict[str, Any], width: int, height: int) -> list[str]:
    flags: list[str] = []
    if box_touches_border(box, width, height):
        flags.append("CLIPPED")
    card = str(detail.get("card") or "")
    rank_margin = float(detail.get("rank_margin") or 0.0)
    suit_margin = float(detail.get("suit_margin") or 0.0)
    if not card or "?" in card:
        flags.append("INCOMPLETE")
    if rank_margin < 0.055:
        flags.append("LOW-RANK")
    if suit_margin < 0.04:
        flags.append("LOW-SUIT")
    return flags


def card_detail_label(prefix: str, slot: int, detail: dict[str, Any], flags: list[str]) -> str:
    rank_conf = float(detail.get("rank_confidence") or 0.0)
    rank_margin = float(detail.get("rank_margin") or 0.0)
    suit_conf = float(detail.get("suit_confidence") or 0.0)
    suit_margin = float(detail.get("suit_margin") or 0.0)
    flag_text = f" [{' '.join(flags)}]" if flags else ""
    return (
        f"{prefix}{slot + 1} {detail.get('card') or '?'} "
        f"r={rank_conf:.2f}/{rank_margin:.2f} s={suit_conf:.2f}/{suit_margin:.2f}{flag_text}"
    )


def card_box_label(prefix: str, slot: int, detail: dict[str, Any], flags: list[str]) -> str:
    flag_text = f" {'/'.join(flags)}" if flags else ""
    return f"{prefix}{slot + 1} {detail.get('card') or '?'}{flag_text}"


def box_touches_border(box: dict[str, Any], width: int, height: int, tolerance: int = 2) -> bool:
    if not box:
        return True
    x = int(box.get("x", 0))
    y = int(box.get("y", 0))
    box_width = int(box.get("width", 0))
    box_height = int(box.get("height", 0))
    return (
        x <= tolerance
        or y <= tolerance
        or x + box_width >= width - tolerance
        or y + box_height >= height - tolerance
    )


def draw_box(
    cv2: Any,
    image: Any,
    box: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    *,
    dashed: bool = False,
    label_row: int = 0,
) -> None:
    if not box:
        return
    x1 = int(box.get("x", 0))
    y1 = int(box.get("y", 0))
    x2 = x1 + int(box.get("width", 0))
    y2 = y1 + int(box.get("height", 0))
    if dashed:
        draw_dashed_rectangle(cv2, image, (x1, y1), (x2, y2), color, 2)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label_y = max(18, y1 - 6 - max(0, label_row) * 20)
    cv2.putText(image, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_dashed_rectangle(
    cv2: Any,
    image: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    dash = 9
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_status_panel(cv2: Any, image: Any, lines: list[str]) -> None:
    if not lines:
        return
    panel_width = min(image.shape[1] - 12, max(420, max(len(line) for line in lines) * 9))
    panel_height = 18 + len(lines) * 24
    overlay = image.copy()
    cv2.rectangle(overlay, (6, 6), (6 + panel_width, 6 + panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    y = 29
    for line in lines:
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 1, cv2.LINE_AA)
        y += 24


def relative_box(box: dict[str, Any], width: int, height: int) -> dict[str, float]:
    return {
        "x": float(box["x"]) / max(width, 1),
        "y": float(box["y"]) / max(height, 1),
        "width": float(box["width"]) / max(width, 1),
        "height": float(box["height"]) / max(height, 1),
    }


def absolute_box(box: dict[str, Any], width: int, height: int) -> dict[str, int]:
    return clamp_box(
        {
            "x": int(round(float(box["x"]) * width)),
            "y": int(round(float(box["y"]) * height)),
            "width": max(1, int(round(float(box["width"]) * width))),
            "height": max(1, int(round(float(box["height"]) * height))),
        },
        width,
        height,
    )


def clamp_box(box: dict[str, Any], width: int, height: int) -> dict[str, int]:
    x1 = max(0, min(width - 1, int(box.get("x", 0))))
    y1 = max(0, min(height - 1, int(box.get("y", 0))))
    x2 = max(x1 + 1, min(width, x1 + max(1, int(box.get("width", 1)))))
    y2 = max(y1 + 1, min(height, y1 + max(1, int(box.get("height", 1)))))
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def expanded_union_box(boxes: list[dict[str, Any]]) -> dict[str, float]:
    x1 = min(float(box["x"]) for box in boxes)
    y1 = min(float(box["y"]) for box in boxes)
    x2 = max(float(box["x"]) + float(box["width"]) for box in boxes)
    y2 = max(float(box["y"]) + float(box["height"]) for box in boxes)
    union_width = max(0.01, x2 - x1)
    union_height = max(0.01, y2 - y1)
    pad_x = max(0.02, union_width * 0.20)
    pad_y = max(0.02, union_height * 0.20)
    left = max(0.0, x1 - pad_x)
    top = max(0.0, y1 - pad_y)
    right = min(1.0, x2 + pad_x)
    bottom = min(1.0, y2 + pad_y)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def validate_relative_box(box: dict[str, Any], label: str) -> None:
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1.001 or y + height > 1.001:
        raise ValueError(f"invalid {label}: normalized box must stay within 0..1")


def write_png(cv2: Any, path: Path, image: Any) -> bool:
    if image is None or image.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True
