from __future__ import annotations

import numpy as np

from gto_cli.live_vision import build_hero_turn
from gto_cli import video_vision


def test_stack_text_cannot_by_itself_create_a_hero_turn(monkeypatch) -> None:
    monkeypatch.setattr(video_vision, "detect_bottom_action_buttons", lambda _frame: [])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr = [
        (
            [[60, 84], [80, 84], [80, 94], [60, 94]],
            "269 BB",
            0.98,
        )
    ]

    controls = video_vision.detect_action_controls(frame, ocr)

    assert not controls["visible"]
    assert controls["actions"] == []
    assert controls["call_amount_bb"] is None


def test_full_client_capture_keeps_bottom_action_button_that_inner_table_crops() -> None:
    cv2, _np = video_vision.load_cv()
    outer = np.zeros((900, 1000, 3), dtype=np.uint8)
    cv2.rectangle(outer, (480, 820), (720, 890), (20, 20, 210), -1)

    cropped_inner_table = outer[:760, :]

    assert not video_vision.detect_action_controls(cropped_inner_table, [])["visible"]
    assert video_vision.detect_action_controls(outer, [])["visible"]


def test_manual_outer_capture_is_the_required_action_input_after_auto_inner_crop() -> None:
    """The original manual capture stays authoritative after auto bbox tightens cards."""
    cv2, _np = video_vision.load_cv()
    manual_outer_capture = np.zeros((900, 1000, 3), dtype=np.uint8)
    cv2.rectangle(manual_outer_capture, (480, 820), (720, 890), (20, 20, 210), -1)

    auto_inner_table = manual_outer_capture[30:760, 20:980]

    assert not video_vision.detect_action_controls(auto_inner_table, [])["visible"]
    assert video_vision.detect_action_controls(manual_outer_capture, [])["visible"]


def test_fold_only_surface_is_fast_fold_not_a_confirmed_hero_turn() -> None:
    hero_turn = build_hero_turn(
        {
            "visible": True,
            "actions": ["fold"],
            "red_button_regions": [{"x": 600, "y": 500, "width": 100, "height": 40}],
        }
    )

    assert not hero_turn["is_turn"]
    assert hero_turn["reason"] == "fast_fold_only"


def test_gray_call_label_next_to_red_quick_fold_is_not_a_hero_turn(monkeypatch) -> None:
    monkeypatch.setattr(
        video_vision,
        "detect_bottom_action_buttons",
        lambda _frame: [{"x": 10, "y": 84, "width": 35, "height": 14, "area": 400.0}],
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    ocr = [
        ([[15, 86], [38, 86], [38, 96], [15, 96]], "\u5feb\u901f\u5f03\u724c", 0.95),
        ([[65, 86], [105, 86], [105, 96], [65, 96]], "\u8ddf\u6ce8 2BB", 0.95),
    ]

    controls = video_vision.detect_action_controls(frame, ocr)
    hero_turn = build_hero_turn(controls)

    assert controls["actions"] == ["fold"]
    assert controls["disabled_actions"] == ["call"]
    assert not hero_turn["is_turn"]
    assert hero_turn["reason"] == "fast_fold_only"


def test_action_text_without_clickable_surface_is_not_a_hero_turn() -> None:
    hero_turn = build_hero_turn(
        {
            "visible": True,
            "actions": ["raise"],
            "red_button_regions": [],
            "bottom_texts": [{"text": "加注"}],
        }
    )

    assert not hero_turn["is_turn"]
    assert hero_turn["reason"] == "action_buttons_not_visible"


def test_pot_label_above_board_is_not_filtered_out() -> None:
    frame = np.zeros((733, 1101, 3), dtype=np.uint8)
    ocr = [
        (
            [[469, 203], [659, 203], [659, 238], [469, 238]],
            "\u5e95\u6c60: 16.8 BB",
            0.73,
        ),
        (
            [[500, 67], [620, 67], [620, 97], [500, 97]],
            "124 BB",
            0.88,
        ),
    ]

    pot = video_vision.detect_pot(frame, ocr)

    assert pot is not None
    assert pot["amount_bb"] == 16.8
