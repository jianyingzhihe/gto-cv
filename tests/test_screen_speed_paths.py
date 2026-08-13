from __future__ import annotations

import numpy as np

from gto_cli.screen_vision import (
    should_refresh_dealer_button,
    should_reuse_stable_ocr,
    should_write_overlay_snapshot,
)
from gto_cli import video_vision
from gto_cli.video_vision import run_ocr_in_roi


def test_screen_vision_imports_the_shared_timing_helper() -> None:
    from gto_cli import screen_vision

    assert screen_vision.elapsed_ms(0.0) >= 0.0


def test_action_ocr_crop_preserves_full_window_coordinates() -> None:
    received_shapes: list[tuple[int, ...]] = []

    def fake_ocr(image):
        received_shapes.append(image.shape)
        return [([[1, 2], [11, 2], [11, 12], [1, 12]], "call 2BB", 0.9)], 0.0

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = run_ocr_in_roi(frame, fake_ocr, (0.25, 0.70, 1.0, 1.0))

    assert received_shapes == [(30, 150, 3)]
    assert result[0][0] == [[51.0, 72.0], [61.0, 72.0], [61.0, 82.0], [51.0, 82.0]]


def test_action_ocr_crop_keeps_call_detection_in_full_window_coordinates(monkeypatch) -> None:
    def fake_ocr(_image):
        return [([[45, 12], [105, 12], [105, 17], [45, 17]], "\u8ddf\u6ce8 2BB", 0.9)], 0.0

    monkeypatch.setattr(
        video_vision,
        "detect_bottom_action_buttons",
        lambda _frame: [{"x": 110, "y": 84, "width": 80, "height": 14, "area": 1120.0}],
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    controls = video_vision.detect_action_controls(
        frame,
        run_ocr_in_roi(frame, fake_ocr, (0.40, 0.82, 1.0, 1.0)),
    )

    assert controls["actions"] == ["fold", "call"]
    assert controls["call_amount_bb"] == 2.0


def test_dealer_refreshes_when_normal_action_panel_first_appears_not_every_frame() -> None:
    common = {
        "dealer_button_cache": {"center": {"x": 1, "y": 1}},
        "dealer_refresh_frames": 4,
        "processed_frames": 2,
        "last_dealer_refresh_frame": 1,
        "visual_diff": 0.0,
        "visual_threshold": 2.4,
        "normal_action_buttons_visible": True,
    }

    assert should_refresh_dealer_button(**common, previous_normal_action_buttons_visible=False)
    assert not should_refresh_dealer_button(**common, previous_normal_action_buttons_visible=True)


def test_dealer_refreshes_when_hero_or_board_cards_change() -> None:
    common = {
        "dealer_button_cache": {"center": {"x": 1, "y": 1}},
        "dealer_refresh_frames": 12,
        "processed_frames": 2,
        "last_dealer_refresh_frame": 1,
        "visual_diff": 0.0,
        "visual_threshold": 2.4,
        "normal_action_buttons_visible": False,
        "previous_normal_action_buttons_visible": False,
    }

    assert should_refresh_dealer_button(**common, card_regions_changed=True)
    assert not should_refresh_dealer_button(**common, card_regions_changed=False)


def test_stable_ocr_cache_is_single_use_and_requires_an_unchanged_frame() -> None:
    common = {
        "cached_ocr": [([[0, 0], [1, 0], [1, 1], [0, 1]], "2BB", 0.9)],
        "normal_action_buttons_visible": False,
        "previous_normal_action_buttons_visible": False,
    }

    assert should_reuse_stable_ocr(**common, reuse_streak=0, visual_diff=0.04)
    assert not should_reuse_stable_ocr(**common, reuse_streak=1, visual_diff=0.04)
    assert not should_reuse_stable_ocr(**common, reuse_streak=0, visual_diff=0.13)
    changed_controls = {
        **common,
        "normal_action_buttons_visible": False,
        "previous_normal_action_buttons_visible": True,
    }
    assert not should_reuse_stable_ocr(
        **changed_controls,
        reuse_streak=0,
        visual_diff=0.04,
    )


def test_stable_ocr_cache_is_disabled_when_hero_action_buttons_are_visible() -> None:
    assert not should_reuse_stable_ocr(
        cached_ocr=[([[0, 0], [1, 0], [1, 1], [0, 1]], "2BB", 0.9)],
        reuse_streak=0,
        visual_diff=0.0,
        normal_action_buttons_visible=True,
        previous_normal_action_buttons_visible=True,
    )


def test_video_analysis_uses_supplied_ocr_without_calling_the_ocr_engine(monkeypatch) -> None:
    monkeypatch.setattr(video_vision, "find_dealer_button", lambda *_args, **_kwargs: {"center": {"x": 50, "y": 50}})
    monkeypatch.setattr(video_vision, "detect_pot", lambda *_args, **_kwargs: {"amount_bb": None})
    monkeypatch.setattr(video_vision, "detect_bets", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(video_vision, "detect_action_controls", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(video_vision, "detect_visible_cards", lambda *_args, **_kwargs: {"hero": [], "board": []})
    monkeypatch.setattr(video_vision, "detect_card_statuses", lambda *_args, **_kwargs: {})

    def should_not_run(_frame):
        raise AssertionError("OCR engine should not run when a cached OCR result is supplied")

    result = video_vision.analyze_video_frame(
        np.zeros((100, 100, 3), dtype=np.uint8),
        template=None,
        ocr=should_not_run,
        ocr_result_hint=[],
    )

    assert result["timing_ms"]["ocr_ms"] == 0.0
    assert result["timing_ms"]["ocr_cached"] == 1.0


def test_overlay_png_snapshot_is_rate_limited_without_delaying_live_overlay() -> None:
    assert should_write_overlay_snapshot(0.0, float("-inf"), 2.0)
    assert not should_write_overlay_snapshot(1.9, 0.0, 2.0)
    assert should_write_overlay_snapshot(2.0, 0.0, 2.0)
    assert should_write_overlay_snapshot(0.1, 0.0, 0.0)
