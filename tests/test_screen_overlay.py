from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gto_cli.screen_overlay import (
    apply_manual_hero_profile,
    box_touches_border,
    diagnostic_status_lines,
    load_manual_hero_profile,
    render_diagnostic_frame,
    render_full_window_diagnostic_frame,
    seat_state_entries,
    select_manual_hero_profile,
)
from gto_cli.screen_vision import card_signature_rois, format_screen_summary, manual_hero_live_bbox_arg
from gto_cli.video_vision import detect_locked_hero_cards, load_cv


class ScreenOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manual_profile = {
            "version": 1,
            "coordinate_space": "analysis_frame_normalized",
            "reference_region": {"left": 100, "top": 50, "width": 1000, "height": 700},
            "frame_size": {"width": 1000, "height": 700},
            "hero_card_boxes": [
                {"x": 0.43, "y": 0.72, "width": 0.08, "height": 0.20},
                {"x": 0.50, "y": 0.72, "width": 0.10, "height": 0.20},
            ],
            "hero_search_box": {"x": 0.39, "y": 0.68, "width": 0.25, "height": 0.28},
        }

    def test_manual_hero_live_command_prefers_manual_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp)
            reviewed = calibration / "analysis_bbox.json"
            manual = calibration / "bbox.json"
            reviewed.write_text('{"left": 1, "top": 2, "width": 3, "height": 4}', encoding="utf-8")
            manual.write_text('{"left": 10, "top": 20, "width": 30, "height": 40}', encoding="utf-8")
            self.assertEqual(
                manual_hero_live_bbox_arg(calibration, "1,2,3,4"),
                f'--bbox-file "{manual}"',
            )

    def test_manual_hero_live_command_falls_back_without_reviewed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                manual_hero_live_bbox_arg(Path(tmp), "1,2,3,4"),
                '--bbox "1,2,3,4"',
            )

    def test_manual_profile_round_trip_and_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hero_card_rois.json"
            path.write_text(json.dumps(self.manual_profile), encoding="utf-8")
            loaded = load_manual_hero_profile(path)

        applied = apply_manual_hero_profile(
            {"id": "auto", "hero_card_source": "auto_visible_cards"},
            loaded,
            (840, 1200, 3),
        )
        self.assertEqual(applied["method"], "manual_hero_cards")
        self.assertEqual(applied["hero_card_source"], "manual_hero_cards")
        self.assertEqual(applied["hero_card_boxes"], self.manual_profile["hero_card_boxes"])
        self.assertEqual(applied["frame_size"], {"width": 1200, "height": 840})

    def test_manual_picker_saves_two_normalized_boxes_and_previews(self) -> None:
        cv2, _np = load_cv()

        class CvProxy:
            def __init__(self, real: object):
                self.real = real
                self.rois = iter(((100, 400, 80, 140), (180, 400, 100, 140)))

            def __getattr__(self, name: str) -> object:
                return getattr(self.real, name)

            def selectROI(self, *_args: object, **_kwargs: object) -> tuple[int, int, int, int]:
                return next(self.rois)

            def destroyWindow(self, _name: str) -> None:
                return None

        frame = np.full((700, 1000, 3), 230, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            payload = select_manual_hero_profile(
                CvProxy(cv2),
                frame,
                {"left": 50, "top": 70, "width": 1000, "height": 700},
                Path(tmp),
            )
            self.assertEqual(payload["hero_card_boxes"][0], {"x": 0.1, "y": 4 / 7, "width": 0.08, "height": 0.2})
            self.assertEqual(payload["hero_card_boxes"][1], {"x": 0.18, "y": 4 / 7, "width": 0.1, "height": 0.2})
            self.assertTrue(Path(payload["files"]["profile"]).exists())
            self.assertTrue(Path(payload["files"]["preview"]).exists())
            self.assertEqual(len(payload["files"]["crops"]), 2)

    def test_manual_layout_uses_exact_selected_boxes(self) -> None:
        frame = np.full((700, 1000, 3), 240, dtype=np.uint8)
        profile = apply_manual_hero_profile({}, self.manual_profile, frame.shape)

        def fake_recognize(_crop: object, source: str, index: int) -> dict[str, object]:
            return {
                "card": ("As", "Jd")[index],
                "rank": ("A", "J")[index],
                "suit": ("s", "d")[index],
                "source": source,
                "index": index,
            }

        with patch("gto_cli.video_vision.recognize_card_crop", side_effect=fake_recognize):
            cards = detect_locked_hero_cards(frame, profile)

        self.assertEqual([card["card"] for card in cards], ["As", "Jd"])
        self.assertEqual([card["roi_mode"] for card in cards], ["manual_hero_card", "manual_hero_card"])
        self.assertEqual(cards[0]["roi_box"], {"x": 430, "y": 504, "width": 80, "height": 140})
        self.assertEqual(cards[1]["roi_box"], {"x": 500, "y": 504, "width": 100, "height": 140})

    def test_card_signature_tracks_manual_boxes(self) -> None:
        profile = apply_manual_hero_profile({}, self.manual_profile, (700, 1000, 3))
        rois = card_signature_rois(profile)
        self.assertEqual(rois[0], (0.43, 0.72, 0.51, 0.9199999999999999))
        self.assertEqual(rois[1], (0.5, 0.72, 0.6, 0.9199999999999999))
        self.assertGreater(len(rois), 2)

    def test_clipped_card_box_is_reported(self) -> None:
        self.assertTrue(box_touches_border({"x": 460, "y": 530, "width": 64, "height": 83}, 985, 613))
        self.assertFalse(box_touches_border({"x": 430, "y": 500, "width": 80, "height": 100}, 985, 700))

    def test_render_diagnostic_frame_draws_manual_and_detected_boxes(self) -> None:
        cv2, _np = load_cv()
        frame = np.full((700, 1000, 3), 32, dtype=np.uint8)
        profile = apply_manual_hero_profile({}, self.manual_profile, frame.shape)
        result = {
            "cards": {
                "hero": ["As", "Jd"],
                "board": [],
                "hero_details": [
                    {
                        "card": "As",
                        "rank_confidence": 0.9,
                        "rank_margin": 0.2,
                        "suit_confidence": 0.9,
                        "suit_margin": 0.2,
                        "roi_box": {"x": 430, "y": 504, "width": 80, "height": 140},
                    }
                ],
                "board_details": [],
            }
        }
        state = {"ok": True, "table": {"street": "preflop"}, "source": {}}
        rendered = render_diagnostic_frame(cv2, frame, result, state, profile)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(np.abs(rendered.astype(np.int16) - frame.astype(np.int16)).sum()), 0)

    def test_full_window_diagnostic_keeps_outer_capture_and_inner_overlay(self) -> None:
        cv2, _np = load_cv()
        outer = np.full((800, 1200, 3), 30, dtype=np.uint8)
        inner = np.full((500, 900, 3), 80, dtype=np.uint8)
        rendered = render_full_window_diagnostic_frame(
            cv2,
            outer,
            {"left": 100, "top": 200, "width": 1200, "height": 800},
            {"left": 240, "top": 350, "width": 900, "height": 500},
            inner,
        )
        self.assertEqual(rendered.shape, outer.shape)
        self.assertTrue(np.array_equal(rendered[300, 300], inner[150, 160]))
        self.assertGreater(int(np.abs(rendered.astype(np.int16) - outer.astype(np.int16)).sum()), 0)

    def test_overlay_reports_each_seat_and_preflop_actions(self) -> None:
        result = {
            "cards": {"hero": ["As", "Kd"], "board": []},
            "seats": [
                {
                    "index": 0,
                    "name": "bottom_hero",
                    "screen": {"x": 500, "y": 610},
                    "position": "CO",
                    "preflop_action_order": 5,
                    "status": "active_or_showdown",
                },
                {
                    "index": 1,
                    "name": "left",
                    "screen": {"x": 120, "y": 370},
                    "position": "UTG",
                    "preflop_action_order": 1,
                    "status": "folded_or_empty",
                },
            ],
        }
        state = {
            "ok": True,
            "table": {"street": "preflop", "pot_bb": 5.4, "dealer_position": "BTN"},
            "hero": {"seat_index": 0, "position": "CO"},
            "hero_turn": {"is_turn": True},
            "confidence": {"pot_ocr": 0.81},
            "seats": [
                {
                    "seat_index": 0,
                    "seat": "bottom_hero",
                    "position": "CO",
                    "preflop_action_order": 5,
                    "status": "active_or_showdown",
                    "bet_bb": 0.0,
                },
                {
                    "seat_index": 1,
                    "seat": "left",
                    "position": "UTG",
                    "preflop_action_order": 1,
                    "status": "folded_or_empty",
                    "bet_bb": 2.0,
                },
            ],
            "preflop": {
                "action_history": [
                    {"seat": "left", "position": "UTG", "action": "raise", "amount_bb": 2.0}
                ]
            },
            "action_controls": {"actions": ["fold", "call", "raise"], "disabled_actions": []},
        }

        lines = diagnostic_status_lines(state, result, {}, (700, 1000, 3))
        entries = seat_state_entries(state, result, (700, 1000, 3))

        self.assertTrue(any("pot=5.4BB" in line and "turn=YES" in line for line in lines))
        self.assertTrue(any("UTG RAISE 2BB" in line for line in lines))
        self.assertTrue(any(entry["text"].startswith("HERO CO #5 TURN BET 0BB") for entry in entries))
        self.assertTrue(any("UTG #1 NO-CARD BET 2BB RAISE 2BB" in entry["text"] for entry in entries))

    def test_manual_picker_summary_prints_saved_command(self) -> None:
        payload = {
            "hero_cards_file": "calibrate/hero_card_rois.json",
            "preview": "calibrate/hero_card_rois_preview.png",
            "command": "python gto.py screen-cv --show-overlay",
            "files": {"command": "calibrate/run_live_overlay_command.txt"},
        }
        text = format_screen_summary(payload)
        self.assertIn("Saved hero card ROIs", text)
        self.assertIn("run_live_overlay_command.txt", text)
        self.assertIn("--show-overlay", text)

    def test_outer_bbox_summary_starts_overlay_without_a_review_step(self) -> None:
        payload = {
            "bbox_text": "10,20,1200,800",
            "overlay_command": "python gto.py screen-cv --show-overlay",
            "files": {
                "bbox": "calibrate/bbox.json",
                "overlay_command": "calibrate/run_live_overlay_command.txt",
            },
        }

        text = format_screen_summary(payload)

        self.assertIn("Start recognition and overlay now:", text)
        self.assertIn("run_live_overlay_command.txt", text)
        self.assertNotIn("review", text.lower())


if __name__ == "__main__":
    unittest.main()
