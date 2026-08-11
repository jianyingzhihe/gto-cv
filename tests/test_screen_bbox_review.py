from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.screen_bbox_review import (
    crop_inner_from_outer,
    draw_bbox_review,
    region_inside_outer,
    review_bbox_interactively,
    review_key_action,
)
from gto_cli.screen_vision import build_reviewed_bbox_commands, write_reviewed_bbox_commands
from gto_cli.video_vision import load_cv


class ScreenBboxReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cv2, _np = load_cv()
        self.outer = {"left": 100, "top": 50, "width": 700, "height": 500}
        self.frame = np.full((500, 700, 3), 24, dtype=np.uint8)
        self.proposal = {"left": 140, "top": 80, "width": 620, "height": 430}

    def test_review_key_actions(self) -> None:
        self.assertEqual(review_key_action(13, has_proposal=True), "accept")
        self.assertEqual(review_key_action(32, has_proposal=False), "redraw")
        self.assertEqual(review_key_action(ord("R"), has_proposal=True), "redraw")
        self.assertEqual(review_key_action(27, has_proposal=True), "cancel")
        self.assertEqual(review_key_action(ord("x"), has_proposal=True), "wait")

    def test_accepts_automatic_proposal(self) -> None:
        result = review_bbox_interactively(
            self.cv2,
            self.frame,
            self.outer,
            self.proposal,
            selector=lambda: self.fail("selector should not be called"),
            key_reader=lambda _image, _title: 13,
        )
        self.assertEqual(result["region"], self.proposal)
        self.assertEqual(result["source"], "auto_accepted")
        self.assertEqual(result["adjustments"], 0)

    def test_redraws_then_accepts_manual_region(self) -> None:
        keys = iter((ord("R"), 32))
        manual = (125, 65, 650, 460)
        result = review_bbox_interactively(
            self.cv2,
            self.frame,
            self.outer,
            self.proposal,
            selector=lambda: manual,
            key_reader=lambda _image, _title: next(keys),
        )
        self.assertEqual(result["region"], {"left": 125, "top": 65, "width": 650, "height": 460})
        self.assertEqual(result["source"], "manual_adjustment")
        self.assertEqual(result["adjustments"], 1)

    def test_relative_box_and_crop_use_absolute_screen_coordinates(self) -> None:
        relative = region_inside_outer(self.proposal, self.outer, 700, 500)
        self.assertEqual(relative, {"x": 40, "y": 30, "width": 620, "height": 430})
        crop = crop_inner_from_outer(self.frame, self.outer, self.proposal)
        self.assertEqual(crop.shape, (430, 620, 3))

    def test_review_preview_draws_boxes(self) -> None:
        preview = draw_bbox_review(self.cv2, self.frame, self.outer, self.proposal, source="auto_accepted")
        self.assertEqual(preview.shape, self.frame.shape)
        self.assertGreater(int(np.abs(preview.astype(np.int16) - self.frame.astype(np.int16)).sum()), 0)

    def test_reviewed_commands_use_manual_full_window_and_never_enable_auto_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bbox_path = output / "analysis_bbox.json"
            manual_bbox_path = output / "bbox.json"
            commands = build_reviewed_bbox_commands(
                analysis_bbox_path=bbox_path,
                calibration_output_dir=output,
                hero_name="于寻欢",
                effective_stack_bb=100,
                villain_profile="standard",
                min_confidence=0.35,
                ocr_scale=0.65,
                dealer_refresh_frames=4,
            )
            files = write_reviewed_bbox_commands(output, commands)

            for command in commands.values():
                self.assertIn(f'--bbox-file "{manual_bbox_path}"', command)
                self.assertNotIn("--auto-bbox", command)
                self.assertNotIn("--auto-bbox-refresh", command)
            self.assertIn("--show-overlay", commands["overlay"])
            self.assertIn("--pick-hero-cards", commands["pick_hero"])
            self.assertIn("--preflight-once", commands["preflight"])
            self.assertTrue(all(Path(path).exists() for path in files.values()))


if __name__ == "__main__":
    unittest.main()
