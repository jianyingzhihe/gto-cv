from __future__ import annotations

import json
import unittest
from pathlib import Path

from gto_cli.video_vision import (
    card_sets_need_backup,
    detail_needs_roi_repair,
    detect_locked_hero_cards,
    hero_card_roi_variants_for_index,
    hero_card_detail_score,
    locked_profile_hero_read_boxes,
    merge_best_hero_cards_by_index,
    weak_hero_rank_window,
    wide_hero_rank_conflicts_with_existing,
)


class HeroRoiQualityTest(unittest.TestCase):
    def test_partial_overlap_crop_needs_backup(self) -> None:
        partial = make_detail(
            card="8c",
            index=0,
            roi_mode="locked_layout_overlap_merged_best",
            white_ratio=0.2786,
            face_fill=0.7641,
            face_cover=0.2646,
            face_aspect=0.6375,
            rank_confidence=0.7435,
            rank_margin=0.0968,
            suit_confidence=0.5549,
            suit_margin=0.141,
        )
        full = make_detail(
            card="Ad",
            index=1,
            roi_mode="locked_layout_auto_validated",
            white_ratio=0.7965,
            face_fill=0.8764,
            face_cover=0.8764,
            face_aspect=1.3133,
            rank_confidence=1.0,
            rank_margin=0.30,
            suit_confidence=1.0,
            suit_margin=0.30,
        )

        self.assertTrue(detail_needs_roi_repair(partial))
        self.assertFalse(detail_needs_roi_repair(full))
        self.assertTrue(card_sets_need_backup([[partial, full]]))

    def test_complete_card_scores_above_partial_overlap_crop(self) -> None:
        partial = make_detail(
            card="8c",
            index=0,
            roi_mode="locked_layout_overlap_merged_best",
            white_ratio=0.2786,
            face_fill=0.7641,
            face_cover=0.2646,
            face_aspect=0.6375,
            rank_confidence=0.7435,
            rank_margin=0.0968,
            suit_confidence=0.5549,
            suit_margin=0.141,
        )
        complete = make_detail(
            card="Jd",
            index=0,
            roi_mode="locked_layout_fixed_refresh",
            white_ratio=0.6079,
            face_fill=0.7617,
            face_cover=0.6682,
            face_aspect=2.18,
            rank_confidence=0.9999,
            rank_margin=0.25,
            suit_confidence=0.9997,
            suit_margin=0.25,
        )

        self.assertGreater(hero_card_detail_score(complete), hero_card_detail_score(partial))
        merged = merge_best_hero_cards_by_index([[partial], [complete]])
        self.assertEqual(merged[0]["card"], "Jd")

    def test_search_crop_wins_over_static_backup_for_complete_card(self) -> None:
        search = make_detail(
            card="8d",
            index=0,
            roi_mode="locked_layout_search",
            white_ratio=0.55,
            face_fill=0.87,
            face_cover=0.67,
            face_aspect=1.51,
            rank_confidence=0.70,
            rank_margin=0.14,
            suit_confidence=0.84,
            suit_margin=0.04,
        )
        fixed = make_detail(
            card="Jh",
            index=0,
            roi_mode="locked_layout_fixed_refresh",
            white_ratio=0.60,
            face_fill=0.76,
            face_cover=0.72,
            face_aspect=1.73,
            rank_confidence=0.79,
            rank_margin=0.31,
            suit_confidence=0.98,
            suit_margin=0.19,
        )

        merged = merge_best_hero_cards_by_index([[search], [fixed]])
        self.assertEqual(merged[0]["card"], "8d")

    def test_overlapped_hero_card_variants_do_not_cross_into_neighbor(self) -> None:
        left_variants = hero_card_roi_variants_for_index(0)
        right_variants = hero_card_roi_variants_for_index(1)

        self.assertTrue(left_variants)
        self.assertTrue(right_variants)
        self.assertTrue(all(dx <= 0.001 for dx, _dy, _width_scale, _height_scale in left_variants))
        self.assertTrue(all(dx >= -0.001 for dx, _dy, _width_scale, _height_scale in right_variants))

    def test_locked_profile_uses_stable_right_card_read_window(self) -> None:
        boxes = locked_profile_hero_read_boxes(
            {
                "hero_card_source": "auto_visible_cards",
                "hero_card_boxes": [
                    {"x": 0.4318, "y": 0.8082, "width": 0.0564, "height": 0.1480},
                    {"x": 0.4709, "y": 0.8082, "width": 0.0791, "height": 0.1480},
                ],
            }
        )

        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(boxes[0]["x"], 0.4318, places=4)
        self.assertAlmostEqual(boxes[1]["x"], 0.480392, places=5)
        self.assertAlmostEqual(boxes[1]["y"], 0.81264, places=5)
        self.assertAlmostEqual(boxes[1]["height"], 0.1332, places=5)

    def test_real_overlapped_ten_uses_locked_read_window(self) -> None:
        frame_path = Path(
            "video_frames/screen_live/card_samples/"
            "sample_20260729_231042_0184_1363p472s/frame.png"
        )
        profile_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/fixed_layout_profile.json"
        )
        if not frame_path.exists() or not profile_path.exists():
            self.skipTest("local live overlap regression frame is not available")

        import cv2

        frame = cv2.imread(str(frame_path))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["hero_card_source"] = "auto_visible_cards"
        cards = detect_locked_hero_cards(frame, profile)

        self.assertEqual([card["card"] for card in cards], ["Ks", "Td"])
        self.assertIn(cards[1]["roi_mode"], {"locked_profile_anchor", "locked_profile_raw"})
        self.assertFalse(detail_needs_roi_repair(cards[1]))

    def test_real_single_character_h2_falls_back_to_raw_profile_box(self) -> None:
        frame_path = Path(
            "video_frames/screen_live/card_samples/"
            "sample_20260729_231042_0011_0066p329s/frame.png"
        )
        profile_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/fixed_layout_profile.json"
        )
        if not frame_path.exists() or not profile_path.exists():
            self.skipTest("local single-character H2 regression frame is not available")

        import cv2

        frame = cv2.imread(str(frame_path))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["hero_card_source"] = "auto_visible_cards"
        cards = detect_locked_hero_cards(frame, profile)

        self.assertEqual([card["card"] for card in cards], ["5h", "4d"])
        self.assertEqual(cards[1]["roi_mode"], "locked_profile_raw")

    def test_real_clean_raw_h2_beats_complete_shifted_j_misread(self) -> None:
        frame_path = Path(
            "video_frames/screen_live/card_samples/"
            "sample_20260729_231042_0017_0151p481s/frame.png"
        )
        profile_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/fixed_layout_profile.json"
        )
        if not frame_path.exists() or not profile_path.exists():
            self.skipTest("local raw-five versus shifted-j frame is not available")

        import cv2

        frame = cv2.imread(str(frame_path))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["hero_card_source"] = "auto_visible_cards"
        cards = detect_locked_hero_cards(frame, profile)

        self.assertEqual([card["card"] for card in cards], ["Ah", "5h"])
        self.assertEqual(cards[1]["roi_mode"], "locked_profile_raw")
        self.assertEqual(cards[1]["rank_source"], "clean_corner")

    def test_low_margin_wide_rank_window_is_rejected_for_hero_cards(self) -> None:
        self.assertTrue(
            weak_hero_rank_window(
                source="hero",
                rank_width=64,
                rank_score=0.86,
                next_rank_score=0.855,
            )
        )
        self.assertFalse(
            weak_hero_rank_window(
                source="hero",
                rank_width=64,
                rank_score=0.96,
                next_rank_score=0.82,
            )
        )
        self.assertFalse(
            weak_hero_rank_window(
                source="hero",
                rank_width=64,
                rank_score=0.9213,
                next_rank_score=0.8680,
            )
        )

    def test_wide_rank_window_loses_to_existing_strong_narrow_candidate(self) -> None:
        best = (0.92, "A", 0.889, 0.662)

        self.assertTrue(
            wide_hero_rank_conflicts_with_existing(
                source="hero",
                rank_width=64,
                rank="6",
                rank_score=0.987,
                next_rank_score=0.789,
                best=best,
            )
        )
        self.assertFalse(
            wide_hero_rank_conflicts_with_existing(
                source="hero",
                rank_width=64,
                rank="6",
                rank_score=0.995,
                next_rank_score=0.60,
                best=best,
            )
        )


def make_detail(**kwargs: object) -> dict[str, object]:
    return {
        "source": "hero",
        "rank": str(kwargs.get("card", "?"))[:1],
        "suit": str(kwargs.get("card", "??"))[1:2],
        **kwargs,
    }


if __name__ == "__main__":
    unittest.main()
