import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gto_cli.card_classifier import classify_rank_glyph
from gto_cli.cv_validate import classify_frame
from gto_cli.video_vision import (
    choose_best_suit_candidate,
    choose_aligned_king_rank_override,
    choose_red_diamond_suit_override,
    choose_red_eight_rank_override,
    choose_red_four_rank_override,
    clean_hero_suit_prediction_is_decisive,
    clean_hero_suit_prediction,
    clean_board_suit_prediction_is_decisive,
    clean_prediction_is_decisive,
    detect_board_cards_default,
    detect_visible_cards_from_layout,
    load_cv,
    merge_card_details_by_index,
    normalized_hero_suit_window,
    normalized_hero_black_suit_component,
    normalized_hero_rank_window,
    normalized_board_suit_window,
    normalized_rank_piece,
    normalized_suit_component_by_label,
    rank_candidate_windows,
    recognize_card_crop,
    remove_duplicate_hero_card_reads,
    select_consensus_hero_card_candidate,
)


class VideoVisionRankOverrideTest(unittest.TestCase):
    def test_aligned_king_votes_beat_shifted_j_fragment(self) -> None:
        crop = np.full((103, 87, 3), 255, dtype=np.uint8)
        crop[8:70, 8:22] = (0, 0, 220)

        result = choose_aligned_king_rank_override(
            crop,
            source="hero",
            rank="J",
            rank_score=0.6594,
            next_rank_score=0.5704,
            aligned_votes=[
                ("K", 0.6539, 0.6419),
                ("K", 0.6425, 0.6123),
                ("Q", 0.5998, 0.5652),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "K")
        self.assertGreaterEqual(result[0] - result[2], 0.04)

    def test_true_j_without_aligned_king_consensus_is_unchanged(self) -> None:
        crop = np.full((103, 87, 3), 255, dtype=np.uint8)
        crop[8:70, 8:22] = (0, 0, 220)

        result = choose_aligned_king_rank_override(
            crop,
            source="hero",
            rank="J",
            rank_score=0.6230,
            next_rank_score=0.4722,
            aligned_votes=[
                ("J", 0.6481, 0.5863),
                ("J", 0.6230, 0.4722),
            ],
        )

        self.assertIsNone(result)

    def test_clear_red_king_is_not_erased_by_generic_margin_gate(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0002_0037p967s/hero_slot1_item_card.png"
        )
        if not card_path.exists():
            self.skipTest("local red-king regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 1, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "Kd")
        self.assertGreaterEqual(detail["rank_confidence"], 0.63)

    def test_clean_tight_rank_candidate_corrects_runtime_five(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0011_0066p329s/hero_slot0_8h_card.png"
        )
        if not card_path.exists():
            self.skipTest("local clean-five regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "5h")
        self.assertGreaterEqual(detail["rank_confidence"], 0.82)

    def test_aligned_wide_queen_beats_shifted_eight_fragment(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0143_1044p558s/hero_slot0_8s_card.png"
        )
        if not card_path.exists():
            self.skipTest("local queen-versus-shifted-eight regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "Qs")
        self.assertGreaterEqual(detail["rank_confidence"], 0.90)

    def test_hero_queen_specialist_overrides_generic_eight_confusion(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_boardrankgate_old/samples/"
            "sample_20260729_231042_0013_0087p908s/hero_slot1_8d_card.png"
        )
        if not card_path.exists():
            self.skipTest("local hero-queen regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 1, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["rank"], "Q")
        self.assertEqual(detail["card"], "Qd")
        self.assertGreaterEqual(detail["rank_confidence"], 0.92)

    def test_hero_queen_specialist_accepts_clear_subpoint92_queen(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_boardqueen_tonight/samples/"
            "sample_20260731_223845_0002_0036p864s/hero_slot1_8d_card.png"
        )
        if not card_path.exists():
            self.skipTest("local subpoint92 hero-queen regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 1, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "Qd")
        self.assertGreaterEqual(detail["rank_confidence"], 0.84)

    def test_clean_red_three_beats_shifted_eight_votes(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0180_1338p033s/hero_slot0_8d_card.png"
        )
        if not card_path.exists():
            self.skipTest("local three-versus-eight regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "3d")
        self.assertEqual(detail["rank_source"], "clean_corner")
        self.assertGreaterEqual(detail["rank_confidence"], 0.86)

    def test_ten_rank_glyph_ignores_card_borders_and_neighbor_edge(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_samples/"
            "sample_20260729_231042_0036_0295p522s/hero_slot0_Kc_card.png"
        )
        if not card_path.exists():
            self.skipTest("local ten-border regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        glyph = normalized_rank_piece(crop[:72, :64], (54, 70))
        prediction = classify_rank_glyph(glyph)
        component_count, _labels = cv2.connectedComponents((glyph > 127).astype("uint8"), 8)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction["label"], "T")
        self.assertGreater(prediction["margin"], 0.20)
        self.assertEqual(component_count - 1, 2)

    def test_decisive_clean_ten_overrides_wrong_runtime_king(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0035_0284p567s/hero_slot0_Kc_card.png"
        )
        if not card_path.exists():
            self.skipTest("local decisive-ten regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["rank"], "T")
        self.assertGreaterEqual(detail["rank_margin"], 0.20)

    def test_board_seven_uses_fixed_font_model_instead_of_shifted_j_fragment(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0038_0315p831s/board_slot3_Js_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-seven regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 3, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["rank"], "7")
        self.assertEqual(detail["rank_source"], "clean_corner")
        self.assertGreaterEqual(detail["rank_confidence"], 0.92)

    def test_board_six_ignores_connected_left_card_border(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0054_0435p227s/board_slot2_4s_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-six regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 2, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "6s")
        self.assertEqual(detail["rank_source"], "clean_corner")
        self.assertGreaterEqual(detail["rank_confidence"], 0.92)

    def test_board_two_keeps_stronger_uninset_corner(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0003_0044p448s/board_slot0_2c_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-two regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 0, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "2c")
        self.assertEqual(detail["rank_source"], "clean_corner")
        self.assertGreaterEqual(detail["rank_confidence"], 0.92)

    def test_board_seven_prefers_decisive_inset_corner_over_shifted_fragment(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260731_tonight/samples/"
            "sample_20260731_223845_0003_0038p492s/board_slot0_9s_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-seven regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 0, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "7s")
        self.assertEqual(detail["rank_source"], "clean_corner")
        self.assertGreaterEqual(detail["rank_margin"], 0.09)

    def test_board_clean_override_requires_high_similarity(self) -> None:
        self.assertTrue(
            clean_prediction_is_decisive(
                {"score": 0.95, "margin": 0.08},
                source="board",
            )
        )
        self.assertFalse(
            clean_prediction_is_decisive(
                {"score": 0.88, "margin": 0.20},
                source="board",
            )
        )

    def test_board_clean_queen_accepts_small_eight_margin(self) -> None:
        self.assertTrue(
            clean_prediction_is_decisive(
                {"label": "Q", "score": 0.912, "margin": 0.015},
                source="board",
            )
        )
        self.assertFalse(
            clean_prediction_is_decisive(
                {"label": "Q", "score": 0.90, "margin": 0.009},
                source="board",
            )
        )

    def test_board_queen_overrides_generic_eight_when_fixed_corner_is_clean(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_qrank_old/samples/"
            "sample_20260729_231042_0047_0383p543s/board_slot0_8d_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-queen regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 0, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["rank"], "Q")
        self.assertEqual(detail["card"], "Qd")

    def test_board_three_specialist_overrides_generic_eight_with_small_margin(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_heroqueen_tonight/samples/"
            "sample_20260731_223845_0129_0953p431s/board_slot2_8h_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-three regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 2, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "3h")
        self.assertEqual(detail["rank_source"], "clean_corner")

    def test_board_club_component_excludes_rank_tail_and_neighbor_suit(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0004_0046p301s/board_slot1_Js_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-club regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 1, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "Jc")
        self.assertGreaterEqual(detail["suit_confidence"], 0.88)

    def test_board_club_clean_component_beats_low_margin_spade_template(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260731_tonight/samples/"
            "sample_20260731_223845_0003_0038p492s/board_slot2_Ts_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-club regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "board", 2, return_rejected=True)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["card"], "Tc")
        self.assertGreaterEqual(detail["suit_confidence"], 0.88)

    def test_board_suit_component_gate_accepts_high_score_clean_shape_despite_near_tie(self) -> None:
        self.assertTrue(
            clean_board_suit_prediction_is_decisive(
                {"label": "c", "score": 0.90, "margin": 0.005}
            )
        )

    def test_board_suit_component_gate_rejects_low_score_shape(self) -> None:
        self.assertFalse(
            clean_board_suit_prediction_is_decisive(
                {"label": "c", "score": 0.87, "margin": 0.20}
            )
        )

    def test_hero_club_suit_window_excludes_card_edge_and_neighbor_blob(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0050_0411p415s/hero_slot1_7_card.png"
        )
        if not card_path.exists():
            self.skipTest("local hero-club regression crop is not available")

        cv2, np_cv = load_cv()
        crop = cv2.imread(str(card_path))
        glyph = normalized_hero_suit_window(crop, (42, 42))

        self.assertIsNotNone(glyph)
        component_count, _labels = cv2.connectedComponents((glyph > 127).astype(np_cv.uint8), 8)
        self.assertEqual(component_count - 1, 1)
        self.assertEqual(int((glyph[0] > 0).sum()), 0)
        self.assertEqual(int((glyph[-1] > 0).sum()), 0)

    def test_hero_suit_debug_component_keeps_the_complete_club(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_logicfix_tonight/samples/"
            "sample_20260731_223845_0031_0249p809s/hero_slot0_7_card.png"
        )
        if not card_path.exists():
            self.skipTest("local hero-club regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        from gto_cli.screen_vision import safe_suit_debug_image
        from gto_cli.video_vision import normalized_suit_component_by_label

        debug_glyph = safe_suit_debug_image(crop, "hero")
        full_component = normalized_suit_component_by_label(crop, (42, 42), source="hero")
        self.assertEqual(debug_glyph.tolist(), full_component.tolist())

    def test_board_suit_window_prefers_fixed_large_pip_over_corner_card_edge(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_blackmodel_tonight/samples/"
            "sample_20260731_223845_0070_0512p828s/board_slot1_2s_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-club regression crop is not available")

        cv2, np_cv = load_cv()
        crop = cv2.imread(str(card_path))
        glyph = normalized_board_suit_window(crop, (42, 42))
        component_count, _labels = cv2.connectedComponents((glyph > 127).astype(np_cv.uint8), 8)

        self.assertEqual(component_count - 1, 1)
        self.assertEqual(int((glyph[:, 0] > 0).sum()), 0)

    def test_hero_suit_component_falls_back_to_clean_window_when_card_edge_merges(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_suitasset_old/samples/"
            "sample_20260729_231042_0064_0505p501s/hero_slot0_Kc_card.png"
        )
        if not card_path.exists():
            self.skipTest("local merged-edge hero-club crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        expected = normalized_hero_suit_window(crop, (42, 42))
        actual = normalized_suit_component_by_label(crop, (42, 42), source="hero")

        self.assertIsNotNone(expected)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_left_hero_spade_is_not_merged_into_the_card_rim(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_samples/"
            "sample_20260802_201231_0017_0154p461s/hero_slot0_8c_card.png"
        )
        if not card_path.exists():
            self.skipTest("local left-spade regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        # This regression is about separating the spade from the dark card
        # border.  It must not depend on whichever live layout happens to be
        # active while the test suite runs.
        self.assertEqual(detail["suit"], "s")
        self.assertGreaterEqual(detail["suit_confidence"], 0.90)

    def test_fixed_hero_rank_window_excludes_suit_and_recovers_two(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_boardfixed_tonight/samples/"
            "sample_20260731_223845_0121_0898p037s/hero_slot0_3h_card.png"
        )
        if not card_path.exists():
            self.skipTest("local hero-two regression crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        glyph = normalized_hero_rank_window(crop, (54, 70))
        prediction = classify_rank_glyph(glyph)
        detail = recognize_card_crop(crop, "hero", 0, allow_partial_hero=True)

        self.assertEqual(prediction["label"], "2")
        self.assertEqual(detail["card"], "2h")

    def test_clean_hero_suit_model_uses_same_component_as_review_asset(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260801_suitasset_old/samples/"
            "sample_20260729_231042_0064_0505p501s/hero_slot0_Kc_card.png"
        )
        if not card_path.exists():
            self.skipTest("local merged-edge hero-club crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        expected = normalized_hero_black_suit_component(crop, (42, 42))
        with patch("gto_cli.video_vision.classify_suit_glyph", return_value={"label": "c"}) as classify:
            clean_hero_suit_prediction(crop, source="hero", allowed=("s", "c"))

        self.assertEqual(classify.call_args.args[0].tolist(), expected.tolist())

    def test_clean_hero_suit_override_requires_high_similarity(self) -> None:
        self.assertTrue(clean_hero_suit_prediction_is_decisive({"score": 0.96, "margin": 0.08}))
        self.assertFalse(clean_hero_suit_prediction_is_decisive({"score": 0.90, "margin": 0.20}))

    def test_red_four_override_accepts_repeated_high_tied_votes(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)
        alternatives = [
            ("4", 0.96, 0.96),
            ("4", 0.95, 0.95),
            ("4", 0.94, 0.94),
            ("4", 0.83, 0.83),
            ("J", 0.76, 0.51),
        ]

        result = choose_red_four_rank_override(crop, "J", 0.76, 0.51, alternatives)

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "4")

    def test_red_four_override_rejects_single_tied_vote(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)
        alternatives = [
            ("4", 0.96, 0.96),
            ("J", 0.76, 0.51),
        ]

        result = choose_red_four_rank_override(crop, "J", 0.76, 0.51, alternatives)

        self.assertIsNone(result)

    def test_red_six_override_requires_multiple_strong_four_votes(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)

        weak_result = choose_red_four_rank_override(
            crop,
            "6",
            0.96,
            0.68,
            [("4", 0.88, 0.77), ("6", 0.96, 0.68)],
        )
        strong_result = choose_red_four_rank_override(
            crop,
            "6",
            0.96,
            0.68,
            [("4", 0.88, 0.77), ("4", 0.86, 0.74), ("6", 0.96, 0.68)],
        )

        self.assertIsNone(weak_result)
        self.assertIsNotNone(strong_result)
        self.assertEqual(strong_result[1], "4")

    def test_red_eight_override_accepts_tied_three_with_eight_vote(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)
        alternatives = [
            ("3", 1.0, 1.0),
            ("3", 1.0, 1.0),
            ("3", 1.0, 1.0),
            ("8", 0.84, 0.81),
        ]

        result = choose_red_eight_rank_override(crop, "3", 1.0, 0.68, alternatives)

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "8")

    def test_red_eight_override_accepts_single_tied_three_with_strong_eight_vote(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)
        alternatives = [
            ("3", 1.0, 1.0),
            ("3", 0.89, 0.68),
            ("8", 0.845, 0.817),
        ]

        result = choose_red_eight_rank_override(crop, "3", 0.89, 0.68, alternatives)

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "8")

    def test_red_eight_override_rejects_confident_true_three(self) -> None:
        crop = np.full((110, 80, 3), 255, dtype=np.uint8)
        crop[8:70, 5:22] = (0, 0, 220)
        alternatives = [
            ("3", 1.0, 0.75),
            ("3", 0.94, 0.82),
            ("8", 0.52, 0.43),
        ]

        result = choose_red_eight_rank_override(crop, "3", 1.0, 0.68, alternatives)

        self.assertIsNone(result)

    def test_diamond_override_requires_multiple_detail_votes(self) -> None:
        weak = choose_red_diamond_suit_override(
            suit="h",
            best_score=0.96,
            detail_candidates=[(0.30, 0.88, "d", 0.58)],
        )
        strong = choose_red_diamond_suit_override(
            suit="h",
            best_score=0.96,
            detail_candidates=[(0.30, 0.88, "d", 0.58), (0.25, 0.82, "d", 0.57)],
        )

        self.assertIsNone(weak)
        self.assertIsNotNone(strong)
        self.assertEqual(strong[2], "d")

    def test_suit_candidate_selection_prefers_clean_margin_near_tie(self) -> None:
        margin, score, suit, second = choose_best_suit_candidate(
            [
                (0.0179, 0.8338, "s", 0.8159),
                (0.1826, 0.8025, "c", 0.6199),
            ]
        )

        self.assertEqual(suit, "c")
        self.assertAlmostEqual(score, 0.8025)
        self.assertAlmostEqual(second, 0.6199)
        self.assertGreater(margin, 0.18)

    def test_consensus_candidate_beats_single_higher_scored_variant(self) -> None:
        candidates = [
            {
                "card": "3s",
                "rank_confidence": 0.94,
                "rank_margin": 0.44,
                "suit_confidence": 0.83,
                "suit_margin": 0.03,
                "white_ratio": 0.48,
            },
            {
                "card": "3c",
                "rank_confidence": 0.96,
                "rank_margin": 0.10,
                "suit_confidence": 0.96,
                "suit_margin": 0.02,
                "white_ratio": 0.56,
            },
            {
                "card": "3c",
                "rank_confidence": 0.86,
                "rank_margin": 0.11,
                "suit_confidence": 0.91,
                "suit_margin": 0.02,
                "white_ratio": 0.58,
            },
            {
                "card": "3c",
                "rank_confidence": 1.00,
                "rank_margin": 0.09,
                "suit_confidence": 1.00,
                "suit_margin": 0.02,
                "white_ratio": 0.54,
            },
        ]

        result = select_consensus_hero_card_candidate(candidates)

        self.assertEqual(result["card"], "3c")
        self.assertEqual(result["variant_support"], 3)

    def test_board_merge_prefers_auto_component_when_suit_is_cleaner(self) -> None:
        fixed = {
            "card": "Qd",
            "rank": "Q",
            "suit": "d",
            "index": 0,
            "rank_confidence": 0.9611,
            "rank_margin": 0.2512,
            "suit_confidence": 0.9316,
            "suit_margin": 0.0636,
        }
        auto = {
            "card": "Qh",
            "rank": "Q",
            "suit": "h",
            "index": 0,
            "rank_confidence": 0.7268,
            "rank_margin": 0.1670,
            "suit_confidence": 0.8874,
            "suit_margin": 0.1400,
            "roi_mode": "auto_board_component",
        }

        merged = merge_card_details_by_index([fixed], [auto])

        self.assertEqual([item["card"] for item in merged], ["Qh"])

    def test_board_rank_windows_include_lower_shifted_candidates(self) -> None:
        windows = rank_candidate_windows("board")

        self.assertIn((0, 0, 55, 60), windows)
        self.assertIn((12, 20, 42, 72), windows)

    def test_duplicate_hero_card_reads_are_not_used_as_a_complete_hand(self) -> None:
        cards = [
            {
                "card": "7s",
                "index": 0,
                "rank_confidence": 0.96,
                "rank_margin": 0.24,
                "suit_confidence": 0.91,
                "suit_margin": 0.08,
                "face_cover": 0.68,
                "face_aspect": 1.48,
            },
            {
                "card": "7s",
                "index": 1,
                "rank_confidence": 0.94,
                "rank_margin": 0.19,
                "suit_confidence": 0.91,
                "suit_margin": 0.08,
                "face_cover": 0.73,
                "face_aspect": 0.91,
            },
        ]

        deduped = remove_duplicate_hero_card_reads(cards)

        self.assertEqual([card["card"] for card in deduped], ["7s"])

    def test_known_board_problem_frames_are_read_correctly_when_available(self) -> None:
        paths = [
            Path(
                "video_frames/promoted_default_suitfix_validation/"
                "04_validation_20260610_111301_d5976d83/problem_frames/problem_000020_complete.png"
            )
        ]
        paths.extend(Path("video_frames/promoted_default_suitfix_validation").glob("08_*/problem_frames/problem_000210_complete.png"))
        existing = [path for path in paths if path.exists()]
        if len(existing) < 2:
            self.skipTest("local validation problem frames are not available")

        cv2, np_cv = load_cv()
        expected = {
            "problem_000020_complete.png": ["Qh", "Qd", "7h", "Jc"],
            "problem_000210_complete.png": ["4c", "As", "Js", "6d"],
        }

        for path in existing:
            frame = cv2.imdecode(np_cv.fromfile(str(path), dtype=np_cv.uint8), cv2.IMREAD_COLOR)
            cards = [detail.get("card") for detail in detect_board_cards_default(frame)]
            self.assertEqual(cards, expected[path.name])

    def test_red_card_back_animation_is_non_problem_class(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[125:160, 130:165] = (0, 0, 190)

        result = classify_frame(frame, ["K?"])

        self.assertEqual(result, "obstructed_animation")


if __name__ == "__main__":
    unittest.main()
