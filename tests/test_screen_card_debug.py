from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.card_glyph_label_queue import prepare_card_glyph_label_queue
from gto_cli.screen_vision import (
    append_card_sample_glyph_rows,
    build_reviewed_bbox_commands,
    card_observation_signature,
    card_sample_reason,
    safe_suit_debug_image,
    state_audit_reason,
    state_audit_signature,
    write_card_debug_assets,
)
from gto_cli.video_vision import load_cv


class ScreenCardDebugTest(unittest.TestCase):
    def test_reviewed_commands_pin_the_active_python_runtime(self) -> None:
        commands = build_reviewed_bbox_commands(
            analysis_bbox_path=Path("calibrate/analysis_bbox.json"),
            calibration_output_dir=Path("calibrate"),
            hero_name="fish",
            effective_stack_bb=100,
            villain_profile="standard",
            min_confidence=0.35,
            ocr_scale=0.65,
            dealer_refresh_frames=4,
        )
        prefix = f'& "{Path(sys.executable).resolve()}" gto.py'
        self.assertTrue(commands["live"].startswith(prefix))
        self.assertTrue(commands["overlay"].startswith(prefix))
        self.assertTrue(commands["pick_hero"].startswith(prefix))

    def test_board_suit_debug_image_matches_clean_component_used_by_recognizer(self) -> None:
        card_path = Path(
            "video_frames/screen_live/card_sample_fixed_replay_20260730/samples/"
            "sample_20260729_231042_0004_0046p301s/board_slot2_2c_card.png"
        )
        if not card_path.exists():
            self.skipTest("local board-club debug crop is not available")

        cv2, _np = load_cv()
        crop = cv2.imread(str(card_path))
        glyph = safe_suit_debug_image(crop, "board")

        self.assertGreater(int((glyph > 0).sum()), 100)
        self.assertEqual(int((glyph[0] > 0).sum()), 0)
        self.assertEqual(int((glyph[-1] > 0).sum()), 0)

    def test_write_card_debug_assets_exports_card_and_glyphs(self) -> None:
        cv2, _np = load_cv()
        frame = np.full((240, 360, 3), 32, dtype=np.uint8)
        card = frame[120:220, 140:210]
        card[:] = (245, 245, 245)
        cv2.putText(card, "A", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.circle(card, (22, 75), 9, (10, 10, 10), -1)

        frame_result = {
            "cards": {
                "hero_details": [
                    {
                        "card": "As",
                        "rank": "A",
                        "suit": "s",
                        "source": "hero",
                        "index": 0,
                        "rank_confidence": 0.9,
                        "rank_margin": 0.2,
                        "suit_confidence": 0.8,
                        "suit_margin": 0.1,
                        "roi_mode": "test",
                        "roi_box": {"x": 140, "y": 120, "width": 70, "height": 100},
                    }
                ],
                "board_details": [],
            }
        }
        state = {
            "source": {"timestamp_sec": 1.23, "frame_index": 4},
            "hero": {"cards": ["As", "?"]},
            "table": {"board": []},
            "confidence": {"cards": {"hero": [{"card": "As"}], "board": []}},
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = write_card_debug_assets(
                cv2=cv2,
                frame=frame,
                frame_result=frame_result,
                state=state,
                output_dir=Path(tmp),
                basename="event_0001_hero_cards_incomplete",
                problem="hero_cards_incomplete",
                context_frame=np.full((300, 500, 3), 24, dtype=np.uint8),
                search_region={"left": 10, "top": 20, "width": 500, "height": 300},
                analysis_region={"left": 70, "top": 50, "width": 360, "height": 240},
                context_scope="monitor_full",
                diagnostic_frame=frame.copy(),
            )

            self.assertIsNotNone(result)
            metadata_path = Path(result["metadata"])
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["problem"], "hero_cards_incomplete")
            self.assertGreaterEqual(result["saved_count"], 1)
            self.assertGreaterEqual(result["fallback_count"], 2)
            self.assertTrue(Path(result["screen_context"]).exists())
            self.assertEqual(result["screen_context_scope"], "monitor_full")
            self.assertEqual(metadata["screen_context_scope"], "monitor_full")
            self.assertTrue(Path(result["diagnostic_overlay"]).exists())
            saved = metadata["saved"][0]
            self.assertTrue(Path(saved["card_path"]).exists())
            self.assertTrue(Path(saved["rank_path"]).exists())
            self.assertTrue(Path(saved["suit_path"]).exists())
            self.assertEqual(len(result["saved"]), 1)

    def test_card_observation_signature_tracks_hero_and_board_predictions(self) -> None:
        result = {
            "cards": {
                "hero_details": [{"index": 0, "card": "7c"}, {"index": 1, "card": "Qh"}],
                "board_details": [{"index": 0, "card": "Td"}],
            }
        }
        self.assertEqual(card_observation_signature(result), "hero:0:7c|hero:1:Qh|board:0:Td")
        self.assertEqual(card_observation_signature({"cards": {}}), "")

    def test_state_audit_signature_captures_distinct_actionable_states(self) -> None:
        state = {
            "table": {"street": "preflop", "dealer_seat": "top"},
            "hero": {"position": "HJ", "cards": ["Ah", "6s"], "is_turn": True},
            "action_controls": {"actions": ["fold", "call", "raise"], "call_amount_bb": 4.8},
            "gto_advice": {"reason": "preflop_context_incomplete"},
            "preflop_tracker": {"reason": "blind_posts_unconfirmed"},
            "bets": [{"seat": "bottom_right", "amount_bb": 4.8}],
        }

        self.assertEqual(state_audit_reason(state), "preflop_context_incomplete")
        first = state_audit_signature(state)
        state["action_controls"]["call_amount_bb"] = 5.6
        self.assertNotEqual(first, state_audit_signature(state))

    def test_state_audit_keeps_passive_betting_transitions(self) -> None:
        state = {
            "ok": True,
            "table": {"street": "preflop", "dealer_seat": "top", "pot_bb": 3.4, "board": []},
            "hero": {"position": "HJ", "cards": ["Ah", "6s"], "is_turn": False},
            "action_controls": {"visible": False, "actions": []},
            "bets": [{"seat": "left", "amount_bb": 2.0}],
            "seats": [{"seat": "left", "position": "UTG", "status": "active", "bet_bb": 2.0}],
        }

        self.assertEqual(state_audit_reason(state), "table_state_change")
        first = state_audit_signature(state)
        state["bets"][0]["amount_bb"] = 5.6
        state["seats"][0]["bet_bb"] = 5.6
        self.assertNotEqual(first, state_audit_signature(state))

    def test_card_sample_manifest_is_glyph_queue_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_path = root / "rank.png"
            suit_path = root / "suit.png"
            rank_path.write_bytes(b"rank")
            suit_path.write_bytes(b"suit")
            csv_path = root / "glyph_predictions.csv"
            saved = [
                {
                    "group": "board",
                    "slot": 2,
                    "card": "8h",
                    "rank": "8",
                    "suit": "h",
                    "rank_path": str(rank_path),
                    "suit_path": str(suit_path),
                    "rank_confidence": 0.55,
                    "rank_margin": 0.02,
                    "suit_confidence": 0.91,
                    "suit_margin": 0.20,
                }
            ]
            written = append_card_sample_glyph_rows(
                csv_path,
                saved,
                sample_id="sample_0001",
                timestamp_sec=12.5,
                frame_index=8,
            )
            self.assertEqual(written, 2)
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["kind"] for row in rows], ["rank", "suit"])
            self.assertEqual(rows[0]["current_label"], "8")
            self.assertEqual(rows[0]["reason"], "live_rank_low_score_or_margin")
            self.assertEqual(rows[1]["reason"], "live_suit_observation")
            self.assertTrue(Path(rows[0]["input_path"]).is_absolute())
            queue = prepare_card_glyph_label_queue(
                predictions_csvs=[csv_path],
                output_dir=root / "queue",
                max_rows=10,
                prefill_final_label="current",
                copy_assets=False,
                render_contact_sheet=False,
            )
            self.assertEqual(queue["selected_count"], 2)
            self.assertEqual(queue["prefilled_count"], 2)

    def test_card_sample_reason_prioritizes_uncertain_digits(self) -> None:
        self.assertEqual(card_sample_reason("rank", 0.4, 0.2), "live_rank_low_score_or_margin")
        self.assertEqual(card_sample_reason("rank", 0.9, 0.2), "live_rank_observation")

    def test_card_sample_manifest_skips_zero_confidence_hero_noncard_but_keeps_board_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_path = root / "rank.png"
            suit_path = root / "suit.png"
            rank_path.write_bytes(b"rank")
            suit_path.write_bytes(b"suit")
            common = {
                "card": "??",
                "rank": "?",
                "suit": "?",
                "rank_path": str(rank_path),
                "suit_path": str(suit_path),
                "rank_confidence": 0.0,
                "rank_margin": 0.0,
                "suit_confidence": 0.0,
                "suit_margin": 0.0,
            }
            csv_path = root / "glyph_predictions.csv"

            written = append_card_sample_glyph_rows(
                csv_path,
                [
                    {**common, "group": "hero", "slot": 0},
                    {
                        **common,
                        "group": "hero",
                        "slot": 1,
                        "card": "7s",
                        "rank": "7",
                        "suit": "s",
                        "rank_confidence": 0.5049,
                        "suit_confidence": 0.687,
                        "face_fill": 0.7171,
                        "face_cover": 0.5675,
                        "face_aspect": 1.046,
                    },
                    {**common, "group": "board", "slot": 0},
                ],
                sample_id="sample_noncard",
                timestamp_sec=1.0,
                frame_index=1,
            )

            self.assertEqual(written, 2)
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["group"] for row in rows}, {"board"})


if __name__ == "__main__":
    unittest.main()
