from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gto_cli.state_action_label_server import (
    action_progress,
    build_action_analysis,
    load_action_queue_csv,
    prepare_state_action_label_queue,
    update_action_queue_csv,
)


class StateActionLabelServerTest(unittest.TestCase):
    def test_analysis_marks_ready_advice_and_existing_preflop_history(self) -> None:
        analysis = build_action_analysis(
            {
                "table": {"street": "preflop", "dealer_position": "BTN"},
                "hero": {"position": "LJ"},
                "hero_turn": {"is_turn": True, "reason": "red_buttons_and_action_text"},
                "action_controls": {"actions": ["call", "fold", "raise"]},
                "gto_advice": {
                    "ready": True,
                    "reason": "hero_action_controls_visible",
                    "summary": "3BET",
                    "preflop_context": {"scenario": "vs_open", "needs": []},
                },
                "preflop_tracker": {},
                "preflop": {
                    "action_history": [
                        {"position": "UTG", "action": "raise"},
                        {"position": "UTG+1", "action": "fold"},
                        {"position": "LJ", "action": "hero_to_act"},
                    ]
                },
            }
        )
        self.assertIn("3BET", analysis)
        self.assertIn("已形成可信的翻前行动记录", analysis)
        self.assertNotIn("尚未建立翻前行动记录", analysis)

    def test_prepares_assets_and_preserves_existing_human_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            self.assertTrue(cv2.imwrite(str(frame), np.full((120, 200, 3), 120, dtype=np.uint8)))
            events = root / "events.jsonl"
            event = {
                "ok": True,
                "event": {"index": 7},
                "source": {"timestamp_sec": 12.5, "card_sample": {"frame": str(frame)}},
                "table": {"street": "preflop", "pot_bb": 3.4, "to_call_bb": 2.0, "board": []},
                "hero": {"position": "UTG", "gto_position": "UTG", "cards": ["As", "Kd"]},
                "hero_turn": {"is_turn": True, "confidence": 0.72, "reason": "visual"},
                "action_controls": {"visible": True, "actions": ["fold", "call", "raise"]},
                "preflop_tracker": {"reason": "blind_posts_unconfirmed"},
                "gto_advice": {"reason": "preflop_context_incomplete"},
                "preflop": {"action_history": [{"position": "UTG", "action": "raise"}]},
                "bets": [{"seat": "left", "amount_bb": 2.0}],
            }
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            output = root / "queue"

            payload = prepare_state_action_label_queue(events_path=events, output_dir=output, max_items=8)
            queue = Path(payload["queue_csv"])
            rows, _ = load_action_queue_csv(queue)
            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(rows[0]["frame_path"]).is_file())
            self.assertTrue(Path(rows[0]["panel_crop_path"]).is_file())
            self.assertEqual(rows[0]["cv_actions"], "fold, call, raise")
            self.assertEqual(rows[0]["cv_advice_reason"], "preflop_context_incomplete")
            self.assertEqual(rows[0]["cv_preflop_tracker_reason"], "blind_posts_unconfirmed")
            self.assertEqual(rows[0]["cv_preflop_history"], "UTG:raise")
            self.assertIn("翻前早期行动没有被完整、可信地记录", rows[0]["cv_analysis"])

            update_action_queue_csv(
                queue,
                {
                    "label_id": rows[0]["label_id"],
                    "final_hero_turn": "yes",
                    "final_panel_template": "preflop_fold_call_raise",
                    "final_actions": ["fold", "call", "raise"],
                    "final_call_amount_bb": "2",
                    "notes": "confirmed from buttons",
                },
            )
            prepare_state_action_label_queue(events_path=events, output_dir=output, max_items=8)
            saved, _ = load_action_queue_csv(queue)
            self.assertEqual(saved[0]["final_hero_turn"], "yes")
            self.assertEqual(saved[0]["final_fast_fold_state"], "uncertain")
            self.assertEqual(saved[0]["final_panel_template"], "preflop_fold_call_raise")
            self.assertEqual(saved[0]["notes"], "confirmed from buttons")
            self.assertEqual(action_progress(saved)["completed"], 1)

    def test_rejects_actions_for_no_action_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue.csv"
            queue.write_text(
                "label_id,final_hero_turn,final_panel_template,final_actions\nA00001_1,,,\n",
                encoding="utf-8-sig",
            )
            with self.assertRaisesRegex(ValueError, "no_action_panel"):
                update_action_queue_csv(
                    queue,
                    {
                        "label_id": "A00001_1",
                        "final_hero_turn": "no",
                        "final_fast_fold_state": "available",
                        "final_panel_template": "no_action_panel",
                        "final_actions": ["fold"],
                    },
                )

    def test_context_frame_is_preferred_and_fast_fold_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi = root / "roi.png"
            context = root / "context.png"
            self.assertTrue(cv2.imwrite(str(roi), np.full((100, 200, 3), 60, dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(context), np.full((180, 240, 3), 180, dtype=np.uint8)))
            events = root / "events.jsonl"
            event = {
                "ok": True,
                "event": {"index": 8},
                "source": {
                    "kind": "screen",
                    "timestamp_sec": 14.0,
                    "monitor_region": {"left": 0, "top": 0, "width": 240, "height": 180},
                    "screen_region": {"left": 20, "top": 20, "width": 200, "height": 120},
                    "card_sample": {
                        "frame": str(roi),
                        "screen_context": str(context),
                        "screen_context_scope": "monitor_full",
                    },
                },
                "table": {"street": "flop", "board": ["2c", "7d", "Th"]},
                "hero": {"position": "CO", "gto_position": "CO", "cards": ["9s", "8s"]},
                "hero_turn": {"is_turn": False},
                "action_controls": {"visible": False, "actions": []},
            }
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            payload = prepare_state_action_label_queue(events_path=events, output_dir=root / "queue", max_items=8)
            rows, _ = load_action_queue_csv(Path(payload["queue_csv"]))
            self.assertEqual(rows[0]["frame_scope"], "monitor_full_context")
            copied = cv2.imread(rows[0]["frame_path"], cv2.IMREAD_COLOR)
            self.assertEqual(copied.shape[:2], (180, 240))
            panel = cv2.imread(rows[0]["panel_crop_path"], cv2.IMREAD_COLOR)
            self.assertEqual(panel.shape[:2], (95, 240))

            update_action_queue_csv(
                Path(payload["queue_csv"]),
                {
                    "label_id": rows[0]["label_id"],
                    "final_hero_turn": "no",
                    "final_fast_fold_state": "available",
                    "final_panel_template": "no_hero_action",
                    "final_actions": [],
                    "final_disabled_actions": ["check", "call"],
                },
            )
            saved, _ = load_action_queue_csv(Path(payload["queue_csv"]))
            self.assertEqual(saved[0]["final_disabled_actions"], "check,call")
            self.assertEqual(action_progress(saved)["fast_fold_available"], 1)

    def test_manual_outer_window_audit_frame_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "manual_outer.png"
            self.assertTrue(cv2.imwrite(str(frame), np.full((180, 300, 3), 180, dtype=np.uint8)))
            events = root / "events.jsonl"
            event = {
                "ok": True,
                "event": {"index": 9},
                "source": {"timestamp_sec": 20.0, "state_audit": {"frame": str(frame), "scope": "manual_outer_bbox"}},
                "table": {"street": "preflop"},
                "hero": {"position": "CO", "gto_position": "CO", "cards": ["As", "Kd"]},
                "hero_turn": {"is_turn": True},
                "action_controls": {"visible": True, "actions": ["fold", "call", "raise"]},
            }
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")

            payload = prepare_state_action_label_queue(events_path=events, output_dir=root / "queue", max_items=8)
            rows, _ = load_action_queue_csv(Path(payload["queue_csv"]))
            self.assertEqual(rows[0]["frame_scope"], "manual_outer_bbox")
            copied = cv2.imread(rows[0]["frame_path"], cv2.IMREAD_COLOR)
            self.assertEqual(copied.shape[:2], (180, 300))


if __name__ == "__main__":
    unittest.main()
