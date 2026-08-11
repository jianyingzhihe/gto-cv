from __future__ import annotations

import unittest

from gto_cli.cv_validate import card_health_issues, format_validation_summary, summarize_card_health


class CvValidateCardHealthTest(unittest.TestCase):
    def test_card_health_issues_detects_hero_board_and_duplicate_problems(self) -> None:
        row = {
            "ok": True,
            "class": "incomplete",
            "hero_cards": ["7?"],
            "board": ["As", "As", "Q?"],
            "hero_turn": {"is_turn": True},
            "board_bad": True,
        }

        issues = card_health_issues(row)

        self.assertIn("hero_card_count_incomplete", issues)
        self.assertIn("hero_card_unknown", issues)
        self.assertIn("hero_turn_cards_not_ready", issues)
        self.assertIn("board_card_unknown", issues)
        self.assertIn("duplicate_cards", issues)

    def test_empty_no_hand_frames_do_not_block_card_health(self) -> None:
        row = {
            "ok": True,
            "class": "empty_or_no_hand",
            "hero_cards": [],
            "board": ["5s", "4s", "5s"],
            "hero_turn": {"is_turn": True},
            "board_bad": False,
        }

        issues = card_health_issues(row)

        self.assertNotIn("hero_turn_cards_not_ready", issues)
        self.assertNotIn("duplicate_cards", issues)

    def test_summarize_card_health_counts_visible_complete_and_blocked_frames(self) -> None:
        rows = [
            {
                "ok": True,
                "time": 1.0,
                "class": "complete",
                "hero_cards": ["As", "Kd"],
                "board": [],
                "hero_turn": {"is_turn": False},
                "board_bad": False,
            },
            {
                "ok": True,
                "time": 2.0,
                "class": "incomplete",
                "hero_cards": ["7?"],
                "board": ["Ah", "Kc", "Q?"],
                "hero_turn": {"is_turn": True},
                "board_bad": True,
            },
            {
                "ok": True,
                "time": 3.0,
                "class": "missed_visible_cards",
                "hero_cards": [],
                "board": [],
                "hero_turn": {"is_turn": False},
                "board_bad": False,
            },
            {
                "ok": True,
                "time": 4.0,
                "class": "empty_or_no_hand",
                "hero_cards": [],
                "board": [],
                "hero_turn": {"is_turn": False},
                "board_bad": False,
            },
            {
                "ok": True,
                "time": 5.0,
                "class": "obstructed_animation",
                "hero_cards": ["7s"],
                "board": [],
                "hero_turn": {"is_turn": False},
                "board_bad": False,
            },
        ]

        summary = summarize_card_health(rows)

        self.assertEqual(summary["hero"]["visible_frames"], 4)
        self.assertEqual(summary["hero"]["complete_frames"], 1)
        self.assertEqual(summary["hero"]["incomplete_or_missed_frames"], 2)
        self.assertEqual(summary["hero"]["turn_blocked_frames"], 1)
        self.assertEqual(summary["board"]["frames_with_board"], 1)
        self.assertEqual(summary["board"]["bad_frames"], 1)
        self.assertEqual(summary["issue_counts"]["hero_visible_cards_missed"], 1)
        self.assertEqual(summary["issue_counts"]["hero_card_unknown"], 1)
        self.assertIn("hero_card_unknown", summary["examples"])

    def test_format_validation_summary_includes_card_health(self) -> None:
        payload = {
            "video": "demo.mp4",
            "sample": {"frame_count": 2, "every_sec": 1},
            "counts": {"complete": 1, "incomplete": 1},
            "real_problem_count": 1,
            "board_bad_count": 0,
            "timing_ms": {},
            "card_health": {
                "hero": {
                    "visible_frames": 2,
                    "complete_frames": 1,
                    "incomplete_or_missed_frames": 1,
                    "turn_blocked_frames": 1,
                },
                "board": {"frames_with_board": 0, "bad_frames": 0},
                "duplicate_frames": 0,
                "issue_counts": {"hero_card_unknown": 1},
            },
            "files": {},
        }

        text = format_validation_summary(payload)

        self.assertIn("Hero card health: visible=2 complete=1 incomplete_or_missed=1 turn_blocked=1", text)
        self.assertIn("Card issues:", text)


if __name__ == "__main__":
    unittest.main()
