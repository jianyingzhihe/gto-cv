from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.card_debug_review import collect_card_debug_review
from gto_cli.card_label_queue import prepare_card_label_queue
from gto_cli.screen_vision import write_card_debug_assets
from gto_cli.video_vision import load_cv


class CardDebugReviewTest(unittest.TestCase):
    def test_collect_card_debug_review_outputs_queue_compatible_csv(self) -> None:
        cv2, _np = load_cv()
        frame = np.full((260, 420, 3), 28, dtype=np.uint8)
        card = frame[130:235, 170:245]
        card[:] = (246, 246, 246)
        cv2.putText(card, "Q", (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.circle(card, (24, 78), 9, (30, 30, 30), -1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            debug_dir = root / "card_debug"
            write_card_debug_assets(
                cv2=cv2,
                frame=frame,
                frame_result={
                    "cards": {
                        "hero_details": [
                            {
                                "card": "Q?",
                                "rank": "Q",
                                "suit": "?",
                                "source": "hero",
                                "index": 0,
                                "rank_confidence": 0.91,
                                "rank_margin": 0.22,
                                "suit_confidence": 0.12,
                                "suit_margin": 0.01,
                                "roi_mode": "test",
                                "roi_box": {"x": 170, "y": 130, "width": 75, "height": 105},
                            }
                        ],
                        "board_details": [],
                    }
                },
                state={
                    "source": {"timestamp_sec": 0.0, "frame_index": 0, "path": "screen://test"},
                    "hero": {"cards": ["Q?"]},
                    "table": {"board": []},
                    "confidence": {"cards": {"hero": [], "board": []}},
                },
                output_dir=debug_dir,
                basename="event_0001_hero_cards_incomplete",
                problem="hero_cards_incomplete",
            )

            review = collect_card_debug_review(input_dir=debug_dir, output_dir=root / "review")
            self.assertEqual(review["rows"], 1)
            self.assertIn("retrain-card-label-queue", review["commands"]["retrain_label_queue"])
            review_csv = Path(review["files"]["review_csv"])
            self.assertTrue(Path(review["files"]["runbook"]).exists())
            self.assertTrue(review_csv.exists())
            with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["review_reason"], "hero_cards_incomplete")
            self.assertEqual(row["card0"], "Q?")
            self.assertTrue(Path(row["card0_card_path"]).exists())
            self.assertTrue(Path(row["card1_card_path"]).exists())

            queue = prepare_card_label_queue(
                review_csvs=[review_csv],
                output_dir=root / "queue",
                max_rows=10,
                copy_assets=True,
            )
            self.assertEqual(queue["selected_count"], 1)
            self.assertTrue(Path(queue["files"]["label_queue_csv"]).exists())
            self.assertTrue((root / "queue" / "assets").exists())

            prepared = collect_card_debug_review(
                input_dir=debug_dir,
                output_dir=root / "review_with_queue",
                prepare_label_queue=True,
                prepare_glyph_label_queue=True,
                queue_max_rows=5,
                glyph_queue_max_rows=10,
            )
            self.assertIsNotNone(prepared["label_queue"])
            self.assertEqual(prepared["label_queue"]["selected_count"], 1)
            self.assertTrue(Path(prepared["label_queue"]["files"]["label_queue_html"]).exists())
            self.assertIsNotNone(prepared["glyph_label_queue"])
            self.assertGreater(prepared["glyph_label_queue"]["selected_count"], 0)
            self.assertTrue(Path(prepared["glyph_label_queue"]["files"]["glyph_label_queue_html"]).exists())
            runbook = Path(prepared["files"]["runbook"]).read_text(encoding="utf-8")
            self.assertIn("Label queue CSV", runbook)
            self.assertIn("Glyph label queue CSV", runbook)


if __name__ == "__main__":
    unittest.main()
