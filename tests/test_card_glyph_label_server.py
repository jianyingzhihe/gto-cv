from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_glyph_label_queue import GLYPH_LABEL_QUEUE_COLUMNS
from gto_cli.card_glyph_label_server import (
    glyph_review_sort_key,
    glyph_progress,
    infer_glyph_context_paths,
    load_glyph_queue_csv,
    public_glyph_rows,
    update_glyph_queue_csv,
)


class CardGlyphLabelServerTest(unittest.TestCase):
    def test_review_sort_groups_suits_before_ranks_by_current_prediction(self) -> None:
        rows = [
            {"label_id": "G4", "kind": "rank", "current_label": "Q"},
            {"label_id": "G3", "kind": "suit", "current_label": "c"},
            {"label_id": "G2", "kind": "rank", "current_label": "A"},
            {"label_id": "G1", "kind": "suit", "current_label": "s"},
            {"label_id": "G5", "kind": "rank", "current_label": "?"},
        ]

        ordered = sorted(rows, key=glyph_review_sort_key)

        self.assertEqual([row["label_id"] for row in ordered], ["G1", "G3", "G2", "G4", "G5"])

    def test_updates_rank_and_suit_labels_and_tracks_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue_csv = Path(tmp) / "glyph_label_queue.csv"
            rows = [
                {
                    "label_id": "G0001",
                    "kind": "rank",
                    "current_label": "8",
                    "final_label": "",
                },
                {
                    "label_id": "G0002",
                    "kind": "suit",
                    "current_label": "s",
                    "final_label": "",
                },
            ]
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            result = update_glyph_queue_csv(
                queue_csv,
                {"label_id": "G0001", "final_label": "3", "notes": "3 was read as 8"},
            )
            self.assertEqual(result["final_label"], "3")
            self.assertEqual(result["progress"]["labeled"], 1)
            saved, _fieldnames = load_glyph_queue_csv(queue_csv)
            self.assertEqual(saved[0]["final_label"], "3")
            self.assertEqual(saved[0]["notes"], "3 was read as 8")

            result = update_glyph_queue_csv(
                queue_csv,
                {"label_id": "G0002", "final_label": "c"},
            )
            self.assertEqual(result["progress"]["labeled"], 2)
            self.assertEqual(glyph_progress(load_glyph_queue_csv(queue_csv)[0])["suit_labeled"], 1)

    def test_rejects_label_for_wrong_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue_csv = Path(tmp) / "glyph_label_queue.csv"
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerow({"label_id": "G0001", "kind": "rank"})

            with self.assertRaisesRegex(ValueError, "invalid rank label"):
                update_glyph_queue_csv(queue_csv, {"label_id": "G0001", "final_label": "s"})

    def test_ignore_card_marks_rank_and_suit_and_removes_them_from_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample_0001"
            sample.mkdir()
            rank = sample / "hero_slot1_7s_rank.png"
            suit = sample / "hero_slot1_7s_suit.png"
            card = sample / "hero_slot1_7s_card.png"
            for path in (rank, suit, card):
                path.write_bytes(b"x")
            queue_csv = Path(tmp) / "glyph_label_queue.csv"
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerows(
                    [
                        {"label_id": "G0001", "kind": "rank", "input_path": str(rank)},
                        {"label_id": "G0002", "kind": "suit", "input_path": str(suit)},
                    ]
                )

            result = update_glyph_queue_csv(queue_csv, {"label_id": "G0001", "ignore_card": True})

            self.assertEqual(set(result["ignored_label_ids"]), {"G0001", "G0002"})
            self.assertEqual(result["progress"]["rows"], 0)
            self.assertEqual(result["progress"]["ignored"], 2)
            saved, _fields = load_glyph_queue_csv(queue_csv)
            self.assertTrue(all(row["ignored"] == "1" for row in saved))

    def test_public_rows_include_current_confidence(self) -> None:
        rows = public_glyph_rows(
            [
                {
                    "label_id": "G0001",
                    "kind": "rank",
                    "current_label": "J",
                    "current_confidence": "0.6594",
                    "current_margin": "0.0891",
                }
            ]
        )

        self.assertEqual(rows[0]["current_confidence"], "0.6594")
        self.assertEqual(rows[0]["current_margin"], "0.0891")

    def test_infers_full_card_and_overlay_from_glyph_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample_0001"
            sample.mkdir()
            rank = sample / "hero_slot0_8s_rank.png"
            card = sample / "hero_slot0_8s_card.png"
            overlay = sample / "diagnostic_overlay.png"
            for path in (rank, card, overlay):
                path.write_bytes(b"x")

            context = infer_glyph_context_paths({"input_path": str(rank)})
            self.assertEqual(context["card_path"], str(card))
            self.assertEqual(context["overlay_path"], str(overlay))
            self.assertEqual(context["sample_id"], "sample_0001")
            self.assertEqual(context["group"], "hero")
            self.assertEqual(context["slot"], "0")
            self.assertEqual(context["observed_card"], "8s")

    def test_infers_context_without_requiring_assets_to_exist(self) -> None:
        rank = Path("missing") / "sample_0002" / "board_slot2_2c_rank.png"

        context = infer_glyph_context_paths({"input_path": str(rank)})

        self.assertEqual(context["card_path"], str(rank.with_name("board_slot2_2c_card.png")))
        self.assertEqual(context["overlay_path"], str(rank.parent / "diagnostic_overlay.png"))
        self.assertEqual(context["frame_path"], str(rank.parent / "frame.png"))
        self.assertEqual(context["group"], "board")
        self.assertEqual(context["slot"], "2")
        self.assertEqual(context["observed_card"], "2c")


if __name__ == "__main__":
    unittest.main()
