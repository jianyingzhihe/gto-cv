from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_glyph_label_queue import (
    GLYPH_LABEL_QUEUE_COLUMNS,
    apply_card_glyph_label_queue,
    prepare_card_glyph_label_queue,
)


PREDICTION_FIELDS = [
    "index",
    "kind",
    "input_path",
    "current_label",
    "teacher_label",
    "teacher_score",
    "teacher_margin",
    "teacher_second_score",
    "teacher_model",
    "accepted",
    "reason",
    "output_path",
    "rank_path",
    "suit_path",
    "card_path",
]


class CardGlyphLabelQueueTest(unittest.TestCase):
    def test_prepare_glyph_queue_from_teacher_disagreements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank = root / "rank_A.png"
            suit = root / "suit_h.png"
            rank.write_bytes(b"rank")
            suit.write_bytes(b"suit")
            predictions = root / "predictions.csv"
            write_predictions(
                predictions,
                [
                    prediction_row(0, "rank", rank, "A", "", "teacher_disagrees"),
                    prediction_row(1, "suit", suit, "h", "h", "accepted"),
                ],
            )

            payload = prepare_card_glyph_label_queue(
                predictions_csvs=[predictions],
                output_dir=root / "queue",
                allowed_reasons=["teacher_disagrees"],
                render_contact_sheet=False,
            )

            self.assertEqual(payload["selected_count"], 1)
            self.assertEqual(payload["reason_counts"], {"teacher_disagrees": 1})
            queue_csv = Path(payload["files"]["glyph_label_queue_csv"])
            with queue_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["kind"], "rank")
            self.assertEqual(rows[0]["current_label"], "A")
            self.assertEqual(rows[0]["current_confidence"], "")
            self.assertEqual(rows[0]["current_margin"], "")
            self.assertEqual(rows[0]["teacher_label"], "")
            self.assertTrue(Path(rows[0]["asset_path"]).exists())

    def test_prepare_glyph_queue_can_prefill_from_current_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank = root / "rank_A.png"
            rank.write_bytes(b"rank")
            predictions = root / "predictions.csv"
            write_predictions(predictions, [prediction_row(0, "rank", rank, "A", "", "teacher_disagrees")])

            payload = prepare_card_glyph_label_queue(
                predictions_csvs=[predictions],
                output_dir=root / "queue",
                allowed_reasons=["teacher_disagrees"],
                prefill_final_label="current",
                render_contact_sheet=False,
            )

            self.assertEqual(payload["prefill_final_label"], "current")
            self.assertEqual(payload["prefilled_count"], 1)
            with Path(payload["files"]["glyph_label_queue_csv"]).open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["final_label"], "A")

    def test_prepare_glyph_queue_from_review_csv_rank_and_suit_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank = root / "rank_Q.png"
            suit = root / "suit_h.png"
            rank.write_bytes(b"rank")
            suit.write_bytes(b"suit")
            review_csv = root / "review.csv"
            with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["review_reason", "card0", "card0_rank_path", "card0_suit_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "review_reason": "hero_cards_incomplete",
                        "card0": "Qh",
                        "card0_rank_path": str(rank),
                        "card0_suit_path": str(suit),
                    }
                )

            payload = prepare_card_glyph_label_queue(
                predictions_csvs=[],
                review_csvs=[review_csv],
                output_dir=root / "queue",
                render_contact_sheet=False,
            )

            self.assertEqual(payload["selected_count"], 2)
            self.assertEqual(payload["kind_counts"], {"rank": 1, "suit": 1})
            with Path(payload["files"]["glyph_label_queue_csv"]).open("r", encoding="utf-8-sig", newline="") as stream:
                rows = sorted(csv.DictReader(stream), key=lambda row: row["kind"])
            self.assertEqual(rows[0]["kind"], "rank")
            self.assertEqual(rows[0]["current_label"], "Q")
            self.assertEqual(rows[1]["kind"], "suit")
            self.assertEqual(rows[1]["current_label"], "h")
            self.assertTrue(Path(rows[0]["asset_path"]).exists())

    def test_apply_glyph_queue_copies_valid_final_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank = root / "rank_A.png"
            suit = root / "suit_h.png"
            rank.write_bytes(b"rank")
            suit.write_bytes(b"suit")
            queue_csv = root / "glyph_label_queue.csv"
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "label_id": "G0001",
                        "kind": "rank",
                        "input_path": str(rank),
                        "current_label": "A",
                        "teacher_label": "",
                        "final_label": "K",
                    }
                )
                writer.writerow(
                    {
                        "label_id": "G0002",
                        "kind": "suit",
                        "input_path": str(suit),
                        "current_label": "h",
                        "teacher_label": "",
                        "final_label": "x",
                    }
                )

            payload = apply_card_glyph_label_queue(queue_csv=queue_csv, output_dir=root / "applied")

            self.assertEqual(payload["copied"], 1)
            self.assertEqual(payload["invalid_count"], 1)
            self.assertTrue((root / "applied" / "rank" / "K").exists())
            self.assertEqual(payload["label_counts"], {"rank:K": 1})

    def test_apply_glyph_queue_prefers_reviewed_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dirty_rank.png"
            asset = root / "clean_rank.png"
            source.write_bytes(b"dirty")
            asset.write_bytes(b"clean")
            queue_csv = root / "glyph_label_queue.csv"
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "label_id": "G0001",
                        "kind": "rank",
                        "input_path": str(source),
                        "asset_path": str(asset),
                        "final_label": "T",
                    }
                )

            payload = apply_card_glyph_label_queue(queue_csv=queue_csv, output_dir=root / "applied")
            copied = next((root / "applied" / "rank" / "T").iterdir())

            self.assertEqual(payload["copied"], 1)
            self.assertEqual(copied.read_bytes(), b"clean")


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prediction_row(index: int, kind: str, path: Path, current: str, teacher: str, reason: str) -> dict[str, object]:
    return {
        "index": index,
        "kind": kind,
        "input_path": str(path),
        "current_label": current,
        "teacher_label": teacher,
        "teacher_score": "0.91",
        "teacher_margin": "0.20",
        "teacher_second_score": "0.05",
        "teacher_model": "teacher",
        "reason": reason,
    }


if __name__ == "__main__":
    unittest.main()
