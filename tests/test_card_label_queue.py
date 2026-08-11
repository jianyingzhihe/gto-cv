from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_label_queue import LABEL_QUEUE_COLUMNS, audit_card_label_queue, prepare_card_diff_label_queue
from gto_cli.card_review_diff import DIFF_COLUMNS


class CardLabelQueueAuditTest(unittest.TestCase):
    def test_audit_counts_labels_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card = root / "card.png"
            rank = root / "rank.png"
            suit = root / "suit.png"
            table = root / "table.png"
            for path in (card, rank, suit, table):
                path.write_bytes(b"x")
            queue_csv = root / "label_queue.csv"
            rows = [
                {
                    "label_id": "D0001",
                    "card0": "Ad",
                    "card0_consensus": "4d",
                    "card0_card_path": str(card),
                    "card0_rank_path": str(rank),
                    "card0_suit_path": str(suit),
                    "asset_table": str(table),
                    "final_card0": "Ad",
                },
                {
                    "label_id": "D0002",
                    "card0": "9d",
                    "card0_consensus": "8d",
                    "card0_card_path": str(card),
                    "card0_rank_path": str(rank),
                    "card0_suit_path": str(suit),
                    "asset_table": str(table),
                    "final_card0": "",
                },
            ]
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LABEL_QUEUE_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

            payload = audit_card_label_queue(queue_csv=queue_csv, output_dir=root / "audit")

        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["total_slots"], 2)
        self.assertEqual(payload["labeled_slots"], 1)
        self.assertEqual(payload["missing_asset_count"], 0)
        self.assertEqual(payload["invalid_label_count"], 0)
        self.assertTrue(payload["ready_to_apply"])
        self.assertFalse(payload["ready_to_retrain"])
        self.assertEqual(payload["label_matches"]["current"], 1)
        self.assertIn("D0002", payload["unlabeled_row_ids"])

    def test_prepare_diff_queue_excludes_same_risk_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            same_card = root / "same_card.png"
            change_card = root / "change_card.png"
            same_table = root / "same_table.png"
            change_table = root / "change_table.png"
            for path in (same_card, change_card, same_table, change_table):
                path.write_bytes(b"x")
            diff_csv = root / "diff.csv"
            with diff_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "same",
                        "risk": "True",
                        "risk_reason": "downgraded_review_reason",
                        "video": "v.mp4",
                        "timestamp_sec": "1.000",
                        "frame_index": "30",
                        "slot": "0",
                        "baseline_card": "Jd",
                        "candidate_card": "Jd",
                        "baseline_card_path": str(same_card),
                        "baseline_table_frame_path": str(same_table),
                    }
                )
                writer.writerow(
                    {
                        "status": "risky_unverified_change",
                        "risk": "True",
                        "risk_reason": "changed_high_confidence_baseline_without_truth",
                        "video": "v.mp4",
                        "timestamp_sec": "2.000",
                        "frame_index": "60",
                        "slot": "1",
                        "baseline_card": "Ad",
                        "candidate_card": "4d",
                        "baseline_card_path": str(change_card),
                        "baseline_table_frame_path": str(change_table),
                    }
                )

            default_payload = prepare_card_diff_label_queue(diff_csv=diff_csv, output_dir=root / "queue_default")
            same_payload = prepare_card_diff_label_queue(diff_csv=diff_csv, output_dir=root / "queue_same", include_same=True)

            self.assertEqual(default_payload["selected_count"], 1)
            self.assertEqual(same_payload["selected_count"], 2)

    def test_prepare_diff_queue_writes_contact_sheet_when_pillow_available(self) -> None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card = root / "card.png"
            table = root / "table.png"
            write_tiny_png(card)
            write_tiny_png(table)
            diff_csv = root / "diff.csv"
            with diff_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "risky_unverified_change",
                        "risk": "True",
                        "risk_reason": "changed_high_confidence_baseline_without_truth",
                        "video": "v.mp4",
                        "timestamp_sec": "2.000",
                        "frame_index": "60",
                        "slot": "1",
                        "baseline_card": "Ad",
                        "candidate_card": "4d",
                        "baseline_card_path": str(card),
                        "baseline_table_frame_path": str(table),
                    }
                )

            payload = prepare_card_diff_label_queue(diff_csv=diff_csv, output_dir=root / "queue")

            sheet = Path(payload["files"]["label_queue_sheet"])
            self.assertTrue(payload["contact_sheet"]["ok"])
            self.assertTrue(sheet.exists())
            self.assertGreater(sheet.stat().st_size, 0)

    def test_prepare_diff_queue_prefers_candidate_crop_paths_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_card = root / "baseline_card.png"
            candidate_card = root / "candidate_card.png"
            baseline_table = root / "baseline_table.png"
            candidate_table = root / "candidate_table.png"
            for path in (baseline_card, candidate_card, baseline_table, candidate_table):
                path.write_bytes(b"x")
            diff_csv = root / "diff.csv"
            with diff_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "regression",
                        "risk": "True",
                        "risk_reason": "manual_truth_regression",
                        "video": "v.mp4",
                        "timestamp_sec": "2.000",
                        "frame_index": "60",
                        "slot": "0",
                        "baseline_card": "Jd",
                        "candidate_card": "8c",
                        "baseline_card_path": str(baseline_card),
                        "candidate_card_path": str(candidate_card),
                        "baseline_table_frame_path": str(baseline_table),
                        "candidate_table_frame_path": str(candidate_table),
                    }
                )

            payload = prepare_card_diff_label_queue(diff_csv=diff_csv, output_dir=root / "queue", render_contact_sheet=False)
            with Path(payload["files"]["label_queue_csv"]).open("r", encoding="utf-8-sig", newline="") as queue_stream:
                queue_rows = list(csv.DictReader(queue_stream))

            self.assertEqual(queue_rows[0]["card0_card_path"], str(candidate_card))
            self.assertEqual(queue_rows[0]["table_frame_path"], str(candidate_table))
            self.assertEqual(payload["asset_preference"], "candidate")

    def test_prepare_diff_queue_can_prefer_baseline_crop_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_card = root / "baseline_card.png"
            candidate_card = root / "candidate_card.png"
            baseline_rank = root / "baseline_rank.png"
            candidate_rank = root / "candidate_rank.png"
            baseline_suit = root / "baseline_suit.png"
            candidate_suit = root / "candidate_suit.png"
            baseline_table = root / "baseline_table.png"
            candidate_table = root / "candidate_table.png"
            for path in (
                baseline_card,
                candidate_card,
                baseline_rank,
                candidate_rank,
                baseline_suit,
                candidate_suit,
                baseline_table,
                candidate_table,
            ):
                path.write_bytes(b"x")
            diff_csv = root / "diff.csv"
            with diff_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "regression",
                        "risk": "True",
                        "risk_reason": "manual_truth_regression",
                        "video": "v.mp4",
                        "timestamp_sec": "2.000",
                        "frame_index": "60",
                        "slot": "0",
                        "baseline_card": "Jd",
                        "candidate_card": "8c",
                        "baseline_card_path": str(baseline_card),
                        "candidate_card_path": str(candidate_card),
                        "baseline_rank_path": str(baseline_rank),
                        "candidate_rank_path": str(candidate_rank),
                        "baseline_suit_path": str(baseline_suit),
                        "candidate_suit_path": str(candidate_suit),
                        "baseline_table_frame_path": str(baseline_table),
                        "candidate_table_frame_path": str(candidate_table),
                    }
                )

            payload = prepare_card_diff_label_queue(
                diff_csv=diff_csv,
                output_dir=root / "queue",
                prefer_candidate_assets=False,
                render_contact_sheet=False,
            )
            with Path(payload["files"]["label_queue_csv"]).open("r", encoding="utf-8-sig", newline="") as queue_stream:
                queue_rows = list(csv.DictReader(queue_stream))

            self.assertEqual(queue_rows[0]["card0_card_path"], str(baseline_card))
            self.assertEqual(queue_rows[0]["card0_rank_path"], str(baseline_rank))
            self.assertEqual(queue_rows[0]["card0_suit_path"], str(baseline_suit))
            self.assertEqual(queue_rows[0]["table_frame_path"], str(baseline_table))
            self.assertEqual(payload["asset_preference"], "baseline")


def write_tiny_png(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (8, 8), (255, 255, 255))
    image.save(path)


if __name__ == "__main__":
    unittest.main()
