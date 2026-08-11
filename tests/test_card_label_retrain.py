from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_label_retrain import merge_manual_truth_into_review, retrain_card_label_queue, temporary_model_env


class CardLabelRetrainTest(unittest.TestCase):
    def test_temporary_model_env_restores_previous_values(self) -> None:
        old_knn = os.environ.get("GTO_CARD_KNN_MODEL")
        old_deep = os.environ.get("GTO_CARD_DEEP_MODEL_DIR")
        old_rank = os.environ.get("GTO_CARD_DEEP_RANK_MODEL_DIR")
        old_suit = os.environ.get("GTO_CARD_DEEP_SUIT_MODEL_DIR")
        os.environ["GTO_CARD_KNN_MODEL"] = "old_knn"
        os.environ["GTO_CARD_DEEP_MODEL_DIR"] = "old_deep"
        os.environ["GTO_CARD_DEEP_RANK_MODEL_DIR"] = "old_rank"
        os.environ["GTO_CARD_DEEP_SUIT_MODEL_DIR"] = "old_suit"
        try:
            with temporary_model_env(knn_model_path=Path("new_knn.npz"), deep_model_dir=Path("new_deep")):
                self.assertEqual(os.environ["GTO_CARD_KNN_MODEL"], "new_knn.npz")
                self.assertEqual(os.environ["GTO_CARD_DEEP_MODEL_DIR"], "new_deep")
                self.assertNotIn("GTO_CARD_DEEP_RANK_MODEL_DIR", os.environ)
                self.assertNotIn("GTO_CARD_DEEP_SUIT_MODEL_DIR", os.environ)
            self.assertEqual(os.environ["GTO_CARD_KNN_MODEL"], "old_knn")
            self.assertEqual(os.environ["GTO_CARD_DEEP_MODEL_DIR"], "old_deep")
            self.assertEqual(os.environ["GTO_CARD_DEEP_RANK_MODEL_DIR"], "old_rank")
            self.assertEqual(os.environ["GTO_CARD_DEEP_SUIT_MODEL_DIR"], "old_suit")
        finally:
            restore_env("GTO_CARD_KNN_MODEL", old_knn)
            restore_env("GTO_CARD_DEEP_MODEL_DIR", old_deep)
            restore_env("GTO_CARD_DEEP_RANK_MODEL_DIR", old_rank)
            restore_env("GTO_CARD_DEEP_SUIT_MODEL_DIR", old_suit)

    def test_temporary_model_env_clears_deep_when_disabled(self) -> None:
        old_deep = os.environ.get("GTO_CARD_DEEP_MODEL_DIR")
        old_rank = os.environ.get("GTO_CARD_DEEP_RANK_MODEL_DIR")
        old_suit = os.environ.get("GTO_CARD_DEEP_SUIT_MODEL_DIR")
        os.environ["GTO_CARD_DEEP_MODEL_DIR"] = "old_deep"
        os.environ["GTO_CARD_DEEP_RANK_MODEL_DIR"] = "old_rank"
        os.environ["GTO_CARD_DEEP_SUIT_MODEL_DIR"] = "old_suit"
        try:
            with temporary_model_env(knn_model_path=Path("new_knn.npz"), deep_model_dir=None):
                self.assertNotIn("GTO_CARD_DEEP_MODEL_DIR", os.environ)
                self.assertNotIn("GTO_CARD_DEEP_RANK_MODEL_DIR", os.environ)
                self.assertNotIn("GTO_CARD_DEEP_SUIT_MODEL_DIR", os.environ)
            self.assertEqual(os.environ["GTO_CARD_DEEP_MODEL_DIR"], "old_deep")
            self.assertEqual(os.environ["GTO_CARD_DEEP_RANK_MODEL_DIR"], "old_rank")
            self.assertEqual(os.environ["GTO_CARD_DEEP_SUIT_MODEL_DIR"], "old_suit")
        finally:
            restore_env("GTO_CARD_DEEP_MODEL_DIR", old_deep)
            restore_env("GTO_CARD_DEEP_RANK_MODEL_DIR", old_rank)
            restore_env("GTO_CARD_DEEP_SUIT_MODEL_DIR", old_suit)

    def test_retrain_stops_when_queue_has_no_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_csv = root / "label_queue.csv"
            write_minimal_unlabeled_queue(queue_csv)

            payload = retrain_card_label_queue(
                queue_csv=queue_csv,
                output_dir=root / "run",
                base_glyph_dirs=[],
                video_paths=[],
                benchmark_review_csvs=[root / "missing_review.csv"],
                baseline_review_csv=root / "missing_review.csv",
                baseline_validation_summary_json=None,
                deep_card_model_dir=None,
            )

            self.assertTrue(payload["stopped"])
            self.assertEqual(payload["reason"], "queue_not_ready_to_apply")
            self.assertTrue((root / "run" / "label_retrain_summary.json").exists())

    def test_merge_manual_truth_uses_original_diff_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_csv = root / "review.csv"
            queue_csv = root / "queue.csv"
            write_review_for_truth_merge(review_csv)
            write_queue_for_truth_merge(queue_csv)

            output_csv = merge_manual_truth_into_review(
                review_csv=review_csv,
                queue_csv=queue_csv,
                output_csv=root / "review_with_truth.csv",
            )

            with output_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["final_card0"], "")
            self.assertEqual(rows[0]["final_card1"], "6s")


def write_minimal_unlabeled_queue(path: Path) -> None:
    fieldnames = [
        "label_id",
        "card0",
        "card0_consensus",
        "card0_card_path",
        "card0_rank_path",
        "card0_suit_path",
        "final_card0",
        "final_card1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "label_id": "L0001",
                "card0": "As",
                "card0_consensus": "As",
                "card0_card_path": "missing_card.png",
                "card0_rank_path": "missing_rank.png",
                "card0_suit_path": "missing_suit.png",
                "final_card0": "",
                "final_card1": "",
            }
        )


def write_review_for_truth_merge(path: Path) -> None:
    fieldnames = [
        "video",
        "timestamp_sec",
        "frame_index",
        "review_reason",
        "card0",
        "card1",
        "final_card0",
        "final_card1",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "video": "video_frames/example.mp4",
                "timestamp_sec": "790.0",
                "frame_index": "23700",
                "review_reason": "ok",
                "card0": "6h",
                "card1": "4s",
                "final_card0": "",
                "final_card1": "",
                "notes": "",
            }
        )


def write_queue_for_truth_merge(path: Path) -> None:
    fieldnames = [
        "label_id",
        "video",
        "timestamp_sec",
        "frame_index",
        "card0",
        "card0_consensus",
        "final_card0",
        "final_card1",
        "notes",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "label_id": "D0001",
                "video": "example.mp4",
                "timestamp_sec": "790.000",
                "frame_index": "23700",
                "card0": "6s",
                "card0_consensus": "4s",
                "final_card0": "6s",
                "final_card1": "",
                "notes": "original_slot=1; baseline=6s; candidate=4s",
                "reason": "diff;risky;slot=1;6s->4s",
            }
        )


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
