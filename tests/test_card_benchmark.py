from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_benchmark import collect_review_samples


class CardBenchmarkSampleTest(unittest.TestCase):
    def test_manual_truth_overrides_pseudo_truth_by_original_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_csv = root / "review.csv"
            queue_csv = root / "queue.csv"
            write_review_csv(review_csv)
            write_queue_csv(queue_csv)

            samples = collect_review_samples(
                [review_csv, queue_csv],
                include_ok_pseudo=True,
                allowed_pseudo_reasons=("ok",),
                max_samples=None,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["truth_source"], "manual")
        self.assertEqual(samples[0]["truth_card"], "Ad")
        self.assertEqual(samples[0]["slot"], 1)


def write_review_csv(path: Path) -> None:
    fieldnames = [
        "video",
        "timestamp_sec",
        "frame_index",
        "review_reason",
        "card0",
        "card1",
        "final_card0",
        "final_card1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "video": "video_frames/example.mp4",
                "timestamp_sec": "100.0",
                "frame_index": "3000.0",
                "review_reason": "ok",
                "card0": "",
                "card1": "4d",
                "final_card0": "",
                "final_card1": "",
            }
        )


def write_queue_csv(path: Path) -> None:
    fieldnames = [
        "video",
        "timestamp_sec",
        "frame_index",
        "review_reason",
        "card0",
        "card1",
        "final_card0",
        "final_card1",
        "reason",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "video": "example.mp4",
                "timestamp_sec": "100.000",
                "frame_index": "3000",
                "review_reason": "",
                "card0": "4d",
                "card1": "",
                "final_card0": "Ad",
                "final_card1": "",
                "reason": "diff;risky;slot=1;Ad->4d",
                "notes": "original_slot=1; baseline=Ad; candidate=4d",
            }
        )


if __name__ == "__main__":
    unittest.main()
