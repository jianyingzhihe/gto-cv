from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_active_learning import apply_card_review


class CardActiveLearningTest(unittest.TestCase):
    def test_apply_card_review_skips_empty_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_csv = root / "review.csv"
            with review_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "final_card0",
                        "card0_rank_path",
                        "card0_suit_path",
                        "card0_card_path",
                    ],
                )
                writer.writeheader()
                writer.writerow({"final_card0": "Ad", "card0_rank_path": "", "card0_suit_path": "", "card0_card_path": ""})

            payload = apply_card_review(review_csv=review_csv, output_dir=root / "applied")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["copied_rank"], 0)
        self.assertEqual(payload["copied_suit"], 0)
        self.assertEqual(payload["copied_card"], 0)
        self.assertEqual(payload["skipped_count"], 3)


if __name__ == "__main__":
    unittest.main()
