from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gto_cli.card_dataset_download import dataset_summary, summarize_card_coverage


class CardDatasetDownloadTest(unittest.TestCase):
    def test_summarize_card_coverage_detects_complete_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for rank in "AKQJT98765432":
                for suit in "shdc":
                    name = f"{rank if rank != 'T' else '10'}{suit.upper()}"
                    path = root / name / f"{name}.png"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"fake")
                    paths.append(path)

            coverage = summarize_card_coverage(paths)

            self.assertTrue(coverage["complete_deck"])
            self.assertEqual(coverage["parsed_card_count"], 52)
            self.assertEqual(coverage["missing_cards"], [])
            self.assertEqual(coverage["duplicate_cards"], {})
            self.assertEqual(coverage["rank_counts"]["A"], 4)
            self.assertEqual(coverage["suit_counts"]["s"], 13)

    def test_dataset_summary_reports_missing_and_duplicate_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("AS", "AS", "KH"):
                path = root / name / f"{name}_{len(list(root.rglob('*.png')))}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake")
            (root / "noise.png").write_bytes(b"fake")

            summary = dataset_summary(root, repo_id="local/cards", repo_type="dataset", downloaded=False, reason="test")

            coverage = summary["card_coverage"]
            self.assertFalse(coverage["complete_deck"])
            self.assertEqual(coverage["duplicate_cards"], {"As": 2})
            self.assertIn("Ah", coverage["missing_cards"])
            self.assertGreaterEqual(coverage["unparsed_count"], 1)


if __name__ == "__main__":
    unittest.main()
