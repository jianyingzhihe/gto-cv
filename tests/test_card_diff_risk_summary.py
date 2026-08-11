from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_diff_risk_summary import summarize_card_diff_risks
from gto_cli.card_review_diff import DIFF_COLUMNS


class CardDiffRiskSummaryTest(unittest.TestCase):
    def test_summarize_card_diff_risks_groups_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diff_csv = root / "card_review_diff_rows.csv"
            with diff_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "risky_unverified_change",
                        "risk": "True",
                        "risk_reason": "changed_high_confidence_baseline_without_truth",
                        "video": "v.mp4",
                        "timestamp_sec": "10.000",
                        "frame_index": "300",
                        "slot": "0",
                        "baseline_card": "Ad",
                        "candidate_card": "4d",
                        "baseline_rank_confidence": "0.98",
                        "candidate_rank_confidence": "0.90",
                    }
                )
                writer.writerow(
                    {
                        "status": "candidate_missing",
                        "risk": "True",
                        "risk_reason": "candidate_lost_card",
                        "video": "v.mp4",
                        "timestamp_sec": "20.000",
                        "frame_index": "600",
                        "slot": "1",
                        "baseline_card": "9s",
                        "candidate_card": "",
                    }
                )
                writer.writerow(
                    {
                        "status": "same",
                        "risk": "True",
                        "risk_reason": "downgraded_review_reason",
                        "video": "v.mp4",
                        "timestamp_sec": "30.000",
                        "frame_index": "900",
                        "slot": "0",
                        "baseline_card": "Jd",
                        "candidate_card": "Jd",
                    }
                )

            payload = summarize_card_diff_risks(diff_csv=diff_csv, output_dir=root / "summary")

            self.assertEqual(payload["selected_count"], 3)
            actions = {group["transition"]: group["recommended_action"] for group in payload["groups"]}
            self.assertEqual(actions["Ad->4d"], "manual_label_card_change")
            self.assertEqual(actions["9s->-"], "manual_label_missing_or_roi")
            self.assertEqual(actions["Jd->Jd"], "review_confidence_downgrade")
            self.assertTrue(Path(payload["files"]["groups_csv"]).exists())
            self.assertTrue(Path(payload["files"]["report_md"]).exists())


if __name__ == "__main__":
    unittest.main()
