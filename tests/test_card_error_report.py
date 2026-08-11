from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from gto_cli.card_error_report import render_error_report, summarize_glyph_queue


class CardErrorReportTest(unittest.TestCase):
    def test_summary_and_gate_remain_blocked_until_every_row_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_csv = root / "queue.csv"
            fields = [
                "label_id",
                "kind",
                "input_path",
                "current_label",
                "current_confidence",
                "current_margin",
                "final_label",
                "reason",
            ]
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "label_id": "G0001",
                            "kind": "rank",
                            "input_path": str(root / "sample_1" / "hero_slot0_7s_rank.png"),
                            "current_label": "7",
                            "final_label": "7",
                        },
                        {
                            "label_id": "G0002",
                            "kind": "suit",
                            "input_path": str(root / "sample_1" / "board_slot0_item_suit.png"),
                            "current_label": "?",
                            "current_confidence": "0",
                            "current_margin": "0",
                            "final_label": "c",
                            "reason": "low_score",
                        },
                        {
                            "label_id": "G0003",
                            "kind": "rank",
                            "input_path": str(root / "sample_2" / "hero_slot1_item_rank.png"),
                            "current_label": "?",
                            "final_label": "",
                        },
                    ]
                )

            summary = summarize_glyph_queue(queue_csv)
            report = render_error_report(
                ledger={
                    "dataset": {"name": "test", "expected_samples": 2},
                    "last_regression": {"tests": "10 passed"},
                    "cases": [],
                    "commands": {},
                },
                queue_summary=summary,
                replay_summary={"source_samples": 2},
                output_path=root / "REPORT.md",
                generated_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            )

            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["labeled"], 2)
            self.assertEqual(len(summary["mismatches"]), 1)
            self.assertIn("版本闸门：`阻塞`", report)
            self.assertIn("G0002", report)
            self.assertIn("人工校对完成：2/3", report)

    def test_ignored_noncard_rows_do_not_count_toward_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_csv = root / "queue.csv"
            with queue_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["label_id", "kind", "input_path", "current_label", "final_label", "ignored"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "label_id": "G0001",
                            "kind": "rank",
                            "input_path": str(root / "sample" / "hero_slot0_7s_rank.png"),
                            "current_label": "7",
                            "ignored": "1",
                        },
                        {
                            "label_id": "G0002",
                            "kind": "rank",
                            "input_path": str(root / "sample" / "hero_slot1_Kc_rank.png"),
                            "current_label": "K",
                            "final_label": "K",
                        },
                    ]
                )

            summary = summarize_glyph_queue(queue_csv)

            self.assertEqual(summary["raw_rows"], 2)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["ignored"], 1)
            self.assertEqual(summary["labeled"], 1)


if __name__ == "__main__":
    unittest.main()
