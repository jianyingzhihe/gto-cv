from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_deep_model import RANK_LABELS, SUIT_LABELS
from gto_cli.card_hf_threshold_sweep import sweep_hf_prediction_thresholds


class CardHfThresholdSweepTest(unittest.TestCase):
    def test_recommends_most_conservative_complete_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            write_predictions(predictions)

            payload = sweep_hf_prediction_thresholds(
                predictions_csv=predictions,
                output_dir=root / "sweep",
                rank_score_thresholds=[0.40, 0.45],
                rank_margin_thresholds=[0.04],
                suit_score_thresholds=[0.65],
                suit_margin_thresholds=[0.06],
            )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["recommended"]["rank_score_threshold"], 0.4)
            self.assertTrue(payload["recommended"]["labels_complete"])
            self.assertTrue(Path(payload["files"]["csv"]).exists())
            self.assertTrue(Path(payload["files"]["report_md"]).exists())


def write_predictions(path: Path) -> None:
    fieldnames = [
        "kind",
        "input_path",
        "current_label",
        "teacher_label",
        "teacher_score",
        "teacher_margin",
        "teacher_second_score",
        "teacher_model",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label in RANK_LABELS:
            score = 0.42 if label == "8" else 0.7
            writer.writerow(
                {
                    "kind": "rank",
                    "input_path": f"{label}.png",
                    "current_label": label,
                    "teacher_label": label,
                    "teacher_score": score,
                    "teacher_margin": 0.2,
                    "teacher_second_score": 0.1,
                    "teacher_model": "fake",
                }
            )
        for label in SUIT_LABELS:
            writer.writerow(
                {
                    "kind": "suit",
                    "input_path": f"{label}.png",
                    "current_label": label,
                    "teacher_label": label,
                    "teacher_score": 0.9,
                    "teacher_margin": 0.5,
                    "teacher_second_score": 0.1,
                    "teacher_model": "fake",
                }
            )


if __name__ == "__main__":
    unittest.main()
