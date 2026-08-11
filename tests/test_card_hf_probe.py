from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.card_hf_probe import (
    LabeledCrop,
    build_probe_payload,
    collect_labeled_crops,
    count_labeled_by_source,
    ensemble_hf_probe_predictions,
    filter_hf_probe_predictions,
    format_source_counts,
    limit_labeled_crops_by_source,
    load_probe,
    predict_from_probe,
    save_probe,
)


class CardHfProbeTest(unittest.TestCase):
    def test_probe_predicts_nearest_prototype_after_save_load(self) -> None:
        records = [
            LabeledCrop(kind="rank", path=Path("a1.png"), label="A"),
            LabeledCrop(kind="rank", path=Path("a2.png"), label="A"),
            LabeledCrop(kind="rank", path=Path("k1.png"), label="K"),
            LabeledCrop(kind="rank", path=Path("k2.png"), label="K"),
        ]
        features = np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype="float32",
        )
        features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        payload = build_probe_payload(
            records=records,
            features=features,
            labels=["A", "K"],
            kind="rank",
            model_name="test-model",
            temperature=0.04,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf_rank_probe.npz"
            save_probe(path, payload)
            loaded = load_probe(path)
            prediction = predict_from_probe(loaded, np.asarray([0.99, 0.01], dtype="float32"))
        self.assertEqual(prediction["label"], "A")
        self.assertGreater(float(prediction["score"]), 0.5)
        self.assertEqual(prediction["backend"], "hf_embedding_probe")

    def test_limit_labeled_crops_balances_sources_within_class_cap(self) -> None:
        live = [LabeledCrop(kind="rank", path=Path(f"live_a_{index}.png"), label="A") for index in range(6)]
        external = [LabeledCrop(kind="rank", path=Path(f"external_a_{index}.png"), label="A") for index in range(2)]

        limited = limit_labeled_crops_by_source([live, external], labels=["A"], max_images_per_class=4)

        self.assertEqual(len(limited), 4)
        self.assertEqual([crop.path.name for crop in limited], ["live_a_0.png", "external_a_0.png", "live_a_1.png", "external_a_1.png"])

    def test_collect_labeled_crops_balances_input_dirs_under_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            external = root / "external"
            for index in range(6):
                write_fake_png(live / "rank" / "A" / f"live_{index}.png")
            for index in range(2):
                write_fake_png(external / "rank" / "A" / f"external_{index}.png")

            crops = collect_labeled_crops(
                input_dirs=[live, external],
                kind="rank",
                labels=["A"],
                template_dir=None,
                include_templates=False,
                max_images_per_class=4,
            )

        self.assertEqual(len(crops), 4)
        self.assertEqual(sum(1 for crop in crops if "external" in str(crop.path)), 2)

    def test_count_labeled_by_source_reports_per_dir_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            external = root / "external"
            records = [
                LabeledCrop(kind="rank", path=live / "rank" / "A" / "a0.png", label="A"),
                LabeledCrop(kind="rank", path=external / "rank" / "A" / "a1.png", label="A"),
                LabeledCrop(kind="rank", path=external / "rank" / "K" / "k0.png", label="K"),
            ]

            counts = count_labeled_by_source(records, source_dirs=[live, external], labels=["A", "K"])

        self.assertEqual(counts[str(live)]["count"], 1)
        self.assertEqual(counts[str(external)]["count"], 2)
        self.assertIn(f"{external}=2", format_source_counts(counts))

    def test_filter_hf_probe_predictions_rescreens_cached_predictions(self) -> None:
        fields = [
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_crop = root / "rank_A.png"
            review_crop = root / "rank_K.png"
            accepted_crop.write_bytes(b"fake")
            review_crop.write_bytes(b"fake")
            predictions_csv = root / "predictions.csv"
            with predictions_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "index": 0,
                        "kind": "rank",
                        "input_path": str(accepted_crop),
                        "current_label": "A",
                        "teacher_label": "A",
                        "teacher_score": "0.95",
                        "teacher_margin": "0.40",
                        "teacher_second_score": "0.10",
                        "teacher_model": "test",
                    }
                )
                writer.writerow(
                    {
                        "index": 1,
                        "kind": "rank",
                        "input_path": str(review_crop),
                        "current_label": "K",
                        "teacher_label": "K",
                        "teacher_score": "0.60",
                        "teacher_margin": "0.30",
                        "teacher_second_score": "0.20",
                        "teacher_model": "test",
                    }
                )

            payload = filter_hf_probe_predictions(
                predictions_csv=predictions_csv,
                output_dir=root / "filtered",
                rank_score_threshold=0.8,
                rank_margin_threshold=0.1,
                suit_score_threshold=0.8,
                suit_margin_threshold=0.1,
                require_current_agreement=True,
            )

            self.assertEqual(payload["processed"], 2)
            self.assertEqual(payload["accepted"], 1)
            self.assertEqual(payload["review"], 1)
            self.assertTrue((root / "filtered" / "rank" / "A").exists())
            self.assertTrue((root / "filtered" / "review" / "rank").exists())
            self.assertTrue(Path(payload["files"]["predictions_csv"]).exists())
            self.assertTrue(Path(payload["files"]["review_csv"]).exists())

    def test_ensemble_hf_probe_predictions_accepts_only_teacher_agreement(self) -> None:
        fields = [
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agree_crop = root / "rank_A.png"
            disagree_crop = root / "rank_K.png"
            agree_crop.write_bytes(b"fake")
            disagree_crop.write_bytes(b"fake")
            csv_a = root / "a.csv"
            csv_b = root / "b.csv"
            write_prediction_csv(
                csv_a,
                fields,
                [
                    row_for(0, agree_crop, "A", "A", 0.95, 0.30, "teacher-a"),
                    row_for(1, disagree_crop, "K", "K", 0.96, 0.30, "teacher-a"),
                ],
            )
            write_prediction_csv(
                csv_b,
                fields,
                [
                    row_for(0, agree_crop, "A", "A", 0.92, 0.25, "teacher-b"),
                    row_for(1, disagree_crop, "K", "Q", 0.91, 0.24, "teacher-b"),
                ],
            )

            payload = ensemble_hf_probe_predictions(
                predictions_csvs=[csv_a, csv_b],
                output_dir=root / "ensemble",
                rank_score_threshold=0.80,
                rank_margin_threshold=0.10,
                suit_score_threshold=0.80,
                suit_margin_threshold=0.10,
                require_current_agreement=True,
            )

            self.assertEqual(payload["processed"], 2)
            self.assertEqual(payload["accepted"], 1)
            self.assertEqual(payload["review"], 1)
            with Path(payload["files"]["predictions_csv"]).open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["teacher_label"], "A")
            self.assertEqual(rows[0]["teacher_score"], "0.920000")
            disagree = [row for row in rows if row["reason"] == "teacher_disagrees"]
            self.assertEqual(len(disagree), 1)
            self.assertEqual(disagree[0]["teacher_label"], "")

    def test_ensemble_hf_probe_predictions_accepts_any_majority_vote(self) -> None:
        fields = [
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop = root / "rank_A.png"
            crop.write_bytes(b"fake")
            csv_a = root / "a.csv"
            csv_b = root / "b.csv"
            csv_c = root / "c.csv"
            write_prediction_csv(csv_a, fields, [row_for(0, crop, "A", "A", 0.95, 0.30, "teacher-a")])
            write_prediction_csv(csv_b, fields, [row_for(0, crop, "A", "Q", 0.96, 0.31, "teacher-b")])
            write_prediction_csv(csv_c, fields, [row_for(0, crop, "A", "A", 0.90, 0.20, "teacher-c")])

            payload = ensemble_hf_probe_predictions(
                predictions_csvs=[csv_a, csv_b, csv_c],
                output_dir=root / "ensemble",
                rank_score_threshold=0.80,
                rank_margin_threshold=0.10,
                suit_score_threshold=0.80,
                suit_margin_threshold=0.10,
                require_current_agreement=True,
                min_teachers=2,
            )

            self.assertEqual(payload["processed"], 1)
            self.assertEqual(payload["accepted"], 1)
            with Path(payload["files"]["predictions_csv"]).open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["teacher_label"], "A")
            self.assertEqual(rows[0]["teacher_score"], "0.900000")
            self.assertEqual(rows[0]["teacher_margin"], "0.200000")
            self.assertIn("teacher-a", rows[0]["teacher_model"])
            self.assertIn("teacher-c", rows[0]["teacher_model"])


def write_prediction_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fake_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def row_for(index: int, path: Path, current: str, teacher: str, score: float, margin: float, model: str) -> dict[str, object]:
    return {
        "index": index,
        "kind": "rank",
        "input_path": str(path),
        "current_label": current,
        "teacher_label": teacher,
        "teacher_score": score,
        "teacher_margin": margin,
        "teacher_second_score": 0.05,
        "teacher_model": model,
    }


if __name__ == "__main__":
    unittest.main()
