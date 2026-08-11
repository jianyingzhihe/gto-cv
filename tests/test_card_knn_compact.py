from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.card_classifier import RANK_LABELS, SUIT_LABELS, feature_length, load_cv
from gto_cli.card_knn_compact import compact_card_classifier


class CardKnnCompactTest(unittest.TestCase):
    def test_compact_writes_smaller_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "source.npz"
            output = root / "compact.npz"
            rank_path = root / "rank.png"
            suit_path = root / "suit.png"
            rows_csv = root / "rows.csv"
            write_gray_image(rank_path, width=54, height=70)
            write_gray_image(suit_path, width=42, height=42)
            write_model(model)
            with rows_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["expected_rank", "expected_suit", "rank_path", "suit_path"])
                writer.writeheader()
                writer.writerow({"expected_rank": "A", "expected_suit": "s", "rank_path": str(rank_path), "suit_path": str(suit_path)})

            payload = compact_card_classifier(
                model_path=model,
                output_model=output,
                benchmark_rows_csvs=[rows_csv],
                top_per_sample=1,
                min_per_label=1,
                max_per_label=1,
            )
            with np.load(output, allow_pickle=False) as compact:
                rank_count = compact["rank_features"].shape[0]
                suit_count = compact["suit_features"].shape[0]

        self.assertTrue(payload["ok"])
        self.assertLess(rank_count, 26)
        self.assertLess(suit_count, 8)


def write_gray_image(path: Path, *, width: int, height: int) -> None:
    cv2, np_mod = load_cv()
    image = np_mod.zeros((height, width), dtype=np_mod.uint8)
    image[height // 4 : height // 2, width // 4 : width // 2] = 255
    cv2.imwrite(str(path), image)


def write_model(path: Path) -> None:
    rng = np.random.default_rng(7)
    rank_labels = np.asarray([label for label in RANK_LABELS for _ in range(2)], dtype="<U2")
    suit_labels = np.asarray([label for label in SUIT_LABELS for _ in range(2)], dtype="<U2")
    rank_features = normalize_rows(rng.normal(size=(len(rank_labels), feature_length("rank"))).astype(np.float32))
    suit_features = normalize_rows(rng.normal(size=(len(suit_labels), feature_length("suit"))).astype(np.float32))
    np.savez_compressed(
        str(path),
        rank_features=rank_features,
        rank_labels=rank_labels,
        suit_features=suit_features,
        suit_labels=suit_labels,
        metadata=json.dumps({"kind": "test"}, ensure_ascii=False),
    )


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return (features / norm).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
