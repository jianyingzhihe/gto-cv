from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gto_cli.card_classifier import classify_rank_glyph, dedupe_feature_table, feature_length, train_card_classifier


class CardClassifierSeedTest(unittest.TestCase):
    def test_train_preserves_seed_model_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed.npz"
            output = root / "candidate.npz"
            rank_features = normalize(np.eye(2, feature_length("rank"), dtype=np.float32))
            suit_features = normalize(np.eye(2, feature_length("suit"), dtype=np.float32))
            np.savez_compressed(
                seed,
                rank_features=rank_features,
                rank_labels=np.asarray(["A", "K"], dtype="<U2"),
                suit_features=suit_features,
                suit_labels=np.asarray(["s", "h"], dtype="<U2"),
                metadata=json.dumps({"kind": "card_glyph_knn"}),
            )

            payload = train_card_classifier(
                model_path=output,
                seed_model_path=seed,
                include_templates=False,
                dataset_dirs=[],
                glyph_dirs=[],
            )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["metadata"]["seed_rank_feature_count"], 2)
            self.assertEqual(payload["metadata"]["seed_suit_feature_count"], 2)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["rank_features"].shape[0], 2)
                self.assertEqual(data["suit_features"].shape[0], 2)

    def test_duplicate_features_keep_later_manual_label(self) -> None:
        feature = normalize(np.ones((1, feature_length("rank")), dtype=np.float32))
        features = np.concatenate([feature, feature], axis=0)
        labels = np.asarray(["3", "8"], dtype="<U2")

        deduped_features, deduped_labels = dedupe_feature_table(features, labels, "rank")

        self.assertEqual(deduped_features.shape[0], 1)
        self.assertEqual(deduped_labels.tolist(), ["8"])

    def test_duplicate_features_can_keep_seed_label(self) -> None:
        feature = normalize(np.ones((1, feature_length("rank")), dtype=np.float32))
        features = np.concatenate([feature, feature], axis=0)
        labels = np.asarray(["3", "8"], dtype="<U2")

        deduped_features, deduped_labels = dedupe_feature_table(
            features,
            labels,
            "rank",
            conflict_policy="keep_seed",
            protected_count=1,
        )

        self.assertEqual(deduped_features.shape[0], 1)
        self.assertEqual(deduped_labels.tolist(), ["3"])

    def test_seed_guard_prefers_strong_seed_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "guarded.npz"
            query = normalize(np.eye(1, feature_length("rank"), dtype=np.float32))[0]
            seed_a = query.copy()
            seed_q = normalize(np.roll(query, 10).reshape(1, -1))[0]
            teacher_k = (query * 0.99 + normalize(np.roll(query, 1).reshape(1, -1))[0] * 0.01).astype(np.float32)
            teacher_k = normalize(teacher_k.reshape(1, -1))[0]
            np.savez_compressed(
                model,
                rank_features=np.stack([seed_a, seed_q, teacher_k]).astype(np.float32),
                rank_labels=np.asarray(["A", "Q", "K"], dtype="<U2"),
                suit_features=np.zeros((0, feature_length("suit")), dtype=np.float32),
                suit_labels=np.asarray([], dtype="<U2"),
                metadata=json.dumps(
                    {
                        "kind": "card_glyph_knn",
                        "seed_guard": True,
                        "seed_rank_feature_count": 2,
                        "seed_guard_rank_score": 0.55,
                        "seed_guard_rank_margin": 0.10,
                    }
                ),
            )

            with patch("gto_cli.card_classifier.glyph_feature", return_value=query):
                result = classify_rank_glyph(np.zeros((70, 54), dtype=np.uint8), model_path=model)

            self.assertIsNotNone(result)
            self.assertEqual(result["label"], "A")
            self.assertTrue(result["seed_guard"])


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-6)).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
