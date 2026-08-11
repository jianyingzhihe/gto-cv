from __future__ import annotations

import os
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gto_cli import card_big_teacher as card_big_teacher_module
from gto_cli.card_big_teacher import (
    default_big_teacher_benchmark_review_csvs,
    distill_big_teacher_runtime,
    merge_runtime_truth_if_available,
    model_resolution_from_reused_probe,
    prepare_runtime_risk_label_queue,
    resolve_big_teacher_model_names,
    temporary_model_env,
)


class CardBigTeacherTest(unittest.TestCase):
    def test_auto_model_picks_first_loadable_local_candidate(self) -> None:
        def fake_check(model: str, *, device: str = "auto") -> tuple[bool, str]:
            del device
            return (model == "openai/clip-vit-base-patch32", "ok" if model == "openai/clip-vit-base-patch32" else "missing")

        rank_model, suit_model, metadata = resolve_big_teacher_model_names(
            model_name="auto",
            rank_model=None,
            suit_model=None,
            kind="both",
            device="cpu",
            local_files_only=False,
            loadable_check=fake_check,
        )

        self.assertEqual(rank_model, "openai/clip-vit-base-patch32")
        self.assertEqual(suit_model, "openai/clip-vit-base-patch32")
        self.assertEqual(metadata["mode"], "auto-local")
        self.assertEqual(metadata["rank"]["source"], "auto-local")
        self.assertGreaterEqual(len(metadata["rank"]["attempts"]), 2)

    def test_explicit_rank_model_overrides_auto_shared_model(self) -> None:
        def fake_check(model: str, *, device: str = "auto") -> tuple[bool, str]:
            del model, device
            return False, "should_not_matter"

        rank_model, suit_model, metadata = resolve_big_teacher_model_names(
            model_name="auto",
            rank_model="rank/model",
            suit_model="suit/model",
            kind="both",
            device="cpu",
            local_files_only=True,
            loadable_check=fake_check,
        )

        self.assertEqual(rank_model, "rank/model")
        self.assertEqual(suit_model, "suit/model")
        self.assertEqual(metadata["rank"]["source"], "explicit")
        self.assertEqual(metadata["suit"]["source"], "explicit")

    def test_runtime_distill_stops_without_accepted_teacher_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = distill_big_teacher_runtime(
                teacher_label_summary={
                    "accepted": 0,
                    "files": {"accepted_dir": str(root / "missing_accepted")},
                },
                output_dir=root / "runtime",
                video_paths=[],
                benchmark_review_csvs=[root / "missing_review.csv"],
                baseline_review_csv=root / "missing_review.csv",
                baseline_validation_summary_json=None,
                deep_card_model_dir=None,
                min_accepted=1,
            )

            self.assertTrue(payload["stopped"])
            self.assertEqual(payload["reason"], "not_enough_accepted_teacher_crops")
            self.assertTrue((root / "runtime" / "runtime_distill_summary.json").exists())
            self.assertTrue((root / "runtime" / "runtime_distill_runbook.md").exists())

    def test_runtime_distill_preserves_seed_model_when_training_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted"
            accepted.mkdir()
            seed = root / "seed.npz"
            model = root / "runtime" / "candidate.npz"

            with (
                patch("gto_cli.card_big_teacher.train_card_classifier") as train,
                patch("gto_cli.card_big_teacher.export_card_review") as review,
                patch("gto_cli.card_big_teacher.validate_cv_videos") as validate,
                patch("gto_cli.card_big_teacher.gate_card_model") as gate,
                patch("gto_cli.card_big_teacher.summarize_card_candidates") as summarize,
                patch("gto_cli.card_big_teacher.clear_glyph_classify_cache") as clear_cache,
            ):
                train.return_value = {"ok": True, "metadata": {}}
                review.return_value = {"files": {"review_csv": str(root / "review.csv")}}
                validate.return_value = {"files": {"summary": str(root / "validation.json")}}
                gate.return_value = {"promote": False, "decision": "reject", "diff": {"counts": {"risk_count": 0}}, "files": {}}
                summarize.return_value = {"ok": True}

                payload = distill_big_teacher_runtime(
                    teacher_label_summary={
                        "accepted": 1,
                        "files": {"accepted_dir": str(accepted)},
                    },
                    output_dir=root / "runtime",
                    model_path=model,
                    video_paths=[],
                    benchmark_review_csvs=[root / "missing_review.csv"],
                    baseline_review_csv=root / "missing_review.csv",
                    baseline_validation_summary_json=None,
                    deep_card_model_dir=None,
                    seed_model_path=seed,
                    seed_conflict_policy="keep_seed",
                    seed_guard=True,
                    seed_guard_rank_score=0.61,
                    seed_guard_rank_margin=0.11,
                    seed_guard_suit_score=0.71,
                    seed_guard_suit_margin=0.05,
                    min_accepted=1,
                    prepare_risk_queue=False,
                )

            self.assertEqual(payload["seed_model_path"], str(seed))
            self.assertEqual(payload["seed_conflict_policy"], "keep_seed")
            self.assertEqual(train.call_args.kwargs["seed_model_path"], seed)
            self.assertEqual(train.call_args.kwargs["seed_conflict_policy"], "keep_seed")
            self.assertTrue(train.call_args.kwargs["seed_guard"])
            self.assertEqual(train.call_args.kwargs["seed_guard_rank_score"], 0.61)
            self.assertEqual(train.call_args.kwargs["seed_guard_suit_margin"], 0.05)
            clear_cache.assert_called_once()

    def test_reused_probe_model_resolution_uses_probe_metadata(self) -> None:
        resolution = model_resolution_from_reused_probe(
            existing_resolution={
                "mode": "auto-local",
                "rank": {"selected": "facebook/dinov2-base", "source": "auto-local"},
                "suit": {"selected": "facebook/dinov2-base", "source": "auto-local"},
            },
            probe_metadata={
                "rank": {"model_name": "facebook/dinov2-large", "model_path": "probe/hf_rank_probe.npz"},
                "suit": {"model_name": "facebook/dinov2-base", "model_path": "probe/hf_suit_probe.npz"},
            },
            effective_rank_model="facebook/dinov2-large",
            effective_suit_model="facebook/dinov2-base",
            kind="both",
        )

        self.assertEqual(resolution["mode"], "probe-reused")
        self.assertEqual(resolution["rank"]["selected"], "facebook/dinov2-large")
        self.assertEqual(resolution["rank"]["source"], "probe_metadata")
        self.assertEqual(resolution["suit"]["selected"], "facebook/dinov2-base")

    def test_default_benchmark_review_csvs_include_manual_truth_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual_truth" / "label_queue.csv"
            fallback = root / "fallback" / "review.csv"
            manual.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            manual.write_text("video,final_card0\n", encoding="utf-8")
            fallback.write_text("video,card0\n", encoding="utf-8")

            with (
                patch.object(card_big_teacher_module, "DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV", manual),
                patch.object(card_big_teacher_module, "DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV", fallback),
            ):
                self.assertEqual(default_big_teacher_benchmark_review_csvs(), [manual, fallback])

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

    def test_prepare_runtime_risk_label_queue_skips_when_no_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = prepare_runtime_risk_label_queue(
                gate={
                    "diff": {"counts": {"risk_count": 0}},
                    "files": {"diff_rows_csv": str(Path(tmp) / "missing.csv")},
                },
                output_dir=Path(tmp) / "queue",
            )

            self.assertIsNone(payload)

    def test_merge_runtime_truth_if_available_applies_queue_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_csv = root / "review.csv"
            queue_csv = root / "queue.csv"
            write_review_for_truth_merge(review_csv)
            write_queue_for_truth_merge(queue_csv)

            output = merge_runtime_truth_if_available(
                review_csv=review_csv,
                truth_csvs=[root / "missing.csv", queue_csv],
                output_dir=root / "out",
            )

            self.assertEqual(output.name, "review_with_truth.csv")
            with output.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["final_card1"], "6s")


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


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


if __name__ == "__main__":
    unittest.main()
