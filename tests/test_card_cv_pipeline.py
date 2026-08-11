from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gto_cli.card_classifier import load_cv
from gto_cli.card_cv_pipeline import (
    build_probe_train_input_dirs,
    build_pipeline_checks,
    build_pipeline_commands,
    choose_next_stage,
    compact_auto_bbox_diagnostics,
    compact_probe_train_run,
    compact_teacher_run,
    filter_ingest_dataset_dirs,
    format_card_cv_pipeline_runbook,
    format_card_cv_pipeline_summary,
    inspect_crop_dirs,
    inspect_probe_dir,
    write_command_files,
)
from gto_cli.card_hf_probe import LabeledCrop, build_probe_payload, save_probe


class CardCvPipelineTest(unittest.TestCase):
    def test_inspect_crop_dirs_counts_rank_and_suit_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_placeholder(root / "rank" / "A" / "a_rank.png")
            write_placeholder(root / "rank" / "K" / "k_rank.png")
            write_placeholder(root / "suit" / "s" / "s_suit.png")

            summary = inspect_crop_dirs([root])

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["rank_count"], 2)
        self.assertEqual(summary["suit_count"], 1)
        self.assertIn("A", summary["by_dir"][0]["rank_labels"])
        self.assertIn("s", summary["by_dir"][0]["suit_labels"])

    def test_inspect_crop_dirs_audits_label_coverage_and_bad_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_png(root / "rank" / "A" / "a_rank.png")
            write_placeholder(root / "rank" / "K" / "bad_rank.png")
            write_png(root / "suit" / "s" / "s_suit.png")

            summary = inspect_crop_dirs([root], audit_images=True, min_rank_per_label=2, min_suit_per_label=1)

        self.assertEqual(summary["unreadable_count"], 1)
        self.assertFalse(summary["rank_labels_complete"])
        self.assertFalse(summary["suit_labels_complete"])
        self.assertIn("Q", summary["missing_rank_labels"])
        self.assertIn("h", summary["missing_suit_labels"])
        self.assertIn("A", summary["rare_rank_labels"])

    def test_inspect_probe_dir_reads_rank_and_suit_probe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_probe(root / "hf_rank_probe.npz", fake_probe_payload("rank", ["A", "K"]))
            save_probe(root / "hf_suit_probe.npz", fake_probe_payload("suit", ["s", "h"]))

            summary = inspect_probe_dir(root)

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["rank"]["kind"], "rank")
        self.assertEqual(summary["suit"]["kind"], "suit")
        self.assertEqual(summary["rank"]["backend"], "hf_embedding_probe")

    def test_choose_next_stage_prioritizes_bbox_before_training(self) -> None:
        checks = [
            {"name": "bbox_concrete", "pass": False},
            {"name": "crop_rank_present", "pass": False},
            {"name": "hf_probe_ready", "pass": False},
        ]

        self.assertEqual(choose_next_stage(checks), "pick_or_fix_bbox")

    def test_build_pipeline_checks_includes_card_health_when_available(self) -> None:
        checks = build_pipeline_checks(
            bbox_info={"concrete": True, "normalized": "1,2,3,4"},
            videos=[Path("video.mp4")],
            crop_summary={
                "rank_count": 13,
                "suit_count": 4,
                "rank_labels_complete": True,
                "suit_labels_complete": True,
                "rank_min_per_label_ok": True,
                "suit_min_per_label_ok": True,
                "images_readable": True,
                "unreadable_count": 0,
            },
            probe={"ready": True, "path": "probe"},
            knn={"exists": True, "rank_labels_ok": True, "suit_labels_ok": True, "path": "knn"},
            deep={"enabled": False},
            validation={
                "ok": True,
                "real_problem_count": 0,
                "board_bad_count": 0,
                "timing_ms": {"median": 30, "p90": 60},
                "card_health": {
                    "hero": {"incomplete_or_missed_frames": 1, "turn_blocked_frames": 0},
                    "board": {"bad_frames": 0},
                    "issue_counts": {"hero_card_unknown": 1},
                },
            },
            gate={"promote": True, "decision": "promote"},
        )
        checks_by_name = {check["name"]: check for check in checks}

        self.assertFalse(checks_by_name["validation_hero_incomplete_or_missed_zero"]["pass"])
        self.assertFalse(checks_by_name["validation_card_issue_count_zero"]["pass"])

    def test_build_pipeline_commands_include_hero_and_split_probe_flow(self) -> None:
        commands = build_pipeline_commands(
            bbox="1,2,300,400",
            hero_name="于寻欢",
            crop_dirs=[Path("video_frames/crops")],
            probe_dir=Path("pict/card_models/probe"),
            dataset_dir=Path("pict/card_datasets/cards"),
            extra_dataset_dirs=[Path("pict/card_datasets/manual")],
            skipped_ingest_dataset_dirs=[],
            ingested_dataset_dir=Path("video_frames/external_glyphs"),
            output_dir=Path("video_frames/pipeline"),
            video_dir=Path("video_frames"),
            knn_model_path=Path("pict/card_models/card_glyph_knn.npz"),
            deep_model_dir=Path("pict/card_models/deep_realtime_v2_temporal"),
            auto_bbox_output_dir=Path("video_frames/auto_bbox"),
            auto_bbox_every_sec=123,
            auto_bbox_max_frames=2,
            auto_bbox_variants=["native", "loose_8"],
        )

        self.assertIn("--hero-name \"于寻欢\"", commands["live"])
        self.assertNotIn("--deep-card-model-dir", commands["live"])
        self.assertNotIn("--deep-card-model-dir", commands["validate_current"])
        self.assertIn("--deep-card-model-dir", commands["export_review"])
        self.assertIn("--input-dir \"video_frames\\crops\"", commands["label_with_probe"].replace("/", "\\"))
        self.assertIn("--input-dir \"video_frames\\crops\"", commands["train_probe"].replace("/", "\\"))
        self.assertIn("--input-dir \"video_frames\\external_glyphs\"", commands["train_probe"].replace("/", "\\"))
        self.assertNotIn("--input-dir \"video_frames\\external_glyphs\"", commands["label_with_probe"].replace("/", "\\"))
        self.assertIn("--dataset-dir \"pict\\card_datasets\\cards\"", commands["ingest_dataset"].replace("/", "\\"))
        self.assertIn("--dataset-dir \"pict\\card_datasets\\manual\"", commands["ingest_dataset"].replace("/", "\\"))
        self.assertIn("--rank-score-threshold", commands["distill_and_gate"])
        self.assertIn("--suit-score-threshold", commands["distill_and_gate"])
        self.assertIn("sweep-card-hf-thresholds", commands["threshold_sweep"])
        self.assertIn("big_teacher_label", commands["threshold_sweep"])
        self.assertIn("filter-card-hf-predictions", commands["filter_predictions_distill"])
        self.assertIn("--distill-runtime", commands["filter_predictions_distill"])
        self.assertIn("prepare-card-glyph-label-queue", commands["prepare_glyph_label_queue"])
        self.assertIn("apply-card-glyph-label-queue", commands["apply_glyph_label_queue"])
        self.assertIn("diagnose-auto-bbox --all", commands["diagnose_auto_bbox"])
        self.assertIn("--variant \"native\"", commands["diagnose_auto_bbox"])
        self.assertIn("--max-frames 2", commands["diagnose_auto_bbox"])

    def test_build_pipeline_commands_omit_optional_deep_export_arg(self) -> None:
        commands = build_pipeline_commands(
            bbox="1,2,300,400",
            hero_name=None,
            crop_dirs=[Path("video_frames/crops")],
            probe_dir=Path("pict/card_models/probe"),
            dataset_dir=Path("pict/card_datasets/cards"),
            extra_dataset_dirs=[],
            skipped_ingest_dataset_dirs=[],
            ingested_dataset_dir=Path("video_frames/external_glyphs"),
            output_dir=Path("video_frames/pipeline"),
            video_dir=Path("video_frames"),
            knn_model_path=Path("pict/card_models/card_glyph_knn.npz"),
            deep_model_dir=None,
        )

        self.assertNotIn("--deep-card-model-dir", commands["export_review"])
        self.assertNotIn('"None"', commands["export_review"])

    def test_build_pipeline_commands_support_split_probe_models(self) -> None:
        commands = build_pipeline_commands(
            bbox="1,2,300,400",
            hero_name=None,
            crop_dirs=[Path("video_frames/crops")],
            probe_dir=Path("pict/card_models/probe"),
            probe_model="facebook/dinov2-base",
            probe_rank_model="facebook/dinov2-large",
            probe_suit_model="openai/clip-vit-large-patch14",
            probe_max_images_per_class=32,
            probe_batch_size=8,
            dataset_dir=Path("pict/card_datasets/cards"),
            extra_dataset_dirs=[],
            skipped_ingest_dataset_dirs=[],
            ingested_dataset_dir=Path("video_frames/external_glyphs"),
            output_dir=Path("video_frames/pipeline"),
            video_dir=Path("video_frames"),
            knn_model_path=Path("pict/card_models/card_glyph_knn.npz"),
            deep_model_dir=None,
        )

        train_probe = commands["train_probe"]

        self.assertIn('--rank-model "facebook/dinov2-large"', train_probe)
        self.assertIn('--suit-model "openai/clip-vit-large-patch14"', train_probe)
        self.assertIn('--input-dir "video_frames\\external_glyphs"', train_probe.replace("/", "\\"))
        self.assertIn("--max-images-per-class 32", train_probe)
        self.assertIn("--batch-size 8", train_probe)

    def test_summary_and_runbook_include_dataset_card_coverage(self) -> None:
        payload = {
            "ok": True,
            "output_dir": "out",
            "next_stage": "ready_for_live_and_training_iteration",
            "ready_for_live": True,
            "ready_for_training": True,
            "videos": {"count": 1},
            "crops": {"rank_count": 13, "suit_count": 4},
            "models": {"hf_probe": {"ready": True}},
            "probe_train_input_dirs": ["crops", "external_glyphs"],
            "dataset": {
                "output_dir": "cards",
                "image_count": 52,
                "label_dir_count": 52,
                "likely_dataset_roots": ["cards/root"],
                "card_coverage": {
                    "parsed_card_count": 52,
                    "expected_card_count": 52,
                    "complete_deck": True,
                    "missing_cards": [],
                    "duplicate_cards": {},
                    "unparsed_count": 0,
                },
            },
            "files": {"runbook": "out/runbook.md"},
            "checks": [],
            "commands": {},
        }

        summary = format_card_cv_pipeline_summary(payload)
        runbook = format_card_cv_pipeline_runbook(payload)

        self.assertIn("Dataset deck: parsed=52/52 complete=True missing=0 duplicate_labels=0 unparsed=0", summary)
        self.assertIn("Probe train inputs: ['crops', 'external_glyphs']", summary)
        self.assertIn("## Dataset Coverage", runbook)
        self.assertIn("- Complete deck: `True`", runbook)
        self.assertIn("- Probe train input dirs: `['crops', 'external_glyphs']`", runbook)

    def test_compact_auto_bbox_diagnostics_keeps_key_metrics(self) -> None:
        compact = compact_auto_bbox_diagnostics(
            {
                "ok": True,
                "video_count": 2,
                "sample": {"row_count": 10},
                "failure_count": 0,
                "counts": {"ok": 10},
                "timing_ms": {"median": 12.3},
                "files": {"report_md": "report.md"},
            }
        )

        self.assertEqual(compact["row_count"], 10)
        self.assertEqual(compact["failure_count"], 0)
        self.assertEqual(compact["timing_ms"]["median"], 12.3)

    def test_filter_ingest_dataset_dirs_skips_output_dir_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()

            kept, skipped = filter_ingest_dataset_dirs([source, output, source], output_dir=output)

        self.assertEqual([path.name for path in kept], ["source"])
        self.assertEqual([path.name for path in skipped], ["output", "source"])

    def test_build_probe_train_input_dirs_adds_ingested_dataset_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            external = root / "external"

            dirs = build_probe_train_input_dirs([crops, external], ingested_dataset_dir=external, include_ingested=True)

        self.assertEqual([path.name for path in dirs], ["crops", "external"])

    def test_write_command_files_uses_utf8_bom_for_windows_powershell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = write_command_files(Path(tmp), {"live": "python gto.py --hero-name \"于寻欢\""})
            data = Path(files["live"]).read_bytes()

        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertIn("于寻欢", data.decode("utf-8-sig"))

    def test_compact_teacher_run_omits_distill_when_not_run(self) -> None:
        compact = compact_teacher_run(
            {
                "ok": True,
                "output_dir": "out",
                "probe_dir": "probe",
                "rank_model": "rank-model",
                "suit_model": "suit-model",
                "label": {"processed": 2, "accepted": 1, "review": 1},
                "files": {},
                "distill_runtime": None,
            }
        )

        self.assertIsNotNone(compact)
        self.assertNotIn("distill_runtime", compact or {})

    def test_compact_probe_train_run_keeps_split_metrics(self) -> None:
        compact = compact_probe_train_run(
            {
                "ok": True,
                "output_dir": "probe",
                "input_dirs": ["crops", "external_glyphs"],
                "kind": "both",
                "rank_model": "rank-model",
                "suit_model": "suit-model",
                "results": {
                    "rank": {
                        "source_count": 13,
                        "source_counts_by_dir": {"crops": {"count": 9}, "external_glyphs": {"count": 4}},
                        "val": {"accuracy": 0.75},
                    },
                    "suit": {
                        "source_count": 4,
                        "source_counts_by_dir": {"crops": {"count": 2}, "external_glyphs": {"count": 2}},
                        "val": {"accuracy": 1.0},
                    },
                },
                "files": {"rank_probe": "rank.npz", "suit_probe": "suit.npz"},
            }
        )

        self.assertEqual(compact["rank_model"], "rank-model")
        self.assertEqual(compact["suit_model"], "suit-model")
        self.assertEqual(compact["input_dirs"], ["crops", "external_glyphs"])
        self.assertEqual(compact["rank_source_counts_by_dir"]["external_glyphs"]["count"], 4)
        self.assertEqual(compact["suit_source_counts_by_dir"]["external_glyphs"]["count"], 2)
        self.assertEqual(compact["rank_val_acc"], 0.75)
        self.assertEqual(compact["suit_val_acc"], 1.0)


def write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image")


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2, _np = load_cv()
    cv2.imwrite(str(path), np.full((8, 8, 3), 255, dtype=np.uint8))


def fake_probe_payload(kind: str, labels: list[str]) -> dict:
    records = [LabeledCrop(kind=kind, path=Path(f"{label}.png"), label=label) for label in labels]
    features = np.eye(len(labels), dtype="float32")
    return build_probe_payload(
        records=records,
        features=features,
        labels=labels,
        kind=kind,
        model_name="fake-model",
        temperature=0.04,
    )


if __name__ == "__main__":
    unittest.main()
