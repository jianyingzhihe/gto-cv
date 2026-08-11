from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from gto_cli.bbox_utils import (
    find_latest_bbox_file,
    load_bbox_text,
    load_outer_bbox_text,
    load_rebased_analysis_bbox_text,
    resolve_bbox_text,
    reviewed_bbox_requires_refresh,
)
from gto_cli.cv_health import (
    build_fast_live_command,
    build_health_checks,
    build_health_command,
    build_live_command,
    build_preflight_command,
    compact_validation,
    format_cv_health_summary,
    inspect_bbox,
    inspect_deep_model_dir,
)


class CvHealthTest(unittest.TestCase):
    def test_inspect_bbox_rejects_placeholder(self) -> None:
        info = inspect_bbox("x,y,w,h")
        self.assertFalse(info["concrete"])
        self.assertEqual(info["reason"], "non_numeric")

    def test_inspect_bbox_accepts_numeric_values(self) -> None:
        info = inspect_bbox("136,123,1534,1058")
        self.assertTrue(info["concrete"])
        self.assertEqual(info["normalized"], "136,123,1534,1058")

    def test_inspect_bbox_rejects_non_positive_size(self) -> None:
        info = inspect_bbox("1,2,0,4")
        self.assertFalse(info["concrete"])
        self.assertEqual(info["reason"], "non_positive_size")

    def test_load_bbox_text_reads_pick_bbox_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bbox.json"
            path.write_text(
                '{"left": 141, "top": 382, "width": 1152, "height": 807, "text": "141,382,1152,807"}',
                encoding="utf-8",
            )

            self.assertEqual(load_bbox_text(path), "141,382,1152,807")
            self.assertEqual(resolve_bbox_text("x,y,w,h", bbox_file=path), "141,382,1152,807")

    def test_reviewed_inner_bbox_retains_full_client_capture_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis_bbox.json"
            path.write_text(
                '{"left":348,"top":468,"width":986,"height":651,'
                '"outer_region":{"left":239,"top":328,"width":1214,"height":854}}',
                encoding="utf-8",
            )

            self.assertEqual(load_bbox_text(path), "348,468,986,651")
            self.assertEqual(load_outer_bbox_text(path), "239,328,1214,854")

    def test_reviewed_inner_bbox_uses_sibling_manual_bbox_as_canonical_outer_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = root / "analysis_bbox.json"
            manual = root / "bbox.json"
            reviewed.write_text(
                '{"text":"348,468,986,651",'
                '"outer_region":{"left":239,"top":328,"width":1214,"height":854}}',
                encoding="utf-8",
            )
            manual.write_text('{"text":"212,350,1233,864"}', encoding="utf-8")

            self.assertEqual(load_outer_bbox_text(reviewed), "212,350,1233,864")

    def test_reviewed_bbox_requires_refresh_after_new_manual_outer_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = root / "analysis_bbox.json"
            manual = root / "bbox.json"
            reviewed.write_text('{"text":"10,20,30,40"}', encoding="utf-8")
            manual.write_text('{"text":"1,2,3,4"}', encoding="utf-8")
            os.utime(reviewed, ns=(1_000_000_000, 1_000_000_000))
            os.utime(manual, ns=(2_000_000_000, 2_000_000_000))

            self.assertTrue(reviewed_bbox_requires_refresh(reviewed))
            self.assertFalse(reviewed_bbox_requires_refresh(manual))

    def test_manual_outer_bbox_reuses_reviewed_inner_table_by_relative_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "bbox.json"
            reviewed = root / "analysis_bbox.json"
            manual.write_text('{"text":"10,20,1200,600"}', encoding="utf-8")
            reviewed.write_text(
                '{"text":"150,250,800,300",'
                '"outer_reference":{"left":100,"top":200,"width":1000,"height":500}}',
                encoding="utf-8",
            )
            self.assertEqual(load_rebased_analysis_bbox_text(manual), "70,80,960,360")

    def test_manual_outer_bbox_rejects_large_aspect_ratio_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "bbox.json"
            reviewed = root / "analysis_bbox.json"
            manual.write_text('{"text":"10,20,600,1200"}', encoding="utf-8")
            reviewed.write_text(
                '{"text":"150,250,800,300",'
                '"outer_reference":{"left":100,"top":200,"width":1000,"height":500}}',
                encoding="utf-8",
            )
            self.assertIsNone(load_rebased_analysis_bbox_text(manual))

    def test_find_latest_bbox_file_picks_newest_nested_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old" / "bbox.json"
            new = root / "new" / "bbox.json"
            old.parent.mkdir()
            new.parent.mkdir()
            old.write_text('{"text": "1,2,3,4"}', encoding="utf-8")
            new.write_text('{"text": "5,6,7,8"}', encoding="utf-8")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))

            self.assertEqual(find_latest_bbox_file(root), new)
            self.assertEqual(resolve_bbox_text(None, latest_bbox=True, latest_root=root), "5,6,7,8")

    def test_deep_model_is_optional_when_disabled(self) -> None:
        deep = inspect_deep_model_dir(None)
        checks = build_health_checks(
            knn_info={"exists": True, "rank_labels_ok": True, "suit_labels_ok": True, "metadata": {"rank_labels": [], "suit_labels": []}},
            deep_info=deep,
            validation=None,
            gate=None,
            bbox_info={"concrete": True, "normalized": "1,2,3,4"},
            allow_placeholder_bbox=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300,
            max_p90_ms=900,
        )

        self.assertFalse(deep["enabled"])
        self.assertTrue(next(item for item in checks if item["name"] == "deep_model_optional")["pass"])
        self.assertNotIn("deep_rank_model_exists", {item["name"] for item in checks})

    def test_validation_card_health_adds_health_checks_and_summary_text(self) -> None:
        validation = {
            "ok": True,
            "real_problem_count": 0,
            "board_bad_count": 0,
            "timing_ms": {"median": 40, "p90": 80},
            "card_health": {
                "hero": {
                    "complete_frames": 10,
                    "incomplete_or_missed_frames": 0,
                    "turn_blocked_frames": 0,
                },
                "board": {"bad_frames": 0},
                "issue_counts": {},
            },
        }

        checks = build_health_checks(
            knn_info={"exists": True, "rank_labels_ok": True, "suit_labels_ok": True, "metadata": {"rank_labels": [], "suit_labels": []}},
            deep_info=inspect_deep_model_dir(None),
            validation=validation,
            gate={"promote": True, "decision": "promote"},
            bbox_info={"concrete": True, "normalized": "1,2,3,4"},
            allow_placeholder_bbox=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300,
            max_p90_ms=900,
        )
        payload = {
            "decision": "ready",
            "ready": True,
            "models": {"knn": {}, "deep": {}},
            "bbox": {},
            "validation": compact_validation(validation),
            "checks": checks,
            "files": {},
        }
        text = format_cv_health_summary(payload)

        self.assertTrue(next(item for item in checks if item["name"] == "validation_card_issue_count")["pass"])
        self.assertEqual(payload["validation"]["card_health"]["hero"]["complete_frames"], 10)
        self.assertIn("Card health: hero_complete=10", text)

    def test_generated_commands_include_live_options(self) -> None:
        bbox = "136,123,1534,1058"
        health = build_health_command(
            bbox=bbox,
            output_dir=Path("video_frames") / "cv_health_promoted",
            hero_name="hero",
        )
        live = build_live_command(
            bbox=bbox,
            screen_output_dir=Path("video_frames") / "screen_live",
            hero_name="hero",
            effective_stack=100,
            villain="standard",
            min_confidence=0.35,
            ocr_scale=0.65,
            dealer_refresh_frames=4,
            auto_bbox_refresh=10,
            deep_model_dir=Path("pict") / "card_models" / "deep_realtime_v2_temporal",
        )
        for command in (health, live):
            self.assertIn('--bbox "136,123,1534,1058"', command)
            self.assertIn('--hero-name "hero"', command)
            self.assertIn('--villain "standard"', command)
            self.assertIn("--ocr-scale 0.65", command)
            self.assertIn("--dealer-refresh-frames 4", command)
        self.assertNotIn("--deep-card-model-dir", health)
        self.assertIn('--deep-card-model-dir "pict\\card_models\\deep_realtime_v2_temporal"', live)
        self.assertNotIn("--ocr-action-only", live)

    def test_generated_fast_live_command_uses_action_only_ocr(self) -> None:
        command = build_fast_live_command(
            bbox="136,123,1534,1058",
            screen_output_dir=Path("video_frames") / "screen_live_fast",
            hero_name="hero",
            effective_stack=100,
            villain="standard",
            min_confidence=0.35,
            ocr_scale=0.65,
            dealer_refresh_frames=4,
            auto_bbox_refresh=10,
            deep_model_dir=Path("pict") / "card_models" / "deep_realtime_v2_temporal",
        )
        self.assertIn("--ocr-action-only", command)
        self.assertIn('--output-dir "video_frames\\screen_live_fast"', command)

    def test_generated_commands_can_use_bbox_file_without_deep_fallback(self) -> None:
        bbox_file = Path("video_frames") / "screen_calibrate" / "bbox.json"
        live = build_live_command(
            bbox="136,123,1534,1058",
            bbox_file=bbox_file,
            screen_output_dir=Path("video_frames") / "screen_live",
            hero_name=None,
            effective_stack=100,
            villain="standard",
            min_confidence=0.35,
            ocr_scale=0.65,
            dealer_refresh_frames=4,
            auto_bbox_refresh=10,
            deep_model_dir=None,
        )
        health = build_health_command(
            bbox="136,123,1534,1058",
            bbox_file=bbox_file,
            output_dir=Path("video_frames") / "cv_health_promoted",
            hero_name=None,
        )

        for command in (live, health):
            self.assertIn('--bbox-file "video_frames\\screen_calibrate\\bbox.json"', command)
            self.assertNotIn('--bbox "136,123,1534,1058"', command)
            self.assertNotIn("--deep-card-model-dir", command)

    def test_generated_preflight_command_saves_single_frame(self) -> None:
        command = build_preflight_command(
            bbox="136,123,1534,1058",
            preflight_output_dir=Path("video_frames") / "screen_preflight",
            hero_name="hero",
            effective_stack=100,
            villain="standard",
            min_confidence=0.35,
            ocr_scale=0.65,
            dealer_refresh_frames=4,
            auto_bbox_refresh=10,
            deep_model_dir=Path("pict") / "card_models" / "deep_realtime_v2_temporal",
        )
        self.assertIn("--preflight-once", command)
        self.assertIn("--save-frames", command)
        self.assertIn("--save-annotated", command)
        self.assertIn('--output-dir "video_frames\\screen_preflight"', command)


if __name__ == "__main__":
    unittest.main()
