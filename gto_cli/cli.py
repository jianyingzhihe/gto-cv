from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .advisor import advise_state
from .card_classifier import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPLATE_DIR,
    format_card_classifier_summary,
    train_card_classifier,
)
from .card_glyph_export import (
    export_card_glyphs,
    format_card_glyph_export_summary,
    format_external_ingest_summary,
    ingest_external_card_images,
)
from .card_review_export import export_card_review, format_card_review_summary
from .card_debug_review import collect_card_debug_review, format_card_debug_review_summary
from .card_teacher_label import (
    format_card_crop_label_summary,
    format_organize_card_crops_summary,
    label_card_crops,
    organize_card_crops,
)
from .card_hf_teacher import (
    DEFAULT_HF_CLIP_MODEL,
    format_hf_card_crop_label_summary,
    label_card_crops_hf,
)
from .card_hf_probe import (
    DEFAULT_HF_PROBE_MODEL,
    apply_hf_probe_to_review,
    ensemble_hf_probe_predictions,
    filter_hf_probe_predictions,
    format_hf_probe_ensemble_summary,
    format_hf_probe_filter_summary,
    format_hf_probe_label_summary,
    format_hf_probe_review_summary,
    format_hf_probe_train_summary,
    label_card_crops_hf_probe,
    train_hf_card_probe,
)
from .card_hf_threshold_sweep import (
    format_threshold_sweep_summary,
    sweep_hf_prediction_thresholds,
)
from .card_synthetic import generate_synthetic_card_glyphs, format_synthetic_summary
from .card_dataset_download import (
    DEFAULT_HF_CARD_DIR,
    DEFAULT_HF_CARD_REPO,
    download_card_dataset,
    format_dataset_download_summary,
)
from .card_cv_pipeline import (
    DEFAULT_PIPELINE_CROP_DIR,
    DEFAULT_PIPELINE_INGEST_DIR,
    DEFAULT_PIPELINE_OUTPUT_DIR,
    DEFAULT_PIPELINE_PROBE_DIR,
    format_card_cv_pipeline_summary,
    inspect_card_cv_pipeline,
)
from .card_hand_audit import audit_card_review, format_hand_audit_summary
from .card_benchmark import benchmark_card_review, format_card_benchmark_summary
from .card_review_diff import diff_card_review, format_card_review_diff_summary
from .card_diff_risk_summary import format_diff_risk_summary, summarize_card_diff_risks
from .card_model_gate import gate_card_model, format_card_model_gate_summary
from .card_candidate_summary import format_candidate_summary, summarize_card_candidates
from .card_knn_compact import compact_card_classifier, format_compact_card_classifier_summary
from .card_big_teacher import (
    DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR,
    DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV,
    DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY,
    DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV,
    DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR,
    DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV,
    DEFAULT_BIG_TEACHER_MODEL,
    distill_big_teacher_runtime,
    format_card_big_teacher_summary,
    run_card_big_teacher,
)
from .cv_health import (
    DEFAULT_DEEP_MODEL_DIR as DEFAULT_CV_DEEP_MODEL_DIR,
    DEFAULT_GATE_SUMMARY,
    DEFAULT_HEALTH_OUTPUT_DIR,
    DEFAULT_VALIDATION_SUMMARY,
    check_cv_health,
    format_cv_health_summary,
)
from .card_label_queue import (
    audit_card_label_queue,
    format_card_label_queue_audit_summary,
    format_card_label_queue_summary,
    prepare_card_diff_label_queue,
    prepare_card_label_queue,
)
from .card_glyph_label_queue import (
    apply_card_glyph_label_queue,
    format_card_glyph_label_apply_summary,
    format_card_glyph_label_queue_summary,
    prepare_card_glyph_label_queue,
)
from .card_label_retrain import (
    DEFAULT_BASE_GLYPH_DIR,
    DEFAULT_BASELINE_REVIEW_CSV,
    DEFAULT_BASELINE_VALIDATION_SUMMARY,
    DEFAULT_BENCHMARK_REVIEW_CSV,
    DEFAULT_DEEP_CARD_MODEL_DIR as DEFAULT_RETRAIN_DEEP_CARD_MODEL_DIR,
    format_label_retrain_summary,
    retrain_card_label_queue,
)
from .card_label_server import format_card_label_server_summary, serve_card_label_queue
from .card_glyph_label_server import (
    format_card_glyph_label_server_summary,
    serve_card_glyph_label_queue,
)
from .card_fixed_replay import format_fixed_replay_summary, replay_fixed_card_samples
from .card_deep_model import (
    DEFAULT_DEEP_MODEL_DIR,
    format_deep_train_summary,
    train_deep_card_classifier,
    warm_deep_card_models,
)
from .card_active_learning import (
    apply_card_review,
    audit_card_glyphs,
    format_apply_review_summary,
    format_glyph_audit_summary,
)
from .cv_validate import (
    find_latest_video,
    format_validation_suite_summary,
    format_validation_summary,
    validate_cv_video,
    validate_cv_videos,
)
from .bbox_diagnostics import (
    diagnose_auto_bbox_videos,
    format_auto_bbox_suite_summary,
)
from .bbox_utils import (
    canonical_live_bbox_file,
    load_outer_bbox_text,
    load_rebased_analysis_bbox_text,
    resolve_bbox_text,
)
from .simulator import (
    build_practice_round,
    generate_spot,
    judge_action,
    plain_action_label,
    spot_title,
)
from .live_vision import analyze_realtime_video, format_realtime_summary
from .ocr_events import build_ocr_events_from_states, format_ocr_event_summary
from .state_review import build_state_review, format_state_review_summary
from .state_action_label_server import (
    format_state_action_label_queue_summary,
    format_state_action_label_server_summary,
    prepare_state_action_label_queue,
    serve_state_action_label_queue,
)
from .screen_vision import analyze_screen_stream, format_screen_summary, parse_bbox
from .stream_overlay import format_dashboard_summary, render_dashboard_video
from .vision import analyze_table_image, format_vision_text
from .video_vision import analyze_video, format_video_summary
from .web import run_web


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "advise":
            return command_advise(args)
        if args.command == "stream":
            return command_stream(args)
        if args.command == "deal":
            return command_deal(args)
        if args.command == "practice":
            return command_practice(args)
        if args.command == "ui":
            return command_ui(args)
        if args.command == "rules":
            return command_rules(args)
        if args.command == "web":
            return command_web(args)
        if args.command == "cv":
            return command_cv(args)
        if args.command == "video-cv":
            return command_video_cv(args)
        if args.command == "live-cv":
            return command_live_cv(args)
        if args.command == "dashboard-video":
            return command_dashboard_video(args)
        if args.command == "ocr-events":
            return command_ocr_events(args)
        if args.command == "screen-cv":
            return command_screen_cv(args)
        if args.command == "review-states":
            return command_review_states(args)
        if args.command == "serve-state-action-label-queue":
            return command_serve_state_action_label_queue(args)
        if args.command == "validate-cv":
            return command_validate_cv(args)
        if args.command == "diagnose-auto-bbox":
            return command_diagnose_auto_bbox(args)
        if args.command == "train-card-classifier":
            return command_train_card_classifier(args)
        if args.command == "compact-card-classifier":
            return command_compact_card_classifier(args)
        if args.command == "export-card-glyphs":
            return command_export_card_glyphs(args)
        if args.command == "export-card-review":
            return command_export_card_review(args)
        if args.command == "collect-card-debug-review":
            return command_collect_card_debug_review(args)
        if args.command == "label-card-crops":
            return command_label_card_crops(args)
        if args.command == "organize-card-crops":
            return command_organize_card_crops(args)
        if args.command == "label-card-crops-hf":
            return command_label_card_crops_hf(args)
        if args.command == "train-card-hf-probe":
            return command_train_card_hf_probe(args)
        if args.command == "label-card-crops-hf-probe":
            return command_label_card_crops_hf_probe(args)
        if args.command == "filter-card-hf-predictions":
            return command_filter_card_hf_predictions(args)
        if args.command == "ensemble-card-hf-predictions":
            return command_ensemble_card_hf_predictions(args)
        if args.command == "sweep-card-hf-thresholds":
            return command_sweep_card_hf_thresholds(args)
        if args.command == "card-big-teacher":
            return command_card_big_teacher(args)
        if args.command == "apply-card-hf-probe-review":
            return command_apply_card_hf_probe_review(args)
        if args.command == "generate-card-synthetic":
            return command_generate_card_synthetic(args)
        if args.command == "download-card-dataset":
            return command_download_card_dataset(args)
        if args.command == "card-cv-pipeline":
            return command_card_cv_pipeline(args)
        if args.command == "audit-card-review":
            return command_audit_card_review(args)
        if args.command == "benchmark-card-review":
            return command_benchmark_card_review(args)
        if args.command == "diff-card-review":
            return command_diff_card_review(args)
        if args.command == "summarize-card-diff-risks":
            return command_summarize_card_diff_risks(args)
        if args.command == "gate-card-model":
            return command_gate_card_model(args)
        if args.command == "summarize-card-candidates":
            return command_summarize_card_candidates(args)
        if args.command == "cv-health":
            return command_cv_health(args)
        if args.command == "prepare-card-label-queue":
            return command_prepare_card_label_queue(args)
        if args.command == "prepare-card-diff-label-queue":
            return command_prepare_card_diff_label_queue(args)
        if args.command == "prepare-card-glyph-label-queue":
            return command_prepare_card_glyph_label_queue(args)
        if args.command == "apply-card-glyph-label-queue":
            return command_apply_card_glyph_label_queue(args)
        if args.command == "audit-card-label-queue":
            return command_audit_card_label_queue(args)
        if args.command == "retrain-card-label-queue":
            return command_retrain_card_label_queue(args)
        if args.command == "serve-card-label-queue":
            return command_serve_card_label_queue(args)
        if args.command == "serve-card-glyph-label-queue":
            return command_serve_card_glyph_label_queue(args)
        if args.command == "replay-fixed-card-samples":
            return command_replay_fixed_card_samples(args)
        if args.command == "train-deep-card-classifier":
            return command_train_deep_card_classifier(args)
        if args.command == "ingest-card-images":
            return command_ingest_card_images(args)
        if args.command == "audit-card-glyphs":
            return command_audit_card_glyphs(args)
        if args.command == "apply-card-review":
            return command_apply_card_review(args)
        if args.command == "schema":
            return command_schema(args)
    except Exception as error:
        print_json({"ok": False, "error": str(error)}, pretty=True)
        return 1

    print("人用练习：python gto.py ui")
    print("零基础规则：python gto.py rules")
    print("机器接口：python gto.py deal --level simple")
    print()
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gto.py",
        description="德州扑克 CLI 决策器：读入牌局 JSON，输出近似 GTO 风格建议。",
    )
    subparsers = parser.add_subparsers(dest="command")

    advise = subparsers.add_parser("advise", help="读取一个牌局状态并输出一次决策")
    advise.add_argument("-s", "--state", type=Path, help="牌局状态 JSON 文件；省略时从 stdin 读取")
    advise.add_argument("-i", "--iterations", type=int, default=1200, help="翻后权益模拟次数")
    advise.add_argument("--format", choices=("json", "text"), default="json", help="输出格式")
    advise.add_argument("--compact", action="store_true", help="JSON 单行输出")
    advise.set_defaults(command="advise")

    stream = subparsers.add_parser("stream", help="从 stdin 逐行读取 JSON，逐行输出 JSON 决策")
    stream.add_argument("-i", "--iterations", type=int, default=800, help="翻后权益模拟次数")
    stream.set_defaults(command="stream")

    deal = subparsers.add_parser("deal", help="生成一个随机模拟盘局面 JSON，给程序或 UI 调用")
    add_simulator_args(deal)
    deal.add_argument("--with-answer", action="store_true", help="同时输出答案")
    deal.add_argument("--format", choices=("json", "text"), default="json", help="输出格式")
    deal.add_argument("--compact", action="store_true", help="JSON 单行输出")
    deal.set_defaults(command="deal")

    practice = subparsers.add_parser("practice", help="进入命令行模拟盘练习")
    add_simulator_args(practice)
    practice.add_argument("-n", "--count", type=int, default=10, help="练习局面数量")
    practice.add_argument("-i", "--iterations", type=int, default=700, help="翻后权益模拟次数")
    practice.set_defaults(command="practice")

    ui = subparsers.add_parser("ui", aliases=["play"], help="打开菜单式命令行 UI")
    ui.set_defaults(command="ui")

    rules = subparsers.add_parser("rules", help="输出零基础德州扑克规则速查")
    rules.set_defaults(command="rules")

    web = subparsers.add_parser("web", help="启动网页练习盘")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.set_defaults(command="web")

    cv = subparsers.add_parser("cv", help="识别截图里的 D 庄家按钮，并推断你的位置")
    cv.add_argument("image", type=Path, help="牌桌截图路径")
    cv.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "pict" / "D.png",
        help="D 庄家按钮模板图，默认使用 pict/D.png",
    )
    cv.add_argument("--seats", type=int, default=8, help="桌上座位数，默认 8")
    cv.add_argument("--min-confidence", type=float, default=0.45, help="D 模板匹配最低置信度")
    cv.add_argument("--min-scale", type=float, default=0.55, help="D 模板最小缩放比例")
    cv.add_argument("--max-scale", type=float, default=1.6, help="D 模板最大缩放比例")
    cv.add_argument("--annotate", type=Path, help="可选：输出带标注的截图")
    cv.add_argument("--format", choices=("json", "text"), default="json", help="输出格式")
    cv.add_argument("--compact", action="store_true", help="JSON 单行输出")
    cv.set_defaults(command="cv")

    video_cv = subparsers.add_parser("video-cv", help="从视频抽帧，识别 D、行动顺序、下注和弃牌状态")
    video_cv.add_argument("video", type=Path, help="牌桌视频路径")
    video_cv.add_argument("--output-dir", type=Path, default=Path("video_frames") / "analysis", help="输出目录")
    video_cv.add_argument("--template", type=Path, help="D 庄家按钮模板；默认优先使用 pict/D_purple.png")
    video_cv.add_argument("--seats", type=int, default=8, help="桌上座位数，默认 8")
    video_cv.add_argument("--start", type=float, help="开始秒数")
    video_cv.add_argument("--end", type=float, help="结束秒数")
    video_cv.add_argument("--middle", action="store_true", help="只分析视频中间 25%-75%")
    video_cv.add_argument("--every", type=float, default=5.0, help="每隔多少秒抽一帧，默认 5 秒")
    video_cv.add_argument("--max-frames", type=int, help="最多分析多少帧")
    video_cv.add_argument("--min-confidence", type=float, default=0.45, help="D 模板匹配最低置信度")
    video_cv.add_argument("--no-frames", action="store_true", help="不保存原始抽帧")
    video_cv.add_argument("--no-annotated", action="store_true", help="不保存标注帧")
    video_cv.add_argument("--format", choices=("json", "text"), default="text", help="输出格式")
    video_cv.add_argument("--compact", action="store_true", help="JSON 单行输出")
    video_cv.set_defaults(command="video-cv")

    live_cv = subparsers.add_parser(
        "live-cv",
        aliases=["realtime-cv"],
        help="simulate realtime poker CV from a video and emit state JSON events",
    )
    live_cv.add_argument("video", type=Path, help="video path")
    live_cv.add_argument("--output-dir", type=Path, default=Path("video_frames") / "live", help="output directory")
    live_cv.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    live_cv.add_argument("--seats", type=int, default=8, help="seat count, default 8")
    live_cv.add_argument("--start", type=float, help="start second")
    live_cv.add_argument("--end", type=float, help="end second")
    live_cv.add_argument("--middle", action="store_true", help="analyze the middle 25%%-75%% of the video")
    live_cv.add_argument("--every", type=float, default=1.0, help="sample interval in seconds, default 1")
    live_cv.add_argument("--max-frames", type=int, help="maximum sampled frames")
    live_cv.add_argument("--min-confidence", type=float, default=0.45, help="minimum dealer template confidence")
    live_cv.add_argument(
        "--trigger",
        choices=("state-change", "frame", "visual-change"),
        default="state-change",
        help="emit only changed states or every sampled frame",
    )
    live_cv.add_argument("--visual-threshold", type=float, default=2.4, help="visual diff threshold for --trigger visual-change")
    live_cv.add_argument("--min-event-gap", type=float, default=1.0, help="minimum seconds between visual-change events")
    live_cv.add_argument(
        "--dealer-refresh-frames",
        type=int,
        help="frame mode default is 30; other modes default to 1",
    )
    live_cv.add_argument("--save-frames", action="store_true", help="save raw frames for emitted events")
    live_cv.add_argument("--save-annotated", action="store_true", help="save annotated frames for emitted events")
    live_cv.add_argument("--no-ocr", action="store_true", help="disable OCR for fast realtime state detection")
    live_cv.add_argument("--ocr-scale", type=float, default=1.0, help="resize frame before OCR; e.g. 0.65 is faster")
    live_cv.add_argument("--ocr-action-only", action="store_true", help="run OCR only when bottom action buttons are visible")
    live_cv.add_argument("--with-advice", action="store_true", help="attach GTO advice when hero action controls are visible")
    live_cv.add_argument("--advice-iterations", type=int, default=600, help="postflop equity iterations for --with-advice")
    live_cv.add_argument("--effective-stack", type=float, default=100.0, help="effective stack in BB for --with-advice")
    live_cv.add_argument("--villain", default="standard", help="villain profile for --with-advice: tight/standard/wide/current")
    live_cv.add_argument("--format", choices=("json", "text"), default="text", help="output format")
    live_cv.add_argument("--compact", action="store_true", help="compact JSON output")
    live_cv.set_defaults(command="live-cv")

    dashboard_video = subparsers.add_parser(
        "dashboard-video",
        help="render a side-by-side CV overlay video and realtime state panel",
    )
    dashboard_video.add_argument("video", type=Path, help="source video path")
    dashboard_video.add_argument("--states-jsonl", type=Path, required=True, help="per-frame state JSONL")
    dashboard_video.add_argument("--ocr-events", type=Path, help="optional OCR event JSONL to merge into the panel")
    dashboard_video.add_argument("--output", type=Path, required=True, help="output mp4 path")
    dashboard_video.add_argument("--start", type=float, help="start second")
    dashboard_video.add_argument("--end", type=float, help="end second")
    dashboard_video.add_argument("--max-frames", type=int, help="maximum frames to render")
    dashboard_video.add_argument("--output-fps", type=float, help="output fps; defaults to source fps")
    dashboard_video.add_argument("--ocr-hold-sec", type=float, default=1.25, help="seconds to hold sparse OCR events")
    dashboard_video.add_argument("--format", choices=("json", "text"), default="text", help="output format")
    dashboard_video.add_argument("--compact", action="store_true", help="compact JSON output")
    dashboard_video.set_defaults(command="dashboard-video")

    ocr_events = subparsers.add_parser(
        "ocr-events",
        help="build OCR event stream from a per-frame visual state JSONL",
    )
    ocr_events.add_argument("video", type=Path, help="source video path")
    ocr_events.add_argument("--states-jsonl", type=Path, required=True, help="per-frame visual state JSONL")
    ocr_events.add_argument("--output-dir", type=Path, required=True, help="output directory")
    ocr_events.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    ocr_events.add_argument("--start", type=float, help="start second")
    ocr_events.add_argument("--end", type=float, help="end second")
    ocr_events.add_argument("--visual-threshold", type=float, default=5.0, help="visual diff threshold")
    ocr_events.add_argument("--visual-min-gap", type=float, default=1.0, help="minimum seconds between visual OCR triggers")
    ocr_events.add_argument("--heartbeat", type=float, default=5.0, help="periodic OCR heartbeat seconds")
    ocr_events.add_argument("--semantic-min-gap", type=float, default=0.5, help="minimum seconds between semantic OCR triggers")
    ocr_events.add_argument("--include-hero-semantic", action="store_true", help="also trigger OCR when hero cards change")
    ocr_events.add_argument("--dealer-refresh-events", type=int, default=30, help="refresh dealer detection every N OCR events")
    ocr_events.add_argument("--max-events", type=int, help="maximum selected OCR frames")
    ocr_events.add_argument("--resume", action="store_true", help="append and skip frames already in events.jsonl")
    ocr_events.add_argument("--format", choices=("json", "text"), default="text", help="output format")
    ocr_events.add_argument("--compact", action="store_true", help="compact JSON output")
    ocr_events.set_defaults(command="ocr-events")

    screen_cv = subparsers.add_parser(
        "screen-cv",
        help="capture a live screen region, run poker CV, and optionally attach GTO advice",
    )
    screen_cv.add_argument("--output-dir", type=Path, default=Path("video_frames") / "screen_live", help="output directory")
    screen_cv.add_argument("--bbox", help="absolute screen capture region: x,y,width,height")
    screen_cv.add_argument("--bbox-file", type=Path, help="bbox.json produced by screen-cv --pick-bbox")
    screen_cv.add_argument("--latest-bbox", action="store_true", help="use the newest bbox.json under video_frames")
    screen_cv.add_argument("--monitor", type=int, default=1, help="mss monitor index; 1 is usually the primary monitor")
    screen_cv.add_argument("--duration", type=float, help="seconds to run; omit to keep running")
    screen_cv.add_argument("--every", type=float, default=1.0, help="capture interval seconds")
    screen_cv.add_argument("--trigger", choices=("frame", "state-change", "visual-change"), default="frame")
    screen_cv.add_argument("--visual-threshold", type=float, default=2.4, help="visual diff threshold for visual-change")
    screen_cv.add_argument("--min-event-gap", type=float, default=1.0, help="minimum seconds between visual-change events")
    screen_cv.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    screen_cv.add_argument("--seats", type=int, default=8, help="seat count")
    screen_cv.add_argument("--min-confidence", type=float, default=0.45, help="dealer template confidence threshold")
    screen_cv.add_argument("--no-ocr", action="store_true", help="disable OCR")
    screen_cv.add_argument("--ocr-scale", type=float, default=0.65, help="resize frame before OCR; lower is faster but less accurate")
    screen_cv.add_argument("--ocr-action-only", action="store_true", help="run OCR only when bottom action buttons are visible")
    screen_cv.add_argument("--with-advice", action="store_true", help="attach GTO advice when hero action controls are visible")
    screen_cv.add_argument("--advice-iterations", type=int, default=600, help="postflop equity iterations for --with-advice")
    screen_cv.add_argument("--effective-stack", type=float, default=100.0, help="effective stack in BB for --with-advice")
    screen_cv.add_argument("--villain", default="standard", help="villain profile for --with-advice")
    screen_cv.add_argument("--save-frames", action="store_true", help="save emitted screen frames")
    screen_cv.add_argument("--save-annotated", action="store_true", help="save annotated emitted frames")
    screen_cv.add_argument("--no-problem-frames", action="store_true", help="disable automatic screenshots for failed or incomplete recognition")
    screen_cv.add_argument("--problem-frame-limit", type=int, default=240, help="maximum automatic problem screenshots to save")
    screen_cv.add_argument(
        "--no-card-samples",
        action="store_true",
        help="disable deduplicated hero/board card and rank/suit crop recording",
    )
    screen_cv.add_argument(
        "--card-sample-interval",
        type=float,
        default=30.0,
        help="seconds before saving another sample of an unchanged card prediction",
    )
    screen_cv.add_argument("--card-sample-limit", type=int, default=1000, help="maximum live card observation packages")
    screen_cv.add_argument(
        "--state-audit-limit",
        type=int,
        default=1000,
        help="maximum complete manual-window screenshots saved for action/state review",
    )
    screen_cv.add_argument("--snapshot-only", action="store_true", help="save one screenshot and exit; use it to choose --bbox")
    screen_cv.add_argument("--pick-bbox", action="store_true", help="open a draggable screenshot selector and print the selected --bbox")
    screen_cv.add_argument(
        "--review-auto-bbox",
        "--calibrate-bbox",
        dest="review_auto_bbox",
        action="store_true",
        help="propose the inner poker-table bbox inside the capture region; Enter accepts and R redraws it",
    )
    screen_cv.add_argument(
        "--pick-hero-cards",
        action="store_true",
        help="capture the analyzed table and let you drag H1/H2 card boxes; saves hero_card_rois.json",
    )
    screen_cv.add_argument(
        "--hero-cards-file",
        type=Path,
        help="manual H1/H2 ROI file produced by --pick-hero-cards; overrides automatic hero-card localization",
    )
    screen_cv.add_argument(
        "--show-overlay",
        action="store_true",
        help="show a transparent click-through screen overlay with table/card boxes, predictions, confidence, and CLIPPED warnings",
    )
    screen_cv.add_argument(
        "--overlay-image-interval",
        type=float,
        default=2.0,
        help="seconds between latest_overlay PNG updates; the transparent overlay still updates every processed frame",
    )
    screen_cv.add_argument("--preflight-once", action="store_true", help="capture, analyze, and save exactly one live screen frame")
    screen_cv.add_argument("--auto-bbox", action="store_true", help="find the poker window/table inside the capture region before analysis")
    screen_cv.add_argument("--auto-bbox-refresh", type=float, default=0.0, help="seconds between automatic bbox refreshes; 0 disables refresh")
    screen_cv.add_argument("--lock-layout", action="store_true", help="calibrate the current UI once and reuse the card layout for this process")
    screen_cv.add_argument("--hero-name", help="stable hero name used as a layout anchor, e.g. 于寻欢")
    screen_cv.add_argument("--deep-card-model-dir", type=Path, help="optional deep rank/suit model directory for low-confidence card fallback")
    screen_cv.add_argument("--deep-rank-card-model-dir", type=Path, help="optional rank-only deep model directory; overrides --deep-card-model-dir for ranks")
    screen_cv.add_argument("--deep-suit-card-model-dir", type=Path, help="optional suit-only deep model directory; overrides --deep-card-model-dir for suits")
    screen_cv.add_argument("--card-knn-model", type=Path, help="optional KNN glyph model .npz")
    screen_cv.add_argument(
        "--dealer-refresh-frames",
        type=int,
        default=12,
        help="refresh dealer detection every N processed frames; cached dealer is used between refreshes",
    )
    screen_cv.add_argument(
        "--console-mode",
        choices=("advice", "full"),
        default="advice",
        help="advice prints compact changed advice/status lines; full restores the detailed per-event line",
    )
    screen_cv.add_argument(
        "--console-heartbeat",
        type=float,
        default=10.0,
        help="seconds between repeated compact console status lines; 0 disables unchanged heartbeats",
    )
    screen_cv.add_argument("--format", choices=("json", "text"), default="text")
    screen_cv.add_argument("--compact", action="store_true", help="compact JSON output")
    screen_cv.set_defaults(command="screen-cv")

    review_states = subparsers.add_parser(
        "review-states",
        aliases=["review-live-states"],
        help="build a screenshot-backed Markdown review of saved screen-CV states",
    )
    review_states.add_argument(
        "--events",
        type=Path,
        default=Path("video_frames") / "screen_live" / "events.jsonl",
        help="saved screen-CV events JSONL",
    )
    review_states.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "screen_live" / "state_review",
        help="Markdown report and copied original screenshots",
    )
    review_states.add_argument("--limit", type=int, default=8, help="maximum number of cases to export")
    review_states.add_argument(
        "--include-watch",
        action="store_true",
        help="also include valid non-Hero-turn screenshots after Hero-turn candidates",
    )
    review_states.add_argument("--format", choices=("json", "text"), default="text")
    review_states.add_argument("--compact", action="store_true", help="compact JSON output")
    review_states.set_defaults(command="review-states")

    action_label_server = subparsers.add_parser(
        "serve-state-action-label-queue",
        help="build and open a local browser UI for manually checking Hero action-panel templates",
    )
    action_label_server.add_argument(
        "--events",
        type=Path,
        default=Path("video_frames") / "screen_live" / "events.jsonl",
        help="saved screen-CV events JSONL",
    )
    action_label_server.add_argument(
        "--extra-events",
        type=Path,
        action="append",
        default=[],
        help="additional saved events JSONL to mix into the same review queue; supports video-frame extraction",
    )
    action_label_server.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "screen_live" / "state_action_label_queue",
        help="queue CSV and copied screenshots",
    )
    action_label_server.add_argument(
        "--max-items",
        type=int,
        default=240,
        help="maximum temporally diverse screenshots to include",
    )
    action_label_server.add_argument("--host", default="127.0.0.1", help="bind host")
    action_label_server.add_argument("--port", type=int, default=8771, help="bind port")
    action_label_server.add_argument("--open-browser", action="store_true", help="open the UI in the default browser")
    action_label_server.add_argument("--format", choices=("json", "text"), default="text")
    action_label_server.add_argument("--compact", action="store_true", help="compact JSON output")
    action_label_server.set_defaults(command="serve-state-action-label-queue")

    validate_cv = subparsers.add_parser(
        "validate-cv",
        help="run a repeatable CV regression pass on a recorded poker video",
    )
    validate_cv.add_argument("video", nargs="?", type=Path, help="video path; omit with --latest")
    validate_cv.add_argument("--latest", action="store_true", help="use newest root mp4 under video_frames")
    validate_cv.add_argument("--all", action="store_true", help="run all root mp4 files under --video-dir")
    validate_cv.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory for --latest/--all")
    validate_cv.add_argument("--output-dir", type=Path, default=Path("video_frames") / "cv_validation", help="output directory")
    validate_cv.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    validate_cv.add_argument("--seats", type=int, default=8, help="seat count")
    validate_cv.add_argument("--start", type=float, help="start second")
    validate_cv.add_argument("--end", type=float, help="end second")
    validate_cv.add_argument("--every", type=float, default=30.0, help="sample interval seconds")
    validate_cv.add_argument("--max-frames", type=int, help="maximum sampled frames")
    validate_cv.add_argument("--min-confidence", type=float, default=0.35, help="dealer confidence threshold")
    validate_cv.add_argument("--auto-bbox-refresh", type=float, default=300.0, help="seconds between auto-bbox refresh attempts")
    validate_cv.add_argument("--dealer-refresh-frames", type=int, default=4, help="frames between dealer template refreshes when layout is locked")
    validate_cv.add_argument("--with-ocr", action="store_true", help="also run OCR for pot/bets on sampled frames")
    validate_cv.add_argument("--ocr-scale", type=float, default=0.65, help="OCR resize scale")
    validate_cv.add_argument("--ocr-action-only", action="store_true", help="with --with-ocr, run OCR only when bottom action buttons are visible")
    validate_cv.add_argument("--deep-card-model-dir", type=Path, help="optional deep rank/suit model directory for low-confidence card fallback")
    validate_cv.add_argument("--deep-rank-card-model-dir", type=Path, help="optional rank-only deep model directory; overrides --deep-card-model-dir for ranks")
    validate_cv.add_argument("--deep-suit-card-model-dir", type=Path, help="optional suit-only deep model directory; overrides --deep-card-model-dir for suits")
    validate_cv.add_argument("--card-knn-model", type=Path, help="optional KNN glyph model .npz")
    validate_cv.add_argument("--no-lock-layout", action="store_true", help="disable one-time locked-layout card search during validation")
    validate_cv.add_argument("--no-problem-frames", action="store_true", help="do not save failed/incomplete frames")
    validate_cv.add_argument("--format", choices=("json", "text"), default="text")
    validate_cv.add_argument("--compact", action="store_true", help="compact JSON output")
    validate_cv.set_defaults(command="validate-cv")

    bbox_diag = subparsers.add_parser(
        "diagnose-auto-bbox",
        help="stress-test automatic poker window/table localization on recorded videos",
    )
    bbox_diag.add_argument("video", nargs="*", type=Path, help="video paths; omit with --latest or --all")
    bbox_diag.add_argument("--latest", action="store_true", help="use newest root mp4 under video_frames")
    bbox_diag.add_argument("--all", action="store_true", help="run all root mp4 files under --video-dir")
    bbox_diag.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory")
    bbox_diag.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "auto_bbox_diagnostics",
        help="output diagnostics directory",
    )
    bbox_diag.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    bbox_diag.add_argument("--start", type=float, help="start second")
    bbox_diag.add_argument("--end", type=float, help="end second")
    bbox_diag.add_argument("--every", type=float, default=30.0, help="sample interval seconds")
    bbox_diag.add_argument("--max-frames", type=int, help="maximum sampled frames per video")
    bbox_diag.add_argument("--min-confidence", type=float, default=0.35, help="dealer confidence threshold")
    bbox_diag.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="bbox stress variant; can be repeated. Defaults: native, loose_8, loose_shift, tight_3, tight_6",
    )
    bbox_diag.add_argument("--no-problem-frames", action="store_true", help="do not save failed diagnostic frames")
    bbox_diag.add_argument("--format", choices=("json", "text"), default="text")
    bbox_diag.add_argument("--compact", action="store_true", help="compact JSON output")
    bbox_diag.set_defaults(command="diagnose-auto-bbox")

    train_cards = subparsers.add_parser(
        "train-card-classifier",
        help="train a rank/suit glyph classifier from local templates and optional public datasets",
    )
    train_cards.add_argument("--dataset-dir", type=Path, action="append", default=[], help="external labeled card image directory")
    train_cards.add_argument("--glyph-dir", type=Path, action="append", default=[], help="labeled glyph dataset directory with rank/<label> and suit/<label>")
    train_cards.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="local rank/suit template directory")
    train_cards.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="output classifier model .npz")
    train_cards.add_argument("--seed-model", type=Path, help="existing KNN .npz whose prototypes should be preserved in the new model")
    train_cards.add_argument(
        "--seed-conflict-policy",
        choices=("manual-override", "keep-seed"),
        default="manual-override",
        help="whether duplicate new glyph features override seed prototypes or keep the seed label",
    )
    train_cards.add_argument("--seed-guard", action="store_true", help="prefer preserved seed prototypes when their score/margin are already strong")
    train_cards.add_argument("--seed-guard-rank-score", type=float, default=0.55)
    train_cards.add_argument("--seed-guard-rank-margin", type=float, default=0.10)
    train_cards.add_argument("--seed-guard-suit-score", type=float, default=0.70)
    train_cards.add_argument("--seed-guard-suit-margin", type=float, default=0.04)
    train_cards.add_argument("--no-templates", action="store_true", help="do not include local pict/card_templates samples")
    train_cards.add_argument("--augment", type=int, default=8, help="small synthetic variants per sample")
    train_cards.add_argument("--external-augment", type=int, help="synthetic variants per external sample; defaults to --augment")
    train_cards.add_argument("--glyph-augment", type=int, help="synthetic variants per glyph sample; defaults to --augment")
    train_cards.add_argument("--max-external", type=int, help="maximum external full-card images to ingest")
    train_cards.add_argument("--format", choices=("json", "text"), default="text")
    train_cards.add_argument("--compact", action="store_true", help="compact JSON output")
    train_cards.set_defaults(command="train-card-classifier")

    compact_cards = subparsers.add_parser(
        "compact-card-classifier",
        help="prune a KNN card glyph model using benchmark rows while preserving per-label coverage",
    )
    compact_cards.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="source KNN glyph model .npz")
    compact_cards.add_argument("--output-model", type=Path, required=True, help="compacted KNN glyph model .npz")
    compact_cards.add_argument(
        "--benchmark-rows-csv",
        type=Path,
        action="append",
        required=True,
        help="card_benchmark_rows.csv used to choose useful prototypes; can be repeated",
    )
    compact_cards.add_argument("--top-per-sample", type=int, default=3, help="nearest same-label prototypes kept per benchmark crop")
    compact_cards.add_argument("--min-per-label", type=int, default=96, help="minimum diverse prototypes retained per rank/suit label")
    compact_cards.add_argument("--max-per-label", type=int, default=256, help="maximum prototypes retained per rank/suit label")
    compact_cards.add_argument("--format", choices=("json", "text"), default="text")
    compact_cards.add_argument("--compact", action="store_true", help="compact JSON output")
    compact_cards.set_defaults(command="compact-card-classifier")

    export_glyphs = subparsers.add_parser(
        "export-card-glyphs",
        help="export split rank/suit glyph crops for external OCR or image classifiers",
    )
    export_glyphs.add_argument("video", nargs="*", type=Path, help="video paths; omit with --latest or --all")
    export_glyphs.add_argument("--latest", action="store_true", help="use newest root mp4 under --video-dir")
    export_glyphs.add_argument("--all", action="store_true", help="export from all root mp4 files under --video-dir")
    export_glyphs.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory")
    export_glyphs.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_glyph_export",
        help="output dataset directory",
    )
    export_glyphs.add_argument("--every", type=float, default=5.0, help="sample interval seconds")
    export_glyphs.add_argument("--max-frames", type=int, help="maximum sampled frames per video")
    export_glyphs.add_argument("--no-lock-layout", action="store_true", help="disable one-time locked-layout card search")
    export_glyphs.add_argument("--no-board", action="store_true", help="export only hero cards")
    export_glyphs.add_argument("--min-rank-confidence", type=float, default=0.0, help="minimum current rank confidence to export")
    export_glyphs.add_argument("--min-suit-confidence", type=float, default=0.0, help="minimum current suit confidence to export")
    export_glyphs.add_argument("--card-knn-model", type=Path, help="optional KNN glyph model .npz")
    export_glyphs.add_argument("--format", choices=("json", "text"), default="text")
    export_glyphs.add_argument("--compact", action="store_true", help="compact JSON output")
    export_glyphs.set_defaults(command="export-card-glyphs")

    export_review = subparsers.add_parser(
        "export-card-review",
        help="export hero-card review crops and CSV without treating predictions as labels",
    )
    export_review.add_argument("video", nargs="*", type=Path, help="video paths; omit with --latest or --all")
    export_review.add_argument("--latest", action="store_true", help="use newest root mp4 under video_frames")
    export_review.add_argument("--all", action="store_true", help="run all root mp4 files under --video-dir")
    export_review.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory")
    export_review.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_review", help="review output directory")
    export_review.add_argument("--template", type=Path, help="dealer button template; defaults to pict/D_purple.png")
    export_review.add_argument("--seats", type=int, default=8, help="seat count")
    export_review.add_argument("--start", type=float, help="start second")
    export_review.add_argument("--end", type=float, help="end second")
    export_review.add_argument("--every", type=float, default=10.0, help="sample interval seconds")
    export_review.add_argument("--max-frames", type=int, help="maximum sampled frames per video")
    export_review.add_argument("--min-confidence", type=float, default=0.35, help="dealer confidence threshold")
    export_review.add_argument("--auto-bbox-refresh", type=float, default=300.0, help="seconds between auto-bbox refresh attempts")
    export_review.add_argument("--no-lock-layout", action="store_true", help="disable one-time locked-layout card search")
    export_review.add_argument("--only-suspicious", action="store_true", help="export only incomplete, stabilized, or low-confidence rows")
    export_review.add_argument("--max-sheet-rows", type=int, default=160, help="maximum rows drawn into review_sheet.jpg")
    export_review.add_argument("--deep-card-model-dir", type=Path, help="optional deep rank/suit model directory")
    export_review.add_argument("--deep-rank-card-model-dir", type=Path, help="optional rank-only deep model directory")
    export_review.add_argument("--deep-suit-card-model-dir", type=Path, help="optional suit-only deep model directory")
    export_review.add_argument("--card-knn-model", type=Path, help="optional KNN glyph model .npz")
    export_review.add_argument("--format", choices=("json", "text"), default="text")
    export_review.add_argument("--compact", action="store_true", help="compact JSON output")
    export_review.set_defaults(command="export-card-review")

    collect_debug = subparsers.add_parser(
        "collect-card-debug-review",
        help="convert live screen card_debug packages into review.csv/label-queue input",
    )
    collect_debug.add_argument(
        "--input-dir",
        type=Path,
        default=Path("video_frames") / "screen_live" / "card_debug",
        help="card_debug directory produced by screen-cv problem capture",
    )
    collect_debug.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_debug_review",
        help="review package output directory",
    )
    collect_debug.add_argument("--max-rows", type=int, help="maximum debug events to include")
    collect_debug.add_argument("--max-sheet-rows", type=int, default=160, help="maximum rows drawn into review_sheet.jpg")
    collect_debug.add_argument("--no-fallback", action="store_true", help="do not use fallback fixed ROI crops for missing detected slots")
    collect_debug.add_argument("--prepare-label-queue", action="store_true", help="also build label_queue.csv/html from the generated review.csv")
    collect_debug.add_argument("--queue-output-dir", type=Path, help="label queue output directory; defaults under --output-dir")
    collect_debug.add_argument("--queue-max-rows", type=int, default=80, help="maximum rows to include when --prepare-label-queue is used")
    collect_debug.add_argument("--prepare-glyph-label-queue", action="store_true", help="also build a split rank/suit glyph label queue from review.csv")
    collect_debug.add_argument("--glyph-queue-output-dir", type=Path, help="glyph label queue output directory; defaults under --output-dir")
    collect_debug.add_argument("--glyph-queue-max-rows", type=int, default=160, help="maximum glyph rows to include when --prepare-glyph-label-queue is used")
    collect_debug.add_argument("--no-copy-queue-assets", action="store_true", help="do not copy image assets when preparing the label queue")
    collect_debug.add_argument("--format", choices=("json", "text"), default="text")
    collect_debug.add_argument("--compact", action="store_true", help="compact JSON output")
    collect_debug.set_defaults(command="collect-card-debug-review")

    label_crops = subparsers.add_parser(
        "label-card-crops",
        help="run larger rank/suit teacher models on cropped card glyphs and build a cleaned dataset",
    )
    label_crops.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing rank/ and/or suit/ crops; can be repeated",
    )
    label_crops.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_teacher_labeled",
        help="output dataset directory",
    )
    label_crops.add_argument("--teacher-model-dir", type=Path, help="teacher model directory containing deep_rank.pt and deep_suit.pt")
    label_crops.add_argument("--teacher-rank-model-dir", type=Path, help="rank-specific teacher model directory")
    label_crops.add_argument("--teacher-suit-model-dir", type=Path, help="suit-specific teacher model directory")
    label_crops.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to label")
    label_crops.add_argument("--max-images", type=int, help="maximum crops to process per kind")
    label_crops.add_argument("--rank-score-threshold", type=float, default=0.90, help="minimum rank teacher probability to auto-accept")
    label_crops.add_argument("--rank-margin-threshold", type=float, default=0.20, help="minimum rank top-vs-second margin")
    label_crops.add_argument("--suit-score-threshold", type=float, default=0.88, help="minimum suit teacher probability to auto-accept")
    label_crops.add_argument("--suit-margin-threshold", type=float, default=0.18, help="minimum suit top-vs-second margin")
    label_crops.add_argument(
        "--require-current-agreement",
        action="store_true",
        help="only auto-accept when the existing folder/filename label agrees with the teacher",
    )
    label_crops.add_argument("--no-copy-accepted", action="store_true", help="do not copy accepted crops into rank/suit label folders")
    label_crops.add_argument("--format", choices=("json", "text"), default="text")
    label_crops.add_argument("--compact", action="store_true", help="compact JSON output")
    label_crops.set_defaults(command="label-card-crops")

    organize_crops = subparsers.add_parser(
        "organize-card-crops",
        help="copy cropped rank/suit glyphs into label folders using labels parsed from filenames",
    )
    organize_crops.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing rank/ and/or suit/ crops; can be repeated",
    )
    organize_crops.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_crops_organized",
        help="output rank/suit training dataset directory",
    )
    organize_crops.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to organize")
    organize_crops.add_argument("--max-images", type=int, help="maximum crops to process per kind")
    organize_crops.add_argument("--review-csv", type=Path, help="optional export-card-review CSV used to filter safe rows")
    organize_crops.add_argument(
        "--allowed-review-reason",
        action="append",
        default=[],
        help="review_reason allowed when --review-csv is provided; default is ok",
    )
    organize_crops.add_argument("--format", choices=("json", "text"), default="text")
    organize_crops.add_argument("--compact", action="store_true", help="compact JSON output")
    organize_crops.set_defaults(command="organize-card-crops")

    hf_label_crops = subparsers.add_parser(
        "label-card-crops-hf",
        help="run external HuggingFace/CLIP zero-shot teachers on cropped rank/suit glyphs",
    )
    hf_label_crops.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing rank/ and/or suit/ crops; can be repeated",
    )
    hf_label_crops.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_hf_labeled",
        help="output dataset directory",
    )
    hf_label_crops.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to label")
    hf_label_crops.add_argument("--rank-model", default=DEFAULT_HF_CLIP_MODEL, help="HuggingFace CLIP model for rank crops")
    hf_label_crops.add_argument("--suit-model", default=DEFAULT_HF_CLIP_MODEL, help="HuggingFace CLIP model for suit crops")
    hf_label_crops.add_argument("--device", default="auto", help="auto/cpu/-1/0/cuda:0")
    hf_label_crops.add_argument("--local-files-only", action="store_true", help="do not download HuggingFace models")
    hf_label_crops.add_argument("--max-images", type=int, help="maximum crops to process per kind")
    hf_label_crops.add_argument("--rank-score-threshold", type=float, default=0.52, help="minimum rank teacher probability to auto-accept")
    hf_label_crops.add_argument("--rank-margin-threshold", type=float, default=0.05, help="minimum rank top-vs-second margin")
    hf_label_crops.add_argument("--suit-score-threshold", type=float, default=0.52, help="minimum suit teacher probability to auto-accept")
    hf_label_crops.add_argument("--suit-margin-threshold", type=float, default=0.05, help="minimum suit top-vs-second margin")
    hf_label_crops.add_argument(
        "--require-current-agreement",
        action="store_true",
        help="only auto-accept when the existing folder/filename label agrees with the external teacher",
    )
    hf_label_crops.add_argument("--no-copy-accepted", action="store_true", help="do not copy accepted crops into rank/suit label folders")
    hf_label_crops.add_argument("--format", choices=("json", "text"), default="text")
    hf_label_crops.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_label_crops.set_defaults(command="label-card-crops-hf")

    hf_probe_train = subparsers.add_parser(
        "train-card-hf-probe",
        help="train frozen HuggingFace vision-model embedding probes for cropped rank/suit glyphs",
    )
    hf_probe_train.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="trusted directory containing labeled rank/ and/or suit/ crops; can be repeated",
    )
    hf_probe_train.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pict") / "card_models" / "hf_probe",
        help="output directory for hf_rank_probe.npz and hf_suit_probe.npz",
    )
    hf_probe_train.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which probe to train")
    hf_probe_train.add_argument("--model", default=DEFAULT_HF_PROBE_MODEL, help="shared HuggingFace vision model")
    hf_probe_train.add_argument("--rank-model", help="rank-specific HuggingFace vision model")
    hf_probe_train.add_argument("--suit-model", help="suit-specific HuggingFace vision model")
    hf_probe_train.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="optional local template directory")
    hf_probe_train.add_argument("--no-templates", action="store_true", help="do not include local pict/card_templates seeds")
    hf_probe_train.add_argument("--max-images-per-class", type=int, help="cap training crops per rank/suit class")
    hf_probe_train.add_argument("--val-split", type=float, default=0.18, help="validation split for reporting only")
    hf_probe_train.add_argument("--seed", type=int, default=20260708)
    hf_probe_train.add_argument("--batch-size", type=int, default=32)
    hf_probe_train.add_argument("--temperature", type=float, default=0.04, help="softmax temperature for prototype scores")
    hf_probe_train.add_argument("--device", default="auto", help="auto/cpu/-1/0/cuda:0")
    hf_probe_train.add_argument("--local-files-only", action="store_true", help="do not download HuggingFace models")
    hf_probe_train.add_argument("--format", choices=("json", "text"), default="text")
    hf_probe_train.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_probe_train.set_defaults(command="train-card-hf-probe")

    hf_probe_label = subparsers.add_parser(
        "label-card-crops-hf-probe",
        help="label cropped rank/suit glyphs using a trained HuggingFace embedding probe",
    )
    hf_probe_label.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing rank/ and/or suit/ crops; can be repeated",
    )
    hf_probe_label.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_hf_probe_labeled",
        help="output dataset directory",
    )
    hf_probe_label.add_argument("--probe-dir", type=Path, required=True, help="directory containing hf_rank_probe.npz/hf_suit_probe.npz")
    hf_probe_label.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to label")
    hf_probe_label.add_argument("--device", default="auto", help="auto/cpu/-1/0/cuda:0")
    hf_probe_label.add_argument("--local-files-only", action="store_true", help="do not download HuggingFace models")
    hf_probe_label.add_argument("--max-images", type=int, help="maximum crops to process per kind")
    hf_probe_label.add_argument("--batch-size", type=int, default=32)
    hf_probe_label.add_argument("--rank-score-threshold", type=float, default=0.82, help="minimum rank probe probability to auto-accept")
    hf_probe_label.add_argument("--rank-margin-threshold", type=float, default=0.10, help="minimum rank top-vs-second margin")
    hf_probe_label.add_argument("--suit-score-threshold", type=float, default=0.82, help="minimum suit probe probability to auto-accept")
    hf_probe_label.add_argument("--suit-margin-threshold", type=float, default=0.10, help="minimum suit top-vs-second margin")
    hf_probe_label.add_argument(
        "--require-current-agreement",
        action="store_true",
        help="only auto-accept when the existing folder/filename label agrees with the probe",
    )
    hf_probe_label.add_argument("--no-copy-accepted", action="store_true", help="do not copy accepted crops into rank/suit label folders")
    hf_probe_label.add_argument("--format", choices=("json", "text"), default="text")
    hf_probe_label.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_probe_label.set_defaults(command="label-card-crops-hf-probe")

    hf_probe_filter = subparsers.add_parser(
        "filter-card-hf-predictions",
        help="re-screen an existing HF probe predictions.csv without recomputing embeddings",
    )
    hf_probe_filter.add_argument("--predictions-csv", type=Path, required=True, help="source predictions.csv from label-card-crops-hf-probe/card-big-teacher")
    hf_probe_filter.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_hf_probe_filtered", help="filtered label output directory")
    hf_probe_filter.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to keep")
    hf_probe_filter.add_argument("--rank-score-threshold", type=float, default=0.82, help="minimum rank probe probability to auto-accept")
    hf_probe_filter.add_argument("--rank-margin-threshold", type=float, default=0.10, help="minimum rank top-vs-second margin")
    hf_probe_filter.add_argument("--suit-score-threshold", type=float, default=0.82, help="minimum suit probe probability to auto-accept")
    hf_probe_filter.add_argument("--suit-margin-threshold", type=float, default=0.10, help="minimum suit top-vs-second margin")
    hf_probe_filter.add_argument("--require-current-agreement", action="store_true", help="only auto-accept when the current folder/filename label agrees with the probe")
    hf_probe_filter.add_argument("--no-copy-accepted", action="store_true", help="do not copy accepted crops into rank/suit label folders")
    hf_probe_filter.add_argument("--distill-runtime", action="store_true", help="train, validate, and gate a fast runtime KNN candidate from filtered accepted crops")
    hf_probe_filter.add_argument("--runtime-output-dir", type=Path, help="runtime candidate output directory; default is <output-dir>\\runtime_candidate")
    hf_probe_filter.add_argument("--runtime-model", type=Path, help="runtime candidate KNN .npz path; default under --runtime-output-dir")
    hf_probe_filter.add_argument("--runtime-candidate-name", help="candidate name in runtime gate reports")
    hf_probe_filter.add_argument("--runtime-base-glyph-dir", type=Path, action="append", default=[], help=f"trusted base rank/suit glyph directory; default uses {DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR} when present")
    hf_probe_filter.add_argument("--runtime-dataset-dir", type=Path, action="append", default=[], help="optional external full-card dataset directory to add during runtime KNN training")
    hf_probe_filter.add_argument("--runtime-video-dir", type=Path, default=Path("video_frames"), help="root video directory for runtime validation")
    hf_probe_filter.add_argument("--runtime-video", type=Path, action="append", default=[], help="specific runtime validation video; repeatable")
    hf_probe_filter.add_argument(
        "--runtime-benchmark-review-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "benchmark review CSV; default uses manual truth when present "
            f"({DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV}) plus {DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV}"
        ),
    )
    hf_probe_filter.add_argument("--runtime-baseline-review-csv", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV)
    hf_probe_filter.add_argument("--runtime-baseline-validation-summary-json", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY)
    hf_probe_filter.add_argument("--runtime-deep-card-model-dir", type=Path, default=DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR)
    hf_probe_filter.add_argument("--runtime-seed-model", type=Path, default=DEFAULT_MODEL_PATH, help="promoted KNN model to preserve while adding accepted teacher crops")
    hf_probe_filter.add_argument("--no-runtime-seed-model", action="store_true", help="train runtime candidate without preserving the promoted KNN prototypes")
    hf_probe_filter.add_argument(
        "--runtime-seed-conflict-policy",
        choices=("manual-override", "keep-seed"),
        default="keep-seed",
        help="how duplicate teacher glyphs interact with preserved seed prototypes",
    )
    hf_probe_filter.add_argument("--runtime-seed-guard", action="store_true", help="prefer preserved seed prototypes when their score/margin are already strong")
    hf_probe_filter.add_argument("--runtime-seed-guard-rank-score", type=float, default=0.55)
    hf_probe_filter.add_argument("--runtime-seed-guard-rank-margin", type=float, default=0.10)
    hf_probe_filter.add_argument("--runtime-seed-guard-suit-score", type=float, default=0.70)
    hf_probe_filter.add_argument("--runtime-seed-guard-suit-margin", type=float, default=0.04)
    hf_probe_filter.add_argument("--runtime-every", type=float, default=10.0, help="runtime validation/export sample interval seconds")
    hf_probe_filter.add_argument("--runtime-max-frames", type=int, default=80, help="maximum sampled frames per validation video")
    hf_probe_filter.add_argument("--runtime-min-confidence", type=float, default=0.35)
    hf_probe_filter.add_argument("--runtime-augment", type=int, default=8)
    hf_probe_filter.add_argument("--runtime-external-augment", type=int)
    hf_probe_filter.add_argument("--runtime-glyph-augment", type=int, default=8)
    hf_probe_filter.add_argument("--runtime-max-external", type=int)
    hf_probe_filter.add_argument("--runtime-min-accepted", type=int, default=1)
    hf_probe_filter.add_argument("--runtime-max-benchmark-samples", type=int, default=300)
    hf_probe_filter.add_argument("--runtime-max-diff-rows", type=int, default=300)
    hf_probe_filter.add_argument("--runtime-max-risk", type=int, default=0)
    hf_probe_filter.add_argument("--runtime-max-real-problem", type=int, default=0)
    hf_probe_filter.add_argument("--runtime-max-board-bad", type=int, default=0)
    hf_probe_filter.add_argument("--runtime-max-median-ms", type=float, default=300.0)
    hf_probe_filter.add_argument("--runtime-max-p90-ms", type=float, default=900.0)
    hf_probe_filter.add_argument("--runtime-max-median-regression-ms", type=float)
    hf_probe_filter.add_argument("--runtime-max-p90-regression-ms", type=float)
    hf_probe_filter.add_argument("--no-runtime-risk-queue", action="store_true", help="do not generate a label queue from risky runtime diff rows")
    hf_probe_filter.add_argument("--runtime-risk-queue-max-rows", type=int, default=80)
    hf_probe_filter.add_argument("--format", choices=("json", "text"), default="text")
    hf_probe_filter.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_probe_filter.set_defaults(command="filter-card-hf-predictions")

    hf_probe_ensemble = subparsers.add_parser(
        "ensemble-card-hf-predictions",
        help="accept cropped rank/suit labels only when multiple HF teacher prediction CSVs agree",
    )
    hf_probe_ensemble.add_argument(
        "--predictions-csv",
        type=Path,
        action="append",
        required=True,
        help="source predictions.csv from label-card-crops-hf-probe/card-big-teacher; repeat at least twice",
    )
    hf_probe_ensemble.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_hf_probe_ensemble")
    hf_probe_ensemble.add_argument("--kind", choices=("rank", "suit", "both"), default="both")
    hf_probe_ensemble.add_argument("--rank-score-threshold", type=float, default=0.82)
    hf_probe_ensemble.add_argument("--rank-margin-threshold", type=float, default=0.10)
    hf_probe_ensemble.add_argument("--suit-score-threshold", type=float, default=0.82)
    hf_probe_ensemble.add_argument("--suit-margin-threshold", type=float, default=0.10)
    hf_probe_ensemble.add_argument("--no-require-current-agreement", action="store_true")
    hf_probe_ensemble.add_argument("--min-teachers", type=int, help="minimum agreeing teachers; default requires every CSV")
    hf_probe_ensemble.add_argument("--no-copy-accepted", action="store_true")
    hf_probe_ensemble.add_argument("--distill-runtime", action="store_true", help="train, validate, and gate a fast runtime KNN candidate from ensemble accepted crops")
    hf_probe_ensemble.add_argument("--runtime-output-dir", type=Path, help="runtime candidate output directory; default is <output-dir>\\runtime_candidate")
    hf_probe_ensemble.add_argument("--runtime-model", type=Path, help="runtime candidate KNN .npz path; default under --runtime-output-dir")
    hf_probe_ensemble.add_argument("--runtime-candidate-name", help="candidate name in runtime gate reports")
    hf_probe_ensemble.add_argument("--runtime-base-glyph-dir", type=Path, action="append", default=[], help=f"trusted base rank/suit glyph directory; default uses {DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR} when present")
    hf_probe_ensemble.add_argument("--runtime-dataset-dir", type=Path, action="append", default=[], help="optional external full-card dataset directory to add during runtime KNN training")
    hf_probe_ensemble.add_argument("--runtime-video-dir", type=Path, default=Path("video_frames"))
    hf_probe_ensemble.add_argument("--runtime-video", type=Path, action="append", default=[])
    hf_probe_ensemble.add_argument(
        "--runtime-benchmark-review-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "benchmark review CSV; default uses manual truth when present "
            f"({DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV}) plus {DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV}"
        ),
    )
    hf_probe_ensemble.add_argument("--runtime-baseline-review-csv", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV)
    hf_probe_ensemble.add_argument("--runtime-baseline-validation-summary-json", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY)
    hf_probe_ensemble.add_argument("--runtime-deep-card-model-dir", type=Path, default=DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR)
    hf_probe_ensemble.add_argument("--runtime-seed-model", type=Path, default=DEFAULT_MODEL_PATH)
    hf_probe_ensemble.add_argument("--no-runtime-seed-model", action="store_true")
    hf_probe_ensemble.add_argument("--runtime-seed-conflict-policy", choices=("manual-override", "keep-seed"), default="keep-seed")
    hf_probe_ensemble.add_argument("--runtime-seed-guard", action="store_true")
    hf_probe_ensemble.add_argument("--runtime-seed-guard-rank-score", type=float, default=0.55)
    hf_probe_ensemble.add_argument("--runtime-seed-guard-rank-margin", type=float, default=0.10)
    hf_probe_ensemble.add_argument("--runtime-seed-guard-suit-score", type=float, default=0.70)
    hf_probe_ensemble.add_argument("--runtime-seed-guard-suit-margin", type=float, default=0.04)
    hf_probe_ensemble.add_argument("--runtime-every", type=float, default=10.0)
    hf_probe_ensemble.add_argument("--runtime-max-frames", type=int, default=80)
    hf_probe_ensemble.add_argument("--runtime-min-confidence", type=float, default=0.35)
    hf_probe_ensemble.add_argument("--runtime-augment", type=int, default=8)
    hf_probe_ensemble.add_argument("--runtime-external-augment", type=int)
    hf_probe_ensemble.add_argument("--runtime-glyph-augment", type=int, default=8)
    hf_probe_ensemble.add_argument("--runtime-max-external", type=int)
    hf_probe_ensemble.add_argument("--runtime-min-accepted", type=int, default=1)
    hf_probe_ensemble.add_argument("--runtime-max-benchmark-samples", type=int, default=300)
    hf_probe_ensemble.add_argument("--runtime-max-diff-rows", type=int, default=300)
    hf_probe_ensemble.add_argument("--runtime-max-risk", type=int, default=0)
    hf_probe_ensemble.add_argument("--runtime-max-real-problem", type=int, default=0)
    hf_probe_ensemble.add_argument("--runtime-max-board-bad", type=int, default=0)
    hf_probe_ensemble.add_argument("--runtime-max-median-ms", type=float, default=300.0)
    hf_probe_ensemble.add_argument("--runtime-max-p90-ms", type=float, default=900.0)
    hf_probe_ensemble.add_argument("--runtime-max-median-regression-ms", type=float)
    hf_probe_ensemble.add_argument("--runtime-max-p90-regression-ms", type=float)
    hf_probe_ensemble.add_argument("--no-runtime-risk-queue", action="store_true")
    hf_probe_ensemble.add_argument("--runtime-risk-queue-max-rows", type=int, default=80)
    hf_probe_ensemble.add_argument("--format", choices=("json", "text"), default="text")
    hf_probe_ensemble.add_argument("--compact", action="store_true")
    hf_probe_ensemble.set_defaults(command="ensemble-card-hf-predictions")

    hf_threshold_sweep = subparsers.add_parser(
        "sweep-card-hf-thresholds",
        help="scan score/margin thresholds over an HF teacher predictions.csv",
    )
    hf_threshold_sweep.add_argument("--predictions-csv", type=Path, required=True, help="source predictions.csv from label-card-crops-hf-probe/card-big-teacher")
    hf_threshold_sweep.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_hf_threshold_sweep")
    hf_threshold_sweep.add_argument("--rank-score-threshold", type=float, action="append", default=[], help="rank score threshold to test; repeatable")
    hf_threshold_sweep.add_argument("--rank-margin-threshold", type=float, action="append", default=[], help="rank margin threshold to test; repeatable")
    hf_threshold_sweep.add_argument("--suit-score-threshold", type=float, action="append", default=[], help="suit score threshold to test; repeatable")
    hf_threshold_sweep.add_argument("--suit-margin-threshold", type=float, action="append", default=[], help="suit margin threshold to test; repeatable")
    hf_threshold_sweep.add_argument("--no-require-current-agreement", action="store_true", help="allow teacher labels that disagree with the current label")
    hf_threshold_sweep.add_argument("--format", choices=("json", "text"), default="text")
    hf_threshold_sweep.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_threshold_sweep.set_defaults(command="sweep-card-hf-thresholds")

    big_teacher = subparsers.add_parser(
        "card-big-teacher",
        help="one-shot offline big-model rank/suit teacher pipeline for cropped card glyphs",
    )
    big_teacher.add_argument("video", nargs="*", type=Path, help="video paths to export crops from; omit if using --input-dir only")
    big_teacher.add_argument("--latest", action="store_true", help="use newest root mp4 under --video-dir")
    big_teacher.add_argument("--all", action="store_true", help="export from all root mp4 files under --video-dir")
    big_teacher.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory")
    big_teacher.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=[],
        help="existing directory containing rank/ and/or suit/ crops; can be repeated",
    )
    big_teacher.add_argument(
        "--trusted-dir",
        type=Path,
        action="append",
        default=[],
        help="trusted labeled rank/suit crops used to train a probe; required unless --probe-dir is provided",
    )
    big_teacher.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_big_teacher",
        help="output directory containing crops/probe/labeled/runbook",
    )
    big_teacher.add_argument("--probe-dir", type=Path, help="reuse an existing hf_rank_probe.npz/hf_suit_probe.npz directory")
    big_teacher.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which crop type to process")
    big_teacher.add_argument("--model", default=DEFAULT_BIG_TEACHER_MODEL, help="shared HuggingFace vision encoder")
    big_teacher.add_argument("--rank-model", help="rank-specific HuggingFace vision encoder")
    big_teacher.add_argument("--suit-model", help="suit-specific HuggingFace vision encoder")
    big_teacher.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="optional local template directory")
    big_teacher.add_argument("--no-templates", action="store_true", help="do not include local pict/card_templates seeds")
    big_teacher.add_argument("--every", type=float, default=5.0, help="video crop export interval seconds")
    big_teacher.add_argument("--max-frames", type=int, help="maximum sampled frames per video during crop export")
    big_teacher.add_argument("--no-lock-layout", action="store_true", help="disable one-time locked-layout card search")
    big_teacher.add_argument("--no-board", action="store_true", help="export only hero cards from videos")
    big_teacher.add_argument("--max-images", type=int, help="maximum crops to label per kind")
    big_teacher.add_argument("--max-images-per-class", type=int, help="cap trusted training crops per rank/suit class")
    big_teacher.add_argument("--rank-score-threshold", type=float, default=0.82, help="minimum rank probe probability to auto-accept")
    big_teacher.add_argument("--rank-margin-threshold", type=float, default=0.10, help="minimum rank top-vs-second margin")
    big_teacher.add_argument("--suit-score-threshold", type=float, default=0.82, help="minimum suit probe probability to auto-accept")
    big_teacher.add_argument("--suit-margin-threshold", type=float, default=0.10, help="minimum suit top-vs-second margin")
    big_teacher.add_argument(
        "--require-current-agreement",
        action="store_true",
        help="only auto-accept when the current folder/filename label agrees with the probe",
    )
    big_teacher.add_argument("--no-copy-accepted", action="store_true", help="do not copy accepted crops into rank/suit label folders")
    big_teacher.add_argument("--batch-size", type=int, default=32)
    big_teacher.add_argument("--temperature", type=float, default=0.04, help="softmax temperature for prototype scores")
    big_teacher.add_argument("--device", default="auto", help="auto/cpu/-1/0/cuda:0")
    big_teacher.add_argument("--local-files-only", action="store_true", help="do not download HuggingFace models")
    big_teacher.add_argument(
        "--distill-runtime",
        action="store_true",
        help="train, validate, and gate a fast runtime KNN candidate from accepted teacher crops",
    )
    big_teacher.add_argument(
        "--runtime-output-dir",
        type=Path,
        help="runtime candidate output directory; default is <output-dir>\\runtime_candidate",
    )
    big_teacher.add_argument("--runtime-model", type=Path, help="runtime candidate KNN .npz path; default under --runtime-output-dir")
    big_teacher.add_argument("--runtime-candidate-name", help="candidate name in runtime gate reports")
    big_teacher.add_argument(
        "--runtime-base-glyph-dir",
        type=Path,
        action="append",
        default=[],
        help=f"trusted base rank/suit glyph directory; default uses {DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR} when present",
    )
    big_teacher.add_argument(
        "--runtime-dataset-dir",
        type=Path,
        action="append",
        default=[],
        help="optional external full-card dataset directory to add during runtime KNN training",
    )
    big_teacher.add_argument("--runtime-video-dir", type=Path, default=Path("video_frames"), help="root video directory for runtime validation")
    big_teacher.add_argument("--runtime-video", type=Path, action="append", default=[], help="specific runtime validation video; repeatable")
    big_teacher.add_argument(
        "--runtime-benchmark-review-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "benchmark review CSV; default uses manual truth when present "
            f"({DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV}) plus {DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV}"
        ),
    )
    big_teacher.add_argument("--runtime-baseline-review-csv", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV)
    big_teacher.add_argument("--runtime-baseline-validation-summary-json", type=Path, default=DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY)
    big_teacher.add_argument("--runtime-deep-card-model-dir", type=Path, default=DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR)
    big_teacher.add_argument("--runtime-seed-model", type=Path, default=DEFAULT_MODEL_PATH, help="promoted KNN model to preserve while adding accepted teacher crops")
    big_teacher.add_argument("--no-runtime-seed-model", action="store_true", help="train runtime candidate without preserving the promoted KNN prototypes")
    big_teacher.add_argument(
        "--runtime-seed-conflict-policy",
        choices=("manual-override", "keep-seed"),
        default="keep-seed",
        help="how duplicate teacher glyphs interact with preserved seed prototypes",
    )
    big_teacher.add_argument("--runtime-seed-guard", action="store_true", help="prefer preserved seed prototypes when their score/margin are already strong")
    big_teacher.add_argument("--runtime-seed-guard-rank-score", type=float, default=0.55)
    big_teacher.add_argument("--runtime-seed-guard-rank-margin", type=float, default=0.10)
    big_teacher.add_argument("--runtime-seed-guard-suit-score", type=float, default=0.70)
    big_teacher.add_argument("--runtime-seed-guard-suit-margin", type=float, default=0.04)
    big_teacher.add_argument("--runtime-every", type=float, default=10.0, help="runtime validation/export sample interval seconds")
    big_teacher.add_argument("--runtime-max-frames", type=int, default=80, help="maximum sampled frames per validation video")
    big_teacher.add_argument("--runtime-min-confidence", type=float, default=0.35)
    big_teacher.add_argument("--runtime-augment", type=int, default=8)
    big_teacher.add_argument("--runtime-external-augment", type=int)
    big_teacher.add_argument("--runtime-glyph-augment", type=int, default=8)
    big_teacher.add_argument("--runtime-max-external", type=int)
    big_teacher.add_argument("--runtime-min-accepted", type=int, default=1)
    big_teacher.add_argument("--runtime-max-benchmark-samples", type=int, default=300)
    big_teacher.add_argument("--runtime-max-diff-rows", type=int, default=300)
    big_teacher.add_argument("--runtime-max-risk", type=int, default=0)
    big_teacher.add_argument("--runtime-max-real-problem", type=int, default=0)
    big_teacher.add_argument("--runtime-max-board-bad", type=int, default=0)
    big_teacher.add_argument("--runtime-max-median-ms", type=float, default=300.0)
    big_teacher.add_argument("--runtime-max-p90-ms", type=float, default=900.0)
    big_teacher.add_argument("--runtime-max-median-regression-ms", type=float)
    big_teacher.add_argument("--runtime-max-p90-regression-ms", type=float)
    big_teacher.add_argument("--no-runtime-risk-queue", action="store_true", help="do not generate a label queue from risky runtime diff rows")
    big_teacher.add_argument("--runtime-risk-queue-max-rows", type=int, default=80)
    big_teacher.add_argument("--format", choices=("json", "text"), default="text")
    big_teacher.add_argument("--compact", action="store_true", help="compact JSON output")
    big_teacher.set_defaults(command="card-big-teacher")

    hf_probe_review = subparsers.add_parser(
        "apply-card-hf-probe-review",
        help="rewrite an export-card-review CSV with HuggingFace probe card predictions for diff/gate checks",
    )
    hf_probe_review.add_argument("--review-csv", type=Path, required=True, help="source export-card-review CSV")
    hf_probe_review.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_review_hf_probe_candidate",
        help="output directory containing candidate review.csv",
    )
    hf_probe_review.add_argument("--probe-dir", type=Path, required=True, help="directory containing hf_rank_probe.npz/hf_suit_probe.npz")
    hf_probe_review.add_argument("--max-rows", type=int, help="maximum source rows to rewrite; omit for all rows")
    hf_probe_review.add_argument("--batch-size", type=int, default=32)
    hf_probe_review.add_argument("--device", default="auto", help="auto/cpu/-1/0/cuda:0")
    hf_probe_review.add_argument("--local-files-only", action="store_true", help="do not download HuggingFace models")
    hf_probe_review.add_argument("--format", choices=("json", "text"), default="text")
    hf_probe_review.add_argument("--compact", action="store_true", help="compact JSON output")
    hf_probe_review.set_defaults(command="apply-card-hf-probe-review")

    synthetic = subparsers.add_parser(
        "generate-card-synthetic",
        help="generate synthetic normalized rank/suit glyphs for class-balancing deep-card training",
    )
    synthetic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_glyph_synthetic",
        help="output directory with rank/ and suit/ folders",
    )
    synthetic.add_argument("--per-class", type=int, default=80, help="synthetic images per rank/suit class")
    synthetic.add_argument("--seed", type=int, default=20260708, help="random seed")
    synthetic.add_argument("--rank-only", action="store_true", help="generate only rank glyphs")
    synthetic.add_argument("--suit-only", action="store_true", help="generate only suit glyphs")
    synthetic.add_argument("--format", choices=("json", "text"), default="text")
    synthetic.add_argument("--compact", action="store_true", help="compact JSON output")
    synthetic.set_defaults(command="generate-card-synthetic")

    download_dataset = subparsers.add_parser(
        "download-card-dataset",
        help="download or inspect a HuggingFace playing-card image dataset",
    )
    download_dataset.add_argument("--repo-id", default=DEFAULT_HF_CARD_REPO, help="HuggingFace dataset repo id")
    download_dataset.add_argument("--repo-type", default="dataset", help="HuggingFace repo type")
    download_dataset.add_argument("--output-dir", type=Path, default=DEFAULT_HF_CARD_DIR, help="local dataset directory")
    download_dataset.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="optional HuggingFace allow pattern; can be repeated",
    )
    download_dataset.add_argument("--refresh", action="store_true", help="force a HuggingFace snapshot refresh")
    download_dataset.add_argument("--local-files-only", action="store_true", help="do not use network; inspect/cache only")
    download_dataset.add_argument("--format", choices=("json", "text"), default="text")
    download_dataset.add_argument("--compact", action="store_true", help="compact JSON output")
    download_dataset.set_defaults(command="download-card-dataset")

    pipeline = subparsers.add_parser(
        "card-cv-pipeline",
        help="inspect and prepare the full table-localization, crop, rank/suit training, and live-CV pipeline",
    )
    pipeline.add_argument("--output-dir", type=Path, default=DEFAULT_PIPELINE_OUTPUT_DIR)
    pipeline.add_argument("--bbox", default="x,y,w,h", help="screen bbox used for live/preflight commands")
    pipeline.add_argument("--bbox-file", type=Path, help="bbox.json produced by screen-cv --pick-bbox")
    pipeline.add_argument("--latest-bbox", action="store_true", help="use the newest bbox.json under video_frames")
    pipeline.add_argument("--hero-name", help="stable hero name used as a live layout anchor")
    pipeline.add_argument("--video-dir", type=Path, default=Path("video_frames"))
    pipeline.add_argument("--crop-dir", type=Path, action="append", default=[], help="rank/suit crop directory; repeatable")
    pipeline.add_argument("--no-crop-image-audit", action="store_true", help="skip imread checks over rank/suit crop images")
    pipeline.add_argument("--min-rank-per-label", type=int, default=1, help="minimum crop count expected for each rank label")
    pipeline.add_argument("--min-suit-per-label", type=int, default=1, help="minimum crop count expected for each suit label")
    pipeline.add_argument("--probe-dir", type=Path, default=DEFAULT_PIPELINE_PROBE_DIR)
    pipeline.add_argument("--probe-model", default="facebook/dinov2-base", help="shared HuggingFace vision model for generated/train probe commands")
    pipeline.add_argument("--probe-rank-model", help="rank-specific HuggingFace vision model for split probe training")
    pipeline.add_argument("--probe-suit-model", help="suit-specific HuggingFace vision model for split probe training")
    pipeline.add_argument("--probe-max-images-per-class", type=int, default=24, help="maximum trusted crops per class for probe training")
    pipeline.add_argument("--probe-batch-size", type=int, default=16)
    pipeline.add_argument("--run-train-probe", action="store_true", help="train the configured split HF probe before inspecting/running teacher")
    pipeline.add_argument("--dataset-repo-id", default=DEFAULT_HF_CARD_REPO, help="primary HuggingFace dataset repo id")
    pipeline.add_argument("--dataset-repo-type", default="dataset", help="primary HuggingFace repo type")
    pipeline.add_argument("--dataset-allow-pattern", action="append", default=[], help="optional dataset allow pattern; repeatable")
    pipeline.add_argument("--dataset-dir", type=Path, default=DEFAULT_HF_CARD_DIR)
    pipeline.add_argument("--extra-dataset-dir", type=Path, action="append", default=[], help="additional local full-card dataset/image directory; repeatable")
    pipeline.add_argument("--ingested-dataset-dir", type=Path, default=DEFAULT_PIPELINE_INGEST_DIR)
    pipeline.add_argument("--knn-model", type=Path, default=DEFAULT_MODEL_PATH)
    pipeline.add_argument("--deep-card-model-dir", type=Path, default=DEFAULT_CV_DEEP_MODEL_DIR)
    pipeline.add_argument("--validation-summary-json", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    pipeline.add_argument("--gate-summary-json", type=Path, default=DEFAULT_GATE_SUMMARY)
    pipeline.add_argument("--download-dataset", action="store_true", help="download/refresh the configured public card dataset")
    pipeline.add_argument("--refresh-dataset", action="store_true", help="force refresh when --download-dataset is set")
    pipeline.add_argument("--local-files-only", action="store_true", help="do not use network for dataset/HF model checks")
    pipeline.add_argument("--ingest-dataset", action="store_true", help="extract rank/suit glyphs from the configured dataset")
    pipeline.add_argument("--max-external-ingest", type=int, default=1200)
    pipeline.add_argument("--run-smoke", action="store_true", help="run a small card-big-teacher smoke over the crop dirs")
    pipeline.add_argument("--smoke-max-images", type=int, default=20)
    pipeline.add_argument("--smoke-output-dir", type=Path)
    pipeline.add_argument("--smoke-online", action="store_true", help="allow HuggingFace downloads during the smoke")
    pipeline.add_argument("--smoke-batch-size", type=int, default=16)
    pipeline.add_argument("--run-teacher", action="store_true", help="run the offline big-model rank/suit teacher over the crop dirs")
    pipeline.add_argument("--teacher-output-dir", type=Path)
    pipeline.add_argument("--teacher-max-images", type=int, help="maximum crops to label per kind during --run-teacher")
    pipeline.add_argument("--teacher-online", action="store_true", help="allow HuggingFace downloads during --run-teacher")
    pipeline.add_argument("--teacher-batch-size", type=int, default=16)
    pipeline.add_argument("--teacher-distill-runtime", action="store_true", help="also train, validate, and gate a runtime KNN candidate")
    pipeline.add_argument("--teacher-runtime-video", type=Path, action="append", default=[], help="specific runtime validation video for teacher distill; repeatable")
    pipeline.add_argument("--teacher-runtime-every", type=float, default=10.0)
    pipeline.add_argument("--teacher-runtime-max-frames", type=int, default=80)
    pipeline.add_argument("--teacher-runtime-max-benchmark-samples", type=int, default=300)
    pipeline.add_argument("--teacher-runtime-max-diff-rows", type=int, default=300)
    pipeline.add_argument("--no-candidate-summary", action="store_true", help="skip scanning existing gate summaries")
    pipeline.add_argument("--candidate-search-dir", type=Path, default=Path("video_frames"))
    pipeline.add_argument("--candidate-output-dir", type=Path)
    pipeline.add_argument("--keep-candidate-duplicates", action="store_true")
    pipeline.add_argument("--run-auto-bbox-diagnostics", action="store_true", help="run recorded-video auto-bbox localization diagnostics")
    pipeline.add_argument("--auto-bbox-output-dir", type=Path)
    pipeline.add_argument("--auto-bbox-every", type=float, default=300.0)
    pipeline.add_argument("--auto-bbox-max-frames", type=int, default=1)
    pipeline.add_argument("--auto-bbox-variant", action="append", default=[], help="auto-bbox stress variant; repeatable")
    pipeline.add_argument("--no-auto-bbox-problem-frames", action="store_true")
    pipeline.add_argument("--format", choices=("json", "text"), default="text")
    pipeline.add_argument("--compact", action="store_true", help="compact JSON output")
    pipeline.set_defaults(command="card-cv-pipeline")

    hand_audit = subparsers.add_parser(
        "audit-card-review",
        help="audit complete hero hands from export-card-review output for possible wrong-card reads",
    )
    hand_audit.add_argument("--review-csv", type=Path, required=True, help="review.csv produced by export-card-review")
    hand_audit.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_hand_audit")
    hand_audit.add_argument("--teacher-model-dir", type=Path, help="offline teacher model directory")
    hand_audit.add_argument("--teacher-rank-model-dir", type=Path, help="rank-specific teacher model directory")
    hand_audit.add_argument("--teacher-suit-model-dir", type=Path, help="suit-specific teacher model directory")
    hand_audit.add_argument("--realtime-model-dir", type=Path, help="realtime fallback model directory")
    hand_audit.add_argument("--realtime-rank-model-dir", type=Path, help="rank-specific realtime model directory")
    hand_audit.add_argument("--realtime-suit-model-dir", type=Path, help="suit-specific realtime model directory")
    hand_audit.add_argument("--rank-confidence-threshold", type=float, default=0.82)
    hand_audit.add_argument("--suit-confidence-threshold", type=float, default=0.72)
    hand_audit.add_argument("--open-suit-score-threshold", type=float, default=0.78)
    hand_audit.add_argument("--open-suit-margin-threshold", type=float, default=0.08)
    hand_audit.add_argument("--max-review", type=int, default=240)
    hand_audit.add_argument("--no-copy-review-assets", action="store_true", help="do not copy reviewed crops/frames")
    hand_audit.add_argument("--format", choices=("json", "text"), default="text")
    hand_audit.add_argument("--compact", action="store_true", help="compact JSON output")
    hand_audit.set_defaults(command="audit-card-review")

    card_benchmark = subparsers.add_parser(
        "benchmark-card-review",
        help="benchmark rank/suit/card recognition against manual or high-confidence review CSV labels",
    )
    card_benchmark.add_argument(
        "--review-csv",
        type=Path,
        action="append",
        required=True,
        help="review.csv produced by export-card-review or audit-card-review; can be repeated",
    )
    card_benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_benchmark",
        help="benchmark output directory",
    )
    card_benchmark.add_argument("--deep-card-model-dir", type=Path, help="deep rank/suit model directory to benchmark")
    card_benchmark.add_argument("--deep-rank-card-model-dir", type=Path, help="rank-specific deep model directory")
    card_benchmark.add_argument("--deep-suit-card-model-dir", type=Path, help="suit-specific deep model directory")
    card_benchmark.add_argument("--knn-model", type=Path, default=DEFAULT_MODEL_PATH, help="KNN glyph model .npz to benchmark")
    card_benchmark.add_argument("--hf-probe-dir", type=Path, help="HuggingFace embedding probe directory to benchmark")
    card_benchmark.add_argument("--hf-probe-device", default="auto", help="auto/cpu/-1/0/cuda:0 for --hf-probe-dir")
    card_benchmark.add_argument("--hf-probe-local-files-only", action="store_true", help="do not download HuggingFace model files for --hf-probe-dir")
    card_benchmark.add_argument(
        "--include-ok-pseudo",
        action="store_true",
        help="also use review_reason=ok rows as high-confidence pseudo-truth",
    )
    card_benchmark.add_argument(
        "--allowed-pseudo-reason",
        action="append",
        default=[],
        help="review_reason accepted as pseudo-truth; default is ok when --include-ok-pseudo is used",
    )
    card_benchmark.add_argument("--no-runtime", action="store_true", help="skip re-running the current runtime recognizer on card crops")
    card_benchmark.add_argument("--max-samples", type=int, help="maximum card slots to evaluate")
    card_benchmark.add_argument("--format", choices=("json", "text"), default="text")
    card_benchmark.add_argument("--compact", action="store_true", help="compact JSON output")
    card_benchmark.set_defaults(command="benchmark-card-review")

    card_diff = subparsers.add_parser(
        "diff-card-review",
        help="compare baseline and candidate export-card-review CSV files for card-recognition regressions",
    )
    card_diff.add_argument("--baseline-review-csv", type=Path, required=True, help="baseline review.csv")
    card_diff.add_argument("--candidate-review-csv", type=Path, required=True, help="candidate review.csv")
    card_diff.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_review_diff",
        help="diff output directory",
    )
    card_diff.add_argument(
        "--risky-baseline-reason",
        action="append",
        default=[],
        help="baseline review_reason treated as high-confidence; default ok",
    )
    card_diff.add_argument("--max-rows", type=int, help="maximum slot rows to compare")
    card_diff.add_argument("--fail-on-risk", action="store_true", help="exit non-zero if risky changes are found")
    card_diff.add_argument("--max-risk", type=int, default=0, help="allowed risky slot changes when --fail-on-risk is set")
    card_diff.add_argument("--format", choices=("json", "text"), default="text")
    card_diff.add_argument("--compact", action="store_true", help="compact JSON output")
    card_diff.set_defaults(command="diff-card-review")

    diff_risks = subparsers.add_parser(
        "summarize-card-diff-risks",
        help="group card_review_diff_rows.csv risk rows into manual-label actions",
    )
    diff_risks.add_argument("--diff-csv", type=Path, required=True, help="card_review_diff_rows.csv")
    diff_risks.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_diff_risk_summary",
        help="risk summary output directory",
    )
    diff_risks.add_argument("--include-safe", action="store_true", help="include non-risk changed rows as well")
    diff_risks.add_argument("--no-include-same", action="store_true", help="exclude same-card confidence downgrade rows")
    diff_risks.add_argument("--max-examples", type=int, default=8)
    diff_risks.add_argument("--format", choices=("json", "text"), default="text")
    diff_risks.add_argument("--compact", action="store_true", help="compact JSON output")
    diff_risks.set_defaults(command="summarize-card-diff-risks")

    card_gate = subparsers.add_parser(
        "gate-card-model",
        help="run benchmark plus review-diff checks and decide whether a card model candidate is promotable",
    )
    card_gate.add_argument("--benchmark-review-csv", type=Path, action="append", required=True, help="review CSV used for benchmark; can be repeated")
    card_gate.add_argument("--baseline-review-csv", type=Path, required=True, help="current promoted baseline review.csv")
    card_gate.add_argument("--candidate-review-csv", type=Path, required=True, help="candidate model review.csv")
    card_gate.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_model_gate", help="gate output directory")
    card_gate.add_argument("--candidate-name", default="candidate", help="name shown in reports")
    card_gate.add_argument(
        "--candidate-evaluator",
        choices=("current_csv", "runtime", "knn", "deep", "hf_probe"),
        default="knn",
        help="benchmark evaluator treated as the candidate",
    )
    card_gate.add_argument("--knn-model", type=Path, default=DEFAULT_MODEL_PATH, help="KNN glyph model .npz to gate")
    card_gate.add_argument("--deep-card-model-dir", type=Path, help="deep rank/suit model directory to gate")
    card_gate.add_argument("--deep-rank-card-model-dir", type=Path, help="rank-specific deep model directory")
    card_gate.add_argument("--deep-suit-card-model-dir", type=Path, help="suit-specific deep model directory")
    card_gate.add_argument("--hf-probe-dir", type=Path, help="HuggingFace embedding probe directory to gate")
    card_gate.add_argument("--hf-probe-device", default="auto", help="auto/cpu/-1/0/cuda:0 for --hf-probe-dir")
    card_gate.add_argument("--hf-probe-local-files-only", action="store_true", help="do not download HuggingFace model files for --hf-probe-dir")
    card_gate.add_argument("--candidate-validation-summary-json", type=Path, help="candidate validate-cv summary JSON")
    card_gate.add_argument("--baseline-validation-summary-json", type=Path, help="baseline validate-cv summary JSON for latency comparison")
    card_gate.add_argument("--include-ok-pseudo", action="store_true", help="also use review_reason=ok rows as high-confidence pseudo-truth")
    card_gate.add_argument("--allowed-pseudo-reason", action="append", default=[], help="accepted pseudo-truth/review-diff high-confidence reason; default ok")
    card_gate.add_argument("--no-runtime", action="store_true", help="skip current runtime recognizer in the benchmark substep")
    card_gate.add_argument("--max-benchmark-samples", type=int, help="maximum card slots to benchmark")
    card_gate.add_argument("--max-diff-rows", type=int, help="maximum slot rows to diff")
    card_gate.add_argument("--max-risk", type=int, default=0, help="allowed risky review-diff changes")
    card_gate.add_argument("--require-validation", action="store_true", help="fail when candidate validation summary is missing")
    card_gate.add_argument("--max-real-problem", type=int, default=0, help="allowed validate-cv real_problem_count")
    card_gate.add_argument("--max-board-bad", type=int, default=0, help="allowed validate-cv board_bad_count")
    card_gate.add_argument("--max-median-ms", type=float, help="absolute allowed validate-cv median latency")
    card_gate.add_argument("--max-p90-ms", type=float, help="absolute allowed validate-cv p90 latency")
    card_gate.add_argument("--max-median-regression-ms", type=float, help="allowed median latency above baseline validation")
    card_gate.add_argument("--max-p90-regression-ms", type=float, help="allowed p90 latency above baseline validation")
    card_gate.add_argument("--min-card-acc", type=float, default=0.999)
    card_gate.add_argument("--min-rank-acc", type=float, default=0.999)
    card_gate.add_argument("--min-suit-acc", type=float, default=0.999)
    card_gate.add_argument("--allow-missing-rows", action="store_true", help="do not fail when candidate review is missing baseline rows")
    card_gate.add_argument("--fail-on-reject", action="store_true", help="exit non-zero when the candidate is rejected")
    card_gate.add_argument("--format", choices=("json", "text"), default="text")
    card_gate.add_argument("--compact", action="store_true", help="compact JSON output")
    card_gate.set_defaults(command="gate-card-model")

    candidate_summary = subparsers.add_parser(
        "summarize-card-candidates",
        help="summarize all card-model promotion gate outputs into one ranked CSV/Markdown report",
    )
    candidate_summary.add_argument(
        "--gate-summary-json",
        type=Path,
        action="append",
        default=[],
        help="specific card_model_gate_summary.json to include; can be repeated",
    )
    candidate_summary.add_argument(
        "--search-dir",
        type=Path,
        default=Path("video_frames"),
        help="directory recursively scanned for card_model_gate_summary.json when no --gate-summary-json is supplied",
    )
    candidate_summary.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_candidate_summary",
        help="summary output directory",
    )
    candidate_summary.add_argument("--keep-duplicates", action="store_true", help="keep multiple gate runs for the same candidate/evaluator")
    candidate_summary.add_argument("--format", choices=("json", "text"), default="text")
    candidate_summary.add_argument("--compact", action="store_true", help="compact JSON output")
    candidate_summary.set_defaults(command="summarize-card-candidates")

    cv_health = subparsers.add_parser(
        "cv-health",
        help="check promoted CV/card model readiness and print the recommended live screen-cv command",
    )
    cv_health.add_argument("--output-dir", type=Path, default=DEFAULT_HEALTH_OUTPUT_DIR, help="health report output directory")
    cv_health.add_argument("--knn-model", type=Path, default=DEFAULT_MODEL_PATH, help="promoted KNN glyph model .npz")
    cv_health.add_argument("--deep-card-model-dir", type=Path, default=DEFAULT_CV_DEEP_MODEL_DIR)
    cv_health.add_argument("--validation-summary-json", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    cv_health.add_argument("--gate-summary-json", type=Path, default=DEFAULT_GATE_SUMMARY)
    cv_health.add_argument("--bbox", default="x,y,w,h", help="live screen bbox to place in the generated command")
    cv_health.add_argument("--bbox-file", type=Path, help="bbox.json produced by screen-cv --pick-bbox")
    cv_health.add_argument("--latest-bbox", action="store_true", help="use the newest bbox.json under video_frames")
    cv_health.add_argument("--allow-placeholder-bbox", action="store_true", help="allow non-numeric placeholder bbox values")
    cv_health.add_argument("--screen-output-dir", type=Path, default=Path("video_frames") / "screen_live")
    cv_health.add_argument("--fast-screen-output-dir", type=Path, default=Path("video_frames") / "screen_live_fast")
    cv_health.add_argument("--preflight-output-dir", type=Path, default=Path("video_frames") / "screen_preflight")
    cv_health.add_argument("--hero-name", help="optional hero name for layout locking")
    cv_health.add_argument("--effective-stack", type=float, default=100.0)
    cv_health.add_argument("--villain", default="standard", help="villain profile for generated --with-advice live command")
    cv_health.add_argument("--min-confidence", type=float, default=0.35)
    cv_health.add_argument("--ocr-scale", type=float, default=0.65)
    cv_health.add_argument("--dealer-refresh-frames", type=int, default=12)
    cv_health.add_argument("--auto-bbox-refresh", type=float, default=10.0)
    cv_health.add_argument("--max-real-problem", type=int, default=0)
    cv_health.add_argument("--max-board-bad", type=int, default=0)
    cv_health.add_argument("--max-median-ms", type=float, default=300.0)
    cv_health.add_argument("--max-p90-ms", type=float, default=900.0)
    cv_health.add_argument("--fail-on-not-ready", action="store_true", help="exit non-zero when any health check fails")
    cv_health.add_argument("--format", choices=("json", "text"), default="text")
    cv_health.add_argument("--compact", action="store_true", help="compact JSON output")
    cv_health.set_defaults(command="cv-health")

    label_queue = subparsers.add_parser(
        "prepare-card-label-queue",
        help="merge review/audit CSV files into a prioritized manual card-labeling queue",
    )
    label_queue.add_argument(
        "--review-csv",
        type=Path,
        action="append",
        required=True,
        help="review.csv or audit.csv to merge; can be repeated",
    )
    label_queue.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_label_queue",
        help="label queue output directory",
    )
    label_queue.add_argument("--max-rows", type=int, default=120, help="maximum rows to include")
    label_queue.add_argument("--include-ok", action="store_true", help="also include review_reason=ok rows")
    label_queue.add_argument("--include-completed", action="store_true", help="keep rows that already have final_card labels")
    label_queue.add_argument("--no-copy-assets", action="store_true", help="do not copy image assets into the queue directory")
    label_queue.add_argument("--no-contact-sheet", action="store_true", help="do not render label_queue_sheet.jpg")
    label_queue.add_argument("--format", choices=("json", "text"), default="text")
    label_queue.add_argument("--compact", action="store_true", help="compact JSON output")
    label_queue.set_defaults(command="prepare-card-label-queue")

    diff_label_queue = subparsers.add_parser(
        "prepare-card-diff-label-queue",
        help="build a manual card-labeling queue from card_review_diff_rows.csv risk/change rows",
    )
    diff_label_queue.add_argument("--diff-csv", type=Path, required=True, help="card_review_diff_rows.csv produced by diff-card-review")
    diff_label_queue.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_diff_label_queue",
        help="label queue output directory",
    )
    diff_label_queue.add_argument("--max-rows", type=int, default=80, help="maximum rows to include")
    diff_label_queue.add_argument("--include-non-risk", action="store_true", help="also include non-risk changed rows")
    diff_label_queue.add_argument("--include-same", action="store_true", help="include same-card rows, mainly for diagnostics")
    diff_label_queue.add_argument(
        "--prefer-baseline-assets",
        action="store_true",
        help="use baseline crop/table assets first; useful when candidate crops are only diagnostic and should not train the model",
    )
    diff_label_queue.add_argument("--no-copy-assets", action="store_true", help="do not copy image assets into the queue directory")
    diff_label_queue.add_argument("--no-contact-sheet", action="store_true", help="do not render label_queue_sheet.jpg")
    diff_label_queue.add_argument("--format", choices=("json", "text"), default="text")
    diff_label_queue.add_argument("--compact", action="store_true", help="compact JSON output")
    diff_label_queue.set_defaults(command="prepare-card-diff-label-queue")

    glyph_label_queue = subparsers.add_parser(
        "prepare-card-glyph-label-queue",
        help="build a manual rank/suit glyph-labeling queue from HF teacher predictions.csv rows",
    )
    glyph_label_queue.add_argument(
        "--predictions-csv",
        type=Path,
        action="append",
        default=[],
        help="predictions.csv from card-big-teacher/ensemble-card-hf-predictions; repeatable",
    )
    glyph_label_queue.add_argument(
        "--review-csv",
        type=Path,
        action="append",
        default=[],
        help="review.csv from export-card-review/collect-card-debug-review; repeatable",
    )
    glyph_label_queue.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_glyph_label_queue",
        help="glyph label queue output directory",
    )
    glyph_label_queue.add_argument("--max-rows", type=int, default=200)
    glyph_label_queue.add_argument("--allowed-reason", action="append", default=[], help="only include this reason; repeatable")
    glyph_label_queue.add_argument("--include-accepted", action="store_true", help="also include accepted teacher rows")
    glyph_label_queue.add_argument(
        "--prefill-final-label",
        choices=("none", "current", "teacher"),
        default="none",
        help="optionally prefill final_label from current_label or teacher_label; default leaves manual labels blank",
    )
    glyph_label_queue.add_argument("--no-copy-assets", action="store_true")
    glyph_label_queue.add_argument("--no-contact-sheet", action="store_true")
    glyph_label_queue.add_argument("--format", choices=("json", "text"), default="text")
    glyph_label_queue.add_argument("--compact", action="store_true")
    glyph_label_queue.set_defaults(command="prepare-card-glyph-label-queue")

    glyph_label_apply = subparsers.add_parser(
        "apply-card-glyph-label-queue",
        help="copy completed glyph label queue rows into rank/suit training folders",
    )
    glyph_label_apply.add_argument("--queue-csv", type=Path, required=True, help="glyph_label_queue.csv with final_label values")
    glyph_label_apply.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_glyph_label_applied",
        help="rank/suit dataset output directory",
    )
    glyph_label_apply.add_argument("--format", choices=("json", "text"), default="text")
    glyph_label_apply.add_argument("--compact", action="store_true")
    glyph_label_apply.set_defaults(command="apply-card-glyph-label-queue")

    label_queue_audit = subparsers.add_parser(
        "audit-card-label-queue",
        help="summarize manual card-label queue progress, invalid labels, missing assets, and next apply command",
    )
    label_queue_audit.add_argument("--queue-csv", type=Path, required=True, help="label_queue.csv to audit")
    label_queue_audit.add_argument(
        "--output-dir",
        type=Path,
        help="directory for label_queue_audit.json/md; defaults to the queue CSV directory",
    )
    label_queue_audit.add_argument(
        "--applied-output-dir",
        type=Path,
        help="output directory to place in the generated apply-card-review command",
    )
    label_queue_audit.add_argument("--no-contact-sheet", action="store_true", help="do not render label_queue_sheet.jpg")
    label_queue_audit.add_argument("--format", choices=("json", "text"), default="text")
    label_queue_audit.add_argument("--compact", action="store_true", help="compact JSON output")
    label_queue_audit.set_defaults(command="audit-card-label-queue")

    label_retrain = subparsers.add_parser(
        "retrain-card-label-queue",
        help="apply a completed card label queue, train a candidate KNN model, validate it, and gate it",
    )
    label_retrain.add_argument("--queue-csv", type=Path, required=True, help="completed label_queue.csv with final_card labels")
    label_retrain.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_label_retrain",
        help="output directory for applied labels, candidate review, validation, gate, and summary",
    )
    label_retrain.add_argument(
        "--base-glyph-dir",
        type=Path,
        action="append",
        default=[],
        help=f"existing trusted rank/suit glyph directory; default uses {DEFAULT_BASE_GLYPH_DIR} when present",
    )
    label_retrain.add_argument("--video-dir", type=Path, default=Path("video_frames"), help="root video directory for validation/export")
    label_retrain.add_argument("--video", type=Path, action="append", default=[], help="specific validation video; repeatable; defaults to all root mp4s")
    label_retrain.add_argument(
        "--benchmark-review-csv",
        type=Path,
        action="append",
        default=[],
        help=f"benchmark review CSV; default {DEFAULT_BENCHMARK_REVIEW_CSV}",
    )
    label_retrain.add_argument("--baseline-review-csv", type=Path, default=DEFAULT_BASELINE_REVIEW_CSV)
    label_retrain.add_argument("--baseline-validation-summary-json", type=Path, default=DEFAULT_BASELINE_VALIDATION_SUMMARY)
    label_retrain.add_argument("--deep-card-model-dir", type=Path, default=DEFAULT_RETRAIN_DEEP_CARD_MODEL_DIR)
    label_retrain.add_argument("--candidate-name", help="candidate name in gate reports; defaults to output directory name")
    label_retrain.add_argument("--model", type=Path, help="candidate KNN .npz path; defaults under --output-dir")
    label_retrain.add_argument("--seed-model", type=Path, default=DEFAULT_MODEL_PATH, help="existing KNN .npz prototypes to preserve; defaults to the promoted model")
    label_retrain.add_argument("--no-seed-model", action="store_true", help="train only from templates/base glyphs/manual labels, without preserving promoted prototypes")
    label_retrain.add_argument(
        "--seed-conflict-policy",
        choices=("manual-override", "keep-seed"),
        default="manual-override",
        help="use keep-seed when a diagnostic/baseline queue repeats existing prototypes and should not overwrite the seed",
    )
    label_retrain.add_argument("--every", type=float, default=10.0, help="validation/export sample interval seconds")
    label_retrain.add_argument("--max-frames", type=int, default=80, help="maximum sampled frames per validation video")
    label_retrain.add_argument("--min-confidence", type=float, default=0.35)
    label_retrain.add_argument("--augment", type=int, default=8)
    label_retrain.add_argument("--glyph-augment", type=int, default=8)
    label_retrain.add_argument("--no-templates", action="store_true", help="do not include local card templates while training")
    label_retrain.add_argument("--allow-partial", action="store_true", help="allow retraining before every slot is labeled")
    label_retrain.add_argument("--max-benchmark-samples", type=int, default=300)
    label_retrain.add_argument("--max-diff-rows", type=int, default=300)
    label_retrain.add_argument("--max-risk", type=int, default=0)
    label_retrain.add_argument("--max-real-problem", type=int, default=0)
    label_retrain.add_argument("--max-board-bad", type=int, default=0)
    label_retrain.add_argument("--max-median-ms", type=float, default=300.0)
    label_retrain.add_argument("--max-p90-ms", type=float, default=900.0)
    label_retrain.add_argument("--max-median-regression-ms", type=float)
    label_retrain.add_argument("--max-p90-regression-ms", type=float)
    label_retrain.add_argument("--format", choices=("json", "text"), default="text")
    label_retrain.add_argument("--compact", action="store_true", help="compact JSON output")
    label_retrain.add_argument("--fail-on-reject", action="store_true", help="exit non-zero when the candidate does not promote")
    label_retrain.set_defaults(command="retrain-card-label-queue")

    label_server = subparsers.add_parser(
        "serve-card-label-queue",
        help="open a local browser UI for editing final_card labels in a label_queue.csv",
    )
    label_server.add_argument(
        "--queue-csv",
        type=Path,
        required=True,
        help="label_queue.csv produced by prepare-card-label-queue",
    )
    label_server.add_argument("--host", default="127.0.0.1", help="bind host")
    label_server.add_argument("--port", type=int, default=8765, help="bind port")
    label_server.add_argument("--open-browser", action="store_true", help="open the UI in the default browser")
    label_server.add_argument("--format", choices=("json", "text"), default="text")
    label_server.add_argument("--compact", action="store_true", help="compact JSON output")
    label_server.set_defaults(command="serve-card-label-queue")

    glyph_label_server = subparsers.add_parser(
        "serve-card-glyph-label-queue",
        help="open a local browser UI for reviewing rank/suit glyph labels",
    )
    glyph_label_server.add_argument(
        "--queue-csv",
        type=Path,
        required=True,
        help="glyph_label_queue.csv produced by prepare-card-glyph-label-queue",
    )
    glyph_label_server.add_argument("--host", default="127.0.0.1", help="bind host")
    glyph_label_server.add_argument("--port", type=int, default=8766, help="bind port")
    glyph_label_server.add_argument("--open-browser", action="store_true", help="open the UI in the default browser")
    glyph_label_server.add_argument("--format", choices=("json", "text"), default="text")
    glyph_label_server.add_argument("--compact", action="store_true", help="compact JSON output")
    glyph_label_server.set_defaults(command="serve-card-glyph-label-queue")

    fixed_card_replay = subparsers.add_parser(
        "replay-fixed-card-samples",
        help="re-crop saved full frames with fixed relative H1/H2 and B1-B5 boxes, then build a fresh glyph queue",
    )
    fixed_card_replay.add_argument("--samples-dir", type=Path, required=True)
    fixed_card_replay.add_argument("--sample-prefix", help="only replay sample directories with this name prefix")
    fixed_card_replay.add_argument("--layout-profile", type=Path, required=True)
    fixed_card_replay.add_argument("--old-queue-csv", type=Path)
    fixed_card_replay.add_argument("--output-dir", type=Path, required=True)
    fixed_card_replay.add_argument("--format", choices=("json", "text"), default="text")
    fixed_card_replay.add_argument("--compact", action="store_true")
    fixed_card_replay.set_defaults(command="replay-fixed-card-samples")

    deep_cards = subparsers.add_parser(
        "train-deep-card-classifier",
        help="train torch rank/suit glyph classifiers from export-card-glyphs output",
    )
    deep_cards.add_argument("--glyph-dir", type=Path, default=Path("video_frames") / "card_glyph_export", help="directory containing rank/ and suit/ folders")
    deep_cards.add_argument("--extra-glyph-dir", type=Path, action="append", default=[], help="additional rank/suit glyph dataset directory")
    deep_cards.add_argument("--kind", choices=("rank", "suit", "both"), default="both", help="which classifier to train")
    deep_cards.add_argument("--model-dir", type=Path, default=DEFAULT_DEEP_MODEL_DIR, help="output model directory")
    deep_cards.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="local rank/suit template directory")
    deep_cards.add_argument("--no-templates", action="store_true", help="do not include local pict/card_templates seed samples")
    deep_arch_choices = (
        "simple_cnn",
        "mobilenet_v3_small",
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b2",
        "convnext_tiny",
        "swin_t",
        "vit_b_16",
    )
    deep_cards.add_argument(
        "--arch",
        choices=deep_arch_choices,
        default="mobilenet_v3_small",
        help="default backbone for rank/suit unless overridden by --rank-arch or --suit-arch",
    )
    deep_cards.add_argument("--rank-arch", choices=deep_arch_choices, help="rank-only backbone override")
    deep_cards.add_argument("--suit-arch", choices=deep_arch_choices, help="suit-only backbone override")
    deep_cards.add_argument("--pretrained", action="store_true", help="use torchvision pretrained weights; may download weights")
    deep_cards.add_argument("--freeze-backbone", action="store_true", help="freeze pretrained feature extractor and train only the classification head")
    deep_cards.add_argument("--class-balanced-loss", action="store_true", help="weight CrossEntropyLoss by inverse class frequency")
    deep_cards.add_argument("--weighted-sampler", action="store_true", help="sample rare rank/suit classes more often during training")
    deep_cards.add_argument("--epochs", type=int, default=8)
    deep_cards.add_argument("--batch-size", type=int, default=48)
    deep_cards.add_argument("--lr", type=float, default=3e-4)
    deep_cards.add_argument("--val-split", type=float, default=0.18)
    deep_cards.add_argument("--max-images-per-class", type=int)
    deep_cards.add_argument("--seed", type=int, default=17)
    deep_cards.add_argument("--image-size", type=int, default=96)
    deep_cards.add_argument("--rank-image-size", type=int, help="rank-only input size override")
    deep_cards.add_argument("--suit-image-size", type=int, help="suit-only input size override")
    deep_cards.add_argument("--num-workers", type=int, default=0)
    deep_cards.add_argument("--format", choices=("json", "text"), default="text")
    deep_cards.add_argument("--compact", action="store_true", help="compact JSON output")
    deep_cards.set_defaults(command="train-deep-card-classifier")

    ingest_cards = subparsers.add_parser(
        "ingest-card-images",
        help="ingest external labeled full-card images into rank/suit glyph folders",
    )
    ingest_cards.add_argument("--dataset-dir", type=Path, action="append", required=True, help="external image directory or image file")
    ingest_cards.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_frames") / "card_glyph_export_external",
        help="output glyph dataset directory",
    )
    ingest_cards.add_argument("--max-images", type=int, help="maximum external images to ingest")
    ingest_cards.add_argument("--format", choices=("json", "text"), default="text")
    ingest_cards.add_argument("--compact", action="store_true", help="compact JSON output")
    ingest_cards.set_defaults(command="ingest-card-images")

    audit_glyphs = subparsers.add_parser(
        "audit-card-glyphs",
        help="rank exported glyphs by low confidence and model disagreement for manual review",
    )
    audit_glyphs.add_argument("--manifest", type=Path, default=Path("video_frames") / "card_glyph_export_v1" / "manifest.jsonl")
    audit_glyphs.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_glyph_audit")
    audit_glyphs.add_argument("--teacher-model-dir", type=Path, help="offline teacher model directory, e.g. pict/card_models/deep_v1")
    audit_glyphs.add_argument("--teacher-rank-model-dir", type=Path, help="rank-specific teacher model directory; overrides --teacher-model-dir for rank")
    audit_glyphs.add_argument("--teacher-suit-model-dir", type=Path, help="suit-specific teacher model directory; overrides --teacher-model-dir for suit")
    audit_glyphs.add_argument("--realtime-model-dir", type=Path, help="realtime fallback model directory, e.g. pict/card_models/deep_realtime_v1")
    audit_glyphs.add_argument("--realtime-rank-model-dir", type=Path, help="rank-specific realtime model directory; overrides --realtime-model-dir for rank")
    audit_glyphs.add_argument("--realtime-suit-model-dir", type=Path, help="suit-specific realtime model directory; overrides --realtime-model-dir for suit")
    audit_glyphs.add_argument("--max-review", type=int, default=240)
    audit_glyphs.add_argument("--rank-confidence-threshold", type=float, default=0.82)
    audit_glyphs.add_argument("--suit-confidence-threshold", type=float, default=0.55)
    audit_glyphs.add_argument("--temporal-window-frames", type=int, default=120, help="neighbor frame window for same video/source/card_index voting")
    audit_glyphs.add_argument("--temporal-min-support", type=int, default=2, help="minimum nearby high-confidence labels for temporal consensus")
    audit_glyphs.add_argument("--no-copy-accepted", action="store_true", help="do not copy consensus accepted samples")
    audit_glyphs.add_argument("--format", choices=("json", "text"), default="text")
    audit_glyphs.add_argument("--compact", action="store_true", help="compact JSON output")
    audit_glyphs.set_defaults(command="audit-card-glyphs")

    apply_review = subparsers.add_parser(
        "apply-card-review",
        help="copy manually reviewed rank/suit glyph labels from review.csv into a training dataset",
    )
    apply_review.add_argument("--review-csv", type=Path, required=True)
    apply_review.add_argument("--output-dir", type=Path, default=Path("video_frames") / "card_glyph_review_applied")
    apply_review.add_argument("--format", choices=("json", "text"), default="text")
    apply_review.add_argument("--compact", action="store_true", help="compact JSON output")
    apply_review.set_defaults(command="apply-card-review")

    schema = subparsers.add_parser("schema", help="输出 CV/上游模块应发送的 JSON 结构示例")
    schema.set_defaults(command="schema")
    return parser


def add_simulator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--level",
        default="simple",
        help="练习难度：simple/easy/beginner/medium/advanced/master，或 简单/中等/高级/大师",
    )
    parser.add_argument("--street", default="random", choices=("random", "preflop", "flop", "turn", "river"))
    parser.add_argument("--position", default="random", help="random/UTG/HJ/CO/BTN/SB/BB")
    parser.add_argument("--scenario", default="random", help="random/rfi/vs_open/vs_3bet")
    parser.add_argument("--villain", default="level", help="level/tight/standard/wide/random")
    parser.add_argument("--seed", type=int, help="随机种子")


def command_advise(args: argparse.Namespace) -> int:
    state = read_state(args.state)
    result = advise_state(state, iterations=args.iterations)
    if args.format == "text":
        print(format_text(result))
    else:
        print_json(result, pretty=not args.compact)
    return 0


def command_review_states(args: argparse.Namespace) -> int:
    payload = build_state_review(
        events_path=args.events,
        output_dir=args.output_dir,
        limit=args.limit,
        include_watch=args.include_watch,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_state_review_summary(payload))
    return 0


def command_serve_state_action_label_queue(args: argparse.Namespace) -> int:
    prepared = prepare_state_action_label_queue(
        events_path=args.events,
        output_dir=args.output_dir,
        max_items=args.max_items,
        extra_events=tuple(args.extra_events),
    )
    if args.format == "json":
        print_json(prepared, pretty=not args.compact)
    else:
        print(format_state_action_label_queue_summary(prepared))
    payload = serve_state_action_label_queue(
        queue_csv=Path(str(prepared["queue_csv"])),
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_state_action_label_server_summary(payload))
    return 0


def command_stream(args: argparse.Namespace) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
            result = advise_state(state, iterations=args.iterations)
        except Exception as error:
            result = {"ok": False, "error": str(error)}
        print_json(result, pretty=False)
        sys.stdout.flush()
    return 0


def command_deal(args: argparse.Namespace) -> int:
    if args.with_answer:
        payload = build_practice_round(
            level=args.level,
            street=args.street,
            position=args.position,
            scenario=args.scenario,
            villain_profile=args.villain,
            seed=args.seed,
        )
    else:
        payload = generate_spot(
            level=args.level,
            street=args.street,
            position=args.position,
            scenario=args.scenario,
            villain_profile=args.villain,
            seed=args.seed,
        )
    if args.format == "text":
        print(format_spot_text(payload))
    else:
        print_json(payload, pretty=not args.compact)
    return 0


def command_practice(args: argparse.Namespace) -> int:
    return run_practice(
        level=args.level,
        count=args.count,
        street=args.street,
        position=args.position,
        scenario=args.scenario,
        villain=args.villain,
        seed=args.seed,
        iterations=args.iterations,
    )


def command_ui(_args: argparse.Namespace) -> int:
    print()
    print("=== 德州扑克练习盘 ===")
    print("1. 简单    只练最基础、最清楚的题")
    print("2. 中等    加入 3 张公共牌后的题")
    print("3. 高级    加入更多后面的牌和更难的出钱情况")
    print("4. 大师    很难，很多题没有绝对简单答案")
    print()
    level = choose_menu_value(
        prompt="选择难度 [1-4，默认 1]: ",
        mapping={"1": "simple", "2": "medium", "3": "advanced", "4": "master"},
        default="simple",
    )
    count = choose_count()
    return run_practice(
        level=level,
        count=count,
        street="random",
        position="random",
        scenario="random",
        villain="level",
        seed=None,
        iterations=700,
    )


def choose_menu_value(prompt: str, mapping: dict[str, str], default: str) -> str:
    raw = input(prompt).strip().lower()
    if not raw:
        return default
    return mapping.get(raw, raw)


def choose_count() -> int:
    raw = input("练几题？默认 10: ").strip()
    if not raw:
        return 10
    try:
        return max(1, int(raw))
    except ValueError:
        print("看不懂题数，先按 10 题来。")
        return 10


def run_practice(
    level: str,
    count: int,
    street: str,
    position: str,
    scenario: str,
    villain: str,
    seed: int | None,
    iterations: int,
) -> int:
    total = max(1, count)
    score = 0.0
    print()
    print("模拟盘开始。你可以输入中文动作，也可以点网页按钮练。输入 h 看提示，q 退出。")
    for index in range(total):
        round_seed = None if seed is None else seed + index
        round_data = build_practice_round(
            level=level,
            street=street,
            position=position,
            scenario=scenario,
            villain_profile=villain,
            seed=round_seed,
            iterations=iterations,
        )
        state = round_data["state"]
        answer = round_data["answer"]
        actions = " / ".join(plain_action_label(item) for item in round_data["actions"])
        print()
        print(f"[{index + 1}/{total}] {spot_title(state)}")
        print_lesson("提示", round_data["lesson"]["before"])
        print(f"可选动作：{actions}")
        while True:
            try:
                raw_action = input("> ").strip()
            except EOFError:
                print("\n输入结束。")
                return 0
            if raw_action.lower() in {"h", "hint", "提示"}:
                print_lesson("提示", round_data["lesson"]["before"])
                continue
            break
        if raw_action.lower() in {"q", "quit", "exit"}:
            break
        judgment = judge_action(raw_action, answer)
        if judgment.get("grade") == "correct":
            score += 1
        elif judgment.get("grade") == "mixed":
            score += 0.5
        print(judgment["message"])
        print(format_answer_brief(answer))
        print_lesson("学习点", round_data["lesson"]["after"])
    print(f"\n练习结束：{score:g}/{total}")
    return 0


def command_schema(_args: argparse.Namespace) -> int:
    print_json(sample_schema(), pretty=True)
    return 0


def command_rules(_args: argparse.Namespace) -> int:
    print(rules_text())
    return 0


def command_web(args: argparse.Namespace) -> int:
    run_web(host=args.host, port=args.port)
    return 0


def command_cv(args: argparse.Namespace) -> int:
    payload = analyze_table_image(
        image_path=args.image,
        template_path=args.template,
        seat_count=args.seats,
        min_confidence=args.min_confidence,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        annotate_path=args.annotate,
    )
    if args.format == "text":
        print(format_vision_text(payload))
    else:
        print_json(payload, pretty=not args.compact)
    return 0


def command_video_cv(args: argparse.Namespace) -> int:
    payload = analyze_video(
        video_path=args.video,
        output_dir=args.output_dir,
        template_path=args.template,
        seat_count=args.seats,
        start_sec=args.start,
        end_sec=args.end,
        every_sec=args.every,
        middle=args.middle,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        save_frames=not args.no_frames,
        save_annotated=not args.no_annotated,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_video_summary(payload))
    return 0


def command_live_cv(args: argparse.Namespace) -> int:
    payload = analyze_realtime_video(
        video_path=args.video,
        output_dir=args.output_dir,
        template_path=args.template,
        seat_count=args.seats,
        start_sec=args.start,
        end_sec=args.end,
        every_sec=args.every,
        middle=args.middle,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        trigger=args.trigger,
        use_ocr=not args.no_ocr,
        visual_threshold=args.visual_threshold,
        min_event_gap_sec=args.min_event_gap,
        dealer_refresh_frames=args.dealer_refresh_frames
        if args.dealer_refresh_frames is not None
        else (30 if args.trigger == "frame" else 1),
        save_frames=args.save_frames,
        save_annotated=args.save_annotated,
        with_advice=args.with_advice,
        advice_iterations=args.advice_iterations,
        effective_stack_bb=args.effective_stack,
        villain_profile=args.villain,
        ocr_scale=args.ocr_scale,
        ocr_action_only=args.ocr_action_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_realtime_summary(payload))
    return 0


def command_dashboard_video(args: argparse.Namespace) -> int:
    payload = render_dashboard_video(
        video_path=args.video,
        states_jsonl=args.states_jsonl,
        ocr_events_jsonl=args.ocr_events,
        output_path=args.output,
        start_sec=args.start,
        end_sec=args.end,
        max_frames=args.max_frames,
        output_fps=args.output_fps,
        ocr_hold_sec=args.ocr_hold_sec,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_dashboard_summary(payload))
    return 0


def command_ocr_events(args: argparse.Namespace) -> int:
    payload = build_ocr_events_from_states(
        video_path=args.video,
        states_jsonl=args.states_jsonl,
        output_dir=args.output_dir,
        template_path=args.template,
        start_sec=args.start,
        end_sec=args.end,
        visual_threshold=args.visual_threshold,
        visual_min_gap_sec=args.visual_min_gap,
        heartbeat_sec=args.heartbeat,
        semantic_min_gap_sec=args.semantic_min_gap,
        include_hero_semantic=args.include_hero_semantic,
        dealer_refresh_events=args.dealer_refresh_events,
        max_events=args.max_events,
        resume=args.resume,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_ocr_event_summary(payload))
    return 0


def command_screen_cv(args: argparse.Namespace) -> int:
    apply_card_knn_model_env(args.card_knn_model)
    apply_deep_card_model_env(
        args.deep_card_model_dir,
        rank_model_dir=args.deep_rank_card_model_dir,
        suit_model_dir=args.deep_suit_card_model_dir,
    )
    live_bbox_file = canonical_live_bbox_file(args.bbox_file)
    bbox_text = resolve_bbox_text(args.bbox, bbox_file=live_bbox_file, latest_bbox=args.latest_bbox)
    outer_bbox_text = load_outer_bbox_text(live_bbox_file) if live_bbox_file is not None else None
    reused_reviewed_inner_bbox = None
    can_reuse_reviewed_inner_bbox = not (
        args.pick_bbox or args.pick_hero_cards or args.review_auto_bbox
    )
    if live_bbox_file is not None and can_reuse_reviewed_inner_bbox:
        reused_reviewed_inner_bbox = load_rebased_analysis_bbox_text(live_bbox_file)
        if reused_reviewed_inner_bbox is not None:
            outer_bbox_text = bbox_text
            bbox_text = reused_reviewed_inner_bbox
    payload = analyze_screen_stream(
        output_dir=args.output_dir,
        bbox=parse_bbox(bbox_text),
        outer_bbox=parse_bbox(outer_bbox_text),
        monitor=args.monitor,
        template_path=args.template,
        seat_count=args.seats,
        duration_sec=args.duration,
        every_sec=args.every,
        trigger=args.trigger,
        visual_threshold=args.visual_threshold,
        min_event_gap_sec=args.min_event_gap,
        min_confidence=args.min_confidence,
        use_ocr=not args.no_ocr,
        with_advice=args.with_advice,
        advice_iterations=args.advice_iterations,
        effective_stack_bb=args.effective_stack,
        villain_profile=args.villain,
        save_frames=args.save_frames,
        save_annotated=args.save_annotated,
        save_problem_frames=not args.no_problem_frames,
        problem_frame_limit=args.problem_frame_limit,
        snapshot_only=args.snapshot_only,
        pick_bbox=args.pick_bbox,
        preflight_once=args.preflight_once,
        print_events=(
            args.format == "text"
            and not args.snapshot_only
            and not args.pick_bbox
            and not args.pick_hero_cards
            and not args.review_auto_bbox
        ),
        auto_bbox=args.auto_bbox and reused_reviewed_inner_bbox is None,
        auto_bbox_refresh_sec=args.auto_bbox_refresh if reused_reviewed_inner_bbox is None else 0.0,
        dealer_refresh_frames=args.dealer_refresh_frames,
        ocr_scale=args.ocr_scale,
        ocr_action_only=args.ocr_action_only,
        lock_layout=args.lock_layout,
        hero_name=args.hero_name,
        show_overlay=args.show_overlay,
        overlay_image_interval_sec=args.overlay_image_interval,
        pick_hero_cards=args.pick_hero_cards,
        hero_cards_file=args.hero_cards_file,
        review_auto_bbox=args.review_auto_bbox,
        record_card_samples=not args.no_card_samples,
        card_sample_interval_sec=args.card_sample_interval,
        card_sample_limit=args.card_sample_limit,
        state_audit_limit=args.state_audit_limit,
        console_mode=args.console_mode,
        console_heartbeat_sec=args.console_heartbeat,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_screen_summary(payload))
    return 0


def command_validate_cv(args: argparse.Namespace) -> int:
    apply_card_knn_model_env(args.card_knn_model)
    apply_deep_card_model_env(
        args.deep_card_model_dir,
        rank_model_dir=args.deep_rank_card_model_dir,
        suit_model_dir=args.deep_suit_card_model_dir,
    )
    if args.all:
        payload = validate_cv_videos(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            template_path=args.template,
            seat_count=args.seats,
            start_sec=args.start,
            end_sec=args.end,
            every_sec=args.every,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            auto_bbox_refresh_sec=args.auto_bbox_refresh,
            use_ocr=args.with_ocr,
            ocr_scale=args.ocr_scale,
            ocr_action_only=args.ocr_action_only,
            save_problem_frames=not args.no_problem_frames,
            lock_layout=not args.no_lock_layout,
            dealer_refresh_frames=args.dealer_refresh_frames,
        )
    else:
        video_path = find_latest_video(args.video_dir) if args.latest or args.video is None else args.video
        payload = validate_cv_video(
            video_path=video_path,
            output_dir=args.output_dir,
            template_path=args.template,
            seat_count=args.seats,
            start_sec=args.start,
            end_sec=args.end,
            every_sec=args.every,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            auto_bbox_refresh_sec=args.auto_bbox_refresh,
            use_ocr=args.with_ocr,
            ocr_scale=args.ocr_scale,
            ocr_action_only=args.ocr_action_only,
            save_problem_frames=not args.no_problem_frames,
            lock_layout=not args.no_lock_layout,
            dealer_refresh_frames=args.dealer_refresh_frames,
        )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    elif args.all:
        print(format_validation_suite_summary(payload))
    else:
        print(format_validation_summary(payload))
    return 0


def command_diagnose_auto_bbox(args: argparse.Namespace) -> int:
    if args.all:
        video_paths = None
    elif args.latest or not args.video:
        video_paths = [find_latest_video(args.video_dir)]
    else:
        video_paths = list(args.video or [])
    payload = diagnose_auto_bbox_videos(
        video_paths=video_paths,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        template_path=args.template,
        start_sec=args.start,
        end_sec=args.end,
        every_sec=args.every,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        variants=tuple(args.variants) if args.variants else None,
        save_problem_frames=not args.no_problem_frames,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_auto_bbox_suite_summary(payload))
    return 0


def command_train_card_classifier(args: argparse.Namespace) -> int:
    payload = train_card_classifier(
        template_dir=args.template_dir,
        dataset_dirs=list(args.dataset_dir or []),
        glyph_dirs=list(args.glyph_dir or []),
        model_path=args.model,
        seed_model_path=args.seed_model,
        seed_conflict_policy=args.seed_conflict_policy.replace("-", "_"),
        seed_guard=args.seed_guard,
        seed_guard_rank_score=args.seed_guard_rank_score,
        seed_guard_rank_margin=args.seed_guard_rank_margin,
        seed_guard_suit_score=args.seed_guard_suit_score,
        seed_guard_suit_margin=args.seed_guard_suit_margin,
        include_templates=not args.no_templates,
        augment=args.augment,
        external_augment=args.external_augment,
        glyph_augment=args.glyph_augment,
        max_external=args.max_external,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_classifier_summary(payload))
    return 0


def command_compact_card_classifier(args: argparse.Namespace) -> int:
    payload = compact_card_classifier(
        model_path=args.model,
        output_model=args.output_model,
        benchmark_rows_csvs=list(args.benchmark_rows_csv or []),
        top_per_sample=args.top_per_sample,
        min_per_label=args.min_per_label,
        max_per_label=args.max_per_label,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_compact_card_classifier_summary(payload))
    return 0


def command_export_card_glyphs(args: argparse.Namespace) -> int:
    apply_card_knn_model_env(args.card_knn_model)
    if args.all:
        video_paths = sorted(Path(args.video_dir).glob("*.mp4"))
    elif args.latest:
        video_paths = [find_latest_video(args.video_dir)]
    else:
        video_paths = list(args.video or [])
    if not video_paths:
        raise ValueError("no video paths provided; use --latest, --all, or pass video files")
    payload = export_card_glyphs(
        video_paths=video_paths,
        output_dir=args.output_dir,
        every_sec=args.every,
        max_frames=args.max_frames,
        lock_layout=not args.no_lock_layout,
        include_board=not args.no_board,
        min_rank_confidence=args.min_rank_confidence,
        min_suit_confidence=args.min_suit_confidence,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_glyph_export_summary(payload))
    return 0


def command_export_card_review(args: argparse.Namespace) -> int:
    apply_card_knn_model_env(args.card_knn_model)
    apply_deep_card_model_env(
        args.deep_card_model_dir,
        rank_model_dir=args.deep_rank_card_model_dir,
        suit_model_dir=args.deep_suit_card_model_dir,
    )
    if args.all:
        video_paths = sorted(Path(args.video_dir).glob("*.mp4"))
    elif args.latest:
        video_paths = [find_latest_video(args.video_dir)]
    else:
        video_paths = list(args.video or [])
    if not video_paths:
        raise ValueError("no video paths provided; use --latest, --all, or pass video files")
    payload = export_card_review(
        video_paths=video_paths,
        output_dir=args.output_dir,
        template_path=args.template,
        seat_count=args.seats,
        start_sec=args.start,
        end_sec=args.end,
        every_sec=args.every,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        auto_bbox_refresh_sec=args.auto_bbox_refresh,
        lock_layout=not args.no_lock_layout,
        only_suspicious=args.only_suspicious,
        max_sheet_rows=args.max_sheet_rows,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_review_summary(payload))
    return 0


def command_collect_card_debug_review(args: argparse.Namespace) -> int:
    payload = collect_card_debug_review(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        include_fallback=not args.no_fallback,
        max_rows=args.max_rows,
        max_sheet_rows=args.max_sheet_rows,
        prepare_label_queue=args.prepare_label_queue,
        queue_output_dir=args.queue_output_dir,
        queue_max_rows=args.queue_max_rows,
        copy_queue_assets=not args.no_copy_queue_assets,
        prepare_glyph_label_queue=args.prepare_glyph_label_queue,
        glyph_queue_output_dir=args.glyph_queue_output_dir,
        glyph_queue_max_rows=args.glyph_queue_max_rows,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_debug_review_summary(payload))
    return 0


def command_label_card_crops(args: argparse.Namespace) -> int:
    payload = label_card_crops(
        input_dirs=list(args.input_dir or []),
        output_dir=args.output_dir,
        teacher_model_dir=args.teacher_model_dir,
        teacher_rank_model_dir=args.teacher_rank_model_dir,
        teacher_suit_model_dir=args.teacher_suit_model_dir,
        kind=args.kind,
        max_images=args.max_images,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=args.require_current_agreement,
        copy_accepted=not args.no_copy_accepted,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_crop_label_summary(payload))
    return 0


def command_organize_card_crops(args: argparse.Namespace) -> int:
    payload = organize_card_crops(
        input_dirs=list(args.input_dir or []),
        output_dir=args.output_dir,
        kind=args.kind,
        max_images=args.max_images,
        review_csv=args.review_csv,
        allowed_review_reasons=list(args.allowed_review_reason or []) or None,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_organize_card_crops_summary(payload))
    return 0


def command_label_card_crops_hf(args: argparse.Namespace) -> int:
    payload = label_card_crops_hf(
        input_dirs=list(args.input_dir or []),
        output_dir=args.output_dir,
        kind=args.kind,
        rank_model=args.rank_model,
        suit_model=args.suit_model,
        max_images=args.max_images,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=args.require_current_agreement,
        copy_accepted=not args.no_copy_accepted,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_card_crop_label_summary(payload))
    return 0


def command_train_card_hf_probe(args: argparse.Namespace) -> int:
    payload = train_hf_card_probe(
        input_dirs=list(args.input_dir or []),
        output_dir=args.output_dir,
        kind=args.kind,
        model_name=args.model,
        rank_model=args.rank_model,
        suit_model=args.suit_model,
        template_dir=args.template_dir,
        include_templates=not args.no_templates,
        max_images_per_class=args.max_images_per_class,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_probe_train_summary(payload))
    return 0


def command_label_card_crops_hf_probe(args: argparse.Namespace) -> int:
    payload = label_card_crops_hf_probe(
        input_dirs=list(args.input_dir or []),
        output_dir=args.output_dir,
        probe_dir=args.probe_dir,
        kind=args.kind,
        max_images=args.max_images,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=args.require_current_agreement,
        copy_accepted=not args.no_copy_accepted,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_probe_label_summary(payload))
    return 0


def command_filter_card_hf_predictions(args: argparse.Namespace) -> int:
    payload = filter_hf_probe_predictions(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        kind=args.kind,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=args.require_current_agreement,
        copy_accepted=not args.no_copy_accepted,
    )
    maybe_distill_runtime(args, payload)
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_probe_filter_summary(payload))
    return 0


def command_ensemble_card_hf_predictions(args: argparse.Namespace) -> int:
    payload = ensemble_hf_probe_predictions(
        predictions_csvs=list(args.predictions_csv or []),
        output_dir=args.output_dir,
        kind=args.kind,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=not args.no_require_current_agreement,
        min_teachers=args.min_teachers,
        copy_accepted=not args.no_copy_accepted,
    )
    maybe_distill_runtime(args, payload)
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_probe_ensemble_summary(payload))
    return 0


def maybe_distill_runtime(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if not getattr(args, "distill_runtime", False):
        return
    output_dir = Path(args.output_dir)
    distill = distill_big_teacher_runtime(
        teacher_label_summary=payload,
        output_dir=args.runtime_output_dir or (output_dir / "runtime_candidate"),
        model_path=args.runtime_model,
        candidate_name=args.runtime_candidate_name or f"{output_dir.name}_runtime",
        base_glyph_dirs=list(args.runtime_base_glyph_dir or []),
        dataset_dirs=list(args.runtime_dataset_dir or []),
        video_dir=args.runtime_video_dir,
        video_paths=list(args.runtime_video or []) if args.runtime_video else None,
        benchmark_review_csvs=list(args.runtime_benchmark_review_csv or []) or None,
        baseline_review_csv=args.runtime_baseline_review_csv,
        baseline_validation_summary_json=args.runtime_baseline_validation_summary_json,
        deep_card_model_dir=args.runtime_deep_card_model_dir,
        seed_model_path=None if args.no_runtime_seed_model else args.runtime_seed_model,
        seed_conflict_policy=args.runtime_seed_conflict_policy.replace("-", "_"),
        seed_guard=getattr(args, "runtime_seed_guard", False),
        seed_guard_rank_score=getattr(args, "runtime_seed_guard_rank_score", 0.55),
        seed_guard_rank_margin=getattr(args, "runtime_seed_guard_rank_margin", 0.10),
        seed_guard_suit_score=getattr(args, "runtime_seed_guard_suit_score", 0.70),
        seed_guard_suit_margin=getattr(args, "runtime_seed_guard_suit_margin", 0.04),
        every_sec=args.runtime_every,
        max_frames=args.runtime_max_frames,
        min_confidence=args.runtime_min_confidence,
        augment=args.runtime_augment,
        external_augment=args.runtime_external_augment,
        glyph_augment=args.runtime_glyph_augment,
        max_external=args.runtime_max_external,
        min_accepted=args.runtime_min_accepted,
        max_benchmark_samples=args.runtime_max_benchmark_samples,
        max_diff_rows=args.runtime_max_diff_rows,
        max_risk=args.runtime_max_risk,
        max_real_problem=args.runtime_max_real_problem,
        max_board_bad=args.runtime_max_board_bad,
        max_median_ms=args.runtime_max_median_ms,
        max_p90_ms=args.runtime_max_p90_ms,
        max_median_regression_ms=args.runtime_max_median_regression_ms,
        max_p90_regression_ms=args.runtime_max_p90_regression_ms,
        prepare_risk_queue=not args.no_runtime_risk_queue,
        risk_queue_max_rows=args.runtime_risk_queue_max_rows,
    )
    payload["distill_runtime"] = distill
    payload.setdefault("files", {})
    payload["files"]["runtime_summary"] = (distill.get("files") or {}).get("summary", "")
    payload["files"]["runtime_runbook"] = (distill.get("files") or {}).get("runbook", "")
    payload["files"]["runtime_gate_report"] = (distill.get("files") or {}).get("gate_report", "")
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def command_sweep_card_hf_thresholds(args: argparse.Namespace) -> int:
    payload = sweep_hf_prediction_thresholds(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        rank_score_thresholds=list(args.rank_score_threshold or []) or None,
        rank_margin_thresholds=list(args.rank_margin_threshold or []) or None,
        suit_score_thresholds=list(args.suit_score_threshold or []) or None,
        suit_margin_thresholds=list(args.suit_margin_threshold or []) or None,
        require_current_agreement=not args.no_require_current_agreement,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_threshold_sweep_summary(payload))
    return 0


def command_card_big_teacher(args: argparse.Namespace) -> int:
    if args.all:
        video_paths = sorted(Path(args.video_dir).glob("*.mp4"))
    elif args.latest:
        video_paths = [find_latest_video(args.video_dir)]
    else:
        video_paths = list(args.video or [])
    payload = run_card_big_teacher(
        video_paths=video_paths,
        input_dirs=list(args.input_dir or []),
        trusted_dirs=list(args.trusted_dir or []),
        output_dir=args.output_dir,
        probe_dir=args.probe_dir,
        kind=args.kind,
        model_name=args.model,
        rank_model=args.rank_model,
        suit_model=args.suit_model,
        template_dir=args.template_dir,
        include_templates=not args.no_templates,
        every_sec=args.every,
        max_frames=args.max_frames,
        lock_layout=not args.no_lock_layout,
        include_board=not args.no_board,
        max_images=args.max_images,
        max_images_per_class=args.max_images_per_class,
        rank_score_threshold=args.rank_score_threshold,
        rank_margin_threshold=args.rank_margin_threshold,
        suit_score_threshold=args.suit_score_threshold,
        suit_margin_threshold=args.suit_margin_threshold,
        require_current_agreement=args.require_current_agreement,
        copy_accepted=not args.no_copy_accepted,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
        local_files_only=args.local_files_only,
        distill_runtime=args.distill_runtime,
        runtime_output_dir=args.runtime_output_dir,
        runtime_model_path=args.runtime_model,
        runtime_candidate_name=args.runtime_candidate_name,
        runtime_base_glyph_dirs=list(args.runtime_base_glyph_dir or []),
        runtime_dataset_dirs=list(args.runtime_dataset_dir or []),
        runtime_video_dir=args.runtime_video_dir,
        runtime_video_paths=list(args.runtime_video or []) if args.runtime_video else None,
        runtime_benchmark_review_csvs=list(args.runtime_benchmark_review_csv or []) or None,
        runtime_baseline_review_csv=args.runtime_baseline_review_csv,
        runtime_baseline_validation_summary_json=args.runtime_baseline_validation_summary_json,
        runtime_deep_card_model_dir=args.runtime_deep_card_model_dir,
        runtime_seed_model_path=None if args.no_runtime_seed_model else args.runtime_seed_model,
        runtime_seed_conflict_policy=args.runtime_seed_conflict_policy.replace("-", "_"),
        runtime_seed_guard=args.runtime_seed_guard,
        runtime_seed_guard_rank_score=args.runtime_seed_guard_rank_score,
        runtime_seed_guard_rank_margin=args.runtime_seed_guard_rank_margin,
        runtime_seed_guard_suit_score=args.runtime_seed_guard_suit_score,
        runtime_seed_guard_suit_margin=args.runtime_seed_guard_suit_margin,
        runtime_every_sec=args.runtime_every,
        runtime_max_frames=args.runtime_max_frames,
        runtime_min_confidence=args.runtime_min_confidence,
        runtime_augment=args.runtime_augment,
        runtime_external_augment=args.runtime_external_augment,
        runtime_glyph_augment=args.runtime_glyph_augment,
        runtime_max_external=args.runtime_max_external,
        runtime_min_accepted=args.runtime_min_accepted,
        runtime_max_benchmark_samples=args.runtime_max_benchmark_samples,
        runtime_max_diff_rows=args.runtime_max_diff_rows,
        runtime_max_risk=args.runtime_max_risk,
        runtime_max_real_problem=args.runtime_max_real_problem,
        runtime_max_board_bad=args.runtime_max_board_bad,
        runtime_max_median_ms=args.runtime_max_median_ms,
        runtime_max_p90_ms=args.runtime_max_p90_ms,
        runtime_max_median_regression_ms=args.runtime_max_median_regression_ms,
        runtime_max_p90_regression_ms=args.runtime_max_p90_regression_ms,
        runtime_prepare_risk_queue=not args.no_runtime_risk_queue,
        runtime_risk_queue_max_rows=args.runtime_risk_queue_max_rows,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_big_teacher_summary(payload))
    return 0


def command_apply_card_hf_probe_review(args: argparse.Namespace) -> int:
    payload = apply_hf_probe_to_review(
        review_csv=args.review_csv,
        output_dir=args.output_dir,
        probe_dir=args.probe_dir,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hf_probe_review_summary(payload))
    return 0


def command_generate_card_synthetic(args: argparse.Namespace) -> int:
    if args.rank_only and args.suit_only:
        raise ValueError("--rank-only and --suit-only cannot be used together")
    payload = generate_synthetic_card_glyphs(
        output_dir=args.output_dir,
        per_class=args.per_class,
        seed=args.seed,
        include_rank=not args.suit_only,
        include_suit=not args.rank_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_synthetic_summary(payload))
    return 0


def command_download_card_dataset(args: argparse.Namespace) -> int:
    payload = download_card_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        repo_type=args.repo_type,
        allow_patterns=list(args.allow_pattern or []),
        refresh=args.refresh,
        local_files_only=args.local_files_only,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_dataset_download_summary(payload))
    return 0


def command_card_cv_pipeline(args: argparse.Namespace) -> int:
    bbox_text = resolve_bbox_text(args.bbox, bbox_file=args.bbox_file, latest_bbox=args.latest_bbox)
    payload = inspect_card_cv_pipeline(
        output_dir=args.output_dir,
        bbox=bbox_text,
        hero_name=args.hero_name,
        video_dir=args.video_dir,
        crop_dirs=list(args.crop_dir or []) or [DEFAULT_PIPELINE_CROP_DIR],
        probe_dir=args.probe_dir,
        probe_model=args.probe_model,
        probe_rank_model=args.probe_rank_model,
        probe_suit_model=args.probe_suit_model,
        probe_max_images_per_class=args.probe_max_images_per_class,
        probe_batch_size=args.probe_batch_size,
        run_train_probe=args.run_train_probe,
        dataset_repo_id=args.dataset_repo_id,
        dataset_repo_type=args.dataset_repo_type,
        dataset_allow_patterns=list(args.dataset_allow_pattern or []),
        dataset_dir=args.dataset_dir,
        extra_dataset_dirs=list(args.extra_dataset_dir or []),
        ingested_dataset_dir=args.ingested_dataset_dir,
        knn_model_path=args.knn_model,
        deep_model_dir=args.deep_card_model_dir,
        validation_summary_json=args.validation_summary_json,
        gate_summary_json=args.gate_summary_json,
        download_dataset_flag=args.download_dataset,
        refresh_dataset=args.refresh_dataset,
        local_files_only=args.local_files_only,
        ingest_dataset_flag=args.ingest_dataset,
        max_external_ingest=args.max_external_ingest,
        run_smoke=args.run_smoke,
        smoke_max_images=args.smoke_max_images,
        smoke_output_dir=args.smoke_output_dir,
        smoke_local_files_only=not args.smoke_online,
        smoke_batch_size=args.smoke_batch_size,
        run_teacher=args.run_teacher,
        teacher_output_dir=args.teacher_output_dir,
        teacher_max_images=args.teacher_max_images,
        teacher_local_files_only=not args.teacher_online,
        teacher_batch_size=args.teacher_batch_size,
        teacher_distill_runtime=args.teacher_distill_runtime,
        teacher_runtime_video_paths=list(args.teacher_runtime_video or []) if args.teacher_runtime_video else None,
        teacher_runtime_every_sec=args.teacher_runtime_every,
        teacher_runtime_max_frames=args.teacher_runtime_max_frames,
        teacher_runtime_max_benchmark_samples=args.teacher_runtime_max_benchmark_samples,
        teacher_runtime_max_diff_rows=args.teacher_runtime_max_diff_rows,
        summarize_candidates=not args.no_candidate_summary,
        candidate_search_dir=args.candidate_search_dir,
        candidate_output_dir=args.candidate_output_dir,
        keep_candidate_duplicates=args.keep_candidate_duplicates,
        audit_crop_images=not args.no_crop_image_audit,
        min_rank_per_label=args.min_rank_per_label,
        min_suit_per_label=args.min_suit_per_label,
        run_auto_bbox_diagnostics=args.run_auto_bbox_diagnostics,
        auto_bbox_output_dir=args.auto_bbox_output_dir,
        auto_bbox_every_sec=args.auto_bbox_every,
        auto_bbox_max_frames=args.auto_bbox_max_frames,
        auto_bbox_variants=list(args.auto_bbox_variant or []) or None,
        auto_bbox_save_problem_frames=not args.no_auto_bbox_problem_frames,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_cv_pipeline_summary(payload))
    return 0


def command_audit_card_review(args: argparse.Namespace) -> int:
    payload = audit_card_review(
        review_csv=args.review_csv,
        output_dir=args.output_dir,
        teacher_model_dir=args.teacher_model_dir,
        teacher_rank_model_dir=args.teacher_rank_model_dir,
        teacher_suit_model_dir=args.teacher_suit_model_dir,
        realtime_model_dir=args.realtime_model_dir,
        realtime_rank_model_dir=args.realtime_rank_model_dir,
        realtime_suit_model_dir=args.realtime_suit_model_dir,
        rank_confidence_threshold=args.rank_confidence_threshold,
        suit_confidence_threshold=args.suit_confidence_threshold,
        open_suit_score_threshold=args.open_suit_score_threshold,
        open_suit_margin_threshold=args.open_suit_margin_threshold,
        max_review=args.max_review,
        copy_review_assets=not args.no_copy_review_assets,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_hand_audit_summary(payload))
    return 0


def command_benchmark_card_review(args: argparse.Namespace) -> int:
    apply_deep_card_model_env(
        args.deep_card_model_dir,
        rank_model_dir=args.deep_rank_card_model_dir,
        suit_model_dir=args.deep_suit_card_model_dir,
    )
    payload = benchmark_card_review(
        review_csvs=list(args.review_csv or []),
        output_dir=args.output_dir,
        deep_model_dir=args.deep_card_model_dir,
        deep_rank_model_dir=args.deep_rank_card_model_dir,
        deep_suit_model_dir=args.deep_suit_card_model_dir,
        knn_model_path=args.knn_model,
        hf_probe_dir=args.hf_probe_dir,
        hf_probe_device=args.hf_probe_device,
        hf_probe_local_files_only=args.hf_probe_local_files_only,
        include_ok_pseudo=args.include_ok_pseudo,
        allowed_pseudo_reasons=tuple(args.allowed_pseudo_reason or ("ok",)),
        run_runtime=not args.no_runtime,
        max_samples=args.max_samples,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_benchmark_summary(payload))
    return 0


def command_diff_card_review(args: argparse.Namespace) -> int:
    payload = diff_card_review(
        baseline_csv=args.baseline_review_csv,
        candidate_csv=args.candidate_review_csv,
        output_dir=args.output_dir,
        risky_baseline_reasons=tuple(args.risky_baseline_reason or ("ok",)),
        max_rows=args.max_rows,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_review_diff_summary(payload))
    risk_count = int(((payload.get("counts") or {}).get("risk_count")) or 0)
    if args.fail_on_risk and risk_count > int(args.max_risk or 0):
        return 2
    return 0


def command_summarize_card_diff_risks(args: argparse.Namespace) -> int:
    payload = summarize_card_diff_risks(
        diff_csv=args.diff_csv,
        output_dir=args.output_dir,
        risk_only=not args.include_safe,
        include_same=not args.no_include_same,
        max_examples=args.max_examples,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_diff_risk_summary(payload))
    return 0


def command_gate_card_model(args: argparse.Namespace) -> int:
    apply_deep_card_model_env(
        args.deep_card_model_dir,
        rank_model_dir=args.deep_rank_card_model_dir,
        suit_model_dir=args.deep_suit_card_model_dir,
    )
    payload = gate_card_model(
        benchmark_review_csvs=list(args.benchmark_review_csv or []),
        baseline_review_csv=args.baseline_review_csv,
        candidate_review_csv=args.candidate_review_csv,
        output_dir=args.output_dir,
        candidate_name=args.candidate_name,
        candidate_evaluator=args.candidate_evaluator,
        knn_model_path=args.knn_model,
        deep_model_dir=args.deep_card_model_dir,
        deep_rank_model_dir=args.deep_rank_card_model_dir,
        deep_suit_model_dir=args.deep_suit_card_model_dir,
        hf_probe_dir=args.hf_probe_dir,
        hf_probe_device=args.hf_probe_device,
        hf_probe_local_files_only=args.hf_probe_local_files_only,
        candidate_validation_summary_json=args.candidate_validation_summary_json,
        baseline_validation_summary_json=args.baseline_validation_summary_json,
        include_ok_pseudo=args.include_ok_pseudo,
        allowed_pseudo_reasons=tuple(args.allowed_pseudo_reason or ("ok",)),
        run_runtime=not args.no_runtime,
        max_benchmark_samples=args.max_benchmark_samples,
        max_diff_rows=args.max_diff_rows,
        max_risk=args.max_risk,
        require_validation=args.require_validation,
        max_real_problem=args.max_real_problem,
        max_board_bad=args.max_board_bad,
        max_median_ms=args.max_median_ms,
        max_p90_ms=args.max_p90_ms,
        max_median_regression_ms=args.max_median_regression_ms,
        max_p90_regression_ms=args.max_p90_regression_ms,
        min_candidate_card_acc=args.min_card_acc,
        min_candidate_rank_acc=args.min_rank_acc,
        min_candidate_suit_acc=args.min_suit_acc,
        require_no_missing_rows=not args.allow_missing_rows,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_model_gate_summary(payload))
    if args.fail_on_reject and not payload.get("promote"):
        return 2
    return 0


def command_summarize_card_candidates(args: argparse.Namespace) -> int:
    payload = summarize_card_candidates(
        gate_paths=list(args.gate_summary_json or []),
        search_dir=args.search_dir,
        output_dir=args.output_dir,
        keep_duplicates=args.keep_duplicates,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_candidate_summary(payload))
    return 0


def command_cv_health(args: argparse.Namespace) -> int:
    bbox_text = resolve_bbox_text(args.bbox, bbox_file=args.bbox_file, latest_bbox=args.latest_bbox)
    payload = check_cv_health(
        output_dir=args.output_dir,
        knn_model_path=args.knn_model,
        deep_model_dir=args.deep_card_model_dir,
        validation_summary_json=args.validation_summary_json,
        gate_summary_json=args.gate_summary_json,
        bbox=bbox_text,
        bbox_file=args.bbox_file,
        allow_placeholder_bbox=args.allow_placeholder_bbox,
        screen_output_dir=args.screen_output_dir,
        fast_screen_output_dir=args.fast_screen_output_dir,
        preflight_output_dir=args.preflight_output_dir,
        hero_name=args.hero_name,
        effective_stack=args.effective_stack,
        villain=args.villain,
        min_confidence=args.min_confidence,
        ocr_scale=args.ocr_scale,
        dealer_refresh_frames=args.dealer_refresh_frames,
        auto_bbox_refresh=args.auto_bbox_refresh,
        max_real_problem=args.max_real_problem,
        max_board_bad=args.max_board_bad,
        max_median_ms=args.max_median_ms,
        max_p90_ms=args.max_p90_ms,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_cv_health_summary(payload))
    if args.fail_on_not_ready and not payload.get("ready"):
        return 2
    return 0


def command_prepare_card_label_queue(args: argparse.Namespace) -> int:
    payload = prepare_card_label_queue(
        review_csvs=list(args.review_csv or []),
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        include_ok=args.include_ok,
        include_completed=args.include_completed,
        copy_assets=not args.no_copy_assets,
        render_contact_sheet=not args.no_contact_sheet,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_label_queue_summary(payload))
    return 0


def command_prepare_card_diff_label_queue(args: argparse.Namespace) -> int:
    payload = prepare_card_diff_label_queue(
        diff_csv=args.diff_csv,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        risk_only=not args.include_non_risk,
        include_same=args.include_same,
        prefer_candidate_assets=not args.prefer_baseline_assets,
        copy_assets=not args.no_copy_assets,
        render_contact_sheet=not args.no_contact_sheet,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_label_queue_summary(payload))
    return 0


def command_prepare_card_glyph_label_queue(args: argparse.Namespace) -> int:
    payload = prepare_card_glyph_label_queue(
        predictions_csvs=list(args.predictions_csv or []),
        review_csvs=list(args.review_csv or []),
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        allowed_reasons=list(args.allowed_reason or []),
        include_accepted=args.include_accepted,
        prefill_final_label=args.prefill_final_label,
        copy_assets=not args.no_copy_assets,
        render_contact_sheet=not args.no_contact_sheet,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_glyph_label_queue_summary(payload))
    return 0


def command_apply_card_glyph_label_queue(args: argparse.Namespace) -> int:
    payload = apply_card_glyph_label_queue(
        queue_csv=args.queue_csv,
        output_dir=args.output_dir,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_glyph_label_apply_summary(payload))
    return 0


def command_audit_card_label_queue(args: argparse.Namespace) -> int:
    payload = audit_card_label_queue(
        queue_csv=args.queue_csv,
        output_dir=args.output_dir,
        applied_output_dir=args.applied_output_dir,
        render_contact_sheet=not args.no_contact_sheet,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_label_queue_audit_summary(payload))
    return 0


def command_retrain_card_label_queue(args: argparse.Namespace) -> int:
    payload = retrain_card_label_queue(
        queue_csv=args.queue_csv,
        output_dir=args.output_dir,
        base_glyph_dirs=list(args.base_glyph_dir or []),
        video_dir=args.video_dir,
        video_paths=list(args.video or []) or None,
        benchmark_review_csvs=list(args.benchmark_review_csv or []) or None,
        baseline_review_csv=args.baseline_review_csv,
        baseline_validation_summary_json=args.baseline_validation_summary_json,
        deep_card_model_dir=args.deep_card_model_dir,
        candidate_name=args.candidate_name,
        model_path=args.model,
        seed_model_path=None if args.no_seed_model else args.seed_model,
        seed_conflict_policy=args.seed_conflict_policy.replace("-", "_"),
        every_sec=args.every,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        augment=args.augment,
        glyph_augment=args.glyph_augment,
        include_templates=not args.no_templates,
        allow_partial=args.allow_partial,
        max_benchmark_samples=args.max_benchmark_samples,
        max_diff_rows=args.max_diff_rows,
        max_risk=args.max_risk,
        max_real_problem=args.max_real_problem,
        max_board_bad=args.max_board_bad,
        max_median_ms=args.max_median_ms,
        max_p90_ms=args.max_p90_ms,
        max_median_regression_ms=args.max_median_regression_ms,
        max_p90_regression_ms=args.max_p90_regression_ms,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_label_retrain_summary(payload))
    if args.fail_on_reject and not payload.get("promote"):
        return 2
    return 0


def command_serve_card_label_queue(args: argparse.Namespace) -> int:
    payload = serve_card_label_queue(
        queue_csv=args.queue_csv,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_label_server_summary(payload))
    return 0


def command_serve_card_glyph_label_queue(args: argparse.Namespace) -> int:
    payload = serve_card_glyph_label_queue(
        queue_csv=args.queue_csv,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_card_glyph_label_server_summary(payload))
    return 0


def command_replay_fixed_card_samples(args: argparse.Namespace) -> int:
    payload = replay_fixed_card_samples(
        samples_dir=args.samples_dir,
        layout_profile_path=args.layout_profile,
        output_dir=args.output_dir,
        old_queue_csv=args.old_queue_csv,
        sample_prefix=args.sample_prefix,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_fixed_replay_summary(payload))
    return 0


def command_train_deep_card_classifier(args: argparse.Namespace) -> int:
    kinds = ("rank", "suit") if args.kind == "both" else (args.kind,)
    results = []
    for kind in kinds:
        arch = args.arch
        image_size = args.image_size
        if kind == "rank":
            arch = args.rank_arch or arch
            image_size = args.rank_image_size or image_size
        elif kind == "suit":
            arch = args.suit_arch or arch
            image_size = args.suit_image_size or image_size
        results.append(
            train_deep_card_classifier(
                glyph_dir=args.glyph_dir,
                extra_glyph_dirs=list(args.extra_glyph_dir or []),
                model_dir=args.model_dir,
                kind=kind,
                template_dir=args.template_dir,
                include_templates=not args.no_templates,
                arch=arch,
                pretrained=args.pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                val_split=args.val_split,
                max_images_per_class=args.max_images_per_class,
                seed=args.seed,
                image_size=image_size,
                num_workers=args.num_workers,
                freeze_backbone=args.freeze_backbone,
                class_balanced_loss=args.class_balanced_loss,
                weighted_sampler=args.weighted_sampler,
            )
        )
    payload = {"ok": True, "model_dir": str(args.model_dir), "results": results}
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print("\n\n".join(format_deep_train_summary(result) for result in results))
    return 0


def command_ingest_card_images(args: argparse.Namespace) -> int:
    payload = ingest_external_card_images(
        dataset_dirs=list(args.dataset_dir or []),
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_external_ingest_summary(payload))
    return 0


def command_audit_card_glyphs(args: argparse.Namespace) -> int:
    payload = audit_card_glyphs(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        teacher_model_dir=args.teacher_model_dir,
        teacher_rank_model_dir=args.teacher_rank_model_dir,
        teacher_suit_model_dir=args.teacher_suit_model_dir,
        realtime_model_dir=args.realtime_model_dir,
        realtime_rank_model_dir=args.realtime_rank_model_dir,
        realtime_suit_model_dir=args.realtime_suit_model_dir,
        max_review=args.max_review,
        rank_confidence_threshold=args.rank_confidence_threshold,
        suit_confidence_threshold=args.suit_confidence_threshold,
        temporal_window_frames=args.temporal_window_frames,
        temporal_min_support=args.temporal_min_support,
        copy_accepted=not args.no_copy_accepted,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_glyph_audit_summary(payload))
    return 0


def command_apply_card_review(args: argparse.Namespace) -> int:
    payload = apply_card_review(
        review_csv=args.review_csv,
        output_dir=args.output_dir,
    )
    if args.format == "json":
        print_json(payload, pretty=not args.compact)
    else:
        print(format_apply_review_summary(payload))
    return 0


def apply_deep_card_model_env(
    model_dir: Path | None,
    *,
    rank_model_dir: Path | None = None,
    suit_model_dir: Path | None = None,
) -> None:
    if model_dir is not None:
        os.environ["GTO_CARD_DEEP_MODEL_DIR"] = str(model_dir)
    if rank_model_dir is not None:
        os.environ["GTO_CARD_DEEP_RANK_MODEL_DIR"] = str(rank_model_dir)
    if suit_model_dir is not None:
        os.environ["GTO_CARD_DEEP_SUIT_MODEL_DIR"] = str(suit_model_dir)
    if model_dir is not None or rank_model_dir is not None or suit_model_dir is not None:
        warm_deep_card_models(model_dir, rank_model_dir=rank_model_dir, suit_model_dir=suit_model_dir)


def apply_card_knn_model_env(model_path: Path | None) -> None:
    if model_path is not None:
        os.environ["GTO_CARD_KNN_MODEL"] = str(model_path)


def read_state(path: Path | None) -> dict[str, Any]:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("no state JSON provided")
    return json.loads(raw)


def print_json(value: dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def format_text(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"错误：{result.get('error')}"
    decision = result["decision"]
    metrics = result.get("metrics", {})
    context = result.get("preflop_context") or {}
    summary = result.get("input_summary") or {}
    if result.get("mode") == "preflop":
        raw_position = context.get("raw_position") or summary.get("raw_position") or "未知"
        solver_position = context.get("solver_position") or summary.get("position") or "未知"
        action_order = context.get("preflop_action_order")
        status_label = {
            "unopened": "前位无人加注",
            "facing_open": "前位有人 open/raise",
            "facing_3bet": "面对 3bet",
            "limped_pot": "前位有人 limp",
            "four_bet_or_more": "面对 4bet 或更高",
            "unknown": "前位行动未知",
            "unsupported_scenario": "场景参数无效",
        }.get(context.get("status"), str(context.get("status") or "未知"))
        lines = [
            f"翻前状态：{status_label}",
            f"位置：{raw_position} -> 策略桶 {solver_position}",
            f"行动顺序：{('第' + str(action_order) + '个') if action_order else '未知'}",
            f"手牌：{metrics.get('hand_code') or '-'}",
        ]
        actions = context.get("actions_before_hero") or []
        if actions:
            action_text = " -> ".join(
                f"{item.get('position') or '?'} {item.get('action')}"
                + (f" {item.get('amount_bb'):g}BB" if item.get("amount_bb") is not None else "")
                for item in actions
            )
            lines.append(f"已知前位行动：{action_text}")
        else:
            lines.append("已知前位行动：无")
        if decision.get("primary_action") == "wait":
            needs = ", ".join(context.get("needs") or ["行动历史"])
            lines.append(f"GTO：暂不建议下注/跟注/弃牌，缺少 {needs}")
            return "\n".join(lines)
        lines.append(
            f"建议：{format_preflop_instruction(decision, context.get('to_call_bb'))}"
        )
        lines.append(f"频率：{format_preflop_mix(decision.get('mix', {}))}")
        return "\n".join(lines)
    lines = [
        f"最推荐：{plain_action_label(decision.get('primary_action'))}",
        f"电脑建议：{format_mix(decision.get('mix', {}))}",
    ]
    if decision.get("recommended_size_bb"):
        lines.append(f"建议多出：{decision['recommended_size_bb']} 份")
    if "equity_pct" in metrics:
        lines.append(f"大概赢面：{metrics['equity_pct']}%")
    if "required_equity_pct" in metrics:
        lines.append(f"继续玩至少需要的赢面：{metrics['required_equity_pct']}%")
    lines.append("提示：网页练习会给更适合新手的讲解。")
    return "\n".join(lines)


def format_preflop_instruction(decision: dict[str, Any], to_call_bb: Any) -> str:
    action = str(decision.get("primary_action") or "wait").lower()
    size = decision.get("recommended_size_bb")
    if action in {"raise", "3bet", "4bet"}:
        verb = {"raise": "OPEN RAISE", "3bet": "3BET", "4bet": "4BET"}[action]
        return f"{verb} TO {size:g} BB" if size is not None else verb
    if action == "call":
        return f"CALL {float(to_call_bb):g} BB" if to_call_bb is not None else "CALL"
    if action == "limp":
        return "LIMP"
    if action == "fold":
        return "FOLD"
    if action == "check":
        return "CHECK"
    return "WAIT"


def format_preflop_mix(mix: dict[str, Any]) -> str:
    labels = {
        "raise": "OPEN RAISE",
        "3bet": "3BET",
        "4bet": "4BET",
        "call": "CALL",
        "fold": "FOLD",
        "limp": "LIMP",
    }
    return " / ".join(f"{labels.get(str(action), str(action).upper())} {value}%" for action, value in mix.items())


def format_spot_text(payload: dict[str, Any]) -> str:
    state = payload.get("state", payload)
    lines = [
        spot_title(state),
        "这是给人看的文本输出。机器接口请用默认 JSON。",
    ]
    if "answer" in payload:
        if "lesson" in payload:
            lines.append(format_lesson("提示", payload["lesson"]["before"]))
        lines.append(format_answer_brief(payload["answer"]))
        if "lesson" in payload:
            lines.append(format_lesson("学习点", payload["lesson"]["after"]))
    return "\n".join(lines)


def format_answer_brief(answer: dict[str, Any]) -> str:
    decision = answer["decision"]
    metrics = answer.get("metrics", {})
    lines = [
        f"答案：{plain_action_label(decision.get('primary_action'))} | 建议：{format_mix(decision.get('mix', {}))}",
    ]
    if decision.get("recommended_size_bb"):
        lines.append(f"建议多出：{decision['recommended_size_bb']} 份")
    if "equity_pct" in metrics:
        lines.append(f"大概赢面：{metrics['equity_pct']}%")
    if "required_equity_pct" in metrics and metrics["required_equity_pct"]:
        lines.append(f"继续玩至少需要的赢面：{metrics['required_equity_pct']}%")
    return "\n".join(lines)


def print_lesson(title: str, items: list[str]) -> None:
    print(format_lesson(title, items))


def format_lesson(title: str, items: list[str]) -> str:
    lines = [f"{title}："]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_mix(mix: dict[str, int]) -> str:
    return "，".join(f"{plain_action_label(action)} {value}%" for action, value in mix.items())


def rules_text() -> str:
    return "\n".join(
        [
            "德州扑克零基础速查",
            "",
            "1. 每个人先拿两张只有自己能看的手牌。",
            "2. 桌面最多发五张大家都能用的牌：先发三张，再发一张，最后再发一张。",
            "3. 最后用自己的两张手牌和公牌里任意五张，组成最大的五张牌。",
            "4. 一手牌会经历几个阶段：没发公共牌、发出三张公共牌、发出第四张、发出第五张。",
            "5. 常见动作：不玩了、跟上、先不出、先出一些、多出一些。",
            "6. 越晚行动越舒服，因为你能先看到别人怎么选。",
            "7. 桌上已有的钱越多，你越可能愿意继续；但如果要你补的钱太多，就要谨慎。",
            "8. 新手先记：手牌明显强就多出一些，手牌明显弱就不玩了。",
            "9. 发出公共牌后，如果你已经凑到好牌，或者很可能变成好牌，才更愿意继续放钱。",
            "",
            "开始练习：python gto.py ui",
        ]
    )


def sample_schema() -> dict[str, Any]:
    return {
        "hero": {
            "cards": ["As", "Ks"],
            "position": "BTN",
            "stack_bb": 100,
        },
        "table": {
            "pot_bb": 6.5,
            "to_call_bb": 0,
            "effective_stack_bb": 100,
            "board": [],
        },
        "action": {
            "scenario": "rfi",
            "street": "preflop",
        },
        "villain": {
            "profile": "standard",
        },
        "practice": {
            "level": "simple",
        },
        "seed": 7,
    }


if __name__ == "__main__":
    raise SystemExit(main())
