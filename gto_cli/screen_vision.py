from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .bbox_utils import parse_bbox_values
from .card_glyph_export import export_rank_glyph_image
from .cv_advisor import attach_gto_advice
from .cv_health import build_fast_live_command, build_health_command, build_live_command, build_preflight_command
from .live_vision import (
    build_error_state,
    build_realtime_state,
    event_source_timing_summary,
    format_realtime_summary,
    roi_signature,
    stabilize_hero_cards,
    state_signature,
)
from .preflop_tracker import PreflopActionTracker
from .screen_bbox_review import crop_inner_from_outer, review_bbox_interactively
from .screen_overlay import (
    LivePokerOverlay,
    apply_manual_hero_profile,
    load_manual_hero_profile,
    render_diagnostic_frame,
    render_full_window_diagnostic_frame,
    render_layout_region_preview,
    select_manual_hero_profile,
)
from .video_vision import (
    BOARD_CARD_ROIS,
    HERO_CARD_ROIS,
    analyze_video_frame,
    annotate_video_frame,
    build_layout_profile,
    choose_template,
    detect_action_controls,
    layout_profile_is_strong,
    layout_profile_quality,
    load_cv,
    load_ocr,
    normalized_suit_component,
    normalized_suit_component_by_label,
    run_ocr,
    run_ocr_in_roi,
    scale_roi,
)
from .vision import find_dealer_button, find_dealer_button_component


DEALER_BUTTON_ANCHORS_8 = (
    (0.565, 0.765),  # bottom_hero
    (0.255, 0.685),  # bottom_left
    (0.185, 0.515),  # left
    (0.305, 0.325),  # top_left
    (0.500, 0.285),  # top
    (0.750, 0.345),  # top_right
    (0.815, 0.515),  # right
    (0.705, 0.675),  # bottom_right
)

AUTO_BBOX_TEMPLATE_RECHECK_LIMIT = 2
AUTO_BBOX_SCORE_MAX_WIDTH = 1280
# The controls live in the lower-right part of the manually selected full
# client. OCR uses this crop, then restores box coordinates to the full client.
ACTION_OCR_ROI = (0.40, 0.82, 1.0, 1.0)


def active_python_gto_command() -> str:
    """Return a quoted CLI prefix pinned to the interpreter running CV."""

    # PowerShell needs an explicit call operator before a quoted executable.
    return f'& "{Path(sys.executable).resolve()}" gto.py'


def pin_command_to_active_python(command: str) -> str:
    return command.replace("python gto.py", active_python_gto_command())


def analyze_screen_stream(
    output_dir: Path,
    bbox: tuple[int, int, int, int] | None = None,
    outer_bbox: tuple[int, int, int, int] | None = None,
    monitor: int = 1,
    template_path: Path | None = None,
    seat_count: int = 8,
    duration_sec: float | None = None,
    every_sec: float = 1.0,
    trigger: str = "frame",
    visual_threshold: float = 2.4,
    min_event_gap_sec: float = 1.0,
    min_confidence: float = 0.45,
    use_ocr: bool = True,
    with_advice: bool = False,
    advice_iterations: int = 600,
    effective_stack_bb: float = 100.0,
    villain_profile: str = "standard",
    save_frames: bool = False,
    save_annotated: bool = False,
    save_problem_frames: bool = True,
    problem_frame_limit: int = 80,
    snapshot_only: bool = False,
    pick_bbox: bool = False,
    preflight_once: bool = False,
    print_events: bool = False,
    auto_bbox: bool = False,
    auto_bbox_refresh_sec: float = 0.0,
    dealer_refresh_frames: int = 4,
    ocr_scale: float = 1.0,
    ocr_action_only: bool = False,
    lock_layout: bool = False,
    hero_name: str | None = None,
    show_overlay: bool = False,
    overlay_image_interval_sec: float = 2.0,
    pick_hero_cards: bool = False,
    hero_cards_file: Path | None = None,
    review_auto_bbox: bool = False,
    record_card_samples: bool = True,
    card_sample_interval_sec: float = 30.0,
    card_sample_limit: int = 1000,
    state_audit_limit: int = 1000,
    console_mode: str = "advice",
    console_heartbeat_sec: float = 10.0,
) -> dict[str, Any]:
    cv2, np = load_cv()
    import mss

    started_at = time.perf_counter()
    card_sample_session_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    frames_dir = output_dir / "event_frames"
    annotated_dir = output_dir / "event_annotated"
    problem_dir = output_dir / "problem_frames"
    card_debug_dir = output_dir / "card_debug"
    card_samples_dir = output_dir / "card_samples"
    state_audit_dir = output_dir / "state_audit"
    card_samples_csv_path = card_samples_dir / "glyph_predictions.csv"
    card_sample_queue_dir = output_dir / "card_sample_label_queue"
    card_sample_prepare_command_path = output_dir / "run_prepare_card_sample_labels_command.txt"
    card_sample_serve_command_path = output_dir / "run_serve_card_sample_labels_command.txt"
    card_sample_apply_command_path = output_dir / "run_apply_card_sample_labels_command.txt"
    record_live_card_samples = bool(
        record_card_samples and not (snapshot_only or pick_bbox or pick_hero_cards or review_auto_bbox)
    )
    record_live_state_audits = not (snapshot_only or pick_bbox or pick_hero_cards or review_auto_bbox)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_frames or snapshot_only or pick_bbox or pick_hero_cards or review_auto_bbox or preflight_once:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    if save_problem_frames:
        problem_dir.mkdir(parents=True, exist_ok=True)
        card_debug_dir.mkdir(parents=True, exist_ok=True)
    if record_live_card_samples:
        card_samples_dir.mkdir(parents=True, exist_ok=True)
        prepare_command = (
            f'{active_python_gto_command()} prepare-card-glyph-label-queue --predictions-csv "{card_samples_csv_path}" '
            f'--output-dir "{card_sample_queue_dir}" --max-rows {min(5000, max(1, int(card_sample_limit)) * 14)} '
            f'--prefill-final-label none --format text'
        )
        serve_command = (
            f'{active_python_gto_command()} serve-card-glyph-label-queue '
            f'--queue-csv "{card_sample_queue_dir / "glyph_label_queue.csv"}" '
            f'--open-browser --format text'
        )
        apply_command = (
            f'{active_python_gto_command()} apply-card-glyph-label-queue '
            f'--queue-csv "{card_sample_queue_dir / "glyph_label_queue.csv"}" '
            f'--output-dir "{output_dir / "card_sample_labeled_dataset"}" --format text'
        )
        card_sample_prepare_command_path.write_text(prepare_command + "\n", encoding="utf-8-sig")
        card_sample_serve_command_path.write_text(serve_command + "\n", encoding="utf-8-sig")
        card_sample_apply_command_path.write_text(apply_command + "\n", encoding="utf-8-sig")
    if record_live_state_audits:
        state_audit_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "events.jsonl"
    events_path = output_dir / "events.json"
    current_path = output_dir / "current_state.json"
    summary_path = output_dir / "screen_summary.json"
    layout_profile_path = output_dir / "layout_profile.json"
    latest_overlay_path = output_dir / "latest_overlay.png"
    latest_overlay_full_window_path = output_dir / "latest_overlay_full_window.png"

    events: list[dict[str, Any]] = []
    previous_signature: str | None = None
    previous_visual: Any | None = None
    last_visual_event_sec = float("-inf")
    processed_frames = 0
    emitted_events = 0
    saved_problem_frames = 0
    saved_card_debug_samples = 0
    saved_card_samples = 0
    saved_state_audits = 0
    last_card_sample_signature: str | None = None
    last_card_sample_sec = float("-inf")
    last_state_audit_signature: str | None = None
    dealer_button_cache: dict[str, Any] | None = None
    last_dealer_refresh_frame = -10**9
    last_normal_action_buttons_visible = False
    dealer_cache_uses = 0
    card_roi_cache_signature: str | None = None
    card_roi_cache: dict[str, Any] | None = None
    card_cache_hits = 0
    card_cache_misses = 0
    hero_card_cache: dict[str, Any] | None = None
    preflop_tracker = PreflopActionTracker()
    last_console_signature: str | None = None
    last_console_emit_sec = float("-inf")
    printed_console_events = 0
    last_overlay_image_sec = float("-inf")

    with mss.mss() as sct:
        # Keep the full poker client for action controls. The reviewed inner
        # table remains the coordinate system for cards, seats, and the dealer.
        search_region = capture_region(sct, monitor=monitor, bbox=outer_bbox or bbox)
        monitor_region = capture_region(sct, monitor=monitor, bbox=None)
        region = capture_region(sct, monitor=monitor, bbox=bbox) if bbox is not None else dict(search_region)
        template_path = None
        template = None
        auto_bbox_info: dict[str, Any] | None = None
        first_outer_frame = grab_bgr(sct, search_region)
        first_frame = crop_inner_from_outer(first_outer_frame, search_region, region)

        if review_auto_bbox:
            template_path = choose_template(template_path)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                raise ValueError(f"cannot read dealer template: {template_path}")
            detection = detect_auto_bbox(cv2, np, first_frame, search_region, template, min_confidence)
            proposed_region = dict(detection["region"]) if detection is not None else None
            print(
                "Review the cyan automatic inner bbox: Enter/Space accepts it, R lets you redraw it, C/Esc cancels.",
                flush=True,
            )
            review = review_bbox_interactively(
                cv2,
                first_frame,
                search_region,
                proposed_region,
                selector=lambda: select_bbox(cv2, first_frame, search_region),
            )
            region = dict(review["region"])
            bbox_text = f'{region["left"]},{region["top"]},{region["width"]},{region["height"]}'
            analysis_bbox_path = output_dir / "analysis_bbox.json"
            review_preview_path = output_dir / "analysis_bbox_review.png"
            layout_preview_path = output_dir / "analysis_layout_preview.png"
            bbox_payload = {
                "left": region["left"],
                "top": region["top"],
                "width": region["width"],
                "height": region["height"],
                "text": bbox_text,
                "reviewed": True,
                "source": review.get("source"),
                "manual_adjustments": int(review.get("adjustments") or 0),
                "manual_outer_bbox_file": str(output_dir / "bbox.json"),
                "outer_reference": dict(search_region),
                "relative_to_outer": {
                    "x": round((region["left"] - search_region["left"]) / search_region["width"], 8),
                    "y": round((region["top"] - search_region["top"]) / search_region["height"], 8),
                    "width": round(region["width"] / search_region["width"], 8),
                    "height": round(region["height"] / search_region["height"], 8),
                },
                "auto_proposal": public_auto_bbox_candidate(detection) if detection is not None else None,
            }
            analysis_bbox_path.write_text(json.dumps(bbox_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            write_png(cv2, review_preview_path, review["preview"])
            inner_frame = crop_inner_from_outer(first_frame, search_region, region)
            layout_profile = build_layout_profile(inner_frame, [], hero_name=hero_name)
            write_png(cv2, layout_preview_path, render_layout_region_preview(cv2, inner_frame, layout_profile))
            commands = build_reviewed_bbox_commands(
                analysis_bbox_path=analysis_bbox_path,
                calibration_output_dir=output_dir,
                hero_name=hero_name,
                effective_stack_bb=effective_stack_bb,
                villain_profile=villain_profile,
                min_confidence=0.35 if abs(float(min_confidence) - 0.45) < 1e-9 else float(min_confidence),
                ocr_scale=ocr_scale,
                dealer_refresh_frames=dealer_refresh_frames,
            )
            command_files = write_reviewed_bbox_commands(output_dir, commands)
            summary = {
                "ok": True,
                "source": source_info(region, monitor),
                "outer_region": dict(search_region),
                "auto_bbox": public_auto_bbox_candidate(detection) if detection is not None else None,
                "analysis_bbox": bbox_payload,
                "analysis_bbox_text": bbox_text,
                "review_source": review.get("source"),
                "manual_adjustments": int(review.get("adjustments") or 0),
                "commands": commands,
                "files": {
                    "analysis_bbox": str(analysis_bbox_path),
                    "bbox_review": str(review_preview_path),
                    "layout_preview": str(layout_preview_path),
                    "summary": str(summary_path),
                    **command_files,
                },
                "hint": "The reviewed inner bbox is locked. Run the overlay command first; use the hero-card picker only if H1/H2 remain off.",
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        if auto_bbox:
            template_path = choose_template(template_path)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                raise ValueError(f"cannot read dealer template: {template_path}")
            detection = detect_auto_bbox(cv2, np, first_frame, search_region, template, min_confidence)
            if detection is None:
                auto_bbox_info = {
                    "method": "manual-full-client-fallback",
                    "region": dict(search_region),
                    "reason": "auto_bbox_not_found",
                }
            else:
                accepted, reject_reason = auto_bbox_preserves_hero_cards(detection["region"], search_region)
                if accepted:
                    region = detection["region"]
                    auto_bbox_info = detection
                    first_frame = grab_bgr(sct, region)
                else:
                    auto_bbox_info = {
                        "method": "manual-full-client-fallback",
                        "region": dict(search_region),
                        "rejected_method": detection.get("method"),
                        "rejected_region": dict(detection["region"]),
                        "reason": reject_reason,
                    }

        if snapshot_only:
            snapshot_path = frames_dir / "screen_snapshot.png"
            cv2.imwrite(str(snapshot_path), first_frame)
            summary = {
                "ok": True,
                "source": source_info(region, monitor),
                "output_dir": str(output_dir),
                "snapshot": str(snapshot_path),
                "auto_bbox": auto_bbox_info,
                "hint": "Use this screenshot to choose --bbox x,y,width,height around only the poker table.",
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        if pick_bbox:
            snapshot_path = frames_dir / "screen_bbox_source.png"
            cv2.imwrite(str(snapshot_path), first_frame)
            selected = select_bbox(cv2, first_frame, region)
            bbox_text = ",".join(str(value) for value in selected)
            bbox_path = output_dir / "bbox.json"
            command_path = output_dir / "run_live_command.txt"
            fast_command_path = output_dir / "run_fast_live_command.txt"
            overlay_command_path = output_dir / "run_overlay_diagnostic_command.txt"
            pick_hero_command_path = output_dir / "run_pick_hero_cards_command.txt"
            review_bbox_command_path = output_dir / "run_review_auto_bbox_command.txt"
            health_command_path = output_dir / "run_health_command.txt"
            preflight_command_path = output_dir / "run_preflight_command.txt"
            generated_min_confidence = 0.35 if abs(float(min_confidence) - 0.45) < 1e-9 else float(min_confidence)
            generated_auto_refresh = float(auto_bbox_refresh_sec) if float(auto_bbox_refresh_sec) > 0 else 10.0
            live_command = pin_command_to_active_python(build_live_command(
                bbox=bbox_text,
                bbox_file=bbox_path,
                screen_output_dir=Path("video_frames") / "screen_live",
                hero_name=hero_name,
                effective_stack=effective_stack_bb,
                villain=villain_profile,
                min_confidence=generated_min_confidence,
                ocr_scale=ocr_scale,
                dealer_refresh_frames=dealer_refresh_frames,
                auto_bbox_refresh=generated_auto_refresh,
                deep_model_dir=None,
            ))
            fast_live_command = pin_command_to_active_python(build_fast_live_command(
                bbox=bbox_text,
                bbox_file=bbox_path,
                screen_output_dir=Path("video_frames") / "screen_live_fast",
                hero_name=hero_name,
                effective_stack=effective_stack_bb,
                villain=villain_profile,
                min_confidence=generated_min_confidence,
                ocr_scale=ocr_scale,
                dealer_refresh_frames=dealer_refresh_frames,
                auto_bbox_refresh=generated_auto_refresh,
                deep_model_dir=None,
            ))
            health_command = pin_command_to_active_python(build_health_command(
                bbox=bbox_text,
                bbox_file=bbox_path,
                output_dir=Path("video_frames") / "cv_health_promoted",
                hero_name=hero_name,
                effective_stack=effective_stack_bb,
                villain=villain_profile,
                min_confidence=generated_min_confidence,
                ocr_scale=ocr_scale,
                dealer_refresh_frames=dealer_refresh_frames,
                auto_bbox_refresh=generated_auto_refresh,
                deep_model_dir=None,
            ))
            preflight_command = pin_command_to_active_python(build_preflight_command(
                bbox=bbox_text,
                bbox_file=bbox_path,
                preflight_output_dir=Path("video_frames") / "screen_preflight",
                hero_name=hero_name,
                effective_stack=effective_stack_bb,
                villain=villain_profile,
                min_confidence=generated_min_confidence,
                ocr_scale=ocr_scale,
                dealer_refresh_frames=dealer_refresh_frames,
                auto_bbox_refresh=generated_auto_refresh,
                deep_model_dir=None,
            ))
            overlay_command = live_command.replace(" --format text", " --show-overlay --format text")
            pick_hero_command = (
                f'{active_python_gto_command()} screen-cv --bbox-file "{bbox_path}" --pick-hero-cards '
                f'--output-dir "{output_dir}" --format text'
            )
            review_bbox_command = (
                f'{active_python_gto_command()} screen-cv --bbox-file "{bbox_path}" --review-auto-bbox '
                f'--output-dir "{output_dir}" --min-confidence {generated_min_confidence:g} '
                f'--ocr-scale {float(ocr_scale):g} --dealer-refresh-frames {int(dealer_refresh_frames)} '
                f'--effective-stack {float(effective_stack_bb):g} --villain "{villain_profile}"'
                f'{f" --hero-name \"{hero_name}\"" if hero_name else ""} --format text'
            )
            bbox_payload = {
                "left": selected[0],
                "top": selected[1],
                "width": selected[2],
                "height": selected[3],
                "text": bbox_text,
            }
            bbox_path.write_text(json.dumps(bbox_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            health_command_path.write_text(health_command + "\n", encoding="utf-8-sig")
            preflight_command_path.write_text(preflight_command + "\n", encoding="utf-8-sig")
            command_path.write_text(live_command + "\n", encoding="utf-8-sig")
            fast_command_path.write_text(fast_live_command + "\n", encoding="utf-8-sig")
            overlay_command_path.write_text(overlay_command + "\n", encoding="utf-8-sig")
            pick_hero_command_path.write_text(pick_hero_command + "\n", encoding="utf-8-sig")
            review_bbox_command_path.write_text(review_bbox_command + "\n", encoding="utf-8-sig")
            summary = {
                "ok": True,
                "source": source_info(region, monitor),
                "output_dir": str(output_dir),
                "snapshot": str(snapshot_path),
                "bbox": bbox_payload,
                "bbox_text": bbox_text,
                "health_command": health_command,
                "preflight_command": preflight_command,
                "command": live_command,
                "fast_command": fast_live_command,
                "overlay_command": overlay_command,
                "pick_hero_command": pick_hero_command,
                "review_bbox_command": review_bbox_command,
                "files": {
                    "bbox": str(bbox_path),
                    "health_command": str(health_command_path),
                    "preflight_command": str(preflight_command_path),
                    "command": str(command_path),
                    "fast_command": str(fast_command_path),
                    "overlay_command": str(overlay_command_path),
                    "pick_hero_command": str(pick_hero_command_path),
                    "review_bbox_command": str(review_bbox_command_path),
                    "snapshot": str(snapshot_path),
                    "summary": str(summary_path),
                },
                "hint": "Drag the poker table region, then press Enter or Space. Press C to cancel.",
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        if pick_hero_cards:
            print(
                "Select H1 (left card), then H2 (right card). "
                "For H2, start at its own rank corner and do not include the overlapping H1 area.",
                flush=True,
            )
            manual_profile = select_manual_hero_profile(cv2, first_frame, region, output_dir)
            manual_profile_path = Path((manual_profile.get("files") or {})["profile"])
            bbox_text = f'{region["left"]},{region["top"]},{region["width"]},{region["height"]}'
            bbox_arg = manual_hero_live_bbox_arg(output_dir, bbox_text)
            generated_min_confidence = 0.35 if abs(float(min_confidence) - 0.45) < 1e-9 else float(min_confidence)
            live_command = (
                f'{active_python_gto_command()} screen-cv {bbox_arg} '
                f'--hero-cards-file "{manual_profile_path}" --show-overlay --lock-layout '
                f'--output-dir "video_frames\\screen_live" --trigger frame --every {float(every_sec):g} '
                f'--with-advice --effective-stack {float(effective_stack_bb):g} '
                f'--villain "{villain_profile}" --min-confidence {generated_min_confidence:g} '
                f'--ocr-scale {float(ocr_scale):g} --dealer-refresh-frames {int(dealer_refresh_frames)} --format text'
            )
            command_path = output_dir / "run_live_overlay_command.txt"
            command_path.write_text(live_command + "\n", encoding="utf-8-sig")
            summary = {
                "ok": True,
                "source": source_info(region, monitor),
                "output_dir": str(output_dir),
                "hero_cards_file": str(manual_profile_path),
                "hero_card_boxes": manual_profile.get("hero_card_boxes"),
                "warnings": manual_profile.get("warnings") or [],
                "preview": (manual_profile.get("files") or {}).get("preview"),
                "command": live_command,
                "files": {
                    **(manual_profile.get("files") or {}),
                    "command": str(command_path),
                    "summary": str(summary_path),
                },
                "hint": "The generated command keeps the reviewed inner table frame, the manual full window, and the two manual hero-card ROIs.",
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        if template_path is None:
            template_path = choose_template(template_path)
        if template is None:
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                raise ValueError(f"cannot read dealer template: {template_path}")
        ocr = load_ocr() if use_ocr else None
        layout_profile: dict[str, Any] | None = None
        layout_locked = False
        manual_hero_profile: dict[str, Any] | None = None
        if hero_cards_file is not None:
            manual_hero_profile = load_manual_hero_profile(Path(hero_cards_file))
            base_profile = build_layout_profile(first_frame, [], hero_name=hero_name)
            layout_profile = apply_manual_hero_profile(base_profile, manual_hero_profile, first_frame.shape)
            layout_locked = True
            layout_profile_path.write_text(json.dumps(layout_profile, ensure_ascii=False, indent=2), encoding="utf-8")
        elif lock_layout:
            calibration_ocr = ocr or load_ocr()
            calibration_ocr_result = run_ocr(first_frame, calibration_ocr, scale=ocr_scale) if calibration_ocr is not None else []
            candidate_profile = build_layout_profile(first_frame, calibration_ocr_result, hero_name=hero_name)
            if layout_profile_is_strong(candidate_profile):
                layout_profile = candidate_profile
                layout_locked = True
                layout_profile_path.write_text(json.dumps(layout_profile, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                layout_profile = None

        overlay: LivePokerOverlay | None = None
        overlay_error: str | None = None
        if show_overlay:
            overlay, overlay_error = LivePokerOverlay.create(search_region)

        deadline = None if duration_sec is None else started_at + max(0.0, float(duration_sec))
        interrupted = False
        next_auto_refresh = (
            started_at + max(0.0, float(auto_bbox_refresh_sec)) if auto_bbox_refresh_sec > 0 and not layout_locked else None
        )
        if print_events:
            bbox_text = f'{region["left"]},{region["top"]},{region["width"]},{region["height"]}'
            print(f"Screen CV live started: bbox={bbox_text} current_state={current_path}", flush=True)
            if auto_bbox_info:
                print(
                    f"Auto bbox: method={auto_bbox_info.get('method')} score={auto_bbox_info.get('score')} "
                    f"dealer_conf={auto_bbox_info.get('dealer_confidence')}",
                    flush=True,
                )
            if layout_profile:
                anchor = layout_profile.get("hero_name_anchor") or {}
                print(
                    f"Layout locked: method={layout_profile.get('method')} "
                    f"hero_name={layout_profile.get('hero_name') or '-'} "
                    f"anchor={anchor.get('text') or '-'} profile={layout_profile_path}",
                    flush=True,
                )
            elif lock_layout:
                print("Layout lock pending: waiting for hero cards or hero-name anchor.", flush=True)
            if show_overlay:
                if overlay is not None:
                    print(
                        "CV overlay active: "
                        f"inner={latest_overlay_path} full_window={latest_overlay_full_window_path}",
                        flush=True,
                    )
                else:
                    print(f"CV overlay window unavailable: {overlay_error}; PNG diagnostics remain enabled.", flush=True)
            if record_live_card_samples:
                print(
                    f"Card sampling active: dir={card_samples_dir} labels={card_samples_csv_path}",
                    flush=True,
                )
            print("Press Ctrl+C to stop.", flush=True)
        try:
            with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
                while deadline is None or time.perf_counter() <= deadline:
                    loop_started = time.perf_counter()
                    timestamp = loop_started - started_at
                    frame_result_for_event: dict[str, Any] | None = None
                    diagnostic: Any | None = None
                    screen_timing: dict[str, float] = {}

                    if next_auto_refresh is not None and loop_started >= next_auto_refresh:
                        search_frame = grab_bgr(sct, search_region)
                        detection = detect_auto_bbox(cv2, np, search_frame, search_region, template, min_confidence)
                        if detection is not None and bbox_changed(region, detection["region"]):
                            accepted, reject_reason = should_accept_bbox_refresh(region, detection["region"], search_region, detection)
                            if accepted:
                                region = detection["region"]
                                auto_bbox_info = detection
                                dealer_button_cache = None
                                if not layout_locked:
                                    layout_profile = None
                                last_dealer_refresh_frame = -10**9
                                last_normal_action_buttons_visible = False
                                card_roi_cache_signature = None
                                card_roi_cache = None
                                previous_visual = None
                                previous_signature = None
                                if print_events:
                                    bbox_text = f'{region["left"]},{region["top"]},{region["width"]},{region["height"]}'
                                    print(
                                        f"Auto bbox refreshed: bbox={bbox_text} method={detection.get('method')} "
                                        f"score={detection.get('score')} dealer_conf={detection.get('dealer_confidence')}",
                                        flush=True,
                                    )
                            elif print_events:
                                candidate = detection["region"]
                                bbox_text = f'{candidate["left"]},{candidate["top"]},{candidate["width"]},{candidate["height"]}'
                                print(
                                    f"Auto bbox refresh ignored: bbox={bbox_text} reason={reject_reason} "
                                    f"method={detection.get('method')} score={detection.get('score')} "
                                    f"dealer_conf={detection.get('dealer_confidence')}",
                                    flush=True,
                                )
                        next_auto_refresh = loop_started + max(1.0, float(auto_bbox_refresh_sec))

                    capture_started = time.perf_counter()
                    outer_frame = grab_bgr(sct, search_region)
                    frame = crop_inner_from_outer(outer_frame, search_region, region)
                    screen_timing["capture_and_crop_ms"] = elapsed_ms(capture_started)
                    processed_frames += 1

                    visual_small = cv2.resize(frame, (160, 112), interpolation=cv2.INTER_AREA)
                    visual_gray = cv2.cvtColor(visual_small, cv2.COLOR_BGR2GRAY)
                    visual_diff = 0.0 if previous_visual is None else float(cv2.absdiff(visual_gray, previous_visual).mean())
                    previous_visual = visual_gray

                    if trigger == "visual-change":
                        enough_gap = timestamp - last_visual_event_sec >= min_event_gap_sec
                        visually_changed = previous_signature is None or (visual_diff >= visual_threshold and enough_gap)
                        if not visually_changed:
                            sleep_until_next(loop_started, every_sec)
                            continue
                        last_visual_event_sec = timestamp

                    table_visibility = None
                    try:
                        table_visible, table_visibility = poker_table_visibility(cv2, frame)
                        if not table_visible:
                            raise ValueError("poker table occluded")
                        if lock_layout and not layout_locked:
                            candidate_profile = build_layout_profile(frame, [], hero_name=hero_name)
                            if layout_profile_is_strong(candidate_profile):
                                layout_profile = candidate_profile
                                layout_locked = True
                                layout_profile_path.write_text(json.dumps(layout_profile, ensure_ascii=False, indent=2), encoding="utf-8")
                                next_auto_refresh = None
                                if print_events:
                                    anchor = layout_profile.get("hero_name_anchor") or {}
                                    print(
                                        f"Layout locked: method={layout_profile.get('method')} "
                                        f"quality={layout_profile_quality(layout_profile)} "
                                        f"hero_name={layout_profile.get('hero_name') or '-'} "
                                        f"anchor={anchor.get('text') or '-'} profile={layout_profile_path}",
                                        flush=True,
                                    )
                        active_layout_profile = layout_profile if layout_locked else None
                        frame_ocr = ocr
                        ocr_mode = "full" if ocr is not None else "disabled"
                        # 先只看红色操作面板。若已经出现正常操作按钮，庄家
                        # 位置绝不能继续沿用上一手的缓存，否则盲注和翻前顺序
                        # 会整体错位。
                        quick_action_controls = detect_action_controls(outer_frame, [])
                        normal_action_buttons_visible = len(quick_action_controls.get("red_button_regions") or []) >= 2
                        if ocr_action_only and ocr is not None:
                            if quick_action_controls.get("visible"):
                                frame_ocr = ocr
                                ocr_mode = "action_only_used"
                            else:
                                frame_ocr = None
                                ocr_mode = "action_only_skipped"
                        used_dealer_cache = False
                        refresh_dealer = should_refresh_dealer_button(
                            dealer_button_cache=dealer_button_cache,
                            dealer_refresh_frames=dealer_refresh_frames,
                            processed_frames=processed_frames,
                            last_dealer_refresh_frame=last_dealer_refresh_frame,
                            visual_diff=visual_diff,
                            visual_threshold=visual_threshold,
                            normal_action_buttons_visible=normal_action_buttons_visible,
                            previous_normal_action_buttons_visible=last_normal_action_buttons_visible,
                        )
                        last_normal_action_buttons_visible = normal_action_buttons_visible
                        cards_hint = None
                        if trigger == "frame":
                            signature_rois = card_signature_rois(layout_profile)
                            card_signature = roi_signature(frame, signature_rois)
                            if card_signature == card_roi_cache_signature:
                                cards_hint = card_roi_cache
                                card_cache_hits += 1
                            else:
                                card_roi_cache_signature = card_signature
                                card_cache_misses += 1
                        try:
                            frame_result = analyze_video_frame(
                                frame,
                                template,
                                seat_count=seat_count,
                                min_confidence=min_confidence,
                                ocr=frame_ocr,
                                dealer_button_hint=None if refresh_dealer else dealer_button_cache,
                                cards_hint=cards_hint,
                                ocr_scale=ocr_scale,
                                layout_profile=active_layout_profile,
                            )
                            frame_result_for_event = frame_result
                            if refresh_dealer:
                                dealer_button_cache = frame_result.get("dealer_button") or dealer_button_cache
                                last_dealer_refresh_frame = processed_frames
                            else:
                                used_dealer_cache = dealer_button_cache is not None
                        except Exception as dealer_error:
                            if dealer_button_cache is None or "dealer" not in str(dealer_error).lower():
                                raise
                            frame_result = analyze_video_frame(
                                frame,
                                template,
                                seat_count=seat_count,
                                min_confidence=min_confidence,
                                ocr=frame_ocr,
                                dealer_button_hint=dealer_button_cache,
                                cards_hint=cards_hint,
                                ocr_scale=ocr_scale,
                                layout_profile=active_layout_profile,
                            )
                            frame_result_for_event = frame_result
                            used_dealer_cache = True
                        if used_dealer_cache:
                            dealer_cache_uses += 1
                        if trigger == "frame" and cards_hint is None:
                            card_roi_cache = frame_result.get("cards")
                        # The first manually selected region is the complete poker client.
                        # Auto bbox may refine only the inner table, never the action-control input.
                        action_ocr = []
                        action_controls_ocr_mode = "skipped_no_button_surface"
                        if ocr is not None and quick_action_controls.get("visible"):
                            action_ocr_started = time.perf_counter()
                            action_ocr = run_ocr_in_roi(outer_frame, ocr, ACTION_OCR_ROI, scale=ocr_scale)
                            screen_timing["action_controls_ocr_ms"] = elapsed_ms(action_ocr_started)
                            action_controls_ocr_mode = "bottom_roi"
                        frame_result["action_controls"] = (
                            detect_action_controls(outer_frame, action_ocr)
                            if action_ocr
                            else quick_action_controls
                        )
                        frame_result["action_controls_capture"] = "full_client"
                        frame_result["ok"] = True
                        state = build_realtime_state(
                            frame_result,
                            video_path=Path(f"screen://monitor/{monitor}"),
                            timestamp_sec=round(timestamp, 3),
                            frame_index=processed_frames - 1,
                            sample_index=processed_frames - 1,
                        )
                        hero_card_cache = stabilize_hero_cards(state, hero_card_cache)
                        state["source"]["dealer_button_cached"] = used_dealer_cache
                        preflop_tracker.update(state)
                        state["source"]["kind"] = "screen"
                        state["source"]["screen_region"] = dict(region)
                        state["source"]["capture_region"] = dict(search_region)
                        state["source"]["action_controls_capture"] = frame_result.get("action_controls_capture")
                        state["source"]["monitor_region"] = dict(monitor_region)
                        state["source"]["auto_bbox"] = auto_bbox_info
                        state["source"]["visual_diff"] = round(float(visual_diff), 4)
                        state["source"]["table_visibility"] = table_visibility
                        state["source"]["ocr_mode"] = ocr_mode
                        state["source"]["action_controls_ocr_mode"] = action_controls_ocr_mode
                        state["source"]["cv_timing_ms"] = frame_result.get("timing_ms") or {}
                        state["source"]["ocr_item_count"] = frame_result.get("ocr_item_count")
                        state["source"]["cards_hint_used"] = frame_result.get("cards_hint_used")
                        state["source"]["card_cache_hit"] = cards_hint is not None
                        if layout_profile:
                            state["source"]["layout_profile"] = {
                                "id": layout_profile.get("id"),
                                "method": layout_profile.get("method"),
                                "hero_name": layout_profile.get("hero_name"),
                                "hero_name_anchor": layout_profile.get("hero_name_anchor"),
                                "hero_card_source": layout_profile.get("hero_card_source"),
                                "locked": layout_locked,
                                "quality": layout_profile_quality(layout_profile),
                            }
                        if with_advice:
                            advice_started = time.perf_counter()
                            attach_gto_advice(
                                state,
                                iterations=advice_iterations,
                                effective_stack_bb=effective_stack_bb,
                                villain_profile=villain_profile,
                            )
                            screen_timing["advice_ms"] = elapsed_ms(advice_started)
                        reason = "frame"
                        signature = "frame" if trigger == "frame" else state_signature(state)
                        should_emit = True if trigger == "frame" else signature != previous_signature
                        previous_signature = signature
                    except Exception as error:
                        state = build_error_state(
                            error,
                            video_path=Path(f"screen://monitor/{monitor}"),
                            timestamp_sec=round(timestamp, 3),
                            frame_index=processed_frames - 1,
                            sample_index=processed_frames - 1,
                        )
                        state["source"]["kind"] = "screen"
                        state["source"]["screen_region"] = dict(region)
                        state["source"]["capture_region"] = dict(search_region)
                        state["source"]["monitor_region"] = dict(monitor_region)
                        state["source"]["auto_bbox"] = auto_bbox_info
                        state["source"]["visual_diff"] = round(float(visual_diff), 4)
                        if "table_visibility" in locals():
                            state["source"]["table_visibility"] = table_visibility
                        reason = "error"
                        signature = "frame-error" if trigger == "frame" else state_signature(state)
                        should_emit = True if trigger == "frame" else signature != previous_signature
                        previous_signature = signature

                    state.setdefault("source", {})["screen_timing_ms"] = screen_timing
                    if show_overlay:
                        state["source"]["overlay_path"] = str(latest_overlay_path)
                        state["source"]["overlay_full_window_path"] = str(latest_overlay_full_window_path)
                        state["source"]["overlay_window_active"] = overlay is not None
                        if overlay_error:
                            state["source"]["overlay_error"] = overlay_error
                        if overlay is not None:
                            overlay_started = time.perf_counter()
                            try:
                                overlay.update(
                                    analysis_region=region,
                                    frame_shape=frame.shape,
                                    frame_result=frame_result_for_event,
                                    state=state,
                                    layout_profile=layout_profile,
                                )
                            except Exception as error:
                                overlay_error = f"{type(error).__name__}: {error}"
                                overlay.close()
                                overlay = None
                                state["source"]["overlay_window_active"] = False
                                state["source"]["overlay_error"] = overlay_error
                            finally:
                                screen_timing["overlay_window_ms"] = elapsed_ms(overlay_started)
                        if should_write_overlay_snapshot(
                            timestamp,
                            last_overlay_image_sec,
                            overlay_image_interval_sec,
                        ):
                            snapshot_started = time.perf_counter()
                            diagnostic = render_diagnostic_frame(
                                cv2,
                                frame,
                                frame_result_for_event,
                                state,
                                layout_profile,
                            )
                            write_png(cv2, latest_overlay_path, diagnostic)
                            full_window_diagnostic = render_full_window_diagnostic_frame(
                                cv2,
                                outer_frame,
                                search_region,
                                region,
                                diagnostic,
                            )
                            write_png(cv2, latest_overlay_full_window_path, full_window_diagnostic)
                            screen_timing["overlay_snapshot_ms"] = elapsed_ms(snapshot_started)
                            last_overlay_image_sec = timestamp

                    if should_emit:
                        artifact_started = time.perf_counter()
                        event_index = emitted_events
                        basename = f"event_{event_index:04d}_{timestamp:08.3f}s".replace(".", "p")
                        event_frame_path = ""
                        annotated_path = ""
                        if save_frames:
                            event_frame_path = str(frames_dir / f"{basename}.png")
                            cv2.imwrite(event_frame_path, frame)
                        if save_annotated and state.get("ok"):
                            annotated_path = str(annotated_dir / f"{basename}.png")
                            annotate_video_frame(frame, frame_result, Path(annotated_path))
                        problem = problem_reason(state)
                        if (
                            save_problem_frames
                            and problem
                            and saved_problem_frames < max(0, int(problem_frame_limit))
                        ):
                            problem_basename = f"{basename}_{problem}"
                            problem_frame_path = str(problem_dir / f"{problem_basename}.png")
                            cv2.imwrite(problem_frame_path, frame)
                            state["source"]["problem_frame_path"] = problem_frame_path
                            state["source"]["problem_reason"] = problem
                            if state.get("ok") and frame_result_for_event is not None:
                                # Keep the actual desktop pixels around the poker client. A
                                # tight analysis ROI must never hide bottom action controls in
                                # a later human review.
                                context_frame = grab_bgr(sct, monitor_region)
                                if diagnostic is None:
                                    diagnostic = render_diagnostic_frame(
                                        cv2,
                                        frame,
                                        frame_result_for_event,
                                        state,
                                        layout_profile,
                                    )
                                card_debug = write_card_debug_assets(
                                    cv2=cv2,
                                    frame=frame,
                                    frame_result=frame_result_for_event,
                                    state=state,
                                    output_dir=card_debug_dir,
                                    basename=problem_basename,
                                    problem=problem,
                                    context_frame=context_frame,
                                    search_region=monitor_region,
                                    analysis_region=region,
                                    context_scope="monitor_full",
                                    diagnostic_frame=diagnostic,
                                )
                                if card_debug:
                                    state["source"]["card_debug"] = card_debug
                                    saved_card_debug_samples += 1
                            saved_problem_frames += 1
                        if record_live_card_samples and state.get("ok") and frame_result_for_event is not None:
                            sample_signature = card_observation_signature(frame_result_for_event)
                            if not sample_signature:
                                last_card_sample_signature = None
                            should_save_card_sample = bool(sample_signature) and (
                                sample_signature != last_card_sample_signature
                                or timestamp - last_card_sample_sec >= max(1.0, float(card_sample_interval_sec))
                            )
                            if should_save_card_sample and saved_card_samples < max(0, int(card_sample_limit)):
                                sample_basename = (
                                    f"sample_{card_sample_session_id}_{saved_card_samples:04d}_{timestamp:08.3f}s"
                                ).replace(".", "p")
                                if diagnostic is None:
                                    diagnostic = render_diagnostic_frame(
                                        cv2,
                                        frame,
                                        frame_result_for_event,
                                        state,
                                        layout_profile,
                                    )
                                # Card recognition runs on the tight ROI for latency, but every
                                # review sample retains the entire display. This also works when
                                # Tencent Meeting renders the poker client inside another window.
                                full_context = grab_bgr(sct, monitor_region)
                                card_sample = write_card_debug_assets(
                                    cv2=cv2,
                                    frame=frame,
                                    frame_result=frame_result_for_event,
                                    state=state,
                                    output_dir=card_samples_dir,
                                    basename=sample_basename,
                                    problem="card_observation",
                                    context_frame=full_context,
                                    search_region=monitor_region,
                                    analysis_region=region,
                                    context_scope="monitor_full",
                                    diagnostic_frame=diagnostic,
                                )
                                if card_sample:
                                    append_card_sample_glyph_rows(
                                        card_samples_csv_path,
                                        card_sample.get("saved") or [],
                                        sample_id=sample_basename,
                                        timestamp_sec=timestamp,
                                        frame_index=processed_frames - 1,
                                    )
                                    state["source"]["card_sample"] = {
                                        key: value for key, value in card_sample.items() if key != "saved"
                                    }
                                    saved_card_samples += 1
                                    last_card_sample_signature = sample_signature
                                    last_card_sample_sec = timestamp
                        if record_live_state_audits and state.get("ok"):
                            audit_reason = state_audit_reason(state)
                            audit_signature = state_audit_signature(state) if audit_reason else ""
                            if (
                                audit_reason
                                and audit_signature != last_state_audit_signature
                                and saved_state_audits < max(0, int(state_audit_limit))
                            ):
                                audit_path = state_audit_dir / f"{basename}_{audit_reason}.png"
                                write_png(cv2, audit_path, outer_frame)
                                state["source"]["state_audit"] = {
                                    "frame": str(audit_path),
                                    "scope": "manual_outer_bbox",
                                    "reason": audit_reason,
                                    "signature": audit_signature,
                                }
                                saved_state_audits += 1
                                last_state_audit_signature = audit_signature
                        state["source"]["frame_path"] = event_frame_path
                        state["source"]["annotated_path"] = annotated_path
                        screen_timing["event_artifacts_ms"] = elapsed_ms(artifact_started)
                        state["source"]["analysis_ms"] = round((time.perf_counter() - loop_started) * 1000, 1)
                        state["event"] = {
                            "index": event_index,
                            "trigger": trigger,
                            "reason": reason,
                            "signature": signature,
                        }
                        stream.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
                        stream.flush()
                        current_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                        events.append(state)
                        emitted_events += 1
                        if print_events:
                            if console_mode == "full":
                                print(format_screen_event_line(state), flush=True)
                                printed_console_events += 1
                            else:
                                console_signature = screen_advice_console_signature(state)
                                heartbeat_due = (
                                    console_heartbeat_sec > 0
                                    and timestamp - last_console_emit_sec >= console_heartbeat_sec
                                )
                                if console_signature != last_console_signature or heartbeat_due:
                                    print(format_screen_advice_line(state), flush=True)
                                    last_console_signature = console_signature
                                    last_console_emit_sec = timestamp
                                    printed_console_events += 1

                    if preflight_once:
                        break
                    sleep_until_next(loop_started, every_sec)
        except KeyboardInterrupt:
            interrupted = True
            if print_events:
                print("\nScreen CV stopped by Ctrl+C.", flush=True)
        finally:
            if overlay is not None:
                overlay.close()

    elapsed_sec = time.perf_counter() - started_at
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ok": True,
        "interrupted": interrupted,
        "source": source_info(region, monitor),
        "search_region": dict(search_region),
        "auto_bbox": auto_bbox_info,
        "template": str(template_path),
        "output_dir": str(output_dir),
        "trigger": trigger,
        "console_mode": console_mode,
        "console_heartbeat_sec": console_heartbeat_sec,
        "preflight_once": bool(preflight_once),
        "ocr_enabled": bool(use_ocr and ocr is not None),
        "ocr_scale": ocr_scale if use_ocr and ocr is not None else None,
        "ocr_action_only": ocr_action_only if use_ocr and ocr is not None else None,
        "lock_layout": lock_layout,
        "layout_profile": layout_profile,
        "manual_hero_profile": str(hero_cards_file) if hero_cards_file is not None else None,
        "overlay_enabled": bool(show_overlay),
        "overlay_image_interval_sec": overlay_image_interval_sec if show_overlay else None,
        "overlay_error": overlay_error,
        "advice_enabled": with_advice,
        "dealer_refresh_frames": dealer_refresh_frames,
        "dealer_cache_uses": dealer_cache_uses,
        "saved_problem_frames": saved_problem_frames,
        "saved_card_debug_samples": saved_card_debug_samples,
        "record_card_samples": record_live_card_samples,
        "saved_card_samples": saved_card_samples,
        "saved_state_audits": saved_state_audits,
        "problem_frames_dir": str(problem_dir) if save_problem_frames else None,
        "card_debug_dir": str(card_debug_dir) if save_problem_frames else None,
        "visual_threshold": visual_threshold if trigger == "visual-change" else None,
        "min_event_gap_sec": min_event_gap_sec if trigger == "visual-change" else None,
        "sample": {
            "duration_sec": duration_sec,
            "every_sec": every_sec,
            "processed_frames": processed_frames,
            "emitted_events": emitted_events,
            "printed_console_events": printed_console_events,
            "wall_time_sec": round(elapsed_sec, 3),
            "avg_processed_frame_ms": round(elapsed_sec * 1000 / processed_frames, 2) if processed_frames else None,
            "effective_processing_fps": round(processed_frames / elapsed_sec, 3) if elapsed_sec > 0 else None,
            "card_cache_hits": card_cache_hits,
            "card_cache_misses": card_cache_misses,
        },
        "timing": event_source_timing_summary(events),
        "files": {
            "events_jsonl": str(jsonl_path),
            "events_json": str(events_path),
            "current_state": str(current_path),
            "summary": str(summary_path),
            "problem_frames": str(problem_dir) if save_problem_frames else "",
            "card_debug": str(card_debug_dir) if save_problem_frames else "",
            "card_samples": str(card_samples_dir) if record_live_card_samples else "",
            "card_sample_predictions": str(card_samples_csv_path) if record_live_card_samples else "",
            "state_audits": str(state_audit_dir) if record_live_state_audits else "",
            "prepare_card_sample_labels_command": str(card_sample_prepare_command_path) if record_live_card_samples else "",
            "serve_card_sample_labels_command": str(card_sample_serve_command_path) if record_live_card_samples else "",
            "apply_card_sample_labels_command": str(card_sample_apply_command_path) if record_live_card_samples else "",
            "layout_profile": str(layout_profile_path) if layout_profile is not None else "",
            "latest_overlay": str(latest_overlay_path) if show_overlay else "",
            "latest_overlay_full_window": str(latest_overlay_full_window_path) if show_overlay else "",
        },
        "events": events,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not current_path.exists():
        current_path.write_text(json.dumps({"ok": False, "error": "no events emitted"}, indent=2), encoding="utf-8")
    return summary


def capture_region(sct: Any, monitor: int, bbox: tuple[int, int, int, int] | None) -> dict[str, int]:
    monitors = sct.monitors
    if monitor < 0 or monitor >= len(monitors):
        raise ValueError(f"monitor must be between 0 and {len(monitors) - 1}")
    if bbox:
        left, top, width, height = bbox
        return {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
    selected = monitors[monitor]
    return {
        "left": int(selected["left"]),
        "top": int(selected["top"]),
        "width": int(selected["width"]),
        "height": int(selected["height"]),
    }


def grab_bgr(sct: Any, region: dict[str, int]) -> Any:
    import numpy as np

    image = np.array(sct.grab(region))
    return image[:, :, :3].copy()


def sleep_until_next(loop_started: float, every_sec: float) -> None:
    elapsed = time.perf_counter() - loop_started
    delay = max(0.0, float(every_sec) - elapsed)
    if delay:
        time.sleep(delay)


def should_refresh_dealer_button(
    *,
    dealer_button_cache: dict[str, Any] | None,
    dealer_refresh_frames: int,
    processed_frames: int,
    last_dealer_refresh_frame: int,
    visual_diff: float,
    visual_threshold: float,
    normal_action_buttons_visible: bool,
    previous_normal_action_buttons_visible: bool,
) -> bool:
    """Refresh when controls first appear, layout changes, or scheduled validation is due."""

    return bool(
        dealer_button_cache is None
        or dealer_refresh_frames <= 1
        or processed_frames - last_dealer_refresh_frame >= max(1, int(dealer_refresh_frames))
        or visual_diff >= max(8.0, visual_threshold * 3.0)
        or (normal_action_buttons_visible and not previous_normal_action_buttons_visible)
    )


def should_write_overlay_snapshot(timestamp: float, last_snapshot_sec: float, interval_sec: float) -> bool:
    """Keep the live window current while rate-limiting large PNG writes."""

    return interval_sec <= 0 or timestamp - last_snapshot_sec >= max(0.0, interval_sec)


def card_signature_rois(
    layout_profile: dict[str, Any] | None,
) -> tuple[tuple[float, float, float, float], ...]:
    hero_rois: list[tuple[float, float, float, float]] = []
    if (layout_profile or {}).get("hero_card_source") == "manual_hero_cards":
        for box in (layout_profile or {}).get("hero_card_boxes") or []:
            x1 = float(box.get("x", 0.0))
            y1 = float(box.get("y", 0.0))
            hero_rois.append(
                (
                    x1,
                    y1,
                    x1 + float(box.get("width", 0.0)),
                    y1 + float(box.get("height", 0.0)),
                )
            )
    return tuple(hero_rois or HERO_CARD_ROIS) + tuple(BOARD_CARD_ROIS)


CARD_SAMPLE_GLYPH_COLUMNS = (
    "sample_id",
    "timestamp_sec",
    "frame_index",
    "group",
    "slot",
    "card",
    "kind",
    "input_path",
    "current_label",
    "confidence",
    "margin",
    "reason",
)


def card_observation_signature(frame_result: dict[str, Any]) -> str:
    cards = frame_result.get("cards") or {}
    observations: list[str] = []
    for group, details in (("hero", cards.get("hero_details") or []), ("board", cards.get("board_details") or [])):
        for position, detail in enumerate(details):
            slot = int(detail.get("index") if detail.get("index") is not None else position)
            card = str(detail.get("card") or "").strip()
            if card:
                observations.append(f"{group}:{slot}:{card}")
    return "|".join(observations)


def state_audit_reason(state: dict[str, Any]) -> str | None:
    """Select states whose full manually selected poker window merits review."""

    advice = state.get("gto_advice") or {}
    controls = state.get("action_controls") or {}
    reason = str(advice.get("reason") or "").strip()
    if reason:
        return reason
    if controls.get("visible"):
        return "hero_action_controls_visible"
    return None


def state_audit_signature(state: dict[str, Any]) -> str:
    """Keep one complete-window screenshot for each distinct actionable state."""

    table = state.get("table") or {}
    hero = state.get("hero") or {}
    controls = state.get("action_controls") or {}
    advice = state.get("gto_advice") or {}
    tracker = state.get("preflop_tracker") or {}
    preflop = state.get("preflop") or {}
    visible_bets = [
        {
            "seat": item.get("seat"),
            "amount_bb": item.get("amount_bb"),
        }
        for item in list(state.get("bets") or [])
        if isinstance(item, dict)
    ]
    payload = {
        "street": table.get("street"),
        "dealer": table.get("dealer_seat"),
        "hero_position": hero.get("position"),
        "hero_cards": list(hero.get("cards") or []),
        "hero_turn": bool(hero.get("is_turn")),
        "actions": list(controls.get("actions") or []),
        "call_amount_bb": controls.get("call_amount_bb"),
        "raise_amount_bb": controls.get("raise_amount_bb"),
        "advice_reason": advice.get("reason"),
        "advice_summary": advice.get("summary"),
        "tracker_reason": tracker.get("reason"),
        "preflop_history": list(preflop.get("action_history") or []),
        "visible_bets": visible_bets,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_card_sample_glyph_rows(
    path: Path,
    saved_cards: list[dict[str, Any]],
    *,
    sample_id: str,
    timestamp_sec: float,
    frame_index: int,
) -> int:
    rows: list[dict[str, Any]] = []
    for item in saved_cards:
        card = str(item.get("card") or "")
        if noncard_hero_glyph_item(item, card=card):
            continue
        for kind in ("rank", "suit"):
            input_path = item.get(f"{kind}_path")
            if not input_path:
                continue
            confidence = item.get(f"{kind}_confidence")
            margin = item.get(f"{kind}_margin")
            current_label = item.get(kind) or card_label_part(card, kind)
            rows.append(
                {
                    "sample_id": sample_id,
                    "timestamp_sec": round(float(timestamp_sec), 3),
                    "frame_index": int(frame_index),
                    "group": item.get("group"),
                    "slot": item.get("slot"),
                    "card": card,
                    "kind": kind,
                    "input_path": str(Path(str(input_path)).resolve()),
                    "current_label": current_label,
                    "confidence": confidence,
                    "margin": margin,
                    "reason": card_sample_reason(kind, confidence, margin),
                }
            )
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if write_header else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CARD_SAMPLE_GLYPH_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def noncard_hero_glyph_item(item: dict[str, Any], *, card: str | None = None) -> bool:
    if str(item.get("group") or "") != "hero":
        return False
    card_text = str(card if card is not None else item.get("card") or "").strip()
    try:
        rank_confidence = float(item.get("rank_confidence") or 0.0)
        suit_confidence = float(item.get("suit_confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    if card_text in {"", "?", "??", "item"} and rank_confidence <= 0.0 and suit_confidence <= 0.0:
        return True
    try:
        face_fill = float(item.get("face_fill") or 0.0)
        face_cover = float(item.get("face_cover") or 0.0)
        face_aspect = float(item.get("face_aspect") or 0.0)
    except (TypeError, ValueError):
        return False
    return (
        0.0 < face_fill < 0.75
        and 0.0 < face_cover < 0.60
        and 0.0 < face_aspect < 1.15
        and rank_confidence < 0.60
    )


def card_label_part(card: str, kind: str) -> str:
    text = str(card or "").strip()
    if kind == "rank":
        return "T" if text.startswith("10") else text[:1]
    return text[-1:] if len(text) >= 2 else ""


def card_sample_reason(kind: str, confidence: Any, margin: Any) -> str:
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    try:
        margin_value = float(margin)
    except (TypeError, ValueError):
        margin_value = 0.0
    confidence_threshold = 0.72 if kind == "rank" else 0.70
    margin_threshold = 0.08 if kind == "rank" else 0.06
    if confidence_value < confidence_threshold or margin_value < margin_threshold:
        return f"live_{kind}_low_score_or_margin"
    return f"live_{kind}_observation"


def problem_reason(state: dict[str, Any]) -> str | None:
    if not state.get("ok"):
        error = str(state.get("error") or "error").lower()
        if "dealer" in error:
            return "dealer_not_found"
        return "error"

    table = state.get("table", {})
    hero = state.get("hero", {})
    action_controls = state.get("action_controls", {})
    advice = state.get("gto_advice", {})
    advice_reason = str(advice.get("reason") or "")
    hero_cards = [card for card in hero.get("cards", []) if card]
    board = [card for card in table.get("board", []) if card]

    if advice_reason in {"hero_cards_incomplete", "board_cards_incomplete", "advisor_error"}:
        return advice_reason
    if hero_cards and (len(hero_cards) != 2 or any("?" in str(card) for card in hero_cards)):
        return "hero_cards_incomplete"
    if hero.get("has_cards") and action_controls.get("visible") and len(hero_cards) != 2:
        return "hero_cards_incomplete"
    if action_controls.get("visible") and (len(hero_cards) != 2 or any("?" in card for card in hero_cards)):
        return "hero_cards_incomplete"
    if any("?" in card for card in board):
        return "board_cards_incomplete"
    known_cards = [card for card in hero_cards + board if "?" not in card]
    if len(known_cards) != len(set(known_cards)):
        return "duplicate_cards"
    expected_board = {"flop": 3, "turn": 4, "river": 5}.get(str(table.get("street") or "").lower())
    if expected_board is not None and 0 < len(board) < expected_board:
        return "board_cards_incomplete"
    return None


def write_card_debug_assets(
    *,
    cv2: Any,
    frame: Any,
    frame_result: dict[str, Any],
    state: dict[str, Any],
    output_dir: Path,
    basename: str,
    problem: str,
    context_frame: Any | None = None,
    search_region: dict[str, int] | None = None,
    analysis_region: dict[str, int] | None = None,
    context_scope: str = "",
    diagnostic_frame: Any | None = None,
) -> dict[str, Any] | None:
    cards = frame_result.get("cards") or {}
    sample_dir = Path(output_dir) / safe_debug_name(basename)
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_path = sample_dir / "frame.png"
    write_png(cv2, frame_path, frame)
    context_path: Path | None = None
    if context_frame is not None and context_frame.size:
        context = context_frame.copy()
        capture_region = (state.get("source") or {}).get("capture_region") or {}
        if search_region and isinstance(capture_region, dict):
            capture_left = int(capture_region.get("left") or search_region["left"])
            capture_top = int(capture_region.get("top") or search_region["top"])
            capture_width = int(capture_region.get("width") or 0)
            capture_height = int(capture_region.get("height") or 0)
            if capture_width > 0 and capture_height > 0:
                cx1 = capture_left - int(search_region["left"])
                cy1 = capture_top - int(search_region["top"])
                cx2 = cx1 + capture_width
                cy2 = cy1 + capture_height
                cv2.rectangle(context, (cx1, cy1), (cx2, cy2), (70, 230, 70), 4)
                cv2.putText(
                    context,
                    "FULL CLIENT (ACTIONS)",
                    (cx1 + 6, max(24, cy1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    context,
                    "FULL CLIENT (ACTIONS)",
                    (cx1 + 6, max(24, cy1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (70, 230, 70),
                    2,
                    cv2.LINE_AA,
                )
        if search_region and analysis_region:
            x1 = int(analysis_region["left"] - search_region["left"])
            y1 = int(analysis_region["top"] - search_region["top"])
            x2 = x1 + int(analysis_region["width"])
            y2 = y1 + int(analysis_region["height"])
            cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(context, "INNER TABLE (CARDS)", (x1 + 6, max(24, y1 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(context, "INNER TABLE (CARDS)", (x1 + 6, max(24, y1 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        context_path = sample_dir / "screen_context.png"
        write_png(cv2, context_path, context)
    diagnostic_path: Path | None = None
    if diagnostic_frame is not None and diagnostic_frame.size:
        diagnostic_path = sample_dir / "diagnostic_overlay.png"
        write_png(cv2, diagnostic_path, diagnostic_frame)

    saved: list[dict[str, Any]] = []
    for group, details in (("hero", cards.get("hero_details") or []), ("board", cards.get("board_details") or [])):
        for position, detail in enumerate(details):
            saved_item = save_card_detail_debug(
                cv2=cv2,
                frame=frame,
                detail=detail,
                sample_dir=sample_dir,
                group=group,
                slot=int(detail.get("index") if detail.get("index") is not None else position),
                fallback=False,
            )
            if saved_item:
                saved.append(saved_item)

    fallback_saved: list[dict[str, Any]] = []
    if problem.startswith("hero_cards"):
        fallback_saved.extend(save_fallback_card_rois(cv2, frame, sample_dir, "hero", HERO_CARD_ROIS, source="hero", max_slots=2))
    if problem.startswith("board_cards") or problem == "duplicate_cards":
        fallback_saved.extend(save_fallback_card_rois(cv2, frame, sample_dir, "board", BOARD_CARD_ROIS, source="board", max_slots=5))

    metadata = {
        "problem": problem,
        "frame": str(frame_path),
        "screen_context": str(context_path) if context_path else "",
        "screen_context_scope": str(context_scope) if context_path else "",
        "diagnostic_overlay": str(diagnostic_path) if diagnostic_path else "",
        "search_region": dict(search_region) if search_region else None,
        "analysis_region": dict(analysis_region) if analysis_region else None,
        "source": dict(state.get("source") or {}),
        "timestamp_sec": (state.get("source") or {}).get("timestamp_sec"),
        "frame_index": (state.get("source") or {}).get("frame_index"),
        "hero_cards": (state.get("hero") or {}).get("cards"),
        "board": (state.get("table") or {}).get("board"),
        "hero_confidence": ((state.get("confidence") or {}).get("cards") or {}).get("hero"),
        "board_confidence": ((state.get("confidence") or {}).get("cards") or {}).get("board"),
        "saved": saved,
        "fallback": fallback_saved,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dir": str(sample_dir),
        "metadata": str(metadata_path),
        "frame": str(frame_path),
        "screen_context": str(context_path) if context_path else "",
        "screen_context_scope": str(context_scope) if context_path else "",
        "diagnostic_overlay": str(diagnostic_path) if diagnostic_path else "",
        "saved_count": len(saved),
        "fallback_count": len(fallback_saved),
        "saved": saved,
    }


def save_card_detail_debug(
    *,
    cv2: Any,
    frame: Any,
    detail: dict[str, Any],
    sample_dir: Path,
    group: str,
    slot: int,
    fallback: bool,
) -> dict[str, Any] | None:
    box = detail.get("roi_box") or {}
    card_crop = crop_box(frame, box)
    if card_crop is None or card_crop.size == 0:
        return None
    source = str(detail.get("source") or group)
    card_label = safe_debug_name(str(detail.get("card") or "unknown"))
    prefix = f"{group}_slot{slot}_{card_label}"
    if fallback:
        prefix = f"fallback_{prefix}"
    card_path = sample_dir / f"{prefix}_card.png"
    rank_path = sample_dir / f"{prefix}_rank.png"
    suit_path = sample_dir / f"{prefix}_suit.png"
    write_png(cv2, card_path, card_crop)
    write_png(cv2, rank_path, safe_rank_debug_image(card_crop, source))
    write_png(cv2, suit_path, safe_suit_debug_image(card_crop, source))
    return {
        "group": group,
        "slot": slot,
        "card": detail.get("card"),
        "rank": detail.get("rank"),
        "suit": detail.get("suit"),
        "rank_confidence": detail.get("rank_confidence"),
        "rank_margin": detail.get("rank_margin"),
        "suit_confidence": detail.get("suit_confidence"),
        "suit_margin": detail.get("suit_margin"),
        "face_fill": detail.get("face_fill"),
        "face_cover": detail.get("face_cover"),
        "face_aspect": detail.get("face_aspect"),
        "roi_mode": detail.get("roi_mode"),
        "roi_box": box,
        "fallback": bool(fallback),
        "card_path": str(card_path),
        "rank_path": str(rank_path),
        "suit_path": str(suit_path),
    }


def save_fallback_card_rois(
    cv2: Any,
    frame: Any,
    sample_dir: Path,
    group: str,
    rois: tuple[tuple[float, float, float, float], ...],
    *,
    source: str,
    max_slots: int,
) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    saved: list[dict[str, Any]] = []
    for slot, roi in enumerate(rois[:max_slots]):
        x1, y1, x2, y2 = scale_roi(roi, width, height)
        detail = {
            "card": "unknown",
            "rank": "",
            "suit": "",
            "source": source,
            "index": slot,
            "roi_mode": "fallback_fixed_roi",
            "roi_box": {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)},
        }
        saved_item = save_card_detail_debug(
            cv2=cv2,
            frame=frame,
            detail=detail,
            sample_dir=sample_dir,
            group=group,
            slot=slot,
            fallback=True,
        )
        if saved_item:
            saved.append(saved_item)
    return saved


def safe_rank_debug_image(card_crop: Any, source: str) -> Any:
    try:
        if card_crop is not None and card_crop.size:
            return export_rank_glyph_image(card_crop, source)
    except Exception:
        pass
    _cv2, np = load_cv()
    return np.zeros((70, 54), dtype=np.uint8)


def safe_suit_debug_image(card_crop: Any, source: str) -> Any:
    try:
        if card_crop is not None and card_crop.size and card_crop.shape[0] >= 16 and card_crop.shape[1] >= 16:
            # Review and training assets must contain the whole suit component.
            # The fast hero window is intentionally narrow for live candidates,
            # but it can clip the lower club/spade stem in a tall card crop.
            return normalized_suit_component_by_label(card_crop, (42, 42), source=source)
    except Exception:
        pass
    _cv2, np = load_cv()
    return np.zeros((42, 42), dtype=np.uint8)


def crop_box(frame: Any, box: dict[str, Any]) -> Any | None:
    if not box:
        return None
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, int(box.get("x", 0))))
    y1 = max(0, min(frame_h - 1, int(box.get("y", 0))))
    x2 = max(x1 + 1, min(frame_w, x1 + int(box.get("width", 0))))
    y2 = max(y1 + 1, min(frame_h, y1 + int(box.get("height", 0))))
    return frame[y1:y2, x1:x2]


def write_png(cv2: Any, path: Path, image: Any) -> bool:
    if image is None or image.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def safe_debug_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._")[:120] or "item"


def select_bbox(cv2: Any, frame: Any, region: dict[str, int]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    scale = min(1600 / max(width, 1), 950 / max(height, 1), 1.0)
    if scale < 1.0:
        display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        display = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    else:
        display = frame

    window_name = "Select poker table, then press Enter or Space"
    roi = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, selected_width, selected_height = (int(value) for value in roi)
    if selected_width <= 0 or selected_height <= 0:
        raise ValueError("bbox selection canceled")

    left = region["left"] + int(round(x / scale))
    top = region["top"] + int(round(y / scale))
    absolute_width = int(round(selected_width / scale))
    absolute_height = int(round(selected_height / scale))
    return left, top, absolute_width, absolute_height


def detect_auto_bbox(
    cv2: Any,
    np: Any,
    frame: Any,
    search_region: dict[str, int],
    template: Any,
    min_confidence: float,
    *,
    allow_native_window: bool = True,
) -> dict[str, Any] | None:
    candidates = []
    if allow_native_window:
        native = detect_native_poker_window_bbox(search_region)
        if native is not None:
            candidates.append({"region": native, "method": "native-window", "bonus": 3.0})
    candidates.extend(detect_dealer_button_bbox_candidates(cv2, frame, search_region, template, min_confidence))
    candidates.extend(detect_visual_poker_window_candidates(cv2, np, frame, search_region))
    if not candidates:
        return None

    dealer_threshold = max(0.25, min_confidence - 0.12)
    scored_by_index: dict[int, dict[str, Any]] = {}
    scored_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        enriched = build_auto_bbox_enriched_candidate(
            cv2,
            frame,
            search_region,
            candidate,
            template,
            dealer_threshold,
            run_template_recheck=False,
        )
        if enriched is None:
            continue
        enriched["_candidate_index"] = index
        scored_by_index[index] = enriched
        scored_candidates.append(enriched)
    strong_candidates = [
        item
        for item in scored_candidates
        if float(item.get("dealer_confidence") or 0.0) >= dealer_threshold
        and float(item.get("score") or 0.0) >= 8.0
    ]
    recheck_candidates = []
    if not strong_candidates:
        recheck_candidates = [
            item
            for item in scored_candidates
            if bool(item.get("_needs_template_recheck")) and str(item.get("method") or "") != "native-window"
        ]
    for item in sorted(recheck_candidates, key=lambda value: float(value.get("score") or 0.0), reverse=True)[
        :AUTO_BBOX_TEMPLATE_RECHECK_LIMIT
    ]:
        index = int(item["_candidate_index"])
        enriched = build_auto_bbox_enriched_candidate(
            cv2,
            frame,
            search_region,
            candidates[index],
            template,
            dealer_threshold,
            run_template_recheck=True,
        )
        if enriched is None:
            continue
        enriched["_candidate_index"] = index
        scored_by_index[index] = enriched
    scored_candidates = list(scored_by_index.values())
    eligible_candidates = [
        item
        for item in scored_candidates
        if str(item.get("method") or "") == "native-window"
        or float(item.get("dealer_confidence") or 0.0) >= dealer_threshold
    ]
    best = max(eligible_candidates, key=lambda item: float(item.get("score") or 0.0), default=None)
    if best is None:
        return None
    titlebar_with_hero = [
        candidate
        for candidate in eligible_candidates
        if str(candidate.get("method") or "").startswith("visual-titlebar")
        and float(candidate.get("hero_anchor_score") or 0.0) >= 0.75
        and float(candidate.get("dealer_confidence") or 0.0) >= dealer_threshold
    ]
    if titlebar_with_hero and best["method"] != "native-window":
        title_best = max(titlebar_with_hero, key=lambda item: item["score"])
        if title_best["score"] >= best["score"] - 5.0:
            best = title_best
    larger_window_candidates = [
        candidate
        for candidate in eligible_candidates
        if str(candidate.get("method") or "").startswith("visual-titlebar")
        and float(candidate.get("dealer_confidence") or 0.0) >= dealer_threshold
        and float(candidate.get("area_ratio") or 0.0) <= 0.86
        and float(candidate.get("score") or 0.0) >= float(best.get("score") or 0.0) - 5.0
        and region_area(candidate.get("region") or {}) >= region_area(best.get("region") or {}) * 1.18
    ]
    if larger_window_candidates and best["method"] != "native-window":
        best = max(larger_window_candidates, key=lambda item: (region_area(item.get("region") or {}), float(item.get("score") or 0.0)))
    if best["dealer_confidence"] < dealer_threshold and best["method"] != "native-window":
        return None
    return public_auto_bbox_candidate(best)


def public_auto_bbox_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def build_auto_bbox_enriched_candidate(
    cv2: Any,
    frame: Any,
    search_region: dict[str, int],
    candidate: dict[str, Any],
    template: Any,
    dealer_threshold: float,
    *,
    run_template_recheck: bool,
) -> dict[str, Any] | None:
    region = candidate["region"]
    x = region["left"] - search_region["left"]
    y = region["top"] - search_region["top"]
    w = region["width"]
    h = region["height"]
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return None
    crop = frame[y : y + h, x : x + w]
    score_crop, score_scale = auto_bbox_score_crop(cv2, crop)
    score_h, score_w = score_crop.shape[:2]
    aspect = w / max(h, 1)
    area_ratio = (w * h) / max(frame.shape[0] * frame.shape[1], 1)
    height_ratio = h / max(frame.shape[0], 1)
    width_ratio = w / max(frame.shape[1], 1)
    method = str(candidate.get("method") or "")
    if method == "visual-dark-region" and area_ratio > 0.85:
        return None
    table_visible, visibility = poker_table_visibility(cv2, score_crop)
    if not table_visible and method != "native-window":
        return None

    dealer_button = candidate_dealer_button(candidate, region, score_scale)
    if dealer_button is None:
        try:
            dealer_button = find_dealer_button_component(score_crop)
        except Exception:
            dealer_button = None
    if dealer_button is None and run_template_recheck:
        try:
            dealer_button = find_dealer_button(
                crop,
                template,
                min_confidence=dealer_threshold,
                min_scale=0.60,
                max_scale=1.80,
            )
        except Exception:
            dealer_button = None

    dealer_confidence = float((dealer_button or {}).get("confidence") or 0.0)
    dealer_box = (dealer_button or {}).get("box")
    content_score = poker_content_score(cv2, score_crop)
    anchor_score = bbox_anchor_score(candidate, dealer_box, score_w, score_h)
    pot_anchor_score = yellow_pot_anchor_score(cv2, score_crop)
    hero_anchor_score = hero_fixed_card_anchor_score(cv2, score_crop)
    aspect_penalty = abs(aspect - 1.42) * 2.5
    oversize_penalty = (
        max(0.0, area_ratio - 0.56) * 10
        + max(0.0, height_ratio - 0.82) * 12
        + max(0.0, width_ratio - 0.88) * 8
    )
    scale_penalty = 0.0
    if method == "dealer-button-anchor":
        scale_penalty = abs(float((candidate.get("anchors") or {}).get("dealer_scale") or 1.0) - 1.0) * 4.0
    score = (
        dealer_confidence * 4
        + content_score * 8
        + anchor_score * 3
        + pot_anchor_score * 2.5
        + hero_anchor_score * 6.0
        + area_ratio * 2
        - aspect_penalty
        - oversize_penalty
        - scale_penalty
        + float(candidate.get("bonus") or 0.0)
    )
    enriched = {
        "method": candidate["method"],
        "region": dict(region),
        "score": round(float(score), 4),
        "dealer_confidence": round(float(dealer_confidence), 4),
        "content_score": round(float(content_score), 4),
        "anchor_score": round(float(anchor_score), 4),
        "pot_anchor_score": round(float(pot_anchor_score), 4),
        "hero_anchor_score": round(float(hero_anchor_score), 4),
        "visibility": visibility,
        "dealer_box": dealer_box,
        "aspect": round(float(aspect), 4),
        "area_ratio": round(float(area_ratio), 4),
        "height_ratio": round(float(height_ratio), 4),
        "width_ratio": round(float(width_ratio), 4),
        "_needs_template_recheck": dealer_confidence < dealer_threshold,
    }
    if candidate.get("anchors"):
        enriched["anchors"] = candidate["anchors"]
    if dealer_button is not None and dealer_button.get("method"):
        enriched["dealer_method"] = dealer_button.get("method")
    return enriched


def auto_bbox_score_crop(cv2: Any, crop: Any) -> tuple[Any, float]:
    if crop.size == 0:
        return crop, 1.0
    height, width = crop.shape[:2]
    if width <= AUTO_BBOX_SCORE_MAX_WIDTH:
        return crop, 1.0
    scale = AUTO_BBOX_SCORE_MAX_WIDTH / max(float(width), 1.0)
    resized = cv2.resize(
        crop,
        (AUTO_BBOX_SCORE_MAX_WIDTH, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def candidate_dealer_button(candidate: dict[str, Any], region: dict[str, int], scale: float = 1.0) -> dict[str, Any] | None:
    anchors = candidate.get("anchors") or {}
    button = anchors.get("dealer_button")
    if not isinstance(button, dict):
        return None
    center_abs_x = float(button.get("x") or 0.0)
    center_abs_y = float(button.get("y") or 0.0)
    width = float(button.get("width") or 0.0)
    height = float(button.get("height") or width)
    confidence = float(button.get("confidence") or 0.0)
    if center_abs_x <= 0 or center_abs_y <= 0 or width <= 0 or height <= 0 or confidence <= 0:
        return None
    center_x = (center_abs_x - float(region["left"])) * scale
    center_y = (center_abs_y - float(region["top"])) * scale
    width *= scale
    height *= scale
    return {
        "box": {
            "x": int(round(center_x - width / 2.0)),
            "y": int(round(center_y - height / 2.0)),
            "width": int(round(width)),
            "height": int(round(height)),
        },
        "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
        "confidence": round(float(confidence), 4),
        "adjusted_score": round(float(confidence), 4),
        "scale": float(anchors.get("dealer_scale") or 0.0),
        "method": "candidate-anchor",
    }


def detect_native_poker_window_bbox(search_region: dict[str, int]) -> dict[str, int] | None:
    try:
        import win32gui
    except Exception:
        return None

    keywords = ("WPT", "PUKE", "Poker", "\u6251\u514b", "\u5fb7\u5dde", "\u6025\u901f")
    base_left = search_region["left"]
    base_top = search_region["top"]
    base_right = base_left + search_region["width"]
    base_bottom = base_top + search_region["height"]
    matches: list[dict[str, int]] = []

    def visit(hwnd: int, _extra: Any) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        if not any(keyword.lower() in title.lower() for keyword in keywords):
            return
        left, top, right, bottom = [int(value) for value in win32gui.GetWindowRect(hwnd)]
        left = max(left, base_left)
        top = max(top, base_top)
        right = min(right, base_right)
        bottom = min(bottom, base_bottom)
        width = right - left
        height = bottom - top
        if width < 500 or height < 350:
            return
        matches.append({"left": left, "top": top, "width": width, "height": height})

    try:
        win32gui.EnumWindows(visit, None)
    except Exception:
        return None
    if not matches:
        return None
    return max(matches, key=lambda item: item["width"] * item["height"])


def detect_visual_poker_window_candidates(
    cv2: Any,
    np: Any,
    frame: Any,
    search_region: dict[str, int],
) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    candidates: list[dict[str, Any]] = []
    candidates.extend(detect_action_button_bbox_candidates(cv2, np, frame, search_region))
    candidates.extend(detect_felt_table_candidates(cv2, np, frame, search_region))
    candidates.extend(detect_titlebar_row_candidates(np, value, search_region))
    current_region_candidate = detect_current_region_table_candidate(cv2, frame, search_region)
    if current_region_candidate is not None:
        candidates.append(current_region_candidate)

    dark = (value < 75).astype("uint8") * 255
    title_mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((45, 5), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(title_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.35 or h < 18 or h > 90:
            continue
        if w / max(h, 1) < 8:
            continue
        estimated_height = int(round(w / 1.415))
        if estimated_height < height * 0.35:
            continue
        region = clamp_region(
            {
                "left": search_region["left"] + x,
                "top": search_region["top"] + y,
                "width": w,
                "height": estimated_height,
            },
            search_region,
        )
        if region["width"] < 500 or region["height"] < 350:
            continue
        candidates.append({"region": region, "method": "visual-titlebar", "bonus": 1.0})

    broad = (value < 145).astype("uint8") * 255
    broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(broad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.35 or h < height * 0.35:
            continue
        aspect = w / max(h, 1)
        if aspect < 1.05 or aspect > 1.90:
            continue
        region = clamp_region(
            {
                "left": search_region["left"] + x,
                "top": search_region["top"] + y,
                "width": w,
                "height": h,
            },
            search_region,
        )
        candidates.append({"region": region, "method": "visual-dark-region", "bonus": 0.0})
    return dedupe_regions(candidates)


def detect_current_region_table_candidate(cv2: Any, frame: Any, search_region: dict[str, int]) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    if width < 700 or height < 500:
        return None
    aspect = width / max(height, 1)
    if aspect < 1.18 or aspect > 1.85:
        return None
    visible, _visibility = poker_table_visibility(cv2, frame)
    if not visible:
        return None
    content_score = poker_content_score(cv2, frame)
    pot_anchor_score = yellow_pot_anchor_score(cv2, frame)
    hero_anchor_score = hero_fixed_card_anchor_score(cv2, frame)
    if content_score < 0.46:
        return None
    if pot_anchor_score < 0.18 and hero_anchor_score < 0.52:
        return None
    return {
        "region": {
            "left": search_region["left"],
            "top": search_region["top"],
            "width": width,
            "height": height,
        },
        "method": "current-region-table",
        "bonus": -1.0,
        "anchors": {
            "current_region": {
                "content_score": round(float(content_score), 4),
                "pot_anchor_score": round(float(pot_anchor_score), 4),
                "hero_anchor_score": round(float(hero_anchor_score), 4),
            }
        },
    }


def detect_dealer_button_bbox_candidates(
    cv2: Any,
    frame: Any,
    search_region: dict[str, int],
    template: Any,
    min_confidence: float,
) -> list[dict[str, Any]]:
    try:
        component = find_dealer_button_component(frame)
        if component is not None and float(component.get("confidence") or 0.0) >= max(0.72, min_confidence):
            button = component
        else:
            button = find_dealer_button(
                frame,
                template,
                min_confidence=max(0.25, min_confidence - 0.12),
                min_scale=0.42,
                max_scale=1.85,
            )
    except Exception:
        return []

    box = button.get("box") or {}
    center = button.get("center") or {}
    button_width = float(box.get("width") or 0.0)
    center_x = float(center.get("x") or 0.0)
    center_y = float(center.get("y") or 0.0)
    if button_width <= 8 or center_x <= 0 or center_y <= 0:
        return []

    base_width = button_width / 0.0208
    frame_height, frame_width = frame.shape[:2]
    candidates: list[dict[str, Any]] = []
    for scale in (0.88, 1.0, 1.12):
        inferred_width = base_width * scale
        inferred_height = inferred_width / 1.415
        if inferred_width < 700 or inferred_height < 500:
            continue
        if inferred_width > frame_width * 0.95 or inferred_height > frame_height * 0.95:
            continue
        for seat_index, anchor in enumerate(DEALER_BUTTON_ANCHORS_8):
            left = int(round(center_x - anchor[0] * inferred_width))
            top = int(round(center_y - anchor[1] * inferred_height))
            margin_x = 12.0
            margin_y = 12.0
            if (
                left < -margin_x
                or top < -margin_y
                or left + inferred_width > frame_width + margin_x
                or top + inferred_height > frame_height + margin_y
            ):
                continue
            region = clamp_region(
                {
                    "left": search_region["left"] + left,
                    "top": search_region["top"] + top,
                    "width": int(round(inferred_width)),
                    "height": int(round(inferred_height)),
                },
                search_region,
            )
            aspect = region["width"] / max(region["height"], 1)
            button_width_ratio = button_width / max(float(region["width"]), 1.0)
            if button_width_ratio < 0.015 or button_width_ratio > 0.030:
                continue
            if region["width"] < 700 or region["height"] < 500 or aspect < 1.25 or aspect > 1.65:
                continue
            candidates.append(
                {
                    "region": region,
                    "method": "dealer-button-anchor",
                    "bonus": 1.6,
                    "anchors": {
                        "dealer_button": {
                            "x": int(round(search_region["left"] + center_x)),
                            "y": int(round(search_region["top"] + center_y)),
                            "width": int(round(button_width)),
                            "height": int(round(float(box.get("height") or 0.0))),
                            "confidence": round(float(button.get("confidence") or 0.0), 4),
                        },
                        "dealer_anchor_seat_index": seat_index,
                        "dealer_anchor": {"x": anchor[0], "y": anchor[1]},
                        "dealer_scale": round(float(scale), 3),
                    },
                }
            )
    return dedupe_regions(candidates)


def detect_felt_table_candidates(cv2: Any, np: Any, frame: Any, search_region: dict[str, int]) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    scale = min(1280 / max(width, 1), 1.0)
    small = frame if scale >= 1.0 else cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    felt = (((hue > 125) & (hue < 176) & (sat > 50) & (val > 42))).astype("uint8") * 255
    felt = cv2.morphologyEx(felt, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)
    contours, _hierarchy = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    min_area = small.shape[0] * small.shape[1] * 0.045
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if aspect < 1.25 or aspect > 2.25:
            continue
        if scale < 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))
        region = clamp_region(
            {
                "left": search_region["left"] + int(x - w * 0.06),
                "top": search_region["top"] + int(y - h * 0.15),
                "width": int(w * 1.10),
                "height": int(h * 1.27),
            },
            search_region,
        )
        if region["width"] < 700 or region["height"] < 500:
            continue
        aspect = region["width"] / max(region["height"], 1)
        if aspect < 1.25:
            target_height = int(round(region["width"] / 1.415))
            if 500 <= target_height < region["height"]:
                region = clamp_region(
                    {
                        "left": region["left"],
                        "top": region["top"],
                        "width": region["width"],
                        "height": target_height,
                    },
                    search_region,
                )
        candidates.append({"region": region, "method": "visual-felt-table", "bonus": 0.0})
    return candidates


def detect_action_button_bbox_candidates(cv2: Any, np: Any, frame: Any, search_region: dict[str, int]) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 70, 55), (12, 255, 255)) | cv2.inRange(hsv, (170, 70, 55), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[dict[str, Any]] = []
    min_area = max(1000.0, width * height * 0.00045)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(55, width * 0.028) or h < max(26, height * 0.018):
            continue
        aspect = w / max(h, 1)
        if aspect < 1.25 or aspect > 4.25:
            continue
        center_y = y + h / 2
        if center_y < height * 0.55:
            continue
        rects.append({"x": x, "y": y, "width": w, "height": h, "area": area, "center_y": center_y})
    if len(rects) < 2:
        return []

    rects.sort(key=lambda item: item["center_y"])
    groups: list[list[dict[str, Any]]] = []
    y_tol = max(28.0, height * 0.035)
    for rect in rects:
        if groups and abs(rect["center_y"] - groups[-1][0]["center_y"]) <= y_tol:
            groups[-1].append(rect)
        else:
            groups.append([rect])

    candidates: list[dict[str, Any]] = []
    for group in groups:
        group = sorted(group, key=lambda item: item["x"])
        if len(group) < 2:
            continue
        left = min(item["x"] for item in group)
        right = max(item["x"] + item["width"] for item in group)
        top = min(item["y"] for item in group)
        bottom = max(item["y"] + item["height"] for item in group)
        span = right - left
        if span < width * 0.16:
            continue
        median_h = float(np.median([item["height"] for item in group]))
        if bottom - top > max(105.0, median_h * 1.9):
            continue

        if len(group) >= 3:
            inferred_width = span / 0.382
            left_from_group_left = left - inferred_width * 0.608
        else:
            inferred_width = span / 0.255
            left_from_group_left = left - inferred_width * 0.735
        left_from_group_right = right - inferred_width * 0.990
        inferred_left = int(round((left_from_group_left + left_from_group_right) / 2))
        inferred_height = int(round(inferred_width / 1.415))
        top_from_button_top = top - inferred_height * 0.910
        top_from_button_bottom = bottom - inferred_height * 0.982
        inferred_top = int(round((top_from_button_top + top_from_button_bottom) / 2))
        region = clamp_region(
            {
                "left": search_region["left"] + inferred_left,
                "top": search_region["top"] + inferred_top,
                "width": int(round(inferred_width)),
                "height": inferred_height,
            },
            search_region,
        )
        aspect = region["width"] / max(region["height"], 1)
        if region["width"] < 700 or region["height"] < 500 or aspect < 1.25 or aspect > 1.65:
            continue
        candidates.append(
            {
                "region": region,
                "method": "action-buttons",
                "bonus": 4.0,
                "anchors": {
                    "buttons": [
                        {
                            "x": int(item["x"] + search_region["left"]),
                            "y": int(item["y"] + search_region["top"]),
                            "width": int(item["width"]),
                            "height": int(item["height"]),
                            "area": round(float(item["area"]), 1),
                        }
                        for item in group
                    ],
                    "button_span": int(span),
                },
            },
        )
    return candidates


def bbox_anchor_score(candidate: dict[str, Any], dealer_box: dict[str, Any] | None, width: int, height: int) -> float:
    anchors = candidate.get("anchors") or {}
    dealer_anchor = anchors.get("dealer_anchor")
    if not dealer_anchor or not dealer_box:
        return 0.0
    center_x = (float(dealer_box.get("x") or 0.0) + float(dealer_box.get("width") or 0.0) / 2.0) / max(float(width), 1.0)
    center_y = (float(dealer_box.get("y") or 0.0) + float(dealer_box.get("height") or 0.0) / 2.0) / max(float(height), 1.0)
    dx = center_x - float(dealer_anchor.get("x") or 0.0)
    dy = center_y - float(dealer_anchor.get("y") or 0.0)
    distance = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - distance / 0.12)


def yellow_pot_anchor_score(cv2: Any, crop: Any) -> float:
    if crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    y1 = int(height * 0.20)
    y2 = int(height * 0.58)
    x1 = int(width * 0.18)
    x2 = int(width * 0.82)
    roi = crop[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 55, 90), (45, 255, 255))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(12.0, width * height * 0.000004):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 6:
            continue
        boxes.append((x1 + x, y1 + y, w, h, area))
    if not boxes:
        return 0.0
    left = min(item[0] for item in boxes)
    top = min(item[1] for item in boxes)
    right = max(item[0] + item[2] for item in boxes)
    bottom = max(item[1] + item[3] for item in boxes)
    total_area = sum(float(item[4]) for item in boxes)
    if total_area < max(45.0, width * height * 0.000025):
        return 0.0
    center_x = (left + right) / 2.0 / max(float(width), 1.0)
    center_y = (top + bottom) / 2.0 / max(float(height), 1.0)
    distance = ((center_x - 0.50) ** 2 + (center_y - 0.385) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 0.22)


def hero_fixed_card_anchor_score(cv2: Any, crop: Any) -> float:
    if crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    ratios = []
    for roi in HERO_CARD_ROIS:
        x1, y1, x2, y2 = scale_roi(roi, width, height)
        slot = crop[y1:y2, x1:x2]
        if slot.size == 0:
            ratios.append(0.0)
            continue
        hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        ratios.append(float(((sat < 86) & (val > 145)).mean()))
    if len(ratios) < 2:
        return 0.0
    strong = [ratio for ratio in ratios if ratio >= 0.18]
    if len(strong) >= 2:
        return min(1.0, sum(ratios[:2]) / 0.85)
    if len(strong) == 1:
        return min(0.45, max(ratios) / 0.55)
    return 0.0


def poker_content_score(cv2: Any, crop: Any) -> float:
    if crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    center = crop[int(height * 0.18) : int(height * 0.84), int(width * 0.08) : int(width * 0.92)]
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    saturated = float(((sat > 55) & (val > 45)).mean())
    magenta_felt = float(((hue > 125) & (hue < 176) & (sat > 45) & (val > 40)).mean())
    white_cards = float(((sat < 75) & (val > 155)).mean())
    return min(1.0, saturated * 1.8 + magenta_felt * 2.2 + white_cards * 1.0)


def poker_table_visibility(cv2: Any, crop: Any) -> tuple[bool, dict[str, float]]:
    if crop.size == 0:
        return False, {"white_center": 1.0, "dark_center": 0.0, "saturated_center": 0.0}
    height, width = crop.shape[:2]
    center = crop[int(height * 0.12) : int(height * 0.88), int(width * 0.05) : int(width * 0.95)]
    if center.size == 0:
        return False, {"white_center": 1.0, "dark_center": 0.0, "saturated_center": 0.0}
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_center = float(((sat < 35) & (val > 205)).mean())
    dark_center = float((val < 95).mean())
    saturated_center = float((sat > 55).mean())
    visible = not (white_center > 0.45 and dark_center < 0.20 and saturated_center < 0.15)
    return visible, {
        "white_center": round(white_center, 4),
        "dark_center": round(dark_center, 4),
        "saturated_center": round(saturated_center, 4),
    }


def detect_titlebar_row_candidates(np: Any, value: Any, search_region: dict[str, int]) -> list[dict[str, Any]]:
    height, width = value.shape[:2]
    mask = value < 75
    rows: list[tuple[int, int, int, int]] = []
    for y in range(height):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        diffs = np.diff(xs)
        breaks = np.where(diffs > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(xs) - 1]
        best_width = 0
        best_left = 0
        best_right = 0
        for start, end in zip(starts, ends):
            left = int(xs[start])
            right = int(xs[end])
            run_width = right - left + 1
            if run_width > best_width:
                best_width = run_width
                best_left = left
                best_right = right
        if best_width >= width * 0.35:
            rows.append((y, best_left, best_right, best_width))

    groups: list[list[tuple[int, int, int, int]]] = []
    for row in rows:
        if groups and row[0] == groups[-1][-1][0] + 1:
            groups[-1].append(row)
        else:
            groups.append([row])

    candidates: list[dict[str, Any]] = []
    for group in groups:
        top = group[0][0]
        bottom = group[-1][0]
        bar_height = bottom - top + 1
        if bar_height < 18 or bar_height > 85:
            continue
        left = int(np.median([row[1] for row in group]))
        right = int(np.median([row[2] for row in group]))
        bar_width = right - left + 1
        if bar_width < width * 0.38:
            continue
        estimated_height = int(round(bar_width / 1.415))
        if estimated_height < height * 0.35:
            continue
        region = clamp_region(
            {
                "left": search_region["left"] + left,
                "top": search_region["top"] + top,
                "width": bar_width,
                "height": estimated_height,
            },
            search_region,
        )
        candidates.append({"region": region, "method": "visual-titlebar-row", "bonus": 2.0})
    return candidates


def clamp_region(region: dict[str, int], bounds: dict[str, int]) -> dict[str, int]:
    left = max(region["left"], bounds["left"])
    top = max(region["top"], bounds["top"])
    right = min(region["left"] + region["width"], bounds["left"] + bounds["width"])
    bottom = min(region["top"] + region["height"], bounds["top"] + bounds["height"])
    return {"left": left, "top": top, "width": max(1, right - left), "height": max(1, bottom - top)}


def dedupe_regions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        region = candidate["region"]
        duplicate = False
        for existing in unique:
            other = existing["region"]
            if abs(region["left"] - other["left"]) < 12 and abs(region["top"] - other["top"]) < 12:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def bbox_changed(current: dict[str, int], new_region: dict[str, int]) -> bool:
    deltas = [
        abs(current["left"] - new_region["left"]),
        abs(current["top"] - new_region["top"]),
        abs(current["width"] - new_region["width"]),
        abs(current["height"] - new_region["height"]),
    ]
    return max(deltas) >= 10


def region_area(region: dict[str, Any]) -> float:
    return max(float(region.get("width") or 0) * float(region.get("height") or 0), 0.0)


def should_accept_bbox_refresh(
    current: dict[str, int],
    new_region: dict[str, int],
    search_region: dict[str, int],
    detection: dict[str, Any],
) -> tuple[bool, str | None]:
    if detection.get("method") == "native-window":
        return True, None

    search_width = max(float(search_region.get("width") or 1), 1.0)
    search_height = max(float(search_region.get("height") or 1), 1.0)
    current_area = max(float(current["width"] * current["height"]), 1.0)
    new_area = max(float(new_region["width"] * new_region["height"]), 1.0)
    search_area = max(search_width * search_height, 1.0)
    current_coverage = current_area / search_area
    new_coverage = new_area / search_area
    new_height_ratio = new_region["height"] / search_height
    new_width_ratio = new_region["width"] / search_width
    method = str(detection.get("method") or "")
    new_aspect = new_region["width"] / max(new_region["height"], 1)
    if method in {"action-buttons", "dealer-button-anchor"}:
        if current_coverage > 0.40 and new_area < current_area * 0.86:
            return False, "strong_anchor_shrink"
        preserves_cards, card_reason = auto_bbox_preserves_hero_cards(new_region, search_region)
        if not preserves_cards:
            return False, card_reason
        if new_coverage < 0.08 or new_coverage > 0.88 or new_aspect < 1.25 or new_aspect > 1.65:
            return False, "bad_strong_anchor_bbox"
        return True, None
    preserves_cards, card_reason = auto_bbox_preserves_hero_cards(new_region, search_region)
    if not preserves_cards:
        return False, card_reason
    if method == "visual-felt-table":
        top_delta = new_region["top"] - current["top"]
        height_ratio_to_current = new_region["height"] / max(float(current["height"]), 1.0)
        width_ratio_to_current = new_region["width"] / max(float(current["width"]), 1.0)
        if top_delta > current["height"] * 0.10 and height_ratio_to_current < 0.82 and width_ratio_to_current < 0.96:
            return False, "felt_table_shrink"

    if new_coverage > 0.56 and new_coverage > current_coverage * 1.25:
        return False, "coverage_jump"
    if new_height_ratio > 0.88 and new_region["height"] > current["height"] * 1.12:
        return False, "too_tall"
    if new_width_ratio > 0.82 and new_region["width"] > current["width"] * 1.18:
        return False, "too_wide"
    area_ratio = new_area / current_area
    if area_ratio < 0.62 or area_ratio > 1.55:
        return False, "area_jump"

    current_center_x = current["left"] + current["width"] / 2
    current_center_y = current["top"] + current["height"] / 2
    new_center_x = new_region["left"] + new_region["width"] / 2
    new_center_y = new_region["top"] + new_region["height"] / 2
    if abs(new_center_x - current_center_x) > search_width * 0.22:
        return False, "center_x_jump"
    if abs(new_center_y - current_center_y) > search_height * 0.18:
        return False, "center_y_jump"

    top_jump = new_region["top"] - current["top"]
    if top_jump > search_height * 0.12:
        return False, "top_moved_down"

    if new_aspect < 1.05 or new_aspect > 2.15:
        return False, "bad_aspect"

    content_score = float(detection.get("content_score") or 0.0)
    dealer_confidence = float(detection.get("dealer_confidence") or 0.0)
    if method.startswith("visual-titlebar") and content_score < 0.24 and dealer_confidence < 0.70:
        return False, "weak_titlebar_candidate"

    return True, None


def auto_bbox_preserves_hero_cards(
    region: dict[str, int], search_region: dict[str, int]
) -> tuple[bool, str | None]:
    """Reject automatic inner crops that cut the bottom Hero card area."""
    search_width = max(float(search_region.get("width") or 1), 1.0)
    search_height = max(float(search_region.get("height") or 1), 1.0)
    width_ratio = float(region.get("width") or 0) / search_width
    bottom_ratio = (
        float(region.get("top") or 0)
        + float(region.get("height") or 0)
        - float(search_region.get("top") or 0)
    ) / search_height
    if width_ratio < 0.68:
        return False, "inner_table_too_narrow_for_cards"
    if bottom_ratio < 0.88:
        return False, "inner_table_cuts_hero_cards"
    return True, None


def format_screen_event_line(state: dict[str, Any]) -> str:
    source = state.get("source") or {}
    event = state.get("event") or {}
    timestamp = source.get("timestamp_sec")
    index = event.get("index", "?")
    analysis_ms = source.get("analysis_ms")
    prefix = f"[{format_seconds(timestamp)} #{index}]"
    if not state.get("ok"):
        return f"{prefix} ERROR {state.get('error') or 'unknown'}"

    table = state.get("table") or {}
    hero = state.get("hero") or {}
    street = table.get("street") or "-"
    dealer = table.get("dealer_seat") or "-"
    dealer_position = table.get("dealer_position") or "-"
    pot = format_bb(table.get("pot_bb"))
    to_call = format_bb(table.get("to_call_bb"))
    hero_position = hero.get("position") or "-"
    hero_status = hero.get("status") or "-"
    hero_cards = "".join(str(card) for card in (hero.get("cards") or [])) or "-"
    board = "".join(str(card) for card in (table.get("board") or [])) or "-"
    bets = format_bets(state.get("bets") or [])
    advice = format_advice(state.get("gto_advice") or {})
    hero_turn = format_hero_turn(state.get("hero_turn") or {})
    hero_turn_reason = (state.get("hero_turn") or {}).get("reason") or "-"
    issue = format_state_issue(state)
    ocr_mode = (state.get("source") or {}).get("ocr_mode") or "-"
    seats = format_seats(state.get("seats") or [])
    fields = [
        f"street={street}",
        f"pot={pot}",
        f"to_call={to_call}",
        f"dealer={dealer}/{dealer_position}",
        f"hero={hero_position} {hero_cards} {hero_status}",
        f"turn={hero_turn}",
        f"turn_reason={hero_turn_reason}",
        f"issue={issue}",
        f"ocr={ocr_mode}",
        f"board={board}",
        f"bets={bets}",
        f"seats={seats}",
        f"advice={advice}",
    ]
    if analysis_ms is not None:
        fields.append(f"latency={analysis_ms}ms")
    return f"{prefix} " + " | ".join(fields)


def format_screen_advice_line(state: dict[str, Any]) -> str:
    source = state.get("source") or {}
    event = state.get("event") or {}
    prefix = f"[{format_seconds(source.get('timestamp_sec'))} #{event.get('index', '?')}]"
    if not state.get("ok"):
        return f"{prefix} ERROR | {state.get('error') or 'unknown'}"

    table = state.get("table") or {}
    hero = state.get("hero") or {}
    advice = state.get("gto_advice") or {}
    hero_turn = state.get("hero_turn") or {}
    cards = "".join(str(card) for card in (hero.get("cards") or [])) or "-"
    board = "".join(str(card) for card in (table.get("board") or [])) or "-"
    position = hero.get("gto_position") or hero.get("position") or "-"
    context = (
        f"hero={position} {cards} | {table.get('street') or '-'} "
        f"pot={format_bb(table.get('pot_bb'))} call={format_bb(table.get('to_call_bb'))} "
        f"board={board}"
    )
    preflop_context = format_live_preflop_context(state, advice)
    if preflop_context:
        context = f"{context} | {preflop_context}"
    analysis_ms = source.get("analysis_ms")
    latency = f" | {analysis_ms}ms" if analysis_ms is not None else ""
    if advice.get("ready"):
        summary = str(advice.get("summary") or advice.get("action") or "ready")
        return f"{prefix} ADVICE | {summary} | {context}{latency}"

    reason = str(advice.get("reason") or format_state_issue(state))
    status = "TURN" if hero_turn.get("is_turn") else "WATCH"
    return f"{prefix} {status} | wait={reason} | {context}{latency}"


def screen_advice_console_signature(state: dict[str, Any]) -> str:
    if not state.get("ok"):
        return json.dumps(
            ["error", str(state.get("error") or "unknown")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    advice = state.get("gto_advice") or {}
    hero_turn = state.get("hero_turn") or {}
    payload = [
        bool(advice.get("ready")),
        str(advice.get("summary") or advice.get("action") or ""),
        str(advice.get("reason") or ""),
        bool(hero_turn.get("is_turn")),
        str(hero.get("gto_position") or hero.get("position") or ""),
        [str(card) for card in (hero.get("cards") or [])],
        str(table.get("street") or ""),
        [str(card) for card in (table.get("board") or [])],
    ]
    if str(table.get("street") or "").lower() == "preflop":
        context = advice.get("preflop_context") or {}
        tracker = state.get("preflop_tracker") or {}
        payload.extend(
            [
                str(context.get("status") or ""),
                str(context.get("scenario") or ""),
                str(context.get("source") or ""),
                list(context.get("actions_before_hero") or []),
                format_position_bets(state.get("seats") or []),
                str(tracker.get("reason") or "") if isinstance(tracker, dict) else "",
            ]
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def format_seconds(value: Any) -> str:
    try:
        return f"{float(value):08.3f}s"
    except (TypeError, ValueError):
        return "--------s"


def format_bb(value: Any) -> str:
    if value is None:
        return "-"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{amount:.2f}".rstrip("0").rstrip(".")
    return f"{text}BB"


def format_bets(bets: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for bet in bets:
        seat = bet.get("seat") or bet.get("seat_name") or bet.get("player") or "?"
        amount = bet.get("amount_bb", bet.get("bet_bb", bet.get("bb")))
        parts.append(f"{seat}:{format_bb(amount)}")
    return ", ".join(parts[:8]) if parts else "-"


def format_live_preflop_context(state: dict[str, Any], advice: dict[str, Any]) -> str:
    table = state.get("table") or {}
    if str(table.get("street") or "").lower() != "preflop":
        return ""

    hero = state.get("hero") or {}
    context = advice.get("preflop_context") or {}
    raw_position = context.get("raw_position") or hero.get("position") or "-"
    solver_position = context.get("solver_position") or hero.get("gto_position") or raw_position
    position = raw_position if raw_position == solver_position else f"{raw_position}->{solver_position}"
    action_order = context.get("preflop_action_order") or hero.get("preflop_action_order")
    status = context.get("scenario") or context.get("status") or "unconfirmed"
    actions = list(context.get("actions_before_hero") or [])
    history = format_preflop_history(actions) if actions else "unconfirmed"
    tracker = state.get("preflop_tracker") or {}
    tracker_reason = tracker.get("reason") if isinstance(tracker, dict) else None
    visible_bets = format_position_bets(state.get("seats") or [])
    parts = [f"PF={status}", f"pos={position}"]
    if action_order is not None:
        parts.append(f"order=#{action_order}")
    parts.append(f"history={history}")
    if tracker_reason and not actions:
        parts.append(f"tracker={tracker_reason}")
    parts.append(f"visible={visible_bets}")
    return " ".join(parts)


def format_preflop_history(actions: list[dict[str, Any]]) -> str:
    labels = {
        "fold": "F",
        "call": "C",
        "limp": "L",
        "raise": "R",
        "3bet": "3B",
        "4bet": "4B",
        "5bet": "5B",
        "all_in": "AI",
    }
    parts: list[str] = []
    for event in actions[:9]:
        action = str(event.get("action") or "")
        if action in {"hero_to_act", "to_act", "hero_turn", "current_hero_turn"}:
            break
        actor = event.get("position") or event.get("seat") or "?"
        label = labels.get(action, action.upper() or "?")
        amount = event.get("amount_bb")
        amount_text = "" if amount is None else format_bb(amount)
        parts.append(f"{actor} {label}{amount_text}")
    return ",".join(parts) if parts else "unconfirmed"


def format_position_bets(seats: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for seat in seats:
        amount = seat.get("bet_bb")
        if amount is None:
            continue
        position = seat.get("position") or seat.get("seat") or "?"
        parts.append(f"{position}:{format_bb(amount)}")
    return ",".join(parts[:8]) if parts else "-"


def format_seats(seats: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for seat in seats[:8]:
        name = str(seat.get("seat") or "?").replace("bottom_hero", "hero")
        position = seat.get("position") or "-"
        status = compact_status(seat.get("status"))
        bet = format_bb(seat.get("bet_bb")) if seat.get("bet_bb") is not None else "--"
        cards = "C" if seat.get("has_cards") else "-"
        parts.append(f"{name}:{position}/{status}/{cards}/{bet}")
    return ", ".join(parts) if parts else "-"


def compact_status(status: Any) -> str:
    text = str(status or "-")
    mapping = {
        "active_or_showdown": "ACT",
        "folded_or_empty": "OUT",
        "folded": "FOLD",
        "active": "ACT",
    }
    return mapping.get(text, text[:6].upper())


def format_hero_turn(hero_turn: dict[str, Any]) -> str:
    if not hero_turn:
        return "no"
    if hero_turn.get("is_turn"):
        confidence = hero_turn.get("confidence")
        if confidence is None:
            return "YES"
        try:
            return f"YES({float(confidence):.2f})"
        except (TypeError, ValueError):
            return "YES"
    return "no"


def format_state_issue(state: dict[str, Any]) -> str:
    problem = problem_reason(state)
    if problem:
        return problem
    hero = state.get("hero") or {}
    hero_cards = [card for card in hero.get("cards", []) if card]
    if hero_cards and (len(hero_cards) != 2 or any("?" in str(card) for card in hero_cards)):
        return "hero_cards_incomplete"
    if hero.get("has_cards") and not hero_cards:
        return "hero_cards_missing"
    source = state.get("source") or {}
    if source.get("ocr_mode") == "action_only_skipped":
        return "ocr_skipped_until_turn"
    advice = state.get("gto_advice") or {}
    reason = advice.get("reason")
    if reason == "hero_action_controls_not_visible":
        return "waiting_for_turn"
    if reason:
        return str(reason)
    return "ok"


def format_advice(advice: dict[str, Any]) -> str:
    if not advice:
        return "-"
    if advice.get("ready"):
        return str(advice.get("summary") or advice.get("action") or "ready")
    reason = advice.get("reason")
    return f"wait({reason})" if reason else "wait"


def source_info(region: dict[str, int], monitor: int) -> dict[str, Any]:
    return {
        "kind": "screen",
        "path": f"screen://monitor/{monitor}",
        "monitor": monitor,
        "region": dict(region),
        "width": region["width"],
        "height": region["height"],
    }


def parse_bbox(text: str | None) -> tuple[int, int, int, int] | None:
    return parse_bbox_values(text)


def manual_hero_live_bbox_arg(calibration_output_dir: Path, bbox_text: str) -> str:
    manual_bbox_path = calibration_output_dir / "bbox.json"
    if manual_bbox_path.is_file():
        return f'--bbox-file "{manual_bbox_path}"'
    reviewed_bbox_path = calibration_output_dir / "analysis_bbox.json"
    if reviewed_bbox_path.is_file():
        return f'--bbox-file "{reviewed_bbox_path}"'
    return f'--bbox "{bbox_text}"'


def build_reviewed_bbox_commands(
    *,
    analysis_bbox_path: Path,
    calibration_output_dir: Path,
    hero_name: str | None,
    effective_stack_bb: float,
    villain_profile: str,
    min_confidence: float,
    ocr_scale: float,
    dealer_refresh_frames: int,
) -> dict[str, str]:
    """Build commands from the full manual window and reviewed inner layout."""
    manual_bbox_path = calibration_output_dir / "bbox.json"
    # The CLI restores the reviewed inner table relative to this full window.
    # Every generated command must keep this outer frame so the action panel is
    # never replaced by an old inner-table crop.
    bbox_arg = f'--bbox-file "{manual_bbox_path}"'
    hero_arg = f' --hero-name "{hero_name}"' if hero_name else ""
    shared = (
        f'{bbox_arg} --lock-layout{hero_arg} --trigger frame --every 1 '
        f'--with-advice --effective-stack {float(effective_stack_bb):g} '
        f'--villain "{villain_profile}" --min-confidence {float(min_confidence):g} '
        f'--ocr-scale {float(ocr_scale):g} --dealer-refresh-frames {int(dealer_refresh_frames)}'
    )
    live = f'{active_python_gto_command()} screen-cv {shared} --output-dir "video_frames\\screen_live" --format text'
    return {
        "preflight": (
            f'{active_python_gto_command()} screen-cv {shared} --preflight-once --save-frames --save-annotated '
            f'--output-dir "video_frames\\screen_preflight" --format text'
        ),
        "live": live,
        "overlay": live.replace(" --format text", " --show-overlay --format text"),
        "pick_hero": (
            f'{active_python_gto_command()} screen-cv {bbox_arg} --pick-hero-cards '
            f'--output-dir "{calibration_output_dir}" --format text'
        ),
    }


def write_reviewed_bbox_commands(output_dir: Path, commands: dict[str, str]) -> dict[str, str]:
    names = {
        "preflight": "run_reviewed_preflight_command.txt",
        "live": "run_reviewed_live_command.txt",
        "overlay": "run_reviewed_overlay_command.txt",
        "pick_hero": "run_reviewed_pick_hero_cards_command.txt",
    }
    files: dict[str, str] = {}
    for key, filename in names.items():
        path = Path(output_dir) / filename
        path.write_text(commands[key] + "\n", encoding="utf-8-sig")
        files[f"{key}_command"] = str(path)
    return files


def format_screen_summary(payload: dict[str, Any], limit: int = 12) -> str:
    if payload.get("analysis_bbox_text"):
        files = payload.get("files") or {}
        commands = payload.get("commands") or {}
        return "\n".join(
            [
                f"Reviewed analysis bbox: {payload['analysis_bbox_text']}",
                f"Review result: {payload.get('review_source')} adjustments={payload.get('manual_adjustments', 0)}",
                f"Saved locked bbox: {files.get('analysis_bbox', '-')}",
                f"Saved bbox review: {files.get('bbox_review', '-')}",
                f"Saved layout preview: {files.get('layout_preview', '-')}",
                f"Saved overlay command: {files.get('overlay_command', '-')}",
                f"Saved live command: {files.get('live_command', '-')}",
                f"Saved hero-card picker: {files.get('pick_hero_command', '-')}",
                "Run overlay first:",
                str(commands.get("overlay") or ""),
            ]
        )
    if payload.get("bbox_text"):
        return "\n".join(
            [
                f"Selected bbox: {payload['bbox_text']}",
                f"Saved bbox: {payload['files']['bbox']}",
                f"Saved health command: {payload['files'].get('health_command', '-')}",
                f"Saved command: {payload['files']['command']}",
                f"Saved overlay command: {payload['files'].get('overlay_command', '-')}",
                f"Saved hero-card picker: {payload['files'].get('pick_hero_command', '-')}",
                f"Saved reviewed-inner-bbox command: {payload['files'].get('review_bbox_command', '-')}",
                "Next calibration command:",
                payload.get("review_bbox_command", ""),
                "Health command:",
                payload.get("health_command", ""),
                "Live command:",
                payload["command"],
            ]
        )
    if payload.get("hero_cards_file") and payload.get("command"):
        return "\n".join(
            [
                f"Saved hero card ROIs: {payload['hero_cards_file']}",
                f"Saved preview: {payload.get('preview') or '-'}",
                f"Saved command: {(payload.get('files') or {}).get('command') or '-'}",
                *(f"WARNING: {warning}" for warning in payload.get("warnings") or []),
                "Live overlay command:",
                str(payload["command"]),
            ]
        )
    if payload.get("snapshot"):
        return "\n".join(
            [
                f"Screen snapshot: {payload['snapshot']}",
                f"Region: {payload['source']['region']}",
                payload.get("hint", ""),
            ]
        )
    return format_realtime_summary(payload, limit=limit)
