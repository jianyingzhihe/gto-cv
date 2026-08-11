from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .live_vision import build_realtime_state, stabilize_hero_cards
from .screen_vision import bbox_changed, detect_auto_bbox, poker_table_visibility, should_accept_bbox_refresh
from .video_vision import (
    analyze_video_frame,
    build_layout_profile,
    choose_template,
    detect_action_controls,
    detect_auto_hero_cards,
    layout_profile_is_strong,
    layout_profile_quality,
    load_cv,
    load_ocr,
    sample_times,
)

NON_PROBLEM_FRAME_CLASSES = {"complete", "empty_or_no_hand", "obstructed_animation", "table_occluded"}


def find_latest_video(video_dir: Path = Path("video_frames")) -> Path:
    candidates = find_root_videos(video_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_root_videos(video_dir: Path = Path("video_frames")) -> list[Path]:
    candidates = [path for path in Path(video_dir).glob("*.mp4") if path.is_file()]
    if not candidates:
        raise ValueError(f"no root mp4 files found in {video_dir}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name.lower()))


def validate_cv_videos(
    video_paths: list[Path] | None = None,
    video_dir: Path = Path("video_frames"),
    output_dir: Path = Path("video_frames") / "cv_validation_all",
    template_path: Path | None = None,
    seat_count: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 30.0,
    max_frames: int | None = None,
    min_confidence: float = 0.35,
    auto_bbox_refresh_sec: float = 300.0,
    use_ocr: bool = False,
    ocr_scale: float = 0.65,
    ocr_action_only: bool = False,
    save_problem_frames: bool = True,
    lock_layout: bool = True,
    dealer_refresh_frames: int = 4,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    videos = [Path(path) for path in (video_paths or find_root_videos(video_dir))]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for index, video_path in enumerate(videos, start=1):
        child_dir = output_dir / f"{index:02d}_{safe_video_slug(video_path)}"
        result = validate_cv_video(
            video_path=video_path,
            output_dir=child_dir,
            template_path=template_path,
            seat_count=seat_count,
            start_sec=start_sec,
            end_sec=end_sec,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
            auto_bbox_refresh_sec=auto_bbox_refresh_sec,
            use_ocr=use_ocr,
            ocr_scale=ocr_scale,
            ocr_action_only=ocr_action_only,
            save_problem_frames=save_problem_frames,
            lock_layout=lock_layout,
            dealer_refresh_frames=dealer_refresh_frames,
        )
        results.append(result)

    counts = merge_counts(result.get("counts") or {} for result in results)
    problem_count = sum(int(result.get("real_problem_count") or 0) for result in results)
    board_bad_count = sum(int(result.get("board_bad_count") or 0) for result in results)
    total_frames = sum(int(result.get("sample", {}).get("frame_count") or 0) for result in results)
    timing_values = [
        float(row["analysis_ms"])
        for result in results
        for row in result.get("rows") or []
        if row.get("ok") and row.get("analysis_ms") is not None
    ]
    all_rows = [row for result in results for row in result.get("rows") or []]
    summary = {
        "ok": True,
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "video_count": len(results),
        "sample": {
            "every_sec": every_sec,
            "max_frames": max_frames,
            "frame_count": total_frames,
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "ocr": {
            "enabled": bool(use_ocr),
            "scale": ocr_scale if use_ocr else None,
            "action_only": bool(ocr_action_only and use_ocr),
            "mode_counts": merge_counts(result.get("ocr_mode_counts") or {} for result in results),
        },
        "counts": counts,
        "real_problem_count": problem_count,
        "board_bad_count": board_bad_count,
        "card_health": summarize_card_health(all_rows),
        "timing_ms": timing_stats(timing_values),
        "cv_timing_ms": component_timing_stats(all_rows),
        "files": {
            "summary": str(output_dir / "cv_validation_all_summary.json"),
            "report_md": str(output_dir / "cv_validation_all_report.md"),
        },
        "videos": [
            {
                "video": result.get("video"),
                "frames": result.get("sample", {}).get("frame_count"),
                "counts": result.get("counts") or {},
                "real_problem_count": result.get("real_problem_count"),
                "board_bad_count": result.get("board_bad_count"),
                "card_health": result.get("card_health") or {},
                "timing_ms": result.get("timing_ms") or {},
                "summary": (result.get("files") or {}).get("summary"),
                "report_md": (result.get("files") or {}).get("report_md"),
                "problem_frames": (result.get("files") or {}).get("problem_frames"),
            }
            for result in results
        ],
    }
    (output_dir / "cv_validation_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cv_validation_all_report.md").write_text(format_validation_suite_markdown(summary), encoding="utf-8")
    return summary


def validate_cv_video(
    video_path: Path,
    output_dir: Path,
    template_path: Path | None = None,
    seat_count: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 30.0,
    max_frames: int | None = None,
    min_confidence: float = 0.35,
    auto_bbox_refresh_sec: float = 300.0,
    use_ocr: bool = False,
    ocr_scale: float = 0.65,
    ocr_action_only: bool = False,
    save_problem_frames: bool = True,
    lock_layout: bool = True,
    dealer_refresh_frames: int = 4,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    cv2, np = load_cv()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    problem_dir = output_dir / "problem_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_problem_frames:
        problem_dir.mkdir(parents=True, exist_ok=True)

    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps else 0.0
    start_sec = 0.0 if start_sec is None else float(start_sec)
    end_sec = duration_sec if end_sec is None else float(end_sec)
    times = sample_times(start_sec, end_sec, every_sec, max_frames)
    ocr = load_ocr() if use_ocr else None

    region: dict[str, int] | None = None
    last_auto_bbox_sec = float("-inf")
    dealer_button_cache: dict[str, Any] | None = None
    last_dealer_refresh_sample = -10**9
    layout_profile: dict[str, Any] | None = None
    layout_locked = False
    hero_card_cache: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    events_jsonl = output_dir / "events.jsonl"

    with events_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for sample_index, timestamp in enumerate(times):
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                row = {
                    "time": round(float(timestamp), 3),
                    "frame_index": frame_index,
                    "ok": False,
                    "class": "read_failed",
                    "error": "could not read frame",
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue

            search_region = {"left": 0, "top": 0, "width": int(frame.shape[1]), "height": int(frame.shape[0])}
            refresh_rejected = None
            refresh_accepted = False
            detection: dict[str, Any] | None = None
            warmup_bbox_refresh = bool(lock_layout and layout_locked and sample_index <= 6)
            should_refresh_bbox = (
                region is None
                or (lock_layout and not layout_locked)
                or warmup_bbox_refresh
                or (float(auto_bbox_refresh_sec) > 0 and timestamp - last_auto_bbox_sec >= max(1.0, auto_bbox_refresh_sec))
            )
            if should_refresh_bbox:
                detection = detect_auto_bbox(
                    cv2,
                    np,
                    frame,
                    search_region,
                    template,
                    min_confidence,
                    allow_native_window=False,
                )
                last_auto_bbox_sec = float(timestamp)
                if detection is not None:
                    if region is None or not bbox_changed(region, detection["region"]):
                        region = detection["region"]
                        dealer_button_cache = None
                        if not layout_locked:
                            layout_profile = None
                        refresh_accepted = True
                    else:
                        accepted, reason = should_accept_bbox_refresh(region, detection["region"], search_region, detection)
                        if accepted:
                            region = detection["region"]
                            dealer_button_cache = None
                            layout_profile = None
                            if lock_layout:
                                layout_locked = False
                            refresh_accepted = True
                        else:
                            refresh_rejected = reason

            if region is None:
                row = {
                    "time": round(float(timestamp), 3),
                    "frame_index": frame_index,
                    "ok": False,
                    "class": "auto_bbox_failed",
                    "error": "auto bbox could not find table",
                }
                if save_problem_frames:
                    row["problem_frame"] = save_problem_frame(cv2, problem_dir, frame, timestamp, row["class"])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue

            x, y, w, h = region["left"], region["top"], region["width"], region["height"]
            crop = frame[y : y + h, x : x + w]
            table_visible, visibility = poker_table_visibility(cv2, crop)
            if not table_visible:
                row = {
                    "time": round(float(timestamp), 3),
                    "frame_index": frame_index,
                    "ok": False,
                    "class": "table_occluded",
                    "region": dict(region),
                    "refresh_rejected": refresh_rejected,
                    "table_visibility": visibility,
                }
                if save_problem_frames:
                    row["problem_frame"] = save_problem_frame(cv2, problem_dir, crop, timestamp, row["class"])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue
            if lock_layout and not layout_locked:
                candidate_profile = build_layout_profile(crop, [], hero_name=None)
                if layout_profile_is_strong(candidate_profile):
                    layout_profile = candidate_profile
                    layout_locked = True
            active_layout_profile = layout_profile if layout_locked else None
            should_refresh_dealer = (
                dealer_button_cache is None
                or dealer_refresh_frames <= 1
                or sample_index - last_dealer_refresh_sample >= max(1, int(dealer_refresh_frames))
            )
            dealer_button_hint = None if should_refresh_dealer else dealer_button_cache
            used_dealer_cache = dealer_button_hint is not None
            frame_ocr = ocr
            ocr_mode = "full" if ocr is not None else "disabled"
            if ocr_action_only and ocr is not None:
                quick_action_controls = detect_action_controls(crop, [])
                if quick_action_controls.get("visible"):
                    frame_ocr = ocr
                    ocr_mode = "action_only_used"
                else:
                    frame_ocr = None
                    ocr_mode = "action_only_skipped"
            analysis_started = time.perf_counter()
            try:
                frame_result = analyze_video_frame(
                    crop,
                    template,
                    seat_count=seat_count,
                    min_confidence=min_confidence,
                    ocr=frame_ocr,
                    dealer_button_hint=dealer_button_hint,
                    ocr_scale=ocr_scale,
                    layout_profile=active_layout_profile,
                )
                dealer_button_cache = frame_result.get("dealer_button") or dealer_button_cache
                if not used_dealer_cache:
                    last_dealer_refresh_sample = sample_index
            except Exception as error:
                if dealer_button_cache is None or "dealer" not in str(error).lower():
                    row = {
                        "time": round(float(timestamp), 3),
                        "frame_index": frame_index,
                        "ok": False,
                        "class": "analysis_error",
                        "error": str(error),
                        "region": dict(region),
                        "refresh_rejected": refresh_rejected,
                    }
                    if save_problem_frames:
                        row["problem_frame"] = save_problem_frame(cv2, problem_dir, crop, timestamp, row["class"])
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    continue
                frame_result = analyze_video_frame(
                    crop,
                    template,
                    seat_count=seat_count,
                    min_confidence=min_confidence,
                    ocr=frame_ocr,
                    dealer_button_hint=dealer_button_cache,
                    ocr_scale=ocr_scale,
                    layout_profile=active_layout_profile,
                )
                used_dealer_cache = True

            elapsed_ms = (time.perf_counter() - analysis_started) * 1000
            state = build_realtime_state(
                frame_result,
                video_path=video_path,
                timestamp_sec=round(float(timestamp), 3),
                frame_index=frame_index,
                sample_index=sample_index,
            )
            hero_card_cache = stabilize_hero_cards(state, hero_card_cache)
            hero_cards = state.get("hero", {}).get("cards") or []
            board = state.get("table", {}).get("board") or []
            row_class = classify_frame(crop, hero_cards)
            board_bad = any("?" in str(card) for card in board)
            row = {
                "time": round(float(timestamp), 3),
                "frame_index": frame_index,
                "ok": True,
                "class": row_class,
                "street": state.get("table", {}).get("street"),
                "dealer": state.get("table", {}).get("dealer_seat"),
                "dealer_position": state.get("table", {}).get("dealer_position"),
                "hero_position": state.get("hero", {}).get("position"),
                "hero_cards": hero_cards,
                "board": board,
                "pot_bb": state.get("table", {}).get("pot_bb"),
                "to_call_bb": state.get("table", {}).get("to_call_bb"),
                "bets": state.get("bets") or [],
                "hero_turn": state.get("hero_turn") or {},
                "region": dict(region),
                "auto_bbox": {
                    "method": detection.get("method") if detection else None,
                    "score": detection.get("score") if detection else None,
                    "dealer_confidence": detection.get("dealer_confidence") if detection else None,
                },
                "used_dealer_cache": used_dealer_cache,
                "refresh_accepted": refresh_accepted,
                "refresh_rejected": refresh_rejected,
                "layout_locked": layout_locked,
                "layout_quality": layout_profile_quality(layout_profile),
                "analysis_ms": round(float(elapsed_ms), 1),
                "cv_timing_ms": frame_result.get("timing_ms") or {},
                "ocr_item_count": frame_result.get("ocr_item_count"),
                "cards_hint_used": frame_result.get("cards_hint_used"),
                "ocr_mode": ocr_mode,
                "board_bad": board_bad,
            }
            row["card_issues"] = card_health_issues(row)
            if save_problem_frames and (row_class in {"incomplete", "missed_visible_cards"} or board_bad or "duplicate_cards" in row["card_issues"]):
                row["problem_frame"] = save_problem_frame(cv2, problem_dir, crop, timestamp, row_class)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    cap.release()
    elapsed_sec = time.perf_counter() - started_at
    counts = count_by_class(rows)
    timing = timing_stats([row["analysis_ms"] for row in rows if row.get("ok") and row.get("analysis_ms") is not None])
    summary = {
        "ok": True,
        "video": str(video_path),
        "template": str(template_path),
        "output_dir": str(output_dir),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(float(duration_sec), 3),
        },
        "sample": {
            "start_sec": round(float(start_sec), 3),
            "end_sec": round(float(end_sec), 3),
            "every_sec": every_sec,
            "frame_count": len(rows),
            "wall_time_sec": round(float(elapsed_sec), 3),
        },
        "ocr_enabled": bool(use_ocr and ocr is not None),
        "ocr_scale": ocr_scale if use_ocr and ocr is not None else None,
        "ocr_action_only": bool(ocr_action_only and use_ocr and ocr is not None),
        "ocr_mode_counts": count_values(row.get("ocr_mode") for row in rows if row.get("ok")),
        "auto_bbox_refresh_sec": auto_bbox_refresh_sec,
        "counts": counts,
        "real_problem_count": real_problem_count(rows),
        "board_bad_count": sum(1 for row in rows if row.get("board_bad")),
        "card_health": summarize_card_health(rows),
        "timing_ms": timing,
        "cv_timing_ms": component_timing_stats(rows),
        "files": {
            "summary": str(output_dir / "cv_validation_summary.json"),
            "events_jsonl": str(events_jsonl),
            "report_md": str(output_dir / "cv_validation_report.md"),
            "problem_frames": str(problem_dir) if save_problem_frames else "",
        },
        "rows": rows,
    }
    (output_dir / "cv_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cv_validation_report.md").write_text(format_validation_markdown(summary), encoding="utf-8")
    return summary


def safe_video_slug(path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem).strip("_")
    stem = stem[:80] or "video"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def merge_counts(counts_iter: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in counts_iter:
        for key, value in counts.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


def real_problem_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if row.get("board_bad"):
            count += 1
            continue
        if str(row.get("class") or "unknown") not in NON_PROBLEM_FRAME_CLASSES:
            count += 1
    return count


def card_health_issues(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    row_class = str(row.get("class") or "")
    hero_cards = [str(card) for card in (row.get("hero_cards") or []) if card]
    board_cards = [str(card) for card in (row.get("board") or []) if card]
    hero_visible = row_class in {"complete", "incomplete", "missed_visible_cards", "obstructed_animation"} or bool(hero_cards)
    hero_complete = len(hero_cards) == 2 and all(card_is_complete(card) for card in hero_cards)
    hero_turn = row.get("hero_turn") or {}
    decision_relevant = row_class != "empty_or_no_hand" and (
        bool(hero_turn.get("is_turn")) or row_class in {"complete", "incomplete", "missed_visible_cards"}
    )

    if row_class == "missed_visible_cards":
        issues.append("hero_visible_cards_missed")
    if hero_visible and len(hero_cards) < 2 and row_class != "obstructed_animation":
        issues.append("hero_card_count_incomplete")
    if hero_cards and any(not card_is_complete(card) for card in hero_cards):
        issues.append("hero_card_unknown")
    if hero_turn.get("is_turn") and row_class != "empty_or_no_hand" and not hero_complete:
        issues.append("hero_turn_cards_not_ready")
    if board_cards and any(not card_is_complete(card) for card in board_cards):
        issues.append("board_card_unknown")

    known_cards = [card for card in hero_cards + board_cards if card_is_complete(card)]
    if decision_relevant and len(known_cards) != len(set(known_cards)):
        issues.append("duplicate_cards")
    return sorted(set(issues))


def summarize_card_health(rows: list[dict[str, Any]], *, max_examples: int = 8) -> dict[str, Any]:
    hero_class_counts = count_values(row.get("class") for row in rows if row.get("ok"))
    issue_counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    hero_visible_frames = 0
    hero_complete_frames = 0
    hero_incomplete_or_missed_frames = 0
    board_frames = 0
    board_bad_frames = 0
    duplicate_frames = 0
    turn_blocked_frames = 0

    for row in rows:
        if not row.get("ok"):
            continue
        hero_cards = [str(card) for card in (row.get("hero_cards") or []) if card]
        board_cards = [str(card) for card in (row.get("board") or []) if card]
        row_class = str(row.get("class") or "")
        if row_class in {"complete", "incomplete", "missed_visible_cards", "obstructed_animation"} or hero_cards:
            hero_visible_frames += 1
        hero_complete = len(hero_cards) == 2 and all(card_is_complete(card) for card in hero_cards)
        if hero_complete:
            hero_complete_frames += 1
        elif row_class in {"incomplete", "missed_visible_cards"} or (hero_cards and row_class != "obstructed_animation"):
            hero_incomplete_or_missed_frames += 1
        if board_cards:
            board_frames += 1
        if row.get("board_bad") or any(not card_is_complete(card) for card in board_cards):
            board_bad_frames += 1

        issues = row.get("card_issues")
        if issues is None:
            issues = card_health_issues(row)
        if "duplicate_cards" in issues:
            duplicate_frames += 1
        if "hero_turn_cards_not_ready" in issues:
            turn_blocked_frames += 1
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            bucket = examples.setdefault(issue, [])
            if len(bucket) < max_examples:
                bucket.append(card_health_example(row))

    return {
        "frames": sum(1 for row in rows if row.get("ok")),
        "hero": {
            "visible_frames": hero_visible_frames,
            "complete_frames": hero_complete_frames,
            "incomplete_or_missed_frames": hero_incomplete_or_missed_frames,
            "class_counts": hero_class_counts,
            "turn_blocked_frames": turn_blocked_frames,
        },
        "board": {
            "frames_with_board": board_frames,
            "bad_frames": board_bad_frames,
        },
        "duplicate_frames": duplicate_frames,
        "issue_counts": dict(sorted(issue_counts.items())),
        "examples": examples,
    }


def card_health_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": row.get("time"),
        "class": row.get("class"),
        "street": row.get("street"),
        "hero_cards": row.get("hero_cards") or [],
        "board": row.get("board") or [],
        "problem_frame": row.get("problem_frame") or "",
    }


def card_is_complete(card: Any) -> bool:
    text = str(card or "").strip()
    return len(text) >= 2 and "?" not in text


def classify_frame(frame: Any, hero_cards: list[Any]) -> str:
    complete = len(hero_cards) == 2 and all("?" not in str(card) for card in hero_cards)
    if complete:
        return "complete"
    if hero_cards and (hero_obstruction_score(frame) > 0.018 or hero_card_back_animation_score(frame) > 0.015):
        return "obstructed_animation"
    if hero_cards:
        return "incomplete"
    return "missed_visible_cards" if detect_auto_hero_cards(frame) else "empty_or_no_hand"


def hero_obstruction_score(frame: Any) -> float:
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.60) : int(height * 0.88), int(width * 0.36) : int(width * 0.64)]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_orange = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 85)
    return float(yellow_orange.mean())


def hero_card_back_animation_score(frame: Any) -> float:
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.60) : int(height * 0.88), int(width * 0.36) : int(width * 0.64)]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_back = (((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 80))
    return float(red_back.mean())


def save_problem_frame(cv2: Any, problem_dir: Path, frame: Any, timestamp: float, reason: str) -> str:
    safe_reason = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in reason)
    path = problem_dir / f"problem_{int(round(timestamp)):06d}_{safe_reason}.png"
    ok, encoded = cv2.imencode(".png", frame)
    if ok:
        path.write_bytes(encoded.tobytes())
    return str(path)


def count_by_class(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("class") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def timing_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "median": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)
    p90_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))
    return {
        "avg": round(float(statistics.mean(ordered)), 1),
        "median": round(float(statistics.median(ordered)), 1),
        "p90": round(float(ordered[p90_index]), 1),
        "max": round(float(max(ordered)), 1),
    }


def component_timing_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        timing = row.get("cv_timing_ms") or {}
        for key, value in timing.items():
            try:
                buckets.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {key: timing_stats(values) for key, values in sorted(buckets.items())}


def format_validation_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    timing = summary.get("timing_ms") or {}
    card_health = summary.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    lines = [
        "# CV Validation Report",
        "",
        f"- Video: `{summary.get('video')}`",
        f"- Frames: `{summary.get('sample', {}).get('frame_count')}` every `{summary.get('sample', {}).get('every_sec')}` sec",
        f"- Counts: `{json.dumps(counts, ensure_ascii=False)}`",
        f"- Real problem count: `{summary.get('real_problem_count')}`",
        f"- Board bad count: `{summary.get('board_bad_count')}`",
        f"- Hero card health: visible `{hero_health.get('visible_frames')}`, complete `{hero_health.get('complete_frames')}`, incomplete/missed `{hero_health.get('incomplete_or_missed_frames')}`, turn blocked `{hero_health.get('turn_blocked_frames')}`",
        f"- Board card health: board frames `{board_health.get('frames_with_board')}`, bad `{board_health.get('bad_frames')}`, duplicates `{card_health.get('duplicate_frames')}`",
        f"- Card issues: `{json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}`",
        f"- Timing ms: avg `{timing.get('avg')}`, median `{timing.get('median')}`, p90 `{timing.get('p90')}`, max `{timing.get('max')}`",
        "",
        "## Component Timing",
        "",
        "| Component | Avg | Median | P90 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, item in (summary.get("cv_timing_ms") or {}).items():
        lines.append(
            f"| {key} | {item.get('avg')} | {item.get('median')} | {item.get('p90')} | {item.get('max')} |"
        )
    lines.extend(
        [
            "",
            "## Frames",
            "",
        "| Time | Class | Street | Dealer | Hero Pos | Turn | Hero Cards | Board | ms | Problem |",
        "|---:|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    for row in summary.get("rows") or []:
        cards = " ".join(str(card) for card in (row.get("hero_cards") or [])) or "-"
        board = " ".join(str(card) for card in (row.get("board") or [])) or "-"
        hero_turn = row.get("hero_turn") or {}
        turn = "yes" if hero_turn.get("is_turn") else "no"
        problem = row.get("problem_frame") or ""
        if problem:
            problem = f"[png]({problem})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("time", "-")),
                    str(row.get("class", "-")),
                    str(row.get("street", "-")),
                    str(row.get("dealer", "-")),
                    str(row.get("hero_position", "-")),
                    turn,
                    cards,
                    board,
                    str(row.get("analysis_ms", "-")),
                    problem,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_validation_suite_markdown(summary: dict[str, Any]) -> str:
    timing = summary.get("timing_ms") or {}
    card_health = summary.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    lines = [
        "# CV Validation Suite",
        "",
        f"- Videos: `{summary.get('video_count')}`",
        f"- Frames: `{summary.get('sample', {}).get('frame_count')}` every `{summary.get('sample', {}).get('every_sec')}` sec",
        f"- Counts: `{json.dumps(summary.get('counts') or {}, ensure_ascii=False)}`",
        f"- Real problem count: `{summary.get('real_problem_count')}`",
        f"- Board bad count: `{summary.get('board_bad_count')}`",
        f"- Hero card health: visible `{hero_health.get('visible_frames')}`, complete `{hero_health.get('complete_frames')}`, incomplete/missed `{hero_health.get('incomplete_or_missed_frames')}`, turn blocked `{hero_health.get('turn_blocked_frames')}`",
        f"- Board card health: board frames `{board_health.get('frames_with_board')}`, bad `{board_health.get('bad_frames')}`, duplicates `{card_health.get('duplicate_frames')}`",
        f"- Card issues: `{json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}`",
        f"- Timing ms: avg `{timing.get('avg')}`, median `{timing.get('median')}`, p90 `{timing.get('p90')}`, max `{timing.get('max')}`",
        "",
        "| Video | Frames | Counts | Real Problems | Board Bad | Report |",
        "|---|---:|---|---:|---:|---|",
    ]
    for item in summary.get("videos") or []:
        video = Path(str(item.get("video") or "")).name
        report = item.get("report_md") or ""
        report_link = f"[md]({report})" if report else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    video,
                    str(item.get("frames", "-")),
                    json.dumps(item.get("counts") or {}, ensure_ascii=False),
                    str(item.get("real_problem_count", "-")),
                    str(item.get("board_bad_count", "-")),
                    report_link,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_validation_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    timing = payload.get("timing_ms") or {}
    files = payload.get("files") or {}
    card_health = payload.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    return "\n".join(
        [
            f"Video: {payload.get('video')}",
            f"Frames: {payload.get('sample', {}).get('frame_count')} every {payload.get('sample', {}).get('every_sec')}s",
            f"Counts: {json.dumps(counts, ensure_ascii=False)}",
            f"Real problem count: {payload.get('real_problem_count')}",
            f"Board bad count: {payload.get('board_bad_count')}",
            (
                "Hero card health: "
                f"visible={hero_health.get('visible_frames')} complete={hero_health.get('complete_frames')} "
                f"incomplete_or_missed={hero_health.get('incomplete_or_missed_frames')} "
                f"turn_blocked={hero_health.get('turn_blocked_frames')}"
            ),
            (
                "Board card health: "
                f"frames={board_health.get('frames_with_board')} bad={board_health.get('bad_frames')} "
                f"duplicates={card_health.get('duplicate_frames')}"
            ),
            f"Card issues: {json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}",
            f"Timing ms: avg={timing.get('avg')} median={timing.get('median')} p90={timing.get('p90')} max={timing.get('max')}",
            f"Summary: {files.get('summary')}",
            f"Report: {files.get('report_md')}",
            f"Problem frames: {files.get('problem_frames')}",
        ]
    )


def format_validation_suite_summary(payload: dict[str, Any]) -> str:
    timing = payload.get("timing_ms") or {}
    card_health = payload.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    lines = [
        f"Videos: {payload.get('video_count')}",
        f"Frames: {payload.get('sample', {}).get('frame_count')} every {payload.get('sample', {}).get('every_sec')}s",
        f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
        f"Real problem count: {payload.get('real_problem_count')}",
        f"Board bad count: {payload.get('board_bad_count')}",
        (
            "Hero card health: "
            f"visible={hero_health.get('visible_frames')} complete={hero_health.get('complete_frames')} "
            f"incomplete_or_missed={hero_health.get('incomplete_or_missed_frames')} "
            f"turn_blocked={hero_health.get('turn_blocked_frames')}"
        ),
        (
            "Board card health: "
            f"frames={board_health.get('frames_with_board')} bad={board_health.get('bad_frames')} "
            f"duplicates={card_health.get('duplicate_frames')}"
        ),
        f"Card issues: {json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}",
        f"Timing ms: avg={timing.get('avg')} median={timing.get('median')} p90={timing.get('p90')} max={timing.get('max')}",
        f"Summary: {(payload.get('files') or {}).get('summary')}",
        f"Report: {(payload.get('files') or {}).get('report_md')}",
        "",
        "Per video:",
    ]
    for item in payload.get("videos") or []:
        name = Path(str(item.get("video") or "")).name
        lines.append(
            f"- {name}: frames={item.get('frames')} real_problem={item.get('real_problem_count')} "
            f"board_bad={item.get('board_bad_count')} counts={json.dumps(item.get('counts') or {}, ensure_ascii=False)}"
        )
    return "\n".join(lines)
