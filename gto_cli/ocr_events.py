from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .live_vision import build_error_state, build_realtime_state
from .video_vision import analyze_video_frame, choose_template, load_cv, load_ocr


def build_ocr_events_from_states(
    video_path: Path,
    states_jsonl: Path,
    output_dir: Path,
    template_path: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    visual_threshold: float = 5.0,
    visual_min_gap_sec: float = 1.0,
    heartbeat_sec: float = 5.0,
    semantic_min_gap_sec: float = 0.5,
    include_hero_semantic: bool = False,
    dealer_refresh_events: int = 30,
    max_events: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    started_at = time.perf_counter()
    video_path = Path(video_path)
    states_jsonl = Path(states_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    events_json_path = output_dir / "events.json"
    selected_path = output_dir / "selected_frames.json"
    summary_path = output_dir / "summary.json"
    progress_path = output_dir / "progress.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 19.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps else 0.0
    start_sec = 0.0 if start_sec is None else max(0.0, float(start_sec))
    end_sec = duration_sec if end_sec is None else min(duration_sec, float(end_sec))

    selected = select_ocr_frames(
        states_jsonl,
        start_sec=start_sec,
        end_sec=end_sec,
        visual_threshold=visual_threshold,
        visual_min_gap_sec=visual_min_gap_sec,
        heartbeat_sec=heartbeat_sec,
        semantic_min_gap_sec=semantic_min_gap_sec,
        include_hero_semantic=include_hero_semantic,
        max_events=max_events,
    )
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_frames = set()
    existing_events: list[dict[str, Any]] = []
    if resume and events_path.exists():
        with events_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                frame_index = (event.get("source") or {}).get("frame_index")
                if frame_index is not None:
                    existing_frames.add(int(frame_index))
                    existing_events.append(event)

    ocr = load_ocr()
    if ocr is None:
        raise RuntimeError("RapidOCR is required for OCR event generation")
    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    dealer_button_cache: dict[str, Any] | None = None
    cached_dealer_seat: str | None = None
    last_dealer_refresh_event = -10**9
    processed = len(existing_frames)
    ok_count = sum(1 for event in existing_events if event.get("ok"))
    error_count = sum(1 for event in existing_events if not event.get("ok"))
    events: list[dict[str, Any]] = list(existing_events)

    mode = "a" if resume and events_path.exists() else "w"
    with events_path.open(mode, encoding="utf-8", newline="\n") as stream:
        for item_index, item in enumerate(selected):
            frame_index = int(item["frame_index"])
            if frame_index in existing_frames:
                continue
            timestamp = float(item["timestamp_sec"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                state = build_error_state(
                    RuntimeError("frame read failed"),
                    video_path=video_path,
                    timestamp_sec=round(timestamp, 3),
                    frame_index=frame_index,
                    sample_index=item_index,
                )
            else:
                base_state = item.get("state") or {}
                base_table = base_state.get("table") or {}
                refresh_dealer = (
                    dealer_button_cache is None
                    or base_table.get("dealer_seat") != cached_dealer_seat
                    or item_index - last_dealer_refresh_event >= dealer_refresh_events
                )
                try:
                    frame_result = analyze_video_frame(
                        frame,
                        template,
                        ocr=ocr,
                        dealer_button_hint=None if refresh_dealer else dealer_button_cache,
                        cards_hint=cards_hint_from_state(base_state),
                    )
                    if refresh_dealer:
                        dealer_button_cache = frame_result.get("dealer_button")
                        cached_dealer_seat = base_table.get("dealer_seat")
                        last_dealer_refresh_event = item_index
                    state = build_realtime_state(
                        frame_result,
                        video_path=video_path,
                        timestamp_sec=round(timestamp, 3),
                        frame_index=frame_index,
                        sample_index=item_index,
                    )
                    state["ok"] = True
                    state["source"]["visual_diff"] = round(float(base_state.get("visual_diff") or 0.0), 4)
                except Exception as error:
                    state = build_error_state(
                        error,
                        video_path=video_path,
                        timestamp_sec=round(timestamp, 3),
                        frame_index=frame_index,
                        sample_index=item_index,
                    )
                    state["source"]["visual_diff"] = round(float(base_state.get("visual_diff") or 0.0), 4)
            state["event"] = {
                "index": len(events),
                "trigger": "visual-state-ocr",
                "reason": item["reason"],
                "signature": item.get("signature", ""),
            }
            stream.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            events.append(state)
            processed += 1
            ok_count += 1 if state.get("ok") else 0
            error_count += 0 if state.get("ok") else 1
            write_progress(
                progress_path,
                total=len(selected),
                processed=processed,
                ok_count=ok_count,
                error_count=error_count,
                current=item,
                started_at=started_at,
            )

    cap.release()
    elapsed = time.perf_counter() - started_at
    events_json_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ok": True,
        "source": {
            "kind": "video",
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
        },
        "states_jsonl": str(states_jsonl),
        "template": str(template_path),
        "output_dir": str(output_dir),
        "selection": {
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "visual_threshold": visual_threshold,
            "visual_min_gap_sec": visual_min_gap_sec,
            "heartbeat_sec": heartbeat_sec,
            "semantic_min_gap_sec": semantic_min_gap_sec,
            "include_hero_semantic": include_hero_semantic,
            "selected_frames": len(selected),
        },
        "sample": {
            "processed_frames": processed,
            "ok": ok_count,
            "errors": error_count,
            "wall_time_sec": round(elapsed, 3),
            "avg_processed_frame_ms": round(elapsed * 1000 / max(processed - len(existing_frames), 1), 2),
        },
        "files": {
            "events_jsonl": str(events_path),
            "events_json": str(events_json_path),
            "selected_frames": str(selected_path),
            "progress": str(progress_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def select_ocr_frames(
    states_jsonl: Path,
    start_sec: float,
    end_sec: float,
    visual_threshold: float,
    visual_min_gap_sec: float,
    heartbeat_sec: float,
    semantic_min_gap_sec: float,
    include_hero_semantic: bool,
    max_events: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_visual = float("-inf")
    last_heartbeat = float("-inf")
    last_semantic = float("-inf")
    previous_signature: str | None = None
    with Path(states_jsonl).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            state = json.loads(line)
            timestamp = float(state.get("timestamp_sec") or (state.get("source") or {}).get("timestamp_sec") or 0.0)
            if timestamp < start_sec:
                continue
            if timestamp > end_sec:
                break
            frame_index = int(state.get("frame_index") or (state.get("source") or {}).get("frame_index") or 0)
            visual_diff = float(state.get("visual_diff") or (state.get("source") or {}).get("visual_diff") or 0.0)
            signature = semantic_signature(state, include_hero=include_hero_semantic)
            reasons = []
            if signature != previous_signature:
                if timestamp - last_semantic >= semantic_min_gap_sec:
                    reasons.append("semantic")
                    last_semantic = timestamp
                previous_signature = signature
            if visual_diff >= visual_threshold and timestamp - last_visual >= visual_min_gap_sec:
                reasons.append("visual")
                last_visual = timestamp
            if timestamp - last_heartbeat >= heartbeat_sec:
                reasons.append("heartbeat")
                last_heartbeat = timestamp
            if not reasons:
                continue
            selected.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp, 3),
                    "reason": "+".join(reasons),
                    "visual_diff": round(visual_diff, 4),
                    "signature": signature,
                    "state": state,
                }
            )
            if max_events is not None and len(selected) >= max_events:
                break
    return selected


def semantic_signature(state: dict[str, Any], include_hero: bool) -> str:
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    payload = {
        "street": table.get("street"),
        "dealer": table.get("dealer_seat"),
        "board": table.get("board") or [],
        "hero_position": hero.get("position"),
    }
    if include_hero:
        payload["hero_cards"] = hero.get("cards") or []
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cards_hint_from_state(state: dict[str, Any]) -> dict[str, Any]:
    table = state.get("table") or {}
    hero = state.get("hero") or {}
    return {
        "hero": list(hero.get("cards") or []),
        "board": list(table.get("board") or []),
        "hero_details": [],
        "board_details": [],
    }


def write_progress(
    path: Path,
    total: int,
    processed: int,
    ok_count: int,
    error_count: int,
    current: dict[str, Any],
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    remaining = max(0, total - processed)
    avg = elapsed / processed if processed else 0.0
    payload = {
        "total": total,
        "processed": processed,
        "ok": ok_count,
        "errors": error_count,
        "remaining": remaining,
        "elapsed_sec": round(elapsed, 3),
        "estimated_remaining_sec": round(avg * remaining, 3) if avg else None,
        "current": {
            "frame_index": current.get("frame_index"),
            "timestamp_sec": current.get("timestamp_sec"),
            "reason": current.get("reason"),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_ocr_event_summary(payload: dict[str, Any]) -> str:
    selection = payload["selection"]
    sample = payload["sample"]
    files = payload["files"]
    return "\n".join(
        [
            f"OCR events: {files['events_jsonl']}",
            f"Selected frames: {selection['selected_frames']}",
            f"Processed: {sample['processed_frames']} ok={sample['ok']} errors={sample['errors']}",
            f"Wall time: {sample['wall_time_sec']}s",
            f"Summary: {files['summary']}",
        ]
    )
