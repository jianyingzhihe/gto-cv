from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .cv_validate import find_root_videos, merge_counts, safe_video_slug, timing_stats
from .screen_vision import (
    detect_auto_bbox,
    hero_fixed_card_anchor_score,
    poker_content_score,
    poker_table_visibility,
    yellow_pot_anchor_score,
)
from .video_vision import choose_template, load_cv, sample_times


DEFAULT_BBOX_VARIANTS = ("native", "loose_8", "loose_shift", "tight_3", "tight_6")


def diagnose_auto_bbox_videos(
    video_paths: list[Path] | None = None,
    video_dir: Path = Path("video_frames"),
    output_dir: Path = Path("video_frames") / "auto_bbox_diagnostics",
    template_path: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 30.0,
    max_frames: int | None = None,
    min_confidence: float = 0.35,
    variants: tuple[str, ...] | None = DEFAULT_BBOX_VARIANTS,
    save_problem_frames: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(variants or DEFAULT_BBOX_VARIANTS)
    videos = [Path(path) for path in (video_paths or find_root_videos(video_dir))]

    cv2, np = load_cv()
    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    results = []
    all_rows: list[dict[str, Any]] = []
    for index, video_path in enumerate(videos, start=1):
        child_dir = output_dir / f"{index:02d}_{safe_video_slug(video_path)}"
        result = diagnose_auto_bbox_video(
            video_path=video_path,
            output_dir=child_dir,
            template=template,
            template_path=template_path,
            start_sec=start_sec,
            end_sec=end_sec,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
            variants=variants,
            save_problem_frames=save_problem_frames,
        )
        results.append(result)
        all_rows.extend(result.get("rows") or [])

    counts = merge_counts(result.get("counts") or {} for result in results)
    method_counts: dict[str, int] = {}
    for row in all_rows:
        method = str(row.get("method") or "-")
        method_counts[method] = method_counts.get(method, 0) + 1
    iou_values = [float(row["iou_to_expected"]) for row in all_rows if row.get("iou_to_expected") is not None]
    timing_values = [float(row["analysis_ms"]) for row in all_rows if row.get("analysis_ms") is not None]
    failures = [row for row in all_rows if row_is_failure(row)]
    summary = {
        "ok": True,
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "template": str(template_path),
        "video_count": len(results),
        "sample": {
            "every_sec": every_sec,
            "max_frames": max_frames,
            "variants": list(variants),
            "row_count": len(all_rows),
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "counts": counts,
        "method_counts": method_counts,
        "failure_count": len(failures),
        "iou_stats": timing_stats(iou_values),
        "timing_ms": timing_stats(timing_values),
        "files": {
            "summary": str(output_dir / "auto_bbox_diagnostics_summary.json"),
            "report_md": str(output_dir / "auto_bbox_diagnostics_report.md"),
            "rows_csv": str(output_dir / "auto_bbox_diagnostics_rows.csv"),
        },
        "videos": [
            {
                "video": result.get("video"),
                "rows": result.get("sample", {}).get("row_count"),
                "counts": result.get("counts") or {},
                "failure_count": result.get("failure_count"),
                "method_counts": result.get("method_counts") or {},
                "summary": (result.get("files") or {}).get("summary"),
                "report_md": (result.get("files") or {}).get("report_md"),
                "problem_frames": (result.get("files") or {}).get("problem_frames"),
            }
            for result in results
        ],
    }
    write_rows_csv(output_dir / "auto_bbox_diagnostics_rows.csv", all_rows)
    (output_dir / "auto_bbox_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "auto_bbox_diagnostics_report.md").write_text(format_auto_bbox_suite_markdown(summary), encoding="utf-8")
    return summary


def diagnose_auto_bbox_video(
    video_path: Path,
    output_dir: Path,
    template: Any,
    template_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 30.0,
    max_frames: int | None = None,
    min_confidence: float = 0.35,
    variants: tuple[str, ...] = DEFAULT_BBOX_VARIANTS,
    save_problem_frames: bool = True,
) -> dict[str, Any]:
    cv2, np = load_cv()
    output_dir = Path(output_dir)
    problem_dir = output_dir / "problem_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_problem_frames:
        problem_dir.mkdir(parents=True, exist_ok=True)

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

    rows: list[dict[str, Any]] = []
    for sample_index, timestamp in enumerate(times):
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            rows.append(
                {
                    "video": str(video_path),
                    "time": round(float(timestamp), 3),
                    "frame_index": frame_index,
                    "variant": "-",
                    "class": "read_failed",
                    "error": "could not read frame",
                }
            )
            continue

        base_expected: dict[str, int] | None = None
        ordered_variants = list(variants)
        if "native" in ordered_variants:
            ordered_variants = ["native"] + [variant for variant in ordered_variants if variant != "native"]

        for variant in ordered_variants:
            test_frame, expected = make_bbox_test_frame(cv2, np, frame, variant, base_expected=base_expected)
            row = diagnose_auto_bbox_frame(
                cv2=cv2,
                np=np,
                frame=test_frame,
                expected=expected,
                template=template,
                min_confidence=min_confidence,
            )
            row.update(
                {
                    "video": str(video_path),
                    "time": round(float(timestamp), 3),
                    "frame_index": frame_index,
                    "sample_index": sample_index,
                    "variant": variant,
                    "source_width": width,
                    "source_height": height,
                }
            )
            if variant == "native" and row.get("region_left") is not None:
                base_expected = row_region(row)
            if save_problem_frames and row_is_failure(row):
                row["problem_frame"] = save_bbox_problem_frame(cv2, problem_dir, test_frame, row, timestamp)
            rows.append(row)
    cap.release()

    counts = count_classes(rows)
    method_counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("method") or "-")
        method_counts[method] = method_counts.get(method, 0) + 1
    failures = [row for row in rows if row_is_failure(row)]
    summary = {
        "ok": True,
        "video": str(video_path),
        "template": str(template_path),
        "output_dir": str(output_dir),
        "video_info": {
            "width": width,
            "height": height,
            "fps": round(float(fps), 3),
            "frame_count": frame_count,
            "duration_sec": round(float(duration_sec), 3),
        },
        "sample": {
            "every_sec": every_sec,
            "max_frames": max_frames,
            "frame_count": len(times),
            "variants": list(variants),
            "row_count": len(rows),
        },
        "counts": counts,
        "method_counts": method_counts,
        "failure_count": len(failures),
        "timing_ms": timing_stats([float(row["analysis_ms"]) for row in rows if row.get("analysis_ms") is not None]),
        "iou_stats": timing_stats([float(row["iou_to_expected"]) for row in rows if row.get("iou_to_expected") is not None]),
        "files": {
            "summary": str(output_dir / "auto_bbox_summary.json"),
            "rows_csv": str(output_dir / "auto_bbox_rows.csv"),
            "report_md": str(output_dir / "auto_bbox_report.md"),
            "problem_frames": str(problem_dir) if save_problem_frames else None,
        },
        "rows": rows,
    }
    write_rows_csv(output_dir / "auto_bbox_rows.csv", rows)
    (output_dir / "auto_bbox_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "auto_bbox_report.md").write_text(format_auto_bbox_video_markdown(summary), encoding="utf-8")
    return summary


def make_bbox_test_frame(
    cv2: Any,
    np: Any,
    frame: Any,
    variant: str,
    *,
    base_expected: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int] | None]:
    height, width = frame.shape[:2]
    if variant == "native":
        return frame, None
    if variant.startswith("loose"):
        if variant == "loose_shift":
            pad_l, pad_t, pad_r, pad_b = int(width * 0.14), int(height * 0.08), int(width * 0.05), int(height * 0.13)
        else:
            ratio = parse_variant_ratio(variant, default=0.08)
            pad_l = pad_r = int(width * ratio)
            pad_t = pad_b = int(height * ratio)
        canvas = np.zeros((height + pad_t + pad_b, width + pad_l + pad_r, 3), dtype=frame.dtype)
        canvas[:] = (32, 31, 34)
        canvas[pad_t : pad_t + height, pad_l : pad_l + width] = frame
        expected = translate_region(base_expected, pad_l, pad_t) if base_expected else {
            "left": pad_l,
            "top": pad_t,
            "width": width,
            "height": height,
        }
        return canvas, expected
    if variant.startswith("tight"):
        ratio = parse_variant_ratio(variant, default=0.03)
        dx = int(width * ratio)
        dy = int(height * ratio)
        if width - dx * 2 < 50 or height - dy * 2 < 50:
            return frame, None
        expected = crop_translate_region(
            base_expected,
            {"left": dx, "top": dy, "width": width - dx * 2, "height": height - dy * 2},
        )
        return frame[dy : height - dy, dx : width - dx], expected
    raise ValueError(f"unknown auto-bbox diagnostic variant: {variant}")


def row_region(row: dict[str, Any]) -> dict[str, int] | None:
    if row.get("region_left") is None:
        return None
    return {
        "left": int(row.get("region_left") or 0),
        "top": int(row.get("region_top") or 0),
        "width": int(row.get("region_width") or 0),
        "height": int(row.get("region_height") or 0),
    }


def translate_region(region: dict[str, int] | None, dx: int, dy: int) -> dict[str, int] | None:
    if not region:
        return None
    return {
        "left": int(region["left"] + dx),
        "top": int(region["top"] + dy),
        "width": int(region["width"]),
        "height": int(region["height"]),
    }


def crop_translate_region(region: dict[str, int] | None, crop: dict[str, int]) -> dict[str, int] | None:
    if not region:
        return None
    left = max(int(region["left"]), int(crop["left"]))
    top = max(int(region["top"]), int(crop["top"]))
    right = min(int(region["left"] + region["width"]), int(crop["left"] + crop["width"]))
    bottom = min(int(region["top"] + region["height"]), int(crop["top"] + crop["height"]))
    if right <= left or bottom <= top:
        return None
    return {
        "left": left - int(crop["left"]),
        "top": top - int(crop["top"]),
        "width": right - left,
        "height": bottom - top,
    }


def parse_variant_ratio(name: str, default: float) -> float:
    suffix = name.rsplit("_", 1)[-1]
    try:
        return max(0.0, float(suffix) / 100.0)
    except ValueError:
        return default


def diagnose_auto_bbox_frame(
    cv2: Any,
    np: Any,
    frame: Any,
    expected: dict[str, int] | None,
    template: Any,
    min_confidence: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    search_region = {"left": 0, "top": 0, "width": int(frame.shape[1]), "height": int(frame.shape[0])}
    detection = detect_auto_bbox(
        cv2,
        np,
        frame,
        search_region,
        template,
        min_confidence,
        allow_native_window=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if detection is None:
        return {
            "class": "not_found",
            "analysis_ms": round(float(elapsed_ms), 1),
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
        }
    region = detection.get("region") or {}
    crop = frame[
        int(region.get("top") or 0) : int(region.get("top") or 0) + int(region.get("height") or 0),
        int(region.get("left") or 0) : int(region.get("left") or 0) + int(region.get("width") or 0),
    ]
    visible, visibility = poker_table_visibility(cv2, crop)
    iou = bbox_iou(region, expected) if expected is not None else None
    quality = classify_bbox_detection(detection, visible, iou)
    return {
        "class": quality,
        "analysis_ms": round(float(elapsed_ms), 1),
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "method": detection.get("method"),
        "score": detection.get("score"),
        "dealer_confidence": detection.get("dealer_confidence"),
        "content_score": detection.get("content_score"),
        "pot_anchor_score": detection.get("pot_anchor_score"),
        "hero_anchor_score": detection.get("hero_anchor_score"),
        "table_visible": bool(visible),
        "table_visibility": visibility,
        "region_left": region.get("left"),
        "region_top": region.get("top"),
        "region_width": region.get("width"),
        "region_height": region.get("height"),
        "iou_to_expected": round(float(iou), 4) if iou is not None else None,
        "expected_left": expected.get("left") if expected else None,
        "expected_top": expected.get("top") if expected else None,
        "expected_width": expected.get("width") if expected else None,
        "expected_height": expected.get("height") if expected else None,
        "post_crop_content_score": round(float(poker_content_score(cv2, crop)), 4) if crop.size else 0.0,
        "post_crop_pot_anchor_score": round(float(yellow_pot_anchor_score(cv2, crop)), 4) if crop.size else 0.0,
        "post_crop_hero_anchor_score": round(float(hero_fixed_card_anchor_score(cv2, crop)), 4) if crop.size else 0.0,
    }


def classify_bbox_detection(detection: dict[str, Any], table_visible: bool, iou: float | None) -> str:
    if not table_visible:
        return "not_table"
    dealer_confidence = float(detection.get("dealer_confidence") or 0.0)
    content_score = float(detection.get("content_score") or 0.0)
    pot_anchor = float(detection.get("pot_anchor_score") or 0.0)
    hero_anchor = float(detection.get("hero_anchor_score") or 0.0)
    strong_anchor = dealer_confidence >= 0.45 or hero_anchor >= 0.70 or (content_score >= 0.70 and pot_anchor >= 0.20)
    if iou is not None and iou < 0.72:
        return "ok_inner_table" if strong_anchor else "bad_iou"
    if dealer_confidence < 0.25 and content_score < 0.22 and pot_anchor < 0.25 and hero_anchor < 0.20:
        return "weak_anchors"
    return "ok"


def row_is_failure(row: dict[str, Any]) -> bool:
    return not str(row.get("class") or "").startswith("ok")


def bbox_iou(a: dict[str, Any], b: dict[str, Any] | None) -> float | None:
    if not b:
        return None
    ax1 = float(a.get("left") or 0)
    ay1 = float(a.get("top") or 0)
    ax2 = ax1 + float(a.get("width") or 0)
    ay2 = ay1 + float(a.get("height") or 0)
    bx1 = float(b.get("left") or 0)
    by1 = float(b.get("top") or 0)
    bx2 = bx1 + float(b.get("width") or 0)
    by2 = by1 + float(b.get("height") or 0)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


def save_bbox_problem_frame(cv2: Any, problem_dir: Path, frame: Any, row: dict[str, Any], timestamp: float) -> str:
    safe_class = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(row.get("class") or "problem"))
    safe_variant = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(row.get("variant") or "variant"))
    path = problem_dir / f"bbox_{int(round(timestamp)):06d}_{safe_variant}_{safe_class}.png"
    image = frame.copy()
    draw_bbox_overlay(cv2, image, row)
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        path.write_bytes(encoded.tobytes())
    return str(path)


def draw_bbox_overlay(cv2: Any, image: Any, row: dict[str, Any]) -> None:
    if row.get("expected_left") is not None:
        pt1 = (int(row["expected_left"]), int(row["expected_top"]))
        pt2 = (int(row["expected_left"] + row["expected_width"]), int(row["expected_top"] + row["expected_height"]))
        cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
    if row.get("region_left") is not None:
        pt1 = (int(row["region_left"]), int(row["region_top"]))
        pt2 = (int(row["region_left"] + row["region_width"]), int(row["region_top"] + row["region_height"]))
        cv2.rectangle(image, pt1, pt2, (0, 255, 255), 2)
    label = f"{row.get('class')} {row.get('variant')} {row.get('method') or '-'}"
    cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def count_classes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("class") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "video",
        "time",
        "frame_index",
        "sample_index",
        "variant",
        "class",
        "method",
        "score",
        "dealer_confidence",
        "content_score",
        "pot_anchor_score",
        "hero_anchor_score",
        "table_visible",
        "region_left",
        "region_top",
        "region_width",
        "region_height",
        "expected_left",
        "expected_top",
        "expected_width",
        "expected_height",
        "iou_to_expected",
        "post_crop_content_score",
        "post_crop_pot_anchor_score",
        "post_crop_hero_anchor_score",
        "analysis_ms",
        "problem_frame",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_auto_bbox_suite_summary(payload: dict[str, Any]) -> str:
    timing = payload.get("timing_ms") or {}
    iou = payload.get("iou_stats") or {}
    lines = [
        f"Videos: {payload.get('video_count')}",
        f"Rows: {payload.get('sample', {}).get('row_count')} variants={payload.get('sample', {}).get('variants')}",
        f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
        f"Methods: {json.dumps(payload.get('method_counts') or {}, ensure_ascii=False)}",
        f"Failures: {payload.get('failure_count')}",
        f"IoU: median={iou.get('median')} p90={iou.get('p90')} min/max-as-max={iou.get('max')}",
        f"Timing ms: avg={timing.get('avg')} median={timing.get('median')} p90={timing.get('p90')} max={timing.get('max')}",
        f"Summary: {(payload.get('files') or {}).get('summary')}",
        f"Report: {(payload.get('files') or {}).get('report_md')}",
        f"Rows CSV: {(payload.get('files') or {}).get('rows_csv')}",
        "",
        "Per video:",
    ]
    for item in payload.get("videos") or []:
        name = Path(str(item.get("video") or "")).name
        lines.append(
            f"- {name}: rows={item.get('rows')} failures={item.get('failure_count')} "
            f"counts={json.dumps(item.get('counts') or {}, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def format_auto_bbox_suite_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Auto BBox Diagnostics",
        "",
        f"- Videos: `{summary.get('video_count')}`",
        f"- Rows: `{summary.get('sample', {}).get('row_count')}`",
        f"- Variants: `{', '.join(summary.get('sample', {}).get('variants') or [])}`",
        f"- Counts: `{json.dumps(summary.get('counts') or {}, ensure_ascii=False)}`",
        f"- Methods: `{json.dumps(summary.get('method_counts') or {}, ensure_ascii=False)}`",
        f"- Failures: `{summary.get('failure_count')}`",
        f"- Rows CSV: `{(summary.get('files') or {}).get('rows_csv')}`",
        "",
        "| Video | Rows | Failures | Counts | Methods | Report |",
        "|---|---:|---:|---|---|---|",
    ]
    for item in summary.get("videos") or []:
        report = (item.get("report_md") or "").replace("\\", "/")
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(str(item.get("video") or "")).name,
                    str(item.get("rows")),
                    str(item.get("failure_count")),
                    f"`{json.dumps(item.get('counts') or {}, ensure_ascii=False)}`",
                    f"`{json.dumps(item.get('method_counts') or {}, ensure_ascii=False)}`",
                    f"[report]({report})",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_auto_bbox_video_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Auto BBox Video Diagnostics",
        "",
        f"- Video: `{summary.get('video')}`",
        f"- Rows: `{summary.get('sample', {}).get('row_count')}`",
        f"- Counts: `{json.dumps(summary.get('counts') or {}, ensure_ascii=False)}`",
        f"- Methods: `{json.dumps(summary.get('method_counts') or {}, ensure_ascii=False)}`",
        f"- Failures: `{summary.get('failure_count')}`",
        "",
        "| Time | Variant | Class | Method | Score | D | Content | Pot | Hero | IoU | Problem |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows") or []:
        problem = str(row.get("problem_frame") or "")
        link = f"[png]({problem.replace(chr(92), '/')})" if problem else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("time")),
                    str(row.get("variant")),
                    str(row.get("class")),
                    str(row.get("method") or "-"),
                    str(row.get("score") or "-"),
                    str(row.get("dealer_confidence") or "-"),
                    str(row.get("content_score") or "-"),
                    str(row.get("pot_anchor_score") or "-"),
                    str(row.get("hero_anchor_score") or "-"),
                    str(row.get("iou_to_expected") or "-"),
                    link,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
