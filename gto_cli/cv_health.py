from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bbox_utils import parse_bbox_values
from .card_classifier import DEFAULT_MODEL_PATH, RANK_LABELS, SUIT_LABELS, load_card_classifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEALTH_OUTPUT_DIR = Path("video_frames") / "cv_health"
DEFAULT_DEEP_MODEL_DIR = None
DEFAULT_VALIDATION_SUMMARY = (
    Path("video_frames")
    / "promoted_default_bigteacher_seedguard_20260709_validation"
    / "cv_validation_all_summary.json"
)
DEFAULT_GATE_SUMMARY = (
    Path("video_frames")
    / "promoted_default_bigteacher_seedguard_20260709_gate"
    / "card_model_gate_summary.json"
)


def check_cv_health(
    *,
    output_dir: Path = DEFAULT_HEALTH_OUTPUT_DIR,
    knn_model_path: Path = DEFAULT_MODEL_PATH,
    deep_model_dir: Path = DEFAULT_DEEP_MODEL_DIR,
    validation_summary_json: Path = DEFAULT_VALIDATION_SUMMARY,
    gate_summary_json: Path = DEFAULT_GATE_SUMMARY,
    bbox: str = "x,y,w,h",
    bbox_file: Path | None = None,
    allow_placeholder_bbox: bool = False,
    screen_output_dir: Path = Path("video_frames") / "screen_live",
    fast_screen_output_dir: Path = Path("video_frames") / "screen_live_fast",
    preflight_output_dir: Path = Path("video_frames") / "screen_preflight",
    hero_name: str | None = None,
    effective_stack: float = 100.0,
    villain: str = "standard",
    min_confidence: float = 0.35,
    ocr_scale: float = 0.65,
    dealer_refresh_frames: int = 12,
    auto_bbox_refresh: float = 10.0,
    max_real_problem: int = 0,
    max_board_bad: int = 0,
    max_median_ms: float = 300.0,
    max_p90_ms: float = 900.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knn_info = inspect_knn_model(Path(knn_model_path))
    deep_info = inspect_deep_model_dir(deep_model_dir)
    validation = load_json_file(Path(validation_summary_json))
    gate = load_json_file(Path(gate_summary_json))
    bbox_info = inspect_bbox(bbox)
    live_command = build_live_command(
        bbox=bbox,
        bbox_file=bbox_file,
        screen_output_dir=screen_output_dir,
        hero_name=hero_name,
        effective_stack=effective_stack,
        villain=villain,
        min_confidence=min_confidence,
        ocr_scale=ocr_scale,
        dealer_refresh_frames=dealer_refresh_frames,
        auto_bbox_refresh=auto_bbox_refresh,
        deep_model_dir=deep_model_dir,
    )
    fast_live_command = build_fast_live_command(
        bbox=bbox,
        bbox_file=bbox_file,
        screen_output_dir=fast_screen_output_dir,
        hero_name=hero_name,
        effective_stack=effective_stack,
        villain=villain,
        min_confidence=min_confidence,
        ocr_scale=ocr_scale,
        dealer_refresh_frames=dealer_refresh_frames,
        auto_bbox_refresh=auto_bbox_refresh,
        deep_model_dir=deep_model_dir,
    )
    preflight_command = build_preflight_command(
        bbox=bbox,
        bbox_file=bbox_file,
        preflight_output_dir=preflight_output_dir,
        hero_name=hero_name,
        effective_stack=effective_stack,
        villain=villain,
        min_confidence=min_confidence,
        ocr_scale=ocr_scale,
        dealer_refresh_frames=dealer_refresh_frames,
        auto_bbox_refresh=auto_bbox_refresh,
        deep_model_dir=deep_model_dir,
    )
    checks = build_health_checks(
        knn_info=knn_info,
        deep_info=deep_info,
        validation=validation,
        gate=gate,
        bbox_info=bbox_info,
        allow_placeholder_bbox=allow_placeholder_bbox,
        max_real_problem=max_real_problem,
        max_board_bad=max_board_bad,
        max_median_ms=max_median_ms,
        max_p90_ms=max_p90_ms,
    )
    ready = all(bool(check.get("pass")) for check in checks)
    summary = {
        "ok": True,
        "ready": ready,
        "decision": "ready" if ready else "not_ready",
        "checks": checks,
        "models": {
            "knn": knn_info,
            "deep": deep_info,
        },
        "validation": compact_validation(validation),
        "gate": compact_gate(gate),
        "bbox": bbox_info,
        "live_command": live_command,
        "fast_live_command": fast_live_command,
        "preflight_command": preflight_command,
        "thresholds": {
            "max_real_problem": int(max_real_problem),
            "max_board_bad": int(max_board_bad),
            "max_median_ms": float(max_median_ms),
            "max_p90_ms": float(max_p90_ms),
        },
        "live_options": {
            "bbox": bbox,
            "bbox_file": str(bbox_file) if bbox_file else "",
            "allow_placeholder_bbox": bool(allow_placeholder_bbox),
            "screen_output_dir": str(screen_output_dir),
            "fast_screen_output_dir": str(fast_screen_output_dir),
            "preflight_output_dir": str(preflight_output_dir),
            "hero_name": hero_name,
            "effective_stack": float(effective_stack),
            "villain": villain,
            "min_confidence": float(min_confidence),
            "ocr_scale": float(ocr_scale),
            "dealer_refresh_frames": int(dealer_refresh_frames),
            "auto_bbox_refresh": float(auto_bbox_refresh),
        },
        "files": {
            "summary": str(output_dir / "cv_health_summary.json"),
            "report_md": str(output_dir / "cv_health_report.md"),
            "run_live_command": str(output_dir / "run_live_command.txt"),
            "run_fast_live_command": str(output_dir / "run_fast_live_command.txt"),
            "run_preflight_command": str(output_dir / "run_preflight_command.txt"),
        },
    }
    (output_dir / "cv_health_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cv_health_report.md").write_text(format_cv_health_markdown(summary), encoding="utf-8")
    (output_dir / "run_live_command.txt").write_text(live_command + "\n", encoding="utf-8-sig")
    (output_dir / "run_fast_live_command.txt").write_text(fast_live_command + "\n", encoding="utf-8-sig")
    (output_dir / "run_preflight_command.txt").write_text(preflight_command + "\n", encoding="utf-8-sig")
    return summary


def inspect_knn_model(model_path: Path) -> dict[str, Any]:
    path = Path(model_path)
    exists = path.exists()
    info: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "metadata": None,
        "rank_labels_ok": False,
        "suit_labels_ok": False,
        "rank_feature_count": 0,
        "suit_feature_count": 0,
    }
    if not exists:
        return info
    model = load_card_classifier(path)
    if not model:
        return info
    metadata = model.get("metadata") or {}
    rank_labels = list(metadata.get("rank_labels") or [])
    suit_labels = list(metadata.get("suit_labels") or [])
    info.update(
        {
            "metadata": metadata,
            "rank_labels_ok": set(rank_labels) == set(RANK_LABELS),
            "suit_labels_ok": set(suit_labels) == set(SUIT_LABELS),
            "rank_feature_count": int(metadata.get("rank_feature_count") or 0),
            "suit_feature_count": int(metadata.get("suit_feature_count") or 0),
            "rank_source_count": int(metadata.get("rank_source_count") or 0),
            "suit_source_count": int(metadata.get("suit_source_count") or 0),
        }
    )
    return info


def inspect_deep_model_dir(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {
            "path": "",
            "enabled": False,
            "exists": False,
            "rank_model": "",
            "suit_model": "",
            "rank_exists": False,
            "suit_exists": False,
            "rank_size_bytes": 0,
            "suit_size_bytes": 0,
        }
    root = Path(model_dir)
    rank_path = root / "deep_rank.pt"
    suit_path = root / "deep_suit.pt"
    return {
        "path": str(root),
        "enabled": True,
        "exists": root.exists(),
        "rank_model": str(rank_path),
        "suit_model": str(suit_path),
        "rank_exists": rank_path.exists(),
        "suit_exists": suit_path.exists(),
        "rank_size_bytes": rank_path.stat().st_size if rank_path.exists() else 0,
        "suit_size_bytes": suit_path.stat().st_size if suit_path.exists() else 0,
    }


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def build_health_checks(
    *,
    knn_info: dict[str, Any],
    deep_info: dict[str, Any],
    validation: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    bbox_info: dict[str, Any],
    allow_placeholder_bbox: bool,
    max_real_problem: int,
    max_board_bad: int,
    max_median_ms: float,
    max_p90_ms: float,
) -> list[dict[str, Any]]:
    timing = (validation or {}).get("timing_ms") or {}
    checks = [
        check("knn_model_exists", knn_info.get("exists"), knn_info.get("path"), "existing .npz"),
        check("knn_rank_labels", knn_info.get("rank_labels_ok"), sorted((knn_info.get("metadata") or {}).get("rank_labels") or []), sorted(RANK_LABELS)),
        check("knn_suit_labels", knn_info.get("suit_labels_ok"), sorted((knn_info.get("metadata") or {}).get("suit_labels") or []), sorted(SUIT_LABELS)),
        check("validation_summary_exists", validation is not None, bool(validation), "validate-cv summary"),
        check("gate_summary_exists", gate is not None, bool(gate), "gate summary"),
        check("gate_promote", bool((gate or {}).get("promote")), (gate or {}).get("decision"), "promote"),
        check(
            "bbox_concrete",
            bool(bbox_info.get("concrete")) or bool(allow_placeholder_bbox),
            bbox_info.get("normalized") or bbox_info.get("raw"),
            "numeric x,y,width,height",
        ),
    ]
    if deep_info.get("enabled"):
        checks.extend(
            [
                check("deep_rank_model_exists", deep_info.get("rank_exists"), deep_info.get("rank_model"), "existing deep_rank.pt"),
                check("deep_suit_model_exists", deep_info.get("suit_exists"), deep_info.get("suit_model"), "existing deep_suit.pt"),
            ]
        )
    else:
        checks.append(check("deep_model_optional", True, "disabled", "optional offline fallback"))
    if validation:
        card_health = validation.get("card_health") or {}
        hero_health = card_health.get("hero") or {}
        card_issue_count = sum(int(value or 0) for value in (card_health.get("issue_counts") or {}).values())
        checks.extend(
            [
                check("validation_ok", bool(validation.get("ok")), bool(validation.get("ok")), True),
                check(
                    "validation_real_problem_count",
                    int(validation.get("real_problem_count") or 0) <= int(max_real_problem),
                    int(validation.get("real_problem_count") or 0),
                    int(max_real_problem),
                ),
                check(
                    "validation_board_bad_count",
                    int(validation.get("board_bad_count") or 0) <= int(max_board_bad),
                    int(validation.get("board_bad_count") or 0),
                    int(max_board_bad),
                ),
                check("validation_median_ms", optional_float(timing.get("median"), 999999.0) <= float(max_median_ms), timing.get("median"), float(max_median_ms)),
                check("validation_p90_ms", optional_float(timing.get("p90"), 999999.0) <= float(max_p90_ms), timing.get("p90"), float(max_p90_ms)),
            ]
        )
        if card_health:
            checks.extend(
                [
                    check(
                        "validation_hero_incomplete_or_missed",
                        int(hero_health.get("incomplete_or_missed_frames") or 0) <= int(max_real_problem),
                        int(hero_health.get("incomplete_or_missed_frames") or 0),
                        int(max_real_problem),
                    ),
                    check(
                        "validation_card_issue_count",
                        card_issue_count <= int(max_real_problem) + int(max_board_bad),
                        card_issue_count,
                        int(max_real_problem) + int(max_board_bad),
                    ),
                ]
            )
    return checks


def inspect_bbox(bbox: str) -> dict[str, Any]:
    raw = str(bbox or "").strip()
    info: dict[str, Any] = {
        "raw": raw,
        "concrete": False,
        "normalized": "",
        "values": None,
        "reason": "",
    }
    try:
        parsed = parse_bbox_values(raw)
    except ValueError as error:
        parts = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
        if len(parts) != 4:
            info["reason"] = "expected_four_values"
        elif "positive" in str(error).lower():
            info["reason"] = "non_positive_size"
        else:
            info["reason"] = "non_numeric"
        return info
    if parsed is None:
        info["reason"] = "empty"
        return info
    normalized_values = list(parsed)
    info.update(
        {
            "concrete": True,
            "values": normalized_values,
            "normalized": ",".join(str(value) for value in normalized_values),
            "reason": "ok",
        }
    )
    return info


def check(name: str, passed: Any, actual: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "actual": actual, "required": required}


def compact_validation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "output_dir": payload.get("output_dir"),
        "video_count": payload.get("video_count"),
        "counts": payload.get("counts"),
        "real_problem_count": payload.get("real_problem_count"),
        "board_bad_count": payload.get("board_bad_count"),
        "card_health": payload.get("card_health") or {},
        "timing_ms": payload.get("timing_ms"),
        "files": payload.get("files"),
    }


def compact_gate(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "decision": payload.get("decision"),
        "promote": payload.get("promote"),
        "candidate_name": payload.get("candidate_name"),
        "checks": payload.get("checks"),
        "files": payload.get("files"),
    }


def build_live_command(
    *,
    bbox: str,
    bbox_file: Path | None = None,
    screen_output_dir: Path,
    hero_name: str | None,
    effective_stack: float,
    villain: str,
    min_confidence: float,
    ocr_scale: float,
    dealer_refresh_frames: int,
    auto_bbox_refresh: float,
    deep_model_dir: Path | None,
    ocr_action_only: bool = False,
) -> str:
    parts = [
        "python",
        "gto.py",
        "screen-cv",
        "--auto-bbox",
        "--auto-bbox-refresh",
        format_number(auto_bbox_refresh),
        "--lock-layout",
    ]
    insert_bbox_args(parts, bbox=bbox, bbox_file=bbox_file, after="screen-cv")
    if hero_name:
        parts.extend(["--hero-name", quote_arg(hero_name)])
    parts.extend(
        [
            "--output-dir",
            quote_arg(str(screen_output_dir)),
            "--trigger",
            "frame",
            "--every",
            "1",
            "--with-advice",
            "--effective-stack",
            format_number(effective_stack),
            "--villain",
            quote_arg(villain),
            "--min-confidence",
            format_number(min_confidence),
            "--ocr-scale",
            format_number(ocr_scale),
            "--dealer-refresh-frames",
            str(int(dealer_refresh_frames)),
            "--format",
            "text",
        ]
    )
    if deep_model_dir is not None:
        parts.extend(["--deep-card-model-dir", quote_arg(str(deep_model_dir))])
    if ocr_action_only:
        parts.append("--ocr-action-only")
    return " ".join(parts)


def build_fast_live_command(
    *,
    bbox: str,
    bbox_file: Path | None = None,
    screen_output_dir: Path,
    hero_name: str | None,
    effective_stack: float,
    villain: str,
    min_confidence: float,
    ocr_scale: float,
    dealer_refresh_frames: int,
    auto_bbox_refresh: float,
    deep_model_dir: Path | None,
) -> str:
    return build_live_command(
        bbox=bbox,
        bbox_file=bbox_file,
        screen_output_dir=screen_output_dir,
        hero_name=hero_name,
        effective_stack=effective_stack,
        villain=villain,
        min_confidence=min_confidence,
        ocr_scale=ocr_scale,
        dealer_refresh_frames=dealer_refresh_frames,
        auto_bbox_refresh=auto_bbox_refresh,
        deep_model_dir=deep_model_dir,
        ocr_action_only=True,
    )


def build_preflight_command(
    *,
    bbox: str,
    bbox_file: Path | None = None,
    preflight_output_dir: Path,
    hero_name: str | None,
    effective_stack: float,
    villain: str,
    min_confidence: float,
    ocr_scale: float,
    dealer_refresh_frames: int,
    auto_bbox_refresh: float,
    deep_model_dir: Path | None,
) -> str:
    parts = [
        "python",
        "gto.py",
        "screen-cv",
        "--auto-bbox",
        "--auto-bbox-refresh",
        format_number(auto_bbox_refresh),
        "--lock-layout",
    ]
    insert_bbox_args(parts, bbox=bbox, bbox_file=bbox_file, after="screen-cv")
    if hero_name:
        parts.extend(["--hero-name", quote_arg(hero_name)])
    parts.extend(
        [
            "--output-dir",
            quote_arg(str(preflight_output_dir)),
            "--trigger",
            "frame",
            "--every",
            "1",
            "--preflight-once",
            "--save-frames",
            "--save-annotated",
            "--with-advice",
            "--effective-stack",
            format_number(effective_stack),
            "--villain",
            quote_arg(villain),
            "--min-confidence",
            format_number(min_confidence),
            "--ocr-scale",
            format_number(ocr_scale),
            "--dealer-refresh-frames",
            str(int(dealer_refresh_frames)),
            "--format",
            "text",
        ]
    )
    if deep_model_dir is not None:
        parts.extend(["--deep-card-model-dir", quote_arg(str(deep_model_dir))])
    return " ".join(parts)


def build_health_command(
    *,
    bbox: str,
    bbox_file: Path | None = None,
    output_dir: Path = Path("video_frames") / "cv_health_promoted",
    hero_name: str | None = None,
    effective_stack: float = 100.0,
    villain: str = "standard",
    min_confidence: float = 0.35,
    ocr_scale: float = 0.65,
    dealer_refresh_frames: int = 12,
    auto_bbox_refresh: float = 10.0,
    deep_model_dir: Path | None = DEFAULT_DEEP_MODEL_DIR,
    fail_on_not_ready: bool = True,
) -> str:
    parts = [
        "python",
        "gto.py",
        "cv-health",
    ]
    insert_bbox_args(parts, bbox=bbox, bbox_file=bbox_file, after="cv-health")
    if hero_name:
        parts.extend(["--hero-name", quote_arg(hero_name)])
    parts.extend(
        [
            "--output-dir",
            quote_arg(str(output_dir)),
            "--effective-stack",
            format_number(effective_stack),
            "--villain",
            quote_arg(villain),
            "--min-confidence",
            format_number(min_confidence),
            "--ocr-scale",
            format_number(ocr_scale),
            "--dealer-refresh-frames",
            str(int(dealer_refresh_frames)),
            "--auto-bbox-refresh",
            format_number(auto_bbox_refresh),
        ]
    )
    if deep_model_dir is not None:
        parts.extend(["--deep-card-model-dir", quote_arg(str(deep_model_dir))])
    if fail_on_not_ready:
        parts.append("--fail-on-not-ready")
    parts.extend(["--format", "text"])
    return " ".join(parts)


def insert_bbox_args(parts: list[str], *, bbox: str, bbox_file: Path | None, after: str) -> None:
    try:
        index = parts.index(after) + 1
    except ValueError:
        index = len(parts)
    if bbox_file is not None:
        parts[index:index] = ["--bbox-file", quote_arg(str(bbox_file))]
    else:
        parts[index:index] = ["--bbox", quote_arg(bbox)]


def quote_arg(value: str) -> str:
    text = str(value)
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'


def format_number(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def optional_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def format_cv_health_summary(payload: dict[str, Any]) -> str:
    validation = payload.get("validation") or {}
    timing = validation.get("timing_ms") or {}
    card_health = validation.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    models = payload.get("models") or {}
    knn = models.get("knn") or {}
    deep = models.get("deep") or {}
    bbox = payload.get("bbox") or {}
    lines = [
        f"Decision: {str(payload.get('decision')).upper()}",
        f"Ready: {payload.get('ready')}",
        f"KNN: {knn.get('path')} rank_features={knn.get('rank_feature_count')} suit_features={knn.get('suit_feature_count')}",
        f"Deep: {deep.get('path')} rank={deep.get('rank_exists')} suit={deep.get('suit_exists')}",
        f"BBox: {bbox.get('normalized') or bbox.get('raw')} concrete={bbox.get('concrete')} reason={bbox.get('reason')}",
        f"Validation: real_problem={validation.get('real_problem_count', '-')} board_bad={validation.get('board_bad_count', '-')} median={timing.get('median', '-')}ms p90={timing.get('p90', '-')}ms",
        (
            "Card health: "
            f"hero_complete={hero_health.get('complete_frames', '-')} "
            f"hero_incomplete_or_missed={hero_health.get('incomplete_or_missed_frames', '-')} "
            f"hero_turn_blocked={hero_health.get('turn_blocked_frames', '-')} "
            f"board_bad={board_health.get('bad_frames', '-')}"
        ),
        f"Card issues: {json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}",
        "",
        "Checks:",
    ]
    for item in payload.get("checks") or []:
        mark = "PASS" if item.get("pass") else "FAIL"
        lines.append(f"- {mark} {item.get('name')}: actual={item.get('actual')} required={item.get('required')}")
    lines.extend(
        [
            "",
            "Preflight command:",
            str(payload.get("preflight_command") or ""),
            "",
            "Live command:",
            str(payload.get("live_command") or ""),
            "",
            "Fast live command (OCR only while action buttons are visible):",
            str(payload.get("fast_live_command") or ""),
        ]
    )
    files = payload.get("files") or {}
    lines.extend(
        [
            "",
            f"Report: {files.get('report_md')}",
            f"Preflight command file: {files.get('run_preflight_command')}",
            f"Live command file: {files.get('run_live_command')}",
            f"Fast live command file: {files.get('run_fast_live_command')}",
        ]
    )
    return "\n".join(lines)


def format_cv_health_markdown(summary: dict[str, Any]) -> str:
    validation = summary.get("validation") or {}
    timing = validation.get("timing_ms") or {}
    card_health = validation.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    models = summary.get("models") or {}
    knn = models.get("knn") or {}
    deep = models.get("deep") or {}
    bbox = summary.get("bbox") or {}
    lines = [
        "# CV Health",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Ready: `{summary.get('ready')}`",
        f"- KNN model: `{knn.get('path')}`",
        f"- Deep model: `{deep.get('path')}`",
        f"- BBox: `{bbox.get('normalized') or bbox.get('raw')}` concrete `{bbox.get('concrete')}`",
        f"- Validation median/p90: `{timing.get('median')}` / `{timing.get('p90')}` ms",
        f"- Card health: hero complete `{hero_health.get('complete_frames')}`, incomplete/missed `{hero_health.get('incomplete_or_missed_frames')}`, turn blocked `{hero_health.get('turn_blocked_frames')}`, board bad `{board_health.get('bad_frames')}`",
        f"- Card issues: `{json.dumps(card_health.get('issue_counts') or {}, ensure_ascii=False)}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Actual | Required |",
        "|---|---|---|---|",
    ]
    for item in summary.get("checks") or []:
        result = "PASS" if item.get("pass") else "FAIL"
        lines.append(f"| {item.get('name')} | {result} | `{item.get('actual')}` | `{item.get('required')}` |")
    lines.extend(
        [
            "",
            "## Preflight Command",
            "",
            "```powershell",
            str(summary.get("preflight_command") or ""),
            "```",
            "",
            "## Live Command",
            "",
            "```powershell",
            str(summary.get("live_command") or ""),
            "```",
            "",
            "## Fast Live Command",
            "",
            "This mode keeps the visual/card stream running but skips OCR until bottom action buttons are visible.",
            "",
            "```powershell",
            str(summary.get("fast_live_command") or ""),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
