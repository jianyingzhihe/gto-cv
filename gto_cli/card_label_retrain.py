from __future__ import annotations

import json
import os
import csv
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .card_active_learning import apply_card_review
from .card_candidate_summary import summarize_card_candidates
from .card_classifier import DEFAULT_MODEL_PATH, train_card_classifier
from .card_label_queue import audit_card_label_queue, clean_card
from .card_model_gate import gate_card_model
from .card_review_export import export_card_review
from .cv_validate import find_root_videos, validate_cv_videos


DEFAULT_BASE_GLYPH_DIR = Path("video_frames") / "card_review_all_after_red5_ok_organized"
DEFAULT_BENCHMARK_REVIEW_CSV = Path("video_frames") / "card_review_all_after_red5_override" / "review.csv"
DEFAULT_BASELINE_REVIEW_CSV = DEFAULT_BENCHMARK_REVIEW_CSV
DEFAULT_BASELINE_VALIDATION_SUMMARY = Path("video_frames") / "promoted_default_suitfix_validation" / "cv_validation_all_summary.json"
DEFAULT_DEEP_CARD_MODEL_DIR: Path | None = None


def retrain_card_label_queue(
    *,
    queue_csv: Path,
    output_dir: Path,
    base_glyph_dirs: list[Path] | None = None,
    video_dir: Path = Path("video_frames"),
    video_paths: list[Path] | None = None,
    benchmark_review_csvs: list[Path] | None = None,
    baseline_review_csv: Path = DEFAULT_BASELINE_REVIEW_CSV,
    baseline_validation_summary_json: Path | None = DEFAULT_BASELINE_VALIDATION_SUMMARY,
    deep_card_model_dir: Path | None = DEFAULT_DEEP_CARD_MODEL_DIR,
    candidate_name: str | None = None,
    model_path: Path | None = None,
    seed_model_path: Path | None = DEFAULT_MODEL_PATH,
    seed_conflict_policy: str = "manual_override",
    every_sec: float = 10.0,
    max_frames: int | None = 80,
    min_confidence: float = 0.35,
    augment: int = 8,
    glyph_augment: int | None = 8,
    include_templates: bool = True,
    allow_partial: bool = False,
    max_benchmark_samples: int | None = 300,
    max_diff_rows: int | None = 300,
    max_risk: int = 0,
    max_real_problem: int = 0,
    max_board_bad: int = 0,
    max_median_ms: float | None = 300.0,
    max_p90_ms: float | None = 900.0,
    max_median_regression_ms: float | None = None,
    max_p90_regression_ms: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = candidate_name or output_dir.name
    model_path = Path(model_path) if model_path else output_dir / f"{candidate_name}.npz"
    base_glyph_dirs = [Path(path) for path in (base_glyph_dirs or [])]
    if not base_glyph_dirs and DEFAULT_BASE_GLYPH_DIR.exists():
        base_glyph_dirs = [DEFAULT_BASE_GLYPH_DIR]
    benchmark_review_csvs = [Path(path) for path in (benchmark_review_csvs or [DEFAULT_BENCHMARK_REVIEW_CSV])]
    videos = [Path(path) for path in video_paths] if video_paths is not None else find_root_videos(video_dir)

    audit_dir = output_dir / "audit"
    applied_dir = output_dir / "applied_labels"
    review_dir = output_dir / "candidate_review"
    validation_dir = output_dir / "candidate_validation"
    gate_dir = output_dir / "gate"
    summary_dir = output_dir / "candidate_summary"

    audit = audit_card_label_queue(
        queue_csv=queue_csv,
        output_dir=audit_dir,
        applied_output_dir=applied_dir,
    )
    if not audit.get("ready_to_apply"):
        payload = build_stopped_payload(
            output_dir=output_dir,
            candidate_name=candidate_name,
            stage="audit",
            reason="queue_not_ready_to_apply",
            audit=audit,
        )
        write_retrain_outputs(output_dir, payload)
        return payload
    if not audit.get("ready_to_retrain") and not allow_partial:
        payload = build_stopped_payload(
            output_dir=output_dir,
            candidate_name=candidate_name,
            stage="audit",
            reason="queue_not_ready_to_retrain",
            audit=audit,
        )
        write_retrain_outputs(output_dir, payload)
        return payload

    applied = apply_card_review(review_csv=queue_csv, output_dir=applied_dir)
    if int(applied.get("copied_rank") or 0) <= 0 or int(applied.get("copied_suit") or 0) <= 0:
        payload = build_stopped_payload(
            output_dir=output_dir,
            candidate_name=candidate_name,
            stage="apply",
            reason="no_rank_or_suit_labels_copied",
            audit=audit,
            applied=applied,
        )
        write_retrain_outputs(output_dir, payload)
        return payload

    glyph_dirs = [*base_glyph_dirs, applied_dir]
    train = train_card_classifier(
        glyph_dirs=glyph_dirs,
        model_path=model_path,
        seed_model_path=seed_model_path,
        seed_conflict_policy=seed_conflict_policy,
        include_templates=include_templates,
        augment=augment,
        glyph_augment=glyph_augment,
    )

    with temporary_model_env(knn_model_path=model_path, deep_model_dir=deep_card_model_dir):
        review = export_card_review(
            video_paths=videos,
            output_dir=review_dir,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
        )
        validation = validate_cv_videos(
            video_paths=videos,
            video_dir=video_dir,
            output_dir=validation_dir,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
        )

    candidate_review_csv = Path((review.get("files") or {}).get("review_csv") or review_dir / "review.csv")
    candidate_review_with_truth_csv = merge_manual_truth_into_review(
        review_csv=candidate_review_csv,
        queue_csv=queue_csv,
        output_csv=review_dir / "review_with_truth.csv",
    )
    gate_benchmark_review_csvs = dedupe_paths([queue_csv, *benchmark_review_csvs])
    gate = gate_card_model(
        benchmark_review_csvs=gate_benchmark_review_csvs,
        baseline_review_csv=baseline_review_csv,
        candidate_review_csv=candidate_review_with_truth_csv,
        output_dir=gate_dir,
        candidate_name=candidate_name,
        candidate_evaluator="knn",
        knn_model_path=model_path,
        deep_model_dir=deep_card_model_dir,
        candidate_validation_summary_json=Path((validation.get("files") or {}).get("summary") or validation_dir / "cv_validation_all_summary.json"),
        baseline_validation_summary_json=baseline_validation_summary_json,
        include_ok_pseudo=True,
        max_benchmark_samples=max_benchmark_samples,
        max_diff_rows=max_diff_rows,
        max_risk=max_risk,
        max_real_problem=max_real_problem,
        max_board_bad=max_board_bad,
        max_median_ms=max_median_ms,
        max_p90_ms=max_p90_ms,
        max_median_regression_ms=max_median_regression_ms,
        max_p90_regression_ms=max_p90_regression_ms,
    )
    candidate_summary = summarize_card_candidates(search_dir=output_dir, output_dir=summary_dir)

    payload = {
        "ok": True,
        "stopped": False,
        "promote": bool(gate.get("promote")),
        "decision": gate.get("decision"),
        "candidate_name": candidate_name,
        "output_dir": str(output_dir),
        "queue_csv": str(queue_csv),
        "base_glyph_dirs": [str(path) for path in base_glyph_dirs],
        "applied_dir": str(applied_dir),
        "model_path": str(model_path),
        "seed_model_path": str(seed_model_path) if seed_model_path is not None else "",
        "seed_conflict_policy": seed_conflict_policy,
        "video_dir": str(video_dir),
        "video_count": len(videos),
        "benchmark_review_csvs": [str(path) for path in gate_benchmark_review_csvs],
        "baseline_review_csv": str(baseline_review_csv),
        "audit": audit,
        "applied": applied,
        "train": train,
        "review": compact_review(review),
        "validation": compact_validation(validation),
        "gate": gate,
        "candidate_summary": candidate_summary,
        "files": {
            "summary": str(output_dir / "label_retrain_summary.json"),
            "runbook": str(output_dir / "label_retrain_runbook.md"),
            "model": str(model_path),
            "candidate_review_csv": (review.get("files") or {}).get("review_csv"),
            "candidate_review_with_truth_csv": str(candidate_review_with_truth_csv),
            "candidate_validation_summary": (validation.get("files") or {}).get("summary"),
            "gate_summary": (gate.get("files") or {}).get("summary"),
            "gate_report": (gate.get("files") or {}).get("report_md"),
            "candidate_summary_md": (candidate_summary.get("files") or {}).get("summary_md"),
        },
    }
    write_retrain_outputs(output_dir, payload)
    return payload


def merge_manual_truth_into_review(*, review_csv: Path, queue_csv: Path, output_csv: Path) -> Path:
    truth = load_queue_truth_by_review_key(queue_csv)
    review_csv = Path(review_csv)
    output_csv = Path(output_csv)
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for column in ("final_card0", "final_card1", "notes"):
        if column not in fieldnames:
            fieldnames.append(column)
    applied = 0
    for row in rows:
        for slot in (0, 1):
            key = review_truth_key(row, slot)
            final_card = truth.get(key)
            if not final_card:
                continue
            row[f"final_card{slot}"] = final_card
            note = str(row.get("notes") or "")
            marker = f"manual_truth_slot{slot}={final_card}"
            row["notes"] = f"{note}; {marker}".strip("; ")
            applied += 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def load_queue_truth_by_review_key(queue_csv: Path) -> dict[tuple[str, str, str, int], str]:
    truth: dict[tuple[str, str, str, int], str] = {}
    with Path(queue_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for label_slot in (0, 1):
                final_card = clean_card(row.get(f"final_card{label_slot}"))
                if not final_card:
                    continue
                actual_slot = parse_original_slot(row, default=label_slot)
                key = review_truth_key(row, actual_slot)
                truth[key] = final_card
    return truth


def parse_original_slot(row: dict[str, Any], *, default: int) -> int:
    text = f"{row.get('notes') or ''};{row.get('reason') or ''}"
    match = re.search(r"(?:original_slot|slot)\s*=\s*([01])", text)
    if match:
        return int(match.group(1))
    return int(default)


def review_truth_key(row: dict[str, Any], slot: int) -> tuple[str, str, str, int]:
    return (
        Path(str(row.get("video") or "")).name,
        normalize_timestamp(row.get("timestamp_sec")),
        normalize_int_text(row.get("frame_index")),
        int(slot),
    )


def normalize_timestamp(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value or "").strip()


def normalize_int_text(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value or "").strip()


def dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(Path(path))
    return result


@contextmanager
def temporary_model_env(*, knn_model_path: Path, deep_model_dir: Path | None) -> Iterator[None]:
    old_values = {
        "GTO_CARD_KNN_MODEL": os.environ.get("GTO_CARD_KNN_MODEL"),
        "GTO_CARD_DEEP_MODEL_DIR": os.environ.get("GTO_CARD_DEEP_MODEL_DIR"),
        "GTO_CARD_DEEP_RANK_MODEL_DIR": os.environ.get("GTO_CARD_DEEP_RANK_MODEL_DIR"),
        "GTO_CARD_DEEP_SUIT_MODEL_DIR": os.environ.get("GTO_CARD_DEEP_SUIT_MODEL_DIR"),
    }
    os.environ["GTO_CARD_KNN_MODEL"] = str(knn_model_path)
    for key in ("GTO_CARD_DEEP_MODEL_DIR", "GTO_CARD_DEEP_RANK_MODEL_DIR", "GTO_CARD_DEEP_SUIT_MODEL_DIR"):
        os.environ.pop(key, None)
    if deep_model_dir is not None:
        os.environ["GTO_CARD_DEEP_MODEL_DIR"] = str(deep_model_dir)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_stopped_payload(
    *,
    output_dir: Path,
    candidate_name: str,
    stage: str,
    reason: str,
    audit: dict[str, Any] | None = None,
    applied: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "stopped": True,
        "stage": stage,
        "reason": reason,
        "promote": False,
        "decision": "stopped",
        "candidate_name": candidate_name,
        "output_dir": str(output_dir),
        "audit": audit or {},
        "applied": applied or {},
        "files": {
            "summary": str(output_dir / "label_retrain_summary.json"),
            "runbook": str(output_dir / "label_retrain_runbook.md"),
        },
    }


def compact_review(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": (payload.get("sample") or {}).get("rows"),
        "counts": payload.get("counts") or {},
        "files": payload.get("files") or {},
    }


def compact_validation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_count": payload.get("video_count"),
        "frame_count": (payload.get("sample") or {}).get("frame_count"),
        "counts": payload.get("counts") or {},
        "real_problem_count": payload.get("real_problem_count"),
        "board_bad_count": payload.get("board_bad_count"),
        "timing_ms": payload.get("timing_ms") or {},
        "files": payload.get("files") or {},
    }


def write_retrain_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_retrain_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_retrain_runbook.md").write_text(format_label_retrain_runbook(payload), encoding="utf-8")


def format_label_retrain_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"retrain-card-label-queue failed: {payload.get('error')}"
    if payload.get("stopped"):
        audit = payload.get("audit") or {}
        return "\n".join(
            [
                f"Stopped: {payload.get('reason')}",
                f"Stage: {payload.get('stage')}",
                f"Output: {payload.get('output_dir')}",
                f"Labeled slots: {audit.get('labeled_slots')} / {audit.get('total_slots')}",
                f"Invalid labels: {audit.get('invalid_label_count')}",
                f"Missing assets: {audit.get('missing_asset_count')}",
                f"Runbook: {(payload.get('files') or {}).get('runbook')}",
            ]
        )
    gate = payload.get("gate") or {}
    validation = payload.get("validation") or {}
    review = payload.get("review") or {}
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Candidate: {payload.get('candidate_name')}",
            f"Decision: {str(gate.get('decision') or payload.get('decision')).upper()}",
            f"Promote: {payload.get('promote')}",
            f"Model: {payload.get('model_path')}",
            f"Seed conflict policy: {payload.get('seed_conflict_policy')}",
            f"Review counts: {json.dumps(review.get('counts') or {}, ensure_ascii=False)}",
            f"Validation: real_problem={validation.get('real_problem_count')} board_bad={validation.get('board_bad_count')} timing={json.dumps(validation.get('timing_ms') or {}, ensure_ascii=False)}",
            f"Gate report: {files.get('gate_report')}",
            f"Summary: {files.get('summary')}",
            f"Runbook: {files.get('runbook')}",
        ]
    )


def format_label_retrain_runbook(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    lines = [
        "# Card Label Retrain Run",
        "",
        f"- Candidate: `{payload.get('candidate_name')}`",
        f"- Decision: `{payload.get('decision')}`",
        f"- Promote: `{payload.get('promote')}`",
        f"- Output: `{payload.get('output_dir')}`",
        f"- Queue CSV: `{payload.get('queue_csv') or ''}`",
        f"- Seed conflict policy: `{payload.get('seed_conflict_policy') or ''}`",
        "",
    ]
    if payload.get("stopped"):
        audit = payload.get("audit") or {}
        lines.extend(
            [
                "## Stopped",
                "",
                f"- Stage: `{payload.get('stage')}`",
                f"- Reason: `{payload.get('reason')}`",
                f"- Labeled slots: `{audit.get('labeled_slots')} / {audit.get('total_slots')}`",
                f"- Invalid labels: `{audit.get('invalid_label_count')}`",
                f"- Missing assets: `{audit.get('missing_asset_count')}`",
                "",
                "Fill the remaining `final_card0` / `final_card1` values and rerun the command.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Model: `{files.get('model')}`",
            f"- Candidate review CSV: `{files.get('candidate_review_csv')}`",
            f"- Candidate validation summary: `{files.get('candidate_validation_summary')}`",
            f"- Gate summary: `{files.get('gate_summary')}`",
            f"- Gate report: `{files.get('gate_report')}`",
            "",
            "## Gate",
            "",
            "```json",
            json.dumps(
                {
                    "decision": payload.get("decision"),
                    "promote": payload.get("promote"),
                    "checks": (payload.get("gate") or {}).get("checks") or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    if payload.get("promote"):
        lines.extend(
            [
                "## Promotion",
                "",
                "The gate passed. Promote manually only after reviewing the report:",
                "",
                "```powershell",
                f"Copy-Item \"{files.get('model')}\" \"pict\\card_models\\card_glyph_knn.npz\"",
                "```",
                "",
            ]
        )
    return "\n".join(lines)
