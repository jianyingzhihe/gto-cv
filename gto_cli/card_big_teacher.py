from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .card_candidate_summary import summarize_card_candidates
from .card_classifier import DEFAULT_MODEL_PATH, DEFAULT_TEMPLATE_DIR, clear_glyph_classify_cache, train_card_classifier
from .card_glyph_export import export_card_glyphs
from .card_hf_probe import (
    label_card_crops_hf_probe,
    load_vision_encoder,
    train_hf_card_probe,
)
from .card_label_queue import prepare_card_diff_label_queue
from .card_label_retrain import merge_manual_truth_into_review
from .card_model_gate import gate_card_model
from .card_review_export import export_card_review
from .cv_validate import find_root_videos, validate_cv_videos


DEFAULT_BIG_TEACHER_MODEL = "auto"
DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR = Path("video_frames") / "card_review_all_after_red5_ok_organized"
DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV = (
    Path("video_frames")
    / "glyph_current_prefill_candidate_20260709"
    / "combined_manual_truth_queue_balanced86"
    / "label_queue.csv"
)
DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV = Path("video_frames") / "card_review_all_after_red5_override" / "review.csv"
DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV = DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV
DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY = Path("video_frames") / "promoted_default_suitfix_validation" / "cv_validation_all_summary.json"
DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR: Path | None = None
AUTO_BIG_TEACHER_MODELS = (
    "facebook/dinov2-base",
    "facebook/dinov2-small",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
)


def default_big_teacher_benchmark_review_csvs() -> list[Path]:
    defaults = [
        DEFAULT_BIG_TEACHER_MANUAL_TRUTH_REVIEW_CSV,
        DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV,
    ]
    existing = [path for path in defaults if path.is_file()]
    return existing or [DEFAULT_BIG_TEACHER_BENCHMARK_REVIEW_CSV]


def run_card_big_teacher(
    *,
    video_paths: list[Path],
    input_dirs: list[Path],
    trusted_dirs: list[Path],
    output_dir: Path,
    probe_dir: Path | None = None,
    kind: str = "both",
    model_name: str = DEFAULT_BIG_TEACHER_MODEL,
    rank_model: str | None = None,
    suit_model: str | None = None,
    template_dir: Path | None = DEFAULT_TEMPLATE_DIR,
    include_templates: bool = True,
    every_sec: float = 5.0,
    max_frames: int | None = None,
    lock_layout: bool = True,
    include_board: bool = True,
    max_images: int | None = None,
    max_images_per_class: int | None = None,
    rank_score_threshold: float = 0.82,
    rank_margin_threshold: float = 0.10,
    suit_score_threshold: float = 0.82,
    suit_margin_threshold: float = 0.10,
    require_current_agreement: bool = False,
    copy_accepted: bool = True,
    batch_size: int = 32,
    temperature: float = 0.04,
    device: str = "auto",
    local_files_only: bool = False,
    distill_runtime: bool = False,
    runtime_output_dir: Path | None = None,
    runtime_model_path: Path | None = None,
    runtime_candidate_name: str | None = None,
    runtime_base_glyph_dirs: list[Path] | None = None,
    runtime_dataset_dirs: list[Path] | None = None,
    runtime_video_dir: Path = Path("video_frames"),
    runtime_video_paths: list[Path] | None = None,
    runtime_benchmark_review_csvs: list[Path] | None = None,
    runtime_baseline_review_csv: Path = DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV,
    runtime_baseline_validation_summary_json: Path | None = DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY,
    runtime_deep_card_model_dir: Path | None = DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR,
    runtime_seed_model_path: Path | None = DEFAULT_MODEL_PATH,
    runtime_seed_conflict_policy: str = "keep_seed",
    runtime_seed_guard: bool = False,
    runtime_seed_guard_rank_score: float = 0.55,
    runtime_seed_guard_rank_margin: float = 0.10,
    runtime_seed_guard_suit_score: float = 0.70,
    runtime_seed_guard_suit_margin: float = 0.04,
    runtime_every_sec: float = 10.0,
    runtime_max_frames: int | None = 80,
    runtime_min_confidence: float = 0.35,
    runtime_augment: int = 8,
    runtime_external_augment: int | None = None,
    runtime_glyph_augment: int | None = 8,
    runtime_max_external: int | None = None,
    runtime_min_accepted: int = 1,
    runtime_max_benchmark_samples: int | None = 300,
    runtime_max_diff_rows: int | None = 300,
    runtime_max_risk: int = 0,
    runtime_max_real_problem: int = 0,
    runtime_max_board_bad: int = 0,
    runtime_max_median_ms: float | None = 300.0,
    runtime_max_p90_ms: float | None = 900.0,
    runtime_max_median_regression_ms: float | None = None,
    runtime_max_p90_regression_ms: float | None = None,
    runtime_prepare_risk_queue: bool = True,
    runtime_risk_queue_max_rows: int = 80,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_dirs = [Path(path) for path in input_dirs]
    export_summary: dict[str, Any] | None = None
    if video_paths:
        crops_dir = output_dir / "crops"
        export_summary = export_card_glyphs(
            video_paths=[Path(path) for path in video_paths],
            output_dir=crops_dir,
            every_sec=every_sec,
            max_frames=max_frames,
            lock_layout=lock_layout,
            include_board=include_board,
        )
        crop_dirs.append(crops_dir)

    if not crop_dirs:
        raise ValueError("provide at least one --input-dir or video path to label")

    rank_model, suit_model, model_resolution = resolve_big_teacher_model_names(
        model_name=model_name,
        rank_model=rank_model,
        suit_model=suit_model,
        kind=kind,
        device=device,
        local_files_only=local_files_only,
    )

    train_summary: dict[str, Any] | None = None
    active_probe_dir = Path(probe_dir) if probe_dir else output_dir / "probe"
    if probe_dir is None:
        if not trusted_dirs:
            raise ValueError("--trusted-dir is required when --probe-dir is not provided")
        train_summary = train_hf_card_probe(
            input_dirs=[Path(path) for path in trusted_dirs],
            output_dir=active_probe_dir,
            kind=kind,
            model_name=model_name,
            rank_model=rank_model,
            suit_model=suit_model,
            template_dir=template_dir,
            include_templates=include_templates,
            max_images_per_class=max_images_per_class,
            batch_size=batch_size,
            temperature=temperature,
            device=device,
            local_files_only=local_files_only,
        )

    label_dir = output_dir / "labeled"
    label_summary = label_card_crops_hf_probe(
        input_dirs=crop_dirs,
        output_dir=label_dir,
        probe_dir=active_probe_dir,
        kind=kind,
        max_images=max_images,
        rank_score_threshold=rank_score_threshold,
        rank_margin_threshold=rank_margin_threshold,
        suit_score_threshold=suit_score_threshold,
        suit_margin_threshold=suit_margin_threshold,
        require_current_agreement=require_current_agreement,
        copy_accepted=copy_accepted,
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )
    probe_metadata = label_summary.get("probes") or {}
    effective_rank_model = str((probe_metadata.get("rank") or {}).get("model_name") or rank_model or model_name)
    effective_suit_model = str((probe_metadata.get("suit") or {}).get("model_name") or suit_model or model_name)
    if probe_dir is not None:
        model_resolution = model_resolution_from_reused_probe(
            existing_resolution=model_resolution,
            probe_metadata=probe_metadata,
            effective_rank_model=effective_rank_model,
            effective_suit_model=effective_suit_model,
            kind=kind,
        )
    distill_summary: dict[str, Any] | None = None
    if distill_runtime:
        distill_summary = distill_big_teacher_runtime(
            teacher_label_summary=label_summary,
            output_dir=runtime_output_dir or (output_dir / "runtime_candidate"),
            model_path=runtime_model_path,
            candidate_name=runtime_candidate_name or f"{output_dir.name}_runtime",
            base_glyph_dirs=runtime_base_glyph_dirs or [],
            dataset_dirs=runtime_dataset_dirs or [],
            video_dir=runtime_video_dir,
            video_paths=runtime_video_paths,
            benchmark_review_csvs=runtime_benchmark_review_csvs,
            baseline_review_csv=runtime_baseline_review_csv,
            baseline_validation_summary_json=runtime_baseline_validation_summary_json,
            deep_card_model_dir=runtime_deep_card_model_dir,
            seed_model_path=runtime_seed_model_path,
            seed_conflict_policy=runtime_seed_conflict_policy,
            seed_guard=runtime_seed_guard,
            seed_guard_rank_score=runtime_seed_guard_rank_score,
            seed_guard_rank_margin=runtime_seed_guard_rank_margin,
            seed_guard_suit_score=runtime_seed_guard_suit_score,
            seed_guard_suit_margin=runtime_seed_guard_suit_margin,
            every_sec=runtime_every_sec,
            max_frames=runtime_max_frames,
            min_confidence=runtime_min_confidence,
            augment=runtime_augment,
            external_augment=runtime_external_augment,
            glyph_augment=runtime_glyph_augment,
            max_external=runtime_max_external,
            min_accepted=runtime_min_accepted,
            max_benchmark_samples=runtime_max_benchmark_samples,
            max_diff_rows=runtime_max_diff_rows,
            max_risk=runtime_max_risk,
            max_real_problem=runtime_max_real_problem,
            max_board_bad=runtime_max_board_bad,
            max_median_ms=runtime_max_median_ms,
            max_p90_ms=runtime_max_p90_ms,
            max_median_regression_ms=runtime_max_median_regression_ms,
            max_p90_regression_ms=runtime_max_p90_regression_ms,
            prepare_risk_queue=runtime_prepare_risk_queue,
            risk_queue_max_rows=runtime_risk_queue_max_rows,
        )

    payload = {
        "ok": True,
        "output_dir": str(output_dir),
        "kind": kind,
        "video_paths": [str(path) for path in video_paths],
        "input_dirs": [str(path) for path in input_dirs],
        "trusted_dirs": [str(path) for path in trusted_dirs],
        "crop_dirs": [str(path) for path in crop_dirs],
        "probe_dir": str(active_probe_dir),
        "probe_reused": probe_dir is not None,
        "model_name": model_name,
        "rank_model": effective_rank_model,
        "suit_model": effective_suit_model,
        "model_resolution": model_resolution,
        "export": export_summary,
        "train": train_summary,
        "label": label_summary,
        "distill_runtime": distill_summary,
        "split": {
            "rank": {
                "classes": list("AKQJT98765432"),
                "score_threshold": float(rank_score_threshold),
                "margin_threshold": float(rank_margin_threshold),
            },
            "suit": {
                "classes": list("shdc"),
                "score_threshold": float(suit_score_threshold),
                "margin_threshold": float(suit_margin_threshold),
            },
        },
        "files": {
            "summary": str(output_dir / "summary.json"),
            "runbook": str(output_dir / "runbook.md"),
            "predictions_csv": str((label_summary.get("files") or {}).get("predictions_csv") or ""),
            "review_csv": str((label_summary.get("files") or {}).get("review_csv") or ""),
            "accepted_dir": str((label_summary.get("files") or {}).get("accepted_dir") or ""),
            "runtime_summary": str((distill_summary or {}).get("files", {}).get("summary") or ""),
            "runtime_runbook": str((distill_summary or {}).get("files", {}).get("runbook") or ""),
            "runtime_gate_report": str((distill_summary or {}).get("files", {}).get("gate_report") or ""),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "runbook.md").write_text(format_card_big_teacher_runbook(payload), encoding="utf-8")
    return payload


def distill_big_teacher_runtime(
    *,
    teacher_label_summary: dict[str, Any],
    output_dir: Path,
    model_path: Path | None = None,
    candidate_name: str = "big_teacher_runtime",
    base_glyph_dirs: list[Path] | None = None,
    dataset_dirs: list[Path] | None = None,
    video_dir: Path = Path("video_frames"),
    video_paths: list[Path] | None = None,
    benchmark_review_csvs: list[Path] | None = None,
    baseline_review_csv: Path = DEFAULT_BIG_TEACHER_BASELINE_REVIEW_CSV,
    baseline_validation_summary_json: Path | None = DEFAULT_BIG_TEACHER_BASELINE_VALIDATION_SUMMARY,
    deep_card_model_dir: Path | None = DEFAULT_BIG_TEACHER_DEEP_CARD_MODEL_DIR,
    seed_model_path: Path | None = DEFAULT_MODEL_PATH,
    seed_conflict_policy: str = "keep_seed",
    seed_guard: bool = False,
    seed_guard_rank_score: float = 0.55,
    seed_guard_rank_margin: float = 0.10,
    seed_guard_suit_score: float = 0.70,
    seed_guard_suit_margin: float = 0.04,
    every_sec: float = 10.0,
    max_frames: int | None = 80,
    min_confidence: float = 0.35,
    augment: int = 8,
    external_augment: int | None = None,
    glyph_augment: int | None = 8,
    max_external: int | None = None,
    min_accepted: int = 1,
    max_benchmark_samples: int | None = 300,
    max_diff_rows: int | None = 300,
    max_risk: int = 0,
    max_real_problem: int = 0,
    max_board_bad: int = 0,
    max_median_ms: float | None = 300.0,
    max_p90_ms: float | None = 900.0,
    max_median_regression_ms: float | None = None,
    max_p90_regression_ms: float | None = None,
    prepare_risk_queue: bool = True,
    risk_queue_max_rows: int = 80,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path) if model_path else output_dir / f"{candidate_name}.npz"
    accepted_dir = Path(str((teacher_label_summary.get("files") or {}).get("accepted_dir") or ""))
    accepted_count = int(teacher_label_summary.get("accepted") or 0)
    if not accepted_dir.exists() or accepted_count < max(0, int(min_accepted)):
        payload = {
            "ok": True,
            "stopped": True,
            "stage": "teacher_label",
            "reason": "not_enough_accepted_teacher_crops",
            "candidate_name": candidate_name,
            "output_dir": str(output_dir),
            "accepted_dir": str(accepted_dir),
            "accepted": accepted_count,
            "min_accepted": int(min_accepted),
            "seed_model_path": str(seed_model_path) if seed_model_path is not None else "",
            "seed_conflict_policy": seed_conflict_policy,
            "seed_guard": bool(seed_guard),
            "promote": False,
            "decision": "stopped",
            "files": {
                "summary": str(output_dir / "runtime_distill_summary.json"),
                "runbook": str(output_dir / "runtime_distill_runbook.md"),
            },
        }
        write_runtime_distill_outputs(output_dir, payload)
        return payload

    base_dirs = [Path(path) for path in (base_glyph_dirs or [])]
    if not base_dirs and DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR.exists():
        base_dirs = [DEFAULT_BIG_TEACHER_BASE_GLYPH_DIR]
    dataset_dirs = [Path(path) for path in (dataset_dirs or [])]
    benchmark_csvs = [Path(path) for path in (benchmark_review_csvs or default_big_teacher_benchmark_review_csvs())]
    videos = [Path(path) for path in video_paths] if video_paths is not None else find_root_videos(video_dir)

    review_dir = output_dir / "candidate_review"
    validation_dir = output_dir / "candidate_validation"
    gate_dir = output_dir / "gate"
    candidate_summary_dir = output_dir / "candidate_summary"

    train = train_card_classifier(
        glyph_dirs=[*base_dirs, accepted_dir],
        dataset_dirs=dataset_dirs,
        model_path=model_path,
        seed_model_path=seed_model_path,
        seed_conflict_policy=seed_conflict_policy,
        seed_guard=seed_guard,
        seed_guard_rank_score=seed_guard_rank_score,
        seed_guard_rank_margin=seed_guard_rank_margin,
        seed_guard_suit_score=seed_guard_suit_score,
        seed_guard_suit_margin=seed_guard_suit_margin,
        include_templates=True,
        augment=augment,
        external_augment=external_augment,
        glyph_augment=glyph_augment,
        max_external=max_external,
    )

    with temporary_model_env(knn_model_path=model_path, deep_model_dir=deep_card_model_dir):
        review = export_card_review(
            video_paths=videos,
            output_dir=review_dir,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
        )
        clear_glyph_classify_cache()
        validation = validate_cv_videos(
            video_paths=videos,
            video_dir=video_dir,
            output_dir=validation_dir,
            every_sec=every_sec,
            max_frames=max_frames,
            min_confidence=min_confidence,
        )

    candidate_review_csv = Path((review.get("files") or {}).get("review_csv") or review_dir / "review.csv")
    candidate_review_for_gate = merge_runtime_truth_if_available(
        review_csv=candidate_review_csv,
        truth_csvs=benchmark_csvs,
        output_dir=review_dir,
    )
    gate = gate_card_model(
        benchmark_review_csvs=benchmark_csvs,
        baseline_review_csv=baseline_review_csv,
        candidate_review_csv=candidate_review_for_gate,
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
    risk_label_queue = prepare_runtime_risk_label_queue(
        gate=gate,
        output_dir=output_dir / "risk_label_queue",
        max_rows=risk_queue_max_rows,
    ) if prepare_risk_queue else None
    candidate_summary = summarize_card_candidates(search_dir=output_dir, output_dir=candidate_summary_dir)
    payload = {
        "ok": True,
        "stopped": False,
        "promote": bool(gate.get("promote")),
        "decision": gate.get("decision"),
        "candidate_name": candidate_name,
        "output_dir": str(output_dir),
        "accepted_dir": str(accepted_dir),
        "accepted": accepted_count,
        "base_glyph_dirs": [str(path) for path in base_dirs],
        "dataset_dirs": [str(path) for path in dataset_dirs],
        "model_path": str(model_path),
        "seed_model_path": str(seed_model_path) if seed_model_path is not None else "",
        "seed_conflict_policy": seed_conflict_policy,
        "seed_guard": bool(seed_guard),
        "seed_guard_thresholds": {
            "rank_score": float(seed_guard_rank_score),
            "rank_margin": float(seed_guard_rank_margin),
            "suit_score": float(seed_guard_suit_score),
            "suit_margin": float(seed_guard_suit_margin),
        },
        "deep_card_model_dir": str(deep_card_model_dir) if deep_card_model_dir is not None else "",
        "video_dir": str(video_dir),
        "video_count": len(videos),
        "benchmark_review_csvs": [str(path) for path in benchmark_csvs],
        "baseline_review_csv": str(baseline_review_csv),
        "train": train,
        "review": compact_review(review),
        "validation": compact_validation(validation),
        "gate": gate,
        "risk_label_queue": risk_label_queue,
        "candidate_summary": candidate_summary,
        "files": {
            "summary": str(output_dir / "runtime_distill_summary.json"),
            "runbook": str(output_dir / "runtime_distill_runbook.md"),
            "model": str(model_path),
            "candidate_review_csv": (review.get("files") or {}).get("review_csv"),
            "candidate_review_with_truth_csv": str(candidate_review_for_gate) if candidate_review_for_gate != candidate_review_csv else "",
            "candidate_validation_summary": (validation.get("files") or {}).get("summary"),
            "gate_summary": (gate.get("files") or {}).get("summary"),
            "gate_report": (gate.get("files") or {}).get("report_md"),
            "risk_label_queue_csv": ((risk_label_queue or {}).get("files") or {}).get("label_queue_csv"),
            "risk_label_queue_html": ((risk_label_queue or {}).get("files") or {}).get("label_queue_html"),
            "risk_label_queue_sheet": ((risk_label_queue or {}).get("files") or {}).get("label_queue_sheet"),
            "candidate_summary_md": (candidate_summary.get("files") or {}).get("summary_md"),
        },
    }
    write_runtime_distill_outputs(output_dir, payload)
    return payload


def merge_runtime_truth_if_available(*, review_csv: Path, truth_csvs: list[Path], output_dir: Path) -> Path:
    review_csv = Path(review_csv)
    if not review_csv.is_file():
        return review_csv
    merged_csv = review_csv
    applied_any = False
    for index, truth_csv in enumerate(truth_csvs):
        truth_csv = Path(truth_csv)
        if not truth_csv.is_file() or not csv_has_final_card_columns(truth_csv):
            continue
        output_csv = output_dir / ("review_with_truth.csv" if not applied_any else f"review_with_truth_{index + 1}.csv")
        merged_csv = merge_manual_truth_into_review(
            review_csv=merged_csv,
            queue_csv=truth_csv,
            output_csv=output_csv,
        )
        applied_any = True
    return merged_csv


def csv_has_final_card_columns(path: Path) -> bool:
    try:
        import csv

        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
    except OSError:
        return False
    return bool({"final_card0", "final_card1"} & fieldnames)


def model_resolution_from_reused_probe(
    *,
    existing_resolution: dict[str, Any],
    probe_metadata: dict[str, Any],
    effective_rank_model: str,
    effective_suit_model: str,
    kind: str,
) -> dict[str, Any]:
    """Describe the real model source when --probe-dir is reused."""
    resolution = dict(existing_resolution or {})
    resolution["mode"] = "probe-reused"
    resolution["note"] = "rank/suit encoders are read from the reused probe metadata"
    if kind in ("rank", "both"):
        rank_meta = probe_metadata.get("rank") or {}
        resolution["rank"] = {
            "selected": effective_rank_model,
            "source": "probe_metadata",
            "probe_path": str(rank_meta.get("model_path") or ""),
        }
    if kind in ("suit", "both"):
        suit_meta = probe_metadata.get("suit") or {}
        resolution["suit"] = {
            "selected": effective_suit_model,
            "source": "probe_metadata",
            "probe_path": str(suit_meta.get("model_path") or ""),
        }
    return resolution


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


def write_runtime_distill_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime_distill_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "runtime_distill_runbook.md").write_text(format_runtime_distill_runbook(payload), encoding="utf-8")


def prepare_runtime_risk_label_queue(*, gate: dict[str, Any], output_dir: Path, max_rows: int = 80) -> dict[str, Any] | None:
    diff = gate.get("diff") or {}
    risk_count = int((diff.get("counts") or {}).get("risk_count") or 0)
    diff_csv = Path(str((gate.get("files") or {}).get("diff_rows_csv") or ""))
    if risk_count <= 0 or not diff_csv.is_file():
        return None
    return prepare_card_diff_label_queue(
        diff_csv=diff_csv,
        output_dir=output_dir,
        max_rows=max_rows,
        risk_only=True,
        include_same=False,
        copy_assets=True,
    )


def resolve_big_teacher_model_names(
    *,
    model_name: str,
    rank_model: str | None,
    suit_model: str | None,
    kind: str,
    device: str,
    local_files_only: bool,
    loadable_check: Any | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    checker = loadable_check or is_hf_model_loadable_locally
    requested = {
        "model": model_name,
        "rank_model": rank_model,
        "suit_model": suit_model,
        "kind": kind,
        "local_files_only": bool(local_files_only),
    }
    resolved: dict[str, Any] = {
        "requested": requested,
        "mode": "explicit",
        "candidates": list(AUTO_BIG_TEACHER_MODELS),
        "rank": {},
        "suit": {},
    }

    def resolve_one(one_kind: str, explicit: str | None) -> str | None:
        if one_kind == "rank" and kind == "suit":
            return explicit
        if one_kind == "suit" and kind == "rank":
            return explicit
        candidate_text = explicit or model_name
        if candidate_text not in ("auto", "auto-local", ""):
            resolved[one_kind] = {"selected": candidate_text, "source": "explicit"}
            return candidate_text

        resolved["mode"] = "auto-local"
        attempts = []
        for candidate in AUTO_BIG_TEACHER_MODELS:
            ok, reason = checker(candidate, device=device)
            attempts.append({"model": candidate, "ok": bool(ok), "reason": reason})
            if ok:
                resolved[one_kind] = {"selected": candidate, "source": "auto-local", "attempts": attempts}
                return candidate
        fallback = AUTO_BIG_TEACHER_MODELS[0]
        resolved[one_kind] = {
            "selected": fallback,
            "source": "auto-fallback-download" if not local_files_only else "auto-fallback-unavailable",
            "attempts": attempts,
        }
        return fallback

    resolved_rank = resolve_one("rank", rank_model)
    resolved_suit = resolve_one("suit", suit_model)
    return resolved_rank, resolved_suit, resolved


def is_hf_model_loadable_locally(model_name: str, *, device: str = "auto") -> tuple[bool, str]:
    try:
        load_vision_encoder(model_name, device=device, local_files_only=True)
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error).splitlines()[0][:240]}"
    return True, "local_load_ok"


def format_card_big_teacher_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"card-big-teacher failed: {payload.get('error')}"
    label = payload.get("label") or {}
    files = payload.get("files") or {}
    train = payload.get("train") or {}
    distill = payload.get("distill_runtime") or {}
    lines = [
        f"Output: {payload.get('output_dir')}",
        f"Probe: {payload.get('probe_dir')} ({'reused' if payload.get('probe_reused') else 'trained'})",
        f"Model: rank={payload.get('rank_model')} suit={payload.get('suit_model')}",
        f"Crop dirs: {', '.join(payload.get('crop_dirs') or [])}",
    ]
    if train:
        for one_kind, result in (train.get("results") or {}).items():
            val = result.get("val") or {}
            accuracy = val.get("accuracy")
            accuracy_text = "-" if accuracy is None else f"{float(accuracy):.3f}"
            lines.append(f"{one_kind} train: source={result.get('source_count')} val_acc={accuracy_text}")
    lines.extend(
        [
            f"Processed: {label.get('processed', 0)}",
            f"Accepted: {label.get('accepted', 0)}",
            f"Review: {label.get('review', 0)}",
            f"Predictions CSV: {files.get('predictions_csv')}",
            f"Review CSV: {files.get('review_csv')}",
            f"Runbook: {files.get('runbook')}",
        ]
    )
    if distill:
        if distill.get("stopped"):
            lines.extend(
                [
                    f"Runtime distill: stopped ({distill.get('reason')})",
                    f"Runtime runbook: {(distill.get('files') or {}).get('runbook')}",
                ]
            )
        else:
            lines.extend(
                [
                    f"Runtime candidate: {distill.get('candidate_name')} decision={str(distill.get('decision') or '').upper()} promote={distill.get('promote')}",
                    f"Runtime model: {distill.get('model_path')}",
                    f"Runtime gate report: {(distill.get('files') or {}).get('gate_report')}",
                    f"Runtime risk queue: {(distill.get('files') or {}).get('risk_label_queue_html') or '-'}",
                    f"Runtime risk sheet: {(distill.get('files') or {}).get('risk_label_queue_sheet') or '-'}",
                ]
            )
    return "\n".join(lines)


def format_runtime_distill_runbook(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    lines = [
        "# Big Teacher Runtime Distill",
        "",
        f"- Candidate: `{payload.get('candidate_name')}`",
        f"- Decision: `{payload.get('decision')}`",
        f"- Promote: `{payload.get('promote')}`",
        f"- Output: `{payload.get('output_dir')}`",
        f"- Accepted teacher crops: `{payload.get('accepted')}`",
        "",
    ]
    if payload.get("stopped"):
        lines.extend(
            [
                "## Stopped",
                "",
                f"- Stage: `{payload.get('stage')}`",
                f"- Reason: `{payload.get('reason')}`",
                f"- Accepted dir: `{payload.get('accepted_dir')}`",
                f"- Minimum accepted: `{payload.get('min_accepted')}`",
                "",
                "Raise teacher acceptance only by improving labels/model confidence, not by relaxing the runtime gate.",
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
            f"- Risk label queue: `{files.get('risk_label_queue_html') or ''}`",
            f"- Risk contact sheet: `{files.get('risk_label_queue_sheet') or ''}`",
            f"- Candidate comparison: `{files.get('candidate_summary_md')}`",
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


def format_card_big_teacher_runbook(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    label = payload.get("label") or {}
    output_dir = Path(str(payload.get("output_dir") or "video_frames/card_big_teacher"))
    accepted_dir = files.get("accepted_dir") or str(output_dir / "labeled")
    candidate_model = Path("pict") / "card_models" / "card_glyph_knn_big_teacher_candidate.npz"
    return "\n".join(
        [
            "# Card Big Teacher Pipeline",
            "",
            "This run uses a frozen HuggingFace vision model as an offline teacher over already cropped glyphs.",
            "",
            "## Split Tasks",
            "",
            "- Rank: 13-way classification over `A K Q J T 9 8 7 6 5 4 3 2`.",
            "- Suit: 4-way classification over `s h d c`.",
            "- Final card: rank result plus suit result, for example `7c` or `Qh`.",
            "",
            "## Current Run",
            "",
            f"- Output: `{payload.get('output_dir')}`",
            f"- Probe: `{payload.get('probe_dir')}`",
            f"- Rank model: `{payload.get('rank_model')}`",
            f"- Suit model: `{payload.get('suit_model')}`",
            f"- Model resolution: `{json.dumps(payload.get('model_resolution') or {}, ensure_ascii=False)}`",
            f"- Crop dirs: `{'; '.join(payload.get('crop_dirs') or [])}`",
            f"- Processed crops: `{label.get('processed', 0)}`",
            f"- Accepted crops: `{label.get('accepted', 0)}`",
            f"- Review crops: `{label.get('review', 0)}`",
            f"- Predictions: `{files.get('predictions_csv')}`",
            f"- Manual review: `{files.get('review_csv')}`",
            f"- Runtime distill summary: `{files.get('runtime_summary') or ''}`",
            "",
            "## Next Commands",
            "",
            "After manually fixing rows in the review CSV, organize accepted/fixed crops and train a runtime KNN candidate:",
            "",
            "```powershell",
            f"python gto.py organize-card-crops --input-dir \"{accepted_dir}\" --output-dir \"{output_dir / 'organized'}\" --format text",
            f"python gto.py train-card-classifier --glyph-dir \"{output_dir / 'organized'}\" --model \"{candidate_model}\" --format text",
            "```",
            "",
            "Then run the existing benchmark/gate commands before promoting the candidate to live.",
            "",
        ]
    )
