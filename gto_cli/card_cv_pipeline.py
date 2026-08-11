from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .card_classifier import DEFAULT_MODEL_PATH, RANK_LABELS, SUIT_LABELS, load_cv
from .card_dataset_download import DEFAULT_HF_CARD_DIR, DEFAULT_HF_CARD_REPO, dataset_summary, download_card_dataset
from .card_glyph_export import ingest_external_card_images
from .card_hf_probe import PROBE_FILENAMES, load_probe, train_hf_card_probe
from .card_teacher_label import collect_crop_records
from .card_candidate_summary import summarize_card_candidates
from .bbox_diagnostics import diagnose_auto_bbox_videos
from .cv_health import (
    DEFAULT_DEEP_MODEL_DIR,
    DEFAULT_GATE_SUMMARY,
    DEFAULT_VALIDATION_SUMMARY,
    inspect_bbox,
    inspect_deep_model_dir,
    inspect_knn_model,
    load_json_file,
)
from .cv_validate import find_root_videos
from .card_big_teacher import run_card_big_teacher


DEFAULT_PIPELINE_OUTPUT_DIR = Path("video_frames") / "card_cv_pipeline"
DEFAULT_PIPELINE_CROP_DIR = Path("video_frames") / "current_runtime_final_review"
DEFAULT_PIPELINE_PROBE_DIR = Path("pict") / "card_models" / "hf_probe_dinov2_base_v1_review"
DEFAULT_PIPELINE_INGEST_DIR = Path("video_frames") / "external_card_glyphs"


def inspect_card_cv_pipeline(
    *,
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT_DIR,
    bbox: str = "x,y,w,h",
    hero_name: str | None = None,
    video_dir: Path = Path("video_frames"),
    crop_dirs: list[Path] | None = None,
    probe_dir: Path = DEFAULT_PIPELINE_PROBE_DIR,
    probe_model: str = "facebook/dinov2-base",
    probe_rank_model: str | None = None,
    probe_suit_model: str | None = None,
    probe_max_images_per_class: int | None = 24,
    probe_batch_size: int = 16,
    run_train_probe: bool = False,
    dataset_repo_id: str = DEFAULT_HF_CARD_REPO,
    dataset_repo_type: str = "dataset",
    dataset_allow_patterns: list[str] | None = None,
    dataset_dir: Path = DEFAULT_HF_CARD_DIR,
    extra_dataset_dirs: list[Path] | None = None,
    ingested_dataset_dir: Path = DEFAULT_PIPELINE_INGEST_DIR,
    knn_model_path: Path = DEFAULT_MODEL_PATH,
    deep_model_dir: Path | None = DEFAULT_DEEP_MODEL_DIR,
    validation_summary_json: Path = DEFAULT_VALIDATION_SUMMARY,
    gate_summary_json: Path = DEFAULT_GATE_SUMMARY,
    download_dataset_flag: bool = False,
    refresh_dataset: bool = False,
    local_files_only: bool = False,
    ingest_dataset_flag: bool = False,
    max_external_ingest: int | None = 1200,
    run_smoke: bool = False,
    smoke_max_images: int = 20,
    smoke_output_dir: Path | None = None,
    smoke_local_files_only: bool = True,
    smoke_batch_size: int = 16,
    run_teacher: bool = False,
    teacher_output_dir: Path | None = None,
    teacher_max_images: int | None = None,
    teacher_local_files_only: bool = True,
    teacher_batch_size: int = 16,
    teacher_distill_runtime: bool = False,
    teacher_runtime_video_paths: list[Path] | None = None,
    teacher_runtime_every_sec: float = 10.0,
    teacher_runtime_max_frames: int | None = 80,
    teacher_runtime_max_benchmark_samples: int | None = 300,
    teacher_runtime_max_diff_rows: int | None = 300,
    summarize_candidates: bool = True,
    candidate_search_dir: Path = Path("video_frames"),
    candidate_output_dir: Path | None = None,
    keep_candidate_duplicates: bool = False,
    audit_crop_images: bool = True,
    min_rank_per_label: int = 1,
    min_suit_per_label: int = 1,
    run_auto_bbox_diagnostics: bool = False,
    auto_bbox_output_dir: Path | None = None,
    auto_bbox_every_sec: float = 300.0,
    auto_bbox_max_frames: int | None = 1,
    auto_bbox_variants: list[str] | None = None,
    auto_bbox_save_problem_frames: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_dirs = [Path(path) for path in (crop_dirs or [DEFAULT_PIPELINE_CROP_DIR])]
    probe_dir = Path(probe_dir)
    dataset_dir = Path(dataset_dir)
    extra_dataset_dirs = [Path(path) for path in (extra_dataset_dirs or [])]
    all_dataset_dirs = [dataset_dir, *extra_dataset_dirs]
    ingested_dataset_dir = Path(ingested_dataset_dir)
    validation = load_json_file(Path(validation_summary_json))
    gate = load_json_file(Path(gate_summary_json))
    videos = find_root_videos(Path(video_dir))

    dataset_download = None
    if download_dataset_flag:
        dataset_download = download_card_dataset(
            repo_id=dataset_repo_id,
            output_dir=dataset_dir,
            repo_type=dataset_repo_type,
            allow_patterns=dataset_allow_patterns or None,
            refresh=refresh_dataset,
            local_files_only=local_files_only,
        )

    dataset = dataset_summary(
        dataset_dir,
        repo_id=dataset_repo_id,
        repo_type=dataset_repo_type,
        downloaded=bool(dataset_download and dataset_download.get("downloaded")),
        reason=str((dataset_download or {}).get("reason") or ("existing_output_dir" if dataset_dir.exists() else "missing")),
    )
    datasets = [
        dataset_summary(
            path,
            repo_id=dataset_repo_id if index == 0 else "local",
            repo_type=dataset_repo_type if index == 0 else "directory",
            downloaded=False,
            reason="primary" if index == 0 else "extra_dataset_dir",
        )
        for index, path in enumerate(all_dataset_dirs)
    ]
    ingest_dataset_dirs, skipped_ingest_dataset_dirs = filter_ingest_dataset_dirs(
        all_dataset_dirs,
        output_dir=ingested_dataset_dir,
    )
    probe_train_input_dirs = build_probe_train_input_dirs(
        crop_dirs,
        ingested_dataset_dir=ingested_dataset_dir,
        include_ingested=bool(ingest_dataset_dirs),
    )

    external_ingest = None
    if ingest_dataset_flag:
        external_ingest = ingest_external_card_images(
            dataset_dirs=ingest_dataset_dirs,
            output_dir=ingested_dataset_dir,
            max_images=max_external_ingest,
        )

    crop_summary = inspect_crop_dirs(
        crop_dirs,
        audit_images=audit_crop_images,
        min_rank_per_label=min_rank_per_label,
        min_suit_per_label=min_suit_per_label,
    )
    probe_train_run = None
    if run_train_probe:
        active_probe_train_dirs = [path for path in probe_train_input_dirs if path.exists()]
        if not active_probe_train_dirs:
            probe_train_run = {"ok": False, "error": "no_existing_crop_dir"}
        else:
            probe_train_run = train_hf_card_probe(
                input_dirs=active_probe_train_dirs,
                output_dir=probe_dir,
                kind="both",
                model_name=probe_model,
                rank_model=probe_rank_model,
                suit_model=probe_suit_model,
                max_images_per_class=probe_max_images_per_class,
                batch_size=max(1, int(probe_batch_size)),
                local_files_only=local_files_only,
            )
    probe = inspect_probe_dir(probe_dir)
    knn = inspect_knn_model(Path(knn_model_path))
    deep = inspect_deep_model_dir(deep_model_dir)
    bbox_info = inspect_bbox(bbox)
    commands = build_pipeline_commands(
        bbox=bbox,
        hero_name=hero_name,
        crop_dirs=crop_dirs,
        probe_dir=probe_dir,
        probe_model=probe_model,
        probe_rank_model=probe_rank_model,
        probe_suit_model=probe_suit_model,
        probe_max_images_per_class=probe_max_images_per_class,
        probe_batch_size=probe_batch_size,
        dataset_dir=dataset_dir,
        extra_dataset_dirs=extra_dataset_dirs,
        skipped_ingest_dataset_dirs=skipped_ingest_dataset_dirs,
        ingested_dataset_dir=ingested_dataset_dir,
        output_dir=output_dir,
        video_dir=Path(video_dir),
        knn_model_path=Path(knn_model_path),
        deep_model_dir=deep_model_dir,
        auto_bbox_output_dir=auto_bbox_output_dir or output_dir / "auto_bbox_diagnostics",
        auto_bbox_every_sec=auto_bbox_every_sec,
        auto_bbox_max_frames=auto_bbox_max_frames,
        auto_bbox_variants=auto_bbox_variants,
    )
    command_files = write_command_files(output_dir / "commands", commands)

    auto_bbox_diagnostics = None
    if run_auto_bbox_diagnostics:
        auto_bbox_diagnostics = diagnose_auto_bbox_videos(
            video_dir=Path(video_dir),
            output_dir=Path(auto_bbox_output_dir or output_dir / "auto_bbox_diagnostics"),
            every_sec=auto_bbox_every_sec,
            max_frames=auto_bbox_max_frames,
            min_confidence=0.35,
            variants=tuple(auto_bbox_variants) if auto_bbox_variants else None,
            save_problem_frames=auto_bbox_save_problem_frames,
        )

    candidate_summary = None
    if summarize_candidates:
        candidate_summary = summarize_card_candidates(
            search_dir=Path(candidate_search_dir),
            output_dir=Path(candidate_output_dir or output_dir / "candidate_summary"),
            keep_duplicates=keep_candidate_duplicates,
        )

    smoke = None
    if run_smoke:
        active_crop_dirs = [path for path in crop_dirs if path.exists()]
        if not active_crop_dirs:
            smoke = {"ok": False, "error": "no_existing_crop_dir"}
        elif not probe.get("ready"):
            smoke = {"ok": False, "error": "probe_not_ready"}
        else:
            smoke = run_card_big_teacher(
                video_paths=[],
                input_dirs=active_crop_dirs,
                trusted_dirs=[],
                output_dir=Path(smoke_output_dir or output_dir / "big_teacher_smoke"),
                probe_dir=probe_dir,
                kind="both",
                max_images=max(0, int(smoke_max_images)),
                batch_size=max(1, int(smoke_batch_size)),
                rank_score_threshold=0.55,
                rank_margin_threshold=0.04,
                suit_score_threshold=0.65,
                suit_margin_threshold=0.06,
                require_current_agreement=True,
                local_files_only=smoke_local_files_only,
            )

    teacher_run = None
    if run_teacher:
        active_crop_dirs = [path for path in crop_dirs if path.exists()]
        if not active_crop_dirs:
            teacher_run = {"ok": False, "error": "no_existing_crop_dir"}
        elif not probe.get("ready"):
            teacher_run = {"ok": False, "error": "probe_not_ready"}
        else:
            teacher_run = run_card_big_teacher(
                video_paths=[],
                input_dirs=active_crop_dirs,
                trusted_dirs=[],
                output_dir=Path(teacher_output_dir or output_dir / "big_teacher_run"),
                probe_dir=probe_dir,
                kind="both",
                max_images=teacher_max_images,
                batch_size=max(1, int(teacher_batch_size)),
                rank_score_threshold=0.55,
                rank_margin_threshold=0.04,
                suit_score_threshold=0.65,
                suit_margin_threshold=0.06,
                require_current_agreement=True,
                local_files_only=teacher_local_files_only,
                distill_runtime=teacher_distill_runtime,
                runtime_video_paths=[Path(path) for path in teacher_runtime_video_paths] if teacher_runtime_video_paths else None,
                runtime_every_sec=teacher_runtime_every_sec,
                runtime_max_frames=teacher_runtime_max_frames,
                runtime_max_benchmark_samples=teacher_runtime_max_benchmark_samples,
                runtime_max_diff_rows=teacher_runtime_max_diff_rows,
            )

    checks = build_pipeline_checks(
        bbox_info=bbox_info,
        videos=videos,
        crop_summary=crop_summary,
        probe=probe,
        knn=knn,
        deep=deep,
        validation=validation,
        gate=gate,
        auto_bbox_diagnostics=auto_bbox_diagnostics,
    )
    ready_for_live = all(
        bool(check.get("pass"))
        for check in checks
        if check.get("scope") in {"live", "shared"}
    )
    ready_for_training = all(
        bool(check.get("pass"))
        for check in checks
        if check.get("scope") in {"training", "shared"}
    )

    payload = {
        "ok": True,
        "output_dir": str(output_dir),
        "ready_for_live": ready_for_live,
        "ready_for_training": ready_for_training,
        "next_stage": choose_next_stage(checks),
        "checks": checks,
        "bbox": bbox_info,
        "hero_name": hero_name or "",
        "videos": {"video_dir": str(video_dir), "count": len(videos), "examples": [str(path) for path in videos[:10]]},
        "models": {"knn": knn, "deep": deep, "hf_probe": probe},
        "probe_train_run": compact_probe_train_run(probe_train_run),
        "crops": crop_summary,
        "dataset": dataset,
        "datasets": datasets,
        "ingest_dataset_dirs": [str(path) for path in ingest_dataset_dirs],
        "skipped_ingest_dataset_dirs": [str(path) for path in skipped_ingest_dataset_dirs],
        "probe_train_input_dirs": [str(path) for path in probe_train_input_dirs],
        "dataset_download": dataset_download,
        "external_ingest": external_ingest,
        "candidate_summary": compact_candidate_summary(candidate_summary),
        "auto_bbox_diagnostics": compact_auto_bbox_diagnostics(auto_bbox_diagnostics),
        "validation": compact_validation(validation),
        "gate": compact_gate(gate),
        "smoke": compact_smoke(smoke),
        "teacher_run": compact_teacher_run(teacher_run),
        "commands": commands,
        "files": {
            "summary": str(output_dir / "card_cv_pipeline_summary.json"),
            "runbook": str(output_dir / "card_cv_pipeline_runbook.md"),
            "commands_dir": str(output_dir / "commands"),
            "command_files": command_files,
        },
    }
    (output_dir / "card_cv_pipeline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "card_cv_pipeline_runbook.md").write_text(format_card_cv_pipeline_runbook(payload), encoding="utf-8-sig")
    return payload


def inspect_crop_dirs(
    crop_dirs: list[Path],
    *,
    audit_images: bool = False,
    min_rank_per_label: int = 1,
    min_suit_per_label: int = 1,
) -> dict[str, Any]:
    return inspect_crop_dirs_detailed(
        crop_dirs,
        audit_images=audit_images,
        min_rank_per_label=min_rank_per_label,
        min_suit_per_label=min_suit_per_label,
    )


def inspect_crop_dirs_detailed(
    crop_dirs: list[Path],
    *,
    audit_images: bool = False,
    min_rank_per_label: int = 1,
    min_suit_per_label: int = 1,
) -> dict[str, Any]:
    by_dir = []
    total_rank = 0
    total_suit = 0
    total_unknown = 0
    total_unreadable = 0
    rank_counts = {label: 0 for label in RANK_LABELS}
    suit_counts = {label: 0 for label in SUIT_LABELS}
    unreadable_examples: list[str] = []
    for path in crop_dirs:
        rank_records = collect_crop_records([path], allowed_kinds=("rank",))
        suit_records = collect_crop_records([path], allowed_kinds=("suit",))
        rank_unknown = sum(1 for record in rank_records if not record.current_label or record.current_label == "_unknown")
        suit_unknown = sum(1 for record in suit_records if not record.current_label or record.current_label == "_unknown")
        dir_rank_counts = count_record_labels(rank_records, RANK_LABELS)
        dir_suit_counts = count_record_labels(suit_records, SUIT_LABELS)
        for label, count in dir_rank_counts.items():
            rank_counts[label] = rank_counts.get(label, 0) + count
        for label, count in dir_suit_counts.items():
            suit_counts[label] = suit_counts.get(label, 0) + count
        unreadable = {"count": 0, "examples": []}
        if audit_images:
            unreadable = audit_readable_images([*rank_records, *suit_records])
            total_unreadable += int(unreadable.get("count") or 0)
            unreadable_examples.extend(str(path) for path in unreadable.get("examples") or [])
        total_rank += len(rank_records)
        total_suit += len(suit_records)
        total_unknown += rank_unknown + suit_unknown
        by_dir.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "rank_count": len(rank_records),
                "suit_count": len(suit_records),
                "unknown_count": rank_unknown + suit_unknown,
                "unreadable_count": unreadable.get("count", 0),
                "unreadable_examples": unreadable.get("examples", []),
                "rank_labels": sorted({record.current_label for record in rank_records if record.current_label}),
                "suit_labels": sorted({record.current_label for record in suit_records if record.current_label}),
                "rank_label_counts": dir_rank_counts,
                "suit_label_counts": dir_suit_counts,
            }
        )
    missing_rank = [label for label in RANK_LABELS if rank_counts.get(label, 0) <= 0]
    missing_suit = [label for label in SUIT_LABELS if suit_counts.get(label, 0) <= 0]
    rare_rank = [label for label in RANK_LABELS if rank_counts.get(label, 0) < int(min_rank_per_label)]
    rare_suit = [label for label in SUIT_LABELS if suit_counts.get(label, 0) < int(min_suit_per_label)]
    return {
        "crop_dirs": [str(path) for path in crop_dirs],
        "by_dir": by_dir,
        "rank_count": total_rank,
        "suit_count": total_suit,
        "unknown_count": total_unknown,
        "audit_images": bool(audit_images),
        "unreadable_count": total_unreadable,
        "unreadable_examples": unreadable_examples[:20],
        "rank_label_counts": rank_counts,
        "suit_label_counts": suit_counts,
        "missing_rank_labels": missing_rank,
        "missing_suit_labels": missing_suit,
        "rare_rank_labels": rare_rank,
        "rare_suit_labels": rare_suit,
        "min_rank_per_label": int(min_rank_per_label),
        "min_suit_per_label": int(min_suit_per_label),
        "rank_labels_complete": not missing_rank,
        "suit_labels_complete": not missing_suit,
        "rank_min_per_label_ok": not rare_rank,
        "suit_min_per_label_ok": not rare_suit,
        "images_readable": total_unreadable == 0,
        "ready": total_rank > 0 and total_suit > 0,
    }


def count_record_labels(records: list[Any], labels: tuple[str, ...]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for record in records:
        label = str(record.current_label or "")
        if label in counts:
            counts[label] += 1
    return counts


def audit_readable_images(records: list[Any]) -> dict[str, Any]:
    cv2, _np = load_cv()
    unreadable: list[str] = []
    for record in records:
        image = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable.append(str(record.path))
    return {"count": len(unreadable), "examples": unreadable[:20]}


def inspect_probe_dir(probe_dir: Path) -> dict[str, Any]:
    root = Path(probe_dir)
    info: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "ready": False,
        "rank_probe": str(root / PROBE_FILENAMES["rank"]),
        "suit_probe": str(root / PROBE_FILENAMES["suit"]),
        "rank_exists": (root / PROBE_FILENAMES["rank"]).exists(),
        "suit_exists": (root / PROBE_FILENAMES["suit"]).exists(),
        "rank": {},
        "suit": {},
    }
    for kind in ("rank", "suit"):
        path = root / PROBE_FILENAMES[kind]
        if not path.exists():
            continue
        try:
            probe = load_probe(path)
            info[kind] = probe.get("metadata") or {}
        except Exception as error:
            info[kind] = {"error": f"{type(error).__name__}: {error}"}
    info["ready"] = bool(info["rank_exists"] and info["suit_exists"] and not info["rank"].get("error") and not info["suit"].get("error"))
    return info


def filter_ingest_dataset_dirs(dataset_dirs: list[Path], *, output_dir: Path) -> tuple[list[Path], list[Path]]:
    output_resolved = safe_resolve(output_dir)
    kept: list[Path] = []
    skipped: list[Path] = []
    seen: set[Path] = set()
    for path in dataset_dirs:
        resolved = safe_resolve(path)
        if resolved in seen:
            skipped.append(path)
            continue
        seen.add(resolved)
        if resolved == output_resolved:
            skipped.append(path)
            continue
        kept.append(path)
    return kept, skipped


def build_probe_train_input_dirs(
    crop_dirs: list[Path],
    *,
    ingested_dataset_dir: Path,
    include_ingested: bool,
) -> list[Path]:
    dirs = [Path(path) for path in crop_dirs]
    if include_ingested:
        dirs.append(Path(ingested_dataset_dir))
    unique_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        resolved = safe_resolve(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_dirs.append(path)
    return unique_dirs


def safe_resolve(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path).absolute()


def build_pipeline_checks(
    *,
    bbox_info: dict[str, Any],
    videos: list[Path],
    crop_summary: dict[str, Any],
    probe: dict[str, Any],
    knn: dict[str, Any],
    deep: dict[str, Any],
    validation: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    auto_bbox_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    timing = (validation or {}).get("timing_ms") or {}
    card_health = (validation or {}).get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    checks = [
        check("bbox_concrete", bool(bbox_info.get("concrete")), bbox_info.get("normalized") or bbox_info.get("raw"), "numeric x,y,width,height", "live"),
        check("root_videos_present", len(videos) > 0, len(videos), ">0", "training"),
        check("crop_rank_present", int(crop_summary.get("rank_count") or 0) > 0, crop_summary.get("rank_count"), ">0", "training"),
        check("crop_suit_present", int(crop_summary.get("suit_count") or 0) > 0, crop_summary.get("suit_count"), ">0", "training"),
        check("crop_rank_labels_complete", bool(crop_summary.get("rank_labels_complete")), crop_summary.get("missing_rank_labels"), "no missing rank labels", "training_quality"),
        check("crop_suit_labels_complete", bool(crop_summary.get("suit_labels_complete")), crop_summary.get("missing_suit_labels"), "no missing suit labels", "training_quality"),
        check("crop_rank_min_per_label", bool(crop_summary.get("rank_min_per_label_ok")), crop_summary.get("rare_rank_labels"), f">={crop_summary.get('min_rank_per_label')} per rank", "training_quality"),
        check("crop_suit_min_per_label", bool(crop_summary.get("suit_min_per_label_ok")), crop_summary.get("rare_suit_labels"), f">={crop_summary.get('min_suit_per_label')} per suit", "training_quality"),
        check("crop_images_readable", bool(crop_summary.get("images_readable")), crop_summary.get("unreadable_count"), 0, "training_quality"),
        check("hf_probe_ready", bool(probe.get("ready")), probe.get("path"), "rank and suit probes", "training"),
        check("knn_model_ready", bool(knn.get("exists") and knn.get("rank_labels_ok") and knn.get("suit_labels_ok")), knn.get("path"), "valid promoted KNN", "live"),
        check("validation_summary_exists", validation is not None, bool(validation), "validate-cv summary", "shared"),
        check("gate_summary_exists", gate is not None, bool(gate), "gate summary", "shared"),
        check("gate_promote", bool((gate or {}).get("promote")), (gate or {}).get("decision"), "promote", "live"),
        check("validation_real_problem_zero", int((validation or {}).get("real_problem_count") or 0) == 0, (validation or {}).get("real_problem_count"), 0, "live"),
        check("validation_board_bad_zero", int((validation or {}).get("board_bad_count") or 0) == 0, (validation or {}).get("board_bad_count"), 0, "live"),
        check("validation_median_fast", optional_float(timing.get("median"), 999999.0) <= 300.0, timing.get("median"), "<=300ms", "live"),
        check("validation_p90_fast", optional_float(timing.get("p90"), 999999.0) <= 900.0, timing.get("p90"), "<=900ms", "live"),
    ]
    if card_health:
        checks.extend(
            [
                check(
                    "validation_hero_incomplete_or_missed_zero",
                    int(hero_health.get("incomplete_or_missed_frames") or 0) == 0,
                    int(hero_health.get("incomplete_or_missed_frames") or 0),
                    0,
                    "live",
                ),
                check(
                    "validation_hero_turn_blocked_zero",
                    int(hero_health.get("turn_blocked_frames") or 0) == 0,
                    int(hero_health.get("turn_blocked_frames") or 0),
                    0,
                    "live",
                ),
                check(
                    "validation_board_health_bad_zero",
                    int(board_health.get("bad_frames") or 0) == 0,
                    int(board_health.get("bad_frames") or 0),
                    0,
                    "live",
                ),
                check("validation_card_issue_count_zero", card_issue_count(card_health) == 0, card_issue_count(card_health), 0, "live"),
            ]
        )
    if deep.get("enabled"):
        checks.append(
            check(
                "deep_model_ready",
                bool(deep.get("rank_exists") and deep.get("suit_exists")),
                deep.get("path"),
                "deep rank/suit fallback",
                "live",
            )
        )
    else:
        checks.append(check("deep_model_optional", True, "disabled", "optional offline fallback", "live"))
    if auto_bbox_diagnostics is not None:
        checks.extend(
            [
                check("auto_bbox_ok", bool(auto_bbox_diagnostics.get("ok")), auto_bbox_diagnostics.get("ok"), True, "localization_quality"),
                check("auto_bbox_failure_count", int(auto_bbox_diagnostics.get("failure_count") or 0) == 0, auto_bbox_diagnostics.get("failure_count"), 0, "localization_quality"),
            ]
        )
    return checks


def card_issue_count(card_health: dict[str, Any]) -> int:
    return sum(int(value or 0) for value in (card_health.get("issue_counts") or {}).values())


def check(name: str, passed: bool, actual: Any, required: Any, scope: str) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "actual": actual, "required": required, "scope": scope}


def choose_next_stage(checks: list[dict[str, Any]]) -> str:
    for stage, names in (
        ("pick_or_fix_bbox", ("bbox_concrete",)),
        ("export_or_label_crops", ("crop_rank_present", "crop_suit_present")),
        ("train_or_reuse_hf_probe", ("hf_probe_ready",)),
        ("rerun_validation_and_gate", ("validation_summary_exists", "gate_summary_exists", "gate_promote")),
        ("live_preflight", ("knn_model_ready",)),
    ):
        for name in names:
            row = next((item for item in checks if item.get("name") == name), None)
            if row is not None and not row.get("pass"):
                return stage
    return "ready_for_live_and_training_iteration"


def build_pipeline_commands(
    *,
    bbox: str,
    hero_name: str | None,
    crop_dirs: list[Path],
    probe_dir: Path,
    dataset_dir: Path,
    extra_dataset_dirs: list[Path] | None,
    skipped_ingest_dataset_dirs: list[Path] | None,
    ingested_dataset_dir: Path,
    output_dir: Path,
    video_dir: Path,
    knn_model_path: Path,
    deep_model_dir: Path | None,
    probe_model: str = "facebook/dinov2-base",
    probe_rank_model: str | None = None,
    probe_suit_model: str | None = None,
    probe_max_images_per_class: int | None = 24,
    probe_batch_size: int = 16,
    auto_bbox_output_dir: Path | None = None,
    auto_bbox_every_sec: float = 300.0,
    auto_bbox_max_frames: int | None = 1,
    auto_bbox_variants: list[str] | None = None,
) -> dict[str, str]:
    crop_arg = " ".join(f'--input-dir "{path}"' for path in crop_dirs)
    dataset_dirs, _skipped = filter_ingest_dataset_dirs([dataset_dir, *(extra_dataset_dirs or [])], output_dir=ingested_dataset_dir)
    dataset_arg = " ".join(f'--dataset-dir "{path}"' for path in dataset_dirs)
    probe_train_dirs = build_probe_train_input_dirs(
        crop_dirs,
        ingested_dataset_dir=ingested_dataset_dir,
        include_ingested=bool(dataset_dirs),
    )
    probe_train_arg = " ".join(f'--input-dir "{path}"' for path in probe_train_dirs)
    hero_arg = f' --hero-name "{hero_name}"' if hero_name else ""
    bbox_variant_arg = " ".join(f'--variant "{variant}"' for variant in (auto_bbox_variants or []))
    bbox_max_frames_arg = "" if auto_bbox_max_frames is None else f" --max-frames {int(auto_bbox_max_frames)}"
    export_review_deep_arg = "" if deep_model_dir is None else f'--deep-card-model-dir "{deep_model_dir}" '
    rank_model_arg = f' --rank-model "{probe_rank_model}"' if probe_rank_model else ""
    suit_model_arg = f' --suit-model "{probe_suit_model}"' if probe_suit_model else ""
    max_per_class_arg = "" if probe_max_images_per_class is None else f" --max-images-per-class {int(probe_max_images_per_class)}"
    teacher_predictions = output_dir / "big_teacher_label" / "labeled" / "predictions.csv"
    glyph_queue_csv = output_dir / "glyph_label_queue" / "glyph_label_queue.csv"
    return {
        "pick_bbox": f'python gto.py screen-cv --pick-bbox{hero_arg} --output-dir "video_frames\\screen_calibrate"',
        "cv_health": (
            f'python gto.py cv-health --bbox "{bbox}"{hero_arg} --output-dir "video_frames\\cv_health_promoted" '
            f'--fail-on-not-ready --format text'
        ),
        "live": (
            f'python gto.py screen-cv --bbox "{bbox}" --auto-bbox --auto-bbox-refresh 10 --lock-layout{hero_arg} '
            f'--output-dir "video_frames\\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 '
            f'--min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text'
        ),
        "download_dataset": f'python gto.py download-card-dataset --output-dir "{dataset_dir}" --format text',
        "ingest_dataset": (
            f'python gto.py ingest-card-images {dataset_arg} '
            f'--output-dir "{ingested_dataset_dir}" --max-images 1200 --format text'
        ) if dataset_arg else f'# No ingest dataset dirs; skipped {[str(path) for path in (skipped_ingest_dataset_dirs or [])]}',
        "export_review": (
            f'python gto.py export-card-review --all --video-dir "{video_dir}" '
            f'--output-dir "{output_dir / "review_export"}" --every 10 --min-confidence 0.35 '
            f'{export_review_deep_arg}--format text'
        ),
        "train_probe": (
            f'python gto.py train-card-hf-probe {probe_train_arg} --output-dir "{probe_dir}" --kind both '
            f'--model "{probe_model}"{rank_model_arg}{suit_model_arg}{max_per_class_arg} '
            f'--batch-size {max(1, int(probe_batch_size))} --format text'
        ),
        "label_with_probe": (
            f'python gto.py card-big-teacher {crop_arg} --probe-dir "{probe_dir}" '
            f'--output-dir "{output_dir / "big_teacher_label"}" --kind both --rank-score-threshold 0.55 '
            f'--rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 '
            f'--require-current-agreement --local-files-only --format text'
        ),
        "threshold_sweep": (
            f'python gto.py sweep-card-hf-thresholds --predictions-csv "{teacher_predictions}" '
            f'--output-dir "{output_dir / "threshold_sweep"}" '
            f'--rank-score-threshold 0.45 --rank-score-threshold 0.50 --rank-score-threshold 0.55 '
            f'--rank-margin-threshold 0.04 --rank-margin-threshold 0.08 --rank-margin-threshold 0.12 '
            f'--suit-score-threshold 0.60 --suit-score-threshold 0.65 --suit-score-threshold 0.70 '
            f'--suit-margin-threshold 0.04 --suit-margin-threshold 0.06 --suit-margin-threshold 0.10 '
            f'--format text'
        ),
        "filter_predictions_distill": (
            f'python gto.py filter-card-hf-predictions --predictions-csv "{teacher_predictions}" '
            f'--output-dir "{output_dir / "filtered_distill"}" --kind both '
            f'--rank-score-threshold 0.50 --rank-margin-threshold 0.12 '
            f'--suit-score-threshold 0.65 --suit-margin-threshold 0.06 '
            f'--require-current-agreement --distill-runtime --runtime-every 10 --runtime-max-frames 80 '
            f'--runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 '
            f'--runtime-risk-queue-max-rows 80 --format text'
        ),
        "prepare_glyph_label_queue": (
            f'python gto.py prepare-card-glyph-label-queue --predictions-csv "{teacher_predictions}" '
            f'--output-dir "{output_dir / "glyph_label_queue"}" --max-rows 200 --format text'
        ),
        "apply_glyph_label_queue": (
            f'python gto.py apply-card-glyph-label-queue --queue-csv "{glyph_queue_csv}" '
            f'--output-dir "{output_dir / "glyph_label_applied"}" --format text'
        ),
        "distill_and_gate": (
            f'python gto.py card-big-teacher {crop_arg} --probe-dir "{probe_dir}" '
            f'--output-dir "{output_dir / "big_teacher_distill"}" --kind both --rank-score-threshold 0.55 '
            f'--rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 '
            f'--require-current-agreement --local-files-only --distill-runtime --runtime-every 10 '
            f'--runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text'
        ),
        "validate_current": (
            f'python gto.py validate-cv --all --video-dir "{video_dir}" --output-dir "{output_dir / "validation_current"}" '
            f'--every 10 --max-frames 80 --min-confidence 0.35 --dealer-refresh-frames 4 '
            f'--card-knn-model "{knn_model_path}" --format text'
        ),
        "diagnose_auto_bbox": (
            f'python gto.py diagnose-auto-bbox --all --video-dir "{video_dir}" '
            f'--output-dir "{auto_bbox_output_dir or output_dir / "auto_bbox_diagnostics"}" '
            f'--every {float(auto_bbox_every_sec):g}{bbox_max_frames_arg} --min-confidence 0.35 '
            f'{bbox_variant_arg} --format text'
        ).replace("  ", " ").strip(),
    }


def write_command_files(output_dir: Path, commands: dict[str, str]) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name, command in commands.items():
        path = output_dir / f"{name}.txt"
        path.write_text(command + "\n", encoding="utf-8-sig")
        files[name] = str(path)
    return files


def compact_validation(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "ok": payload.get("ok"),
        "video_count": payload.get("video_count"),
        "real_problem_count": payload.get("real_problem_count"),
        "board_bad_count": payload.get("board_bad_count"),
        "card_health": payload.get("card_health") or {},
        "timing_ms": payload.get("timing_ms") or {},
        "files": payload.get("files") or {},
    }


def compact_gate(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "ok": payload.get("ok"),
        "decision": payload.get("decision"),
        "promote": payload.get("promote"),
        "candidate_name": payload.get("candidate_name"),
        "files": payload.get("files") or {},
    }


def compact_candidate_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    best = payload.get("best_candidate") or {}
    files = payload.get("files") or {}
    return {
        "ok": payload.get("ok"),
        "candidate_count": payload.get("candidate_count"),
        "promote_count": payload.get("promote_count"),
        "best_candidate": {
            "candidate_name": best.get("candidate_name"),
            "decision": best.get("decision"),
            "promote": best.get("promote"),
            "card_acc": best.get("card_acc"),
            "diff_risk": best.get("diff_risk"),
            "missing_rows": best.get("missing_rows"),
            "median_ms": best.get("median_ms"),
            "summary_path": best.get("summary_path"),
            "report_path": best.get("report_path"),
        } if best else None,
        "files": files,
    }


def compact_auto_bbox_diagnostics(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    files = payload.get("files") or {}
    return {
        "ok": payload.get("ok"),
        "video_count": payload.get("video_count"),
        "row_count": (payload.get("sample") or {}).get("row_count"),
        "failure_count": payload.get("failure_count"),
        "counts": payload.get("counts") or {},
        "method_counts": payload.get("method_counts") or {},
        "iou_stats": payload.get("iou_stats") or {},
        "timing_ms": payload.get("timing_ms") or {},
        "files": files,
    }


def compact_smoke(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not payload.get("ok"):
        return payload
    label = payload.get("label") or {}
    return {
        "ok": True,
        "output_dir": payload.get("output_dir"),
        "probe_dir": payload.get("probe_dir"),
        "rank_model": payload.get("rank_model"),
        "suit_model": payload.get("suit_model"),
        "processed": label.get("processed"),
        "accepted": label.get("accepted"),
        "review": label.get("review"),
        "files": payload.get("files") or {},
    }


def compact_probe_train_run(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not payload.get("ok"):
        return payload
    results = payload.get("results") or {}
    return {
        "ok": True,
        "output_dir": payload.get("output_dir"),
        "input_dirs": payload.get("input_dirs") or [],
        "kind": payload.get("kind"),
        "rank_model": payload.get("rank_model"),
        "suit_model": payload.get("suit_model"),
        "rank_source_count": (results.get("rank") or {}).get("source_count"),
        "suit_source_count": (results.get("suit") or {}).get("source_count"),
        "rank_source_counts_by_dir": (results.get("rank") or {}).get("source_counts_by_dir") or {},
        "suit_source_counts_by_dir": (results.get("suit") or {}).get("source_counts_by_dir") or {},
        "rank_val_acc": ((results.get("rank") or {}).get("val") or {}).get("accuracy"),
        "suit_val_acc": ((results.get("suit") or {}).get("val") or {}).get("accuracy"),
        "files": payload.get("files") or {},
    }


def brief_source_counts(source_counts_by_dir: dict[str, Any]) -> dict[str, int]:
    return {
        path: int((info or {}).get("count") or 0)
        for path, info in (source_counts_by_dir or {}).items()
    }


def compact_teacher_run(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    compact = compact_smoke(payload)
    if compact is None or not compact.get("ok"):
        return compact
    distill = (payload or {}).get("distill_runtime") or {}
    if distill:
        compact["distill_runtime"] = {
            "stopped": distill.get("stopped"),
            "decision": distill.get("decision"),
            "promote": distill.get("promote"),
            "model_path": distill.get("model_path"),
            "gate_report": (distill.get("files") or {}).get("gate_report"),
        }
    return compact


def optional_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def dataset_coverage_line(dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    coverage = dataset.get("card_coverage") or {}
    if not coverage:
        return "Dataset deck: unknown"
    duplicate_cards = coverage.get("duplicate_cards") or {}
    missing_cards = coverage.get("missing_cards") or []
    return (
        "Dataset deck: "
        f"parsed={coverage.get('parsed_card_count', 0)}/{coverage.get('expected_card_count', 52)} "
        f"complete={coverage.get('complete_deck')} "
        f"missing={len(missing_cards)} "
        f"duplicate_labels={len(duplicate_cards)} "
        f"unparsed={coverage.get('unparsed_count', 0)}"
    )


def format_card_cv_pipeline_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"card-cv-pipeline failed: {payload.get('error')}"
    dataset = payload.get("dataset") or {}
    lines = [
        f"Output: {payload.get('output_dir')}",
        f"Next stage: {payload.get('next_stage')}",
        f"Ready for live: {payload.get('ready_for_live')}",
        f"Ready for training: {payload.get('ready_for_training')}",
        f"Videos: {(payload.get('videos') or {}).get('count', 0)}",
        f"Crops: rank={(payload.get('crops') or {}).get('rank_count', 0)} suit={(payload.get('crops') or {}).get('suit_count', 0)}",
        f"Probe ready: {((payload.get('models') or {}).get('hf_probe') or {}).get('ready')}",
        f"Dataset images: {dataset.get('image_count', 0)}",
        dataset_coverage_line(dataset),
        f"Probe train inputs: {payload.get('probe_train_input_dirs') or []}",
        f"Runbook: {(payload.get('files') or {}).get('runbook')}",
    ]
    crops = payload.get("crops") or {}
    if crops:
        lines.append(
            "Crop quality: "
            f"missing_rank={crops.get('missing_rank_labels') or []} "
            f"missing_suit={crops.get('missing_suit_labels') or []} "
            f"unreadable={crops.get('unreadable_count')}"
        )
    candidates = payload.get("candidate_summary") or {}
    if candidates:
        best = candidates.get("best_candidate") or {}
        lines.append(
            f"Candidates: {candidates.get('candidate_count')} promotable={candidates.get('promote_count')} "
            f"best={best.get('candidate_name') or '-'}"
        )
    auto_bbox = payload.get("auto_bbox_diagnostics") or {}
    if auto_bbox:
        lines.append(
            f"Auto-bbox: rows={auto_bbox.get('row_count')} failures={auto_bbox.get('failure_count')} "
            f"median_ms={(auto_bbox.get('timing_ms') or {}).get('median')}"
        )
    smoke = payload.get("smoke")
    probe_train = payload.get("probe_train_run")
    if probe_train:
        lines.append(
            "Probe train: "
            f"rank_model={probe_train.get('rank_model')} "
            f"suit_model={probe_train.get('suit_model')} "
            f"rank_val={probe_train.get('rank_val_acc')} "
            f"suit_val={probe_train.get('suit_val_acc')}"
        )
    if smoke:
        lines.append(
            f"Smoke: processed={smoke.get('processed')} accepted={smoke.get('accepted')} review={smoke.get('review')}"
        )
    teacher_run = payload.get("teacher_run")
    if teacher_run:
        lines.append(
            f"Teacher: processed={teacher_run.get('processed')} accepted={teacher_run.get('accepted')} review={teacher_run.get('review')}"
        )
        distill = teacher_run.get("distill_runtime") or {}
        if distill:
            lines.append(f"Teacher distill: decision={distill.get('decision')} promote={distill.get('promote')}")
    failed = [item for item in payload.get("checks") or [] if not item.get("pass")]
    if failed:
        lines.append("Failed checks:")
        for item in failed[:12]:
            lines.append(f"- {item.get('name')}: actual={item.get('actual')} required={item.get('required')}")
    return "\n".join(lines)


def format_card_cv_pipeline_runbook(payload: dict[str, Any]) -> str:
    commands = payload.get("commands") or {}
    checks = payload.get("checks") or []
    failed = [item for item in checks if not item.get("pass")]
    lines = [
        "# Card CV Pipeline",
        "",
        f"- Next stage: `{payload.get('next_stage')}`",
        f"- Ready for live: `{payload.get('ready_for_live')}`",
        f"- Ready for training iteration: `{payload.get('ready_for_training')}`",
        f"- Output: `{payload.get('output_dir')}`",
        "",
        "## Checks",
        "",
        "| Check | Pass | Actual | Required | Scope |",
        "|---|---:|---|---|---|",
    ]
    for item in checks:
        lines.append(
            f"| {item.get('name')} | {item.get('pass')} | `{item.get('actual')}` | `{item.get('required')}` | {item.get('scope')} |"
        )
    if failed:
        lines.extend(["", "## Fix First", ""])
        for item in failed[:10]:
            lines.append(f"- `{item.get('name')}`: actual `{item.get('actual')}`, required `{item.get('required')}`.")
    crops = payload.get("crops") or {}
    if crops:
        lines.extend(
            [
                "",
                "## Crop Quality",
                "",
                f"- Rank labels complete: `{crops.get('rank_labels_complete')}`",
                f"- Suit labels complete: `{crops.get('suit_labels_complete')}`",
                f"- Missing rank labels: `{crops.get('missing_rank_labels') or []}`",
                f"- Missing suit labels: `{crops.get('missing_suit_labels') or []}`",
                f"- Rare rank labels: `{crops.get('rare_rank_labels') or []}`",
                f"- Rare suit labels: `{crops.get('rare_suit_labels') or []}`",
                f"- Unreadable crop images: `{crops.get('unreadable_count')}`",
                "",
                "| Rank | Count |",
                "|---|---:|",
            ]
        )
        for label, count in (crops.get("rank_label_counts") or {}).items():
            lines.append(f"| {label} | {count} |")
        lines.extend(["", "| Suit | Count |", "|---|---:|"])
        for label, count in (crops.get("suit_label_counts") or {}).items():
            lines.append(f"| {label} | {count} |")
    dataset = payload.get("dataset") or {}
    if dataset:
        coverage = dataset.get("card_coverage") or {}
        roots = dataset.get("likely_dataset_roots") or []
        missing_cards = coverage.get("missing_cards") or []
        duplicate_cards = coverage.get("duplicate_cards") or {}
        lines.extend(
            [
                "",
                "## Dataset Coverage",
                "",
                f"- Dataset dir: `{dataset.get('output_dir')}`",
                f"- Images: `{dataset.get('image_count', 0)}`",
                f"- Label dirs: `{dataset.get('label_dir_count', 0)}`",
                f"- Parsed cards: `{coverage.get('parsed_card_count', 0)}/{coverage.get('expected_card_count', 52)}`",
                f"- Complete deck: `{coverage.get('complete_deck')}`",
                f"- Missing cards: `{missing_cards}`",
                f"- Duplicate cards: `{duplicate_cards}`",
                f"- Unparsed images: `{coverage.get('unparsed_count', 0)}`",
                f"- Likely dataset root: `{roots[0] if roots else ''}`",
                f"- Probe train input dirs: `{payload.get('probe_train_input_dirs') or []}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```powershell",
        ]
    )
    for name in (
        "pick_bbox",
        "cv_health",
        "live",
        "download_dataset",
        "ingest_dataset",
        "export_review",
        "train_probe",
        "label_with_probe",
        "threshold_sweep",
        "filter_predictions_distill",
        "prepare_glyph_label_queue",
        "apply_glyph_label_queue",
        "distill_and_gate",
        "validate_current",
        "diagnose_auto_bbox",
    ):
        command = commands.get(name)
        if command:
            lines.extend([f"# {name}", command, ""])
    lines.append("```")
    probe_train = payload.get("probe_train_run")
    if probe_train:
        lines.extend(
            [
                "",
                "## Probe Train",
                "",
                f"- Output: `{probe_train.get('output_dir')}`",
                f"- Input dirs: `{probe_train.get('input_dirs') or []}`",
                f"- Rank model: `{probe_train.get('rank_model')}`",
                f"- Suit model: `{probe_train.get('suit_model')}`",
                f"- Rank source crops: `{probe_train.get('rank_source_count')}`",
                f"- Suit source crops: `{probe_train.get('suit_source_count')}`",
                f"- Rank sources by dir: `{brief_source_counts(probe_train.get('rank_source_counts_by_dir') or {})}`",
                f"- Suit sources by dir: `{brief_source_counts(probe_train.get('suit_source_counts_by_dir') or {})}`",
                f"- Rank validation accuracy: `{probe_train.get('rank_val_acc')}`",
                f"- Suit validation accuracy: `{probe_train.get('suit_val_acc')}`",
            ]
        )
    smoke = payload.get("smoke")
    if smoke:
        lines.extend(
            [
                "",
                "## Smoke",
                "",
                f"- Output: `{smoke.get('output_dir')}`",
                f"- Processed: `{smoke.get('processed')}`",
                f"- Accepted: `{smoke.get('accepted')}`",
                f"- Review: `{smoke.get('review')}`",
            ]
        )
    teacher_run = payload.get("teacher_run")
    if teacher_run:
        distill = teacher_run.get("distill_runtime") or {}
        lines.extend(
            [
                "",
                "## Teacher Run",
                "",
                f"- Output: `{teacher_run.get('output_dir')}`",
                f"- Processed: `{teacher_run.get('processed')}`",
                f"- Accepted: `{teacher_run.get('accepted')}`",
                f"- Review: `{teacher_run.get('review')}`",
                f"- Runtime decision: `{distill.get('decision') or ''}`",
                f"- Runtime promote: `{distill.get('promote') if distill else ''}`",
                f"- Runtime gate report: `{distill.get('gate_report') or ''}`",
            ]
        )
    candidates = payload.get("candidate_summary") or {}
    if candidates:
        best = candidates.get("best_candidate") or {}
        lines.extend(
            [
                "",
                "## Candidate Summary",
                "",
                f"- Candidates: `{candidates.get('candidate_count')}`",
                f"- Promotable: `{candidates.get('promote_count')}`",
                f"- Best: `{best.get('candidate_name') or ''}`",
                f"- Best decision: `{best.get('decision') or ''}`",
                f"- Best report: `{best.get('report_path') or ''}`",
                f"- Summary: `{(candidates.get('files') or {}).get('summary_md') or ''}`",
            ]
        )
    auto_bbox = payload.get("auto_bbox_diagnostics") or {}
    if auto_bbox:
        lines.extend(
            [
                "",
                "## Auto-Bbox Diagnostics",
                "",
                f"- Videos: `{auto_bbox.get('video_count')}`",
                f"- Rows: `{auto_bbox.get('row_count')}`",
                f"- Failures: `{auto_bbox.get('failure_count')}`",
                f"- Counts: `{auto_bbox.get('counts')}`",
                f"- Median ms: `{(auto_bbox.get('timing_ms') or {}).get('median')}`",
                f"- P90 ms: `{(auto_bbox.get('timing_ms') or {}).get('p90')}`",
                f"- Report: `{(auto_bbox.get('files') or {}).get('report_md') or ''}`",
                f"- Rows CSV: `{(auto_bbox.get('files') or {}).get('rows_csv') or ''}`",
            ]
        )
    return "\n".join(lines)
