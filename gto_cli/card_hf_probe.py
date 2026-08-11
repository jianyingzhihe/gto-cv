from __future__ import annotations

import json
import math
import random
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .card_classifier import DEFAULT_TEMPLATE_DIR
from .card_deep_model import RANK_LABELS, SUIT_LABELS
from .card_glyph_export import safe_label, safe_stem
from .card_hf_teacher import glyph_to_pil, load_cv, resolve_device
from .card_teacher_label import (
    CropRecord,
    collect_crop_records,
    copy_labeled_crop,
    copy_review_crop,
    count_rows,
    format_float,
    should_accept_prediction,
    write_predictions_csv,
)


DEFAULT_HF_PROBE_MODEL = "openai/clip-vit-base-patch32"
PROBE_FILENAMES = {"rank": "hf_rank_probe.npz", "suit": "hf_suit_probe.npz"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MODEL_CACHE: dict[tuple[str, bool, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class LabeledCrop:
    kind: str
    path: Path
    label: str


def train_hf_card_probe(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    kind: str = "both",
    model_name: str = DEFAULT_HF_PROBE_MODEL,
    rank_model: str | None = None,
    suit_model: str | None = None,
    template_dir: Path | None = DEFAULT_TEMPLATE_DIR,
    include_templates: bool = True,
    max_images_per_class: int | None = None,
    val_split: float = 0.18,
    seed: int = 20260708,
    batch_size: int = 32,
    temperature: float = 0.04,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_kinds = ("rank", "suit") if kind == "both" else (kind,)

    started_at = time.perf_counter()
    results: dict[str, Any] = {}
    for one_kind in allowed_kinds:
        labels = list(RANK_LABELS if one_kind == "rank" else SUIT_LABELS)
        model_for_kind = rank_model if one_kind == "rank" else suit_model
        model_for_kind = model_for_kind or model_name
        records = collect_labeled_crops(
            input_dirs=[Path(path) for path in input_dirs],
            kind=one_kind,
            labels=labels,
            template_dir=template_dir,
            include_templates=include_templates,
            max_images_per_class=max_images_per_class,
        )
        if len({record.label for record in records}) < 2:
            raise ValueError(f"need at least 2 labeled classes for {one_kind}, found {len(records)} images")

        features = extract_probe_features(
            records,
            kind=one_kind,
            model_name=model_for_kind,
            batch_size=batch_size,
            device=device,
            local_files_only=local_files_only,
        )
        train_indices, val_indices = stratified_split_indices(records, val_split=val_split, seed=seed)
        train_payload = build_probe_payload(
            records=[records[index] for index in train_indices],
            features=features[train_indices],
            labels=labels,
            kind=one_kind,
            model_name=model_for_kind,
            temperature=temperature,
        )
        val_metrics = evaluate_probe(
            probe=train_payload,
            records=[records[index] for index in val_indices],
            features=features[val_indices] if val_indices else features[:0],
        )
        final_payload = build_probe_payload(
            records=records,
            features=features,
            labels=labels,
            kind=one_kind,
            model_name=model_for_kind,
            temperature=temperature,
        )
        model_path = output_dir / PROBE_FILENAMES[one_kind]
        save_probe(model_path, final_payload)
        results[one_kind] = {
            "kind": one_kind,
            "model_name": model_for_kind,
            "model_path": str(model_path),
            "source_count": len(records),
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "label_counts": count_labeled(records, labels),
            "source_counts_by_dir": count_labeled_by_source(
                records,
                source_dirs=probe_source_dirs(
                    input_dirs=[Path(path) for path in input_dirs],
                    template_dir=template_dir,
                    include_templates=include_templates,
                ),
                labels=labels,
            ),
            "val": val_metrics,
        }

    summary = {
        "ok": True,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "kind": kind,
        "model_name": model_name,
        "rank_model": rank_model or model_name,
        "suit_model": suit_model or model_name,
        "include_templates": bool(include_templates),
        "template_dir": str(template_dir) if template_dir else "",
        "max_images_per_class": max_images_per_class,
        "val_split": float(val_split),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "device": device,
        "local_files_only": bool(local_files_only),
        "results": results,
        "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        "files": {
            "rank_probe": str(output_dir / PROBE_FILENAMES["rank"]) if "rank" in results else "",
            "suit_probe": str(output_dir / PROBE_FILENAMES["suit"]) if "suit" in results else "",
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def label_card_crops_hf_probe(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    probe_dir: Path,
    kind: str = "both",
    max_images: int | None = None,
    rank_score_threshold: float = 0.82,
    rank_margin_threshold: float = 0.10,
    suit_score_threshold: float = 0.82,
    suit_margin_threshold: float = 0.10,
    require_current_agreement: bool = False,
    copy_accepted: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any]:
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    output_dir = Path(output_dir)
    review_dir = output_dir / "review"
    for directory in (output_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    allowed_kinds = ("rank", "suit") if kind == "both" else (kind,)
    records = collect_crop_records([Path(path) for path in input_dirs], allowed_kinds=allowed_kinds)
    if max_images is not None:
        records = limit_records_total_per_kind(records, max_per_kind=max(0, int(max_images)))

    probes = {one_kind: load_probe(Path(probe_dir) / PROBE_FILENAMES[one_kind]) for one_kind in allowed_kinds}
    rows: list[dict[str, Any]] = []
    accepted = 0
    copied_accepted = 0
    review = 0
    unreadable = 0
    started_at = time.perf_counter()

    by_kind: dict[str, list[tuple[int, CropRecord]]] = {one_kind: [] for one_kind in allowed_kinds}
    for index, record in enumerate(records):
        by_kind.setdefault(record.kind, []).append((index, record))

    predictions_by_index: dict[int, dict[str, Any] | None] = {}
    for one_kind, indexed_records in by_kind.items():
        if not indexed_records:
            continue
        probe = probes[one_kind]
        crop_records = [record for _index, record in indexed_records]
        feature_records = [LabeledCrop(kind=record.kind, path=record.path, label=record.current_label) for record in crop_records]
        features, failed_indices = extract_probe_features_with_failures(
            feature_records,
            kind=one_kind,
            model_name=str(probe["metadata"]["model_name"]),
            batch_size=batch_size,
            device=device,
            local_files_only=local_files_only,
        )
        feature_cursor = 0
        for local_index, (global_index, _record) in enumerate(indexed_records):
            if local_index in failed_indices:
                predictions_by_index[global_index] = None
                continue
            prediction = predict_from_probe(probe, features[feature_cursor])
            feature_cursor += 1
            predictions_by_index[global_index] = prediction

    for index, record in enumerate(records):
        prediction = predictions_by_index.get(index)
        if prediction is None:
            row = build_probe_row(index, record, None, accepted=False, reason="image_or_teacher_missing", output_path="")
            rows.append(row)
            unreadable += 1
            review += 1
            continue
        accepted_flag, reason = should_accept_prediction(
            record,
            prediction,
            rank_score_threshold=rank_score_threshold,
            rank_margin_threshold=rank_margin_threshold,
            suit_score_threshold=suit_score_threshold,
            suit_margin_threshold=suit_margin_threshold,
            require_current_agreement=require_current_agreement,
        )
        output_path = ""
        if accepted_flag:
            accepted += 1
            if copy_accepted:
                output_path = str(copy_labeled_crop(record, prediction, output_dir, index))
                copied_accepted += 1
        else:
            review_path = copy_review_crop(record, prediction, review_dir, index)
            output_path = str(review_path) if review_path else ""
            review += 1
        rows.append(build_probe_row(index, record, prediction, accepted=accepted_flag, reason=reason, output_path=output_path))

    predictions_csv = output_dir / "predictions.csv"
    review_csv = output_dir / "review.csv"
    write_predictions_csv(predictions_csv, rows)
    write_predictions_csv(review_csv, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "probe_dir": str(probe_dir),
        "kind": kind,
        "processed": len(rows),
        "accepted": accepted,
        "copied_accepted": copied_accepted,
        "review": review,
        "unreadable": unreadable,
        "thresholds": {
            "rank_score": float(rank_score_threshold),
            "rank_margin": float(rank_margin_threshold),
            "suit_score": float(suit_score_threshold),
            "suit_margin": float(suit_margin_threshold),
            "require_current_agreement": bool(require_current_agreement),
        },
        "counts": count_rows(rows),
        "probes": {one_kind: probe["metadata"] for one_kind, probe in probes.items()},
        "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        "files": {
            "predictions_csv": str(predictions_csv),
            "review_csv": str(review_csv),
            "accepted_dir": str(output_dir) if copy_accepted else "",
            "review_dir": str(review_dir),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def filter_hf_probe_predictions(
    *,
    predictions_csv: Path,
    output_dir: Path,
    kind: str = "both",
    rank_score_threshold: float = 0.82,
    rank_margin_threshold: float = 0.10,
    suit_score_threshold: float = 0.82,
    suit_margin_threshold: float = 0.10,
    require_current_agreement: bool = False,
    copy_accepted: bool = True,
) -> dict[str, Any]:
    """Re-screen an existing HF probe predictions.csv without recomputing embeddings."""
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.is_file():
        raise ValueError(f"predictions CSV not found: {predictions_csv}")
    output_dir = Path(output_dir)
    review_dir = output_dir / "review"
    for directory in (output_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    allowed_kinds = {"rank", "suit"} if kind == "both" else {kind}
    rows: list[dict[str, Any]] = []
    input_rows = 0
    skipped = 0
    accepted = 0
    copied_accepted = 0
    review = 0
    missing_images = 0
    started_at = time.perf_counter()

    with predictions_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for input_index, source_row in enumerate(reader):
            input_rows += 1
            row_kind = str(source_row.get("kind") or "").strip().lower()
            if row_kind not in allowed_kinds:
                skipped += 1
                continue
            crop_path = resolve_prediction_asset_path(
                source_row.get("input_path") or source_row.get("rank_path") or source_row.get("suit_path") or "",
                predictions_csv.parent,
            )
            current_label = str(source_row.get("current_label") or "").strip()
            record = CropRecord(kind=row_kind, path=crop_path, current_label=current_label)
            prediction = {
                "label": str(source_row.get("teacher_label") or ""),
                "score": source_row.get("teacher_score"),
                "margin": source_row.get("teacher_margin"),
                "second_score": source_row.get("teacher_second_score"),
                "model": str(source_row.get("teacher_model") or ""),
                "backend": "hf_embedding_probe_cached",
            }
            if not crop_path.exists():
                accepted_flag = False
                reason = "image_missing"
                missing_images += 1
            else:
                accepted_flag, reason = should_accept_prediction(
                    record,
                    prediction,
                    rank_score_threshold=rank_score_threshold,
                    rank_margin_threshold=rank_margin_threshold,
                    suit_score_threshold=suit_score_threshold,
                    suit_margin_threshold=suit_margin_threshold,
                    require_current_agreement=require_current_agreement,
                )
            output_path = ""
            if accepted_flag:
                accepted += 1
                if copy_accepted:
                    output_path = str(copy_labeled_crop(record, prediction, output_dir, input_index))
                    copied_accepted += 1
            else:
                review_path = copy_review_crop(record, prediction, review_dir, input_index)
                output_path = str(review_path) if review_path else ""
                review += 1
            rows.append(build_probe_row(input_index, record, prediction, accepted=accepted_flag, reason=reason, output_path=output_path))

    output_predictions = output_dir / "predictions.csv"
    output_review = output_dir / "review.csv"
    write_predictions_csv(output_predictions, rows)
    write_predictions_csv(output_review, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "predictions_csv": str(predictions_csv),
        "output_dir": str(output_dir),
        "kind": kind,
        "input_rows": input_rows,
        "processed": len(rows),
        "skipped": skipped,
        "accepted": accepted,
        "copied_accepted": copied_accepted,
        "review": review,
        "missing_images": missing_images,
        "thresholds": {
            "rank_score": float(rank_score_threshold),
            "rank_margin": float(rank_margin_threshold),
            "suit_score": float(suit_score_threshold),
            "suit_margin": float(suit_margin_threshold),
            "require_current_agreement": bool(require_current_agreement),
        },
        "counts": count_rows(rows),
        "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        "files": {
            "predictions_csv": str(output_predictions),
            "review_csv": str(output_review),
            "accepted_dir": str(output_dir) if copy_accepted else "",
            "review_dir": str(review_dir),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def ensemble_hf_probe_predictions(
    *,
    predictions_csvs: list[Path],
    output_dir: Path,
    kind: str = "both",
    rank_score_threshold: float = 0.82,
    rank_margin_threshold: float = 0.10,
    suit_score_threshold: float = 0.82,
    suit_margin_threshold: float = 0.10,
    require_current_agreement: bool = True,
    min_teachers: int | None = None,
    copy_accepted: bool = True,
) -> dict[str, Any]:
    """Accept crops only when multiple cached HF teachers agree on the label."""
    if kind not in ("rank", "suit", "both"):
        raise ValueError("kind must be rank, suit, or both")
    source_csvs = [Path(path) for path in predictions_csvs]
    if len(source_csvs) < 2:
        raise ValueError("provide at least two --predictions-csv files for ensemble")
    for path in source_csvs:
        if not path.is_file():
            raise ValueError(f"predictions CSV not found: {path}")

    output_dir = Path(output_dir)
    review_dir = output_dir / "review"
    for directory in (output_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    allowed_kinds = {"rank", "suit"} if kind == "both" else {kind}
    teacher_count = len(source_csvs)
    required_teachers = teacher_count if min_teachers is None else max(1, int(min_teachers))
    if required_teachers > teacher_count:
        raise ValueError("--min-teachers cannot exceed the number of --predictions-csv files")

    indexed_sources = [load_prediction_rows_by_key(path, allowed_kinds=allowed_kinds) for path in source_csvs]
    all_keys = sorted(set().union(*(set(source) for source in indexed_sources)), key=lambda key: (key[0], key[1]))

    rows: list[dict[str, Any]] = []
    accepted = 0
    copied_accepted = 0
    review = 0
    missing_images = 0
    started_at = time.perf_counter()

    for output_index, key in enumerate(all_keys):
        source_rows = [source.get(key) for source in indexed_sources]
        present_rows = [row for row in source_rows if row is not None]
        first_row = present_rows[0] if present_rows else {}
        crop_path = resolve_prediction_asset_path(
            first_row.get("input_path") or first_row.get("rank_path") or first_row.get("suit_path") or key[1],
            source_csvs[0].parent,
        )
        current_labels = sorted({str(row.get("current_label") or "").strip() for row in present_rows if str(row.get("current_label") or "").strip()})
        current_label = current_labels[0] if len(current_labels) == 1 else ""
        record = CropRecord(kind=key[0], path=crop_path, current_label=current_label)
        prediction, pre_reason = build_ensemble_prediction(
            present_rows,
            required_teachers=required_teachers,
            current_labels=current_labels,
        )
        if not crop_path.exists():
            accepted_flag = False
            reason = "image_missing"
            missing_images += 1
        elif pre_reason:
            accepted_flag = False
            reason = pre_reason
        else:
            accepted_flag, reason = should_accept_prediction(
                record,
                prediction,
                rank_score_threshold=rank_score_threshold,
                rank_margin_threshold=rank_margin_threshold,
                suit_score_threshold=suit_score_threshold,
                suit_margin_threshold=suit_margin_threshold,
                require_current_agreement=require_current_agreement,
            )
        output_path = ""
        if accepted_flag and prediction is not None:
            accepted += 1
            if copy_accepted:
                output_path = str(copy_labeled_crop(record, prediction, output_dir, output_index))
                copied_accepted += 1
        else:
            review_path = copy_review_crop(record, prediction, review_dir, output_index)
            output_path = str(review_path) if review_path else ""
            review += 1
        rows.append(build_probe_row(output_index, record, prediction, accepted=accepted_flag, reason=reason, output_path=output_path))

    output_predictions = output_dir / "predictions.csv"
    output_review = output_dir / "review.csv"
    write_predictions_csv(output_predictions, rows)
    write_predictions_csv(output_review, [row for row in rows if not row.get("accepted")], include_final_columns=True)
    summary = {
        "ok": True,
        "predictions_csvs": [str(path) for path in source_csvs],
        "output_dir": str(output_dir),
        "kind": kind,
        "teacher_count": teacher_count,
        "min_teachers": required_teachers,
        "processed": len(rows),
        "accepted": accepted,
        "copied_accepted": copied_accepted,
        "review": review,
        "missing_images": missing_images,
        "thresholds": {
            "rank_score": float(rank_score_threshold),
            "rank_margin": float(rank_margin_threshold),
            "suit_score": float(suit_score_threshold),
            "suit_margin": float(suit_margin_threshold),
            "require_current_agreement": bool(require_current_agreement),
        },
        "counts": count_rows(rows),
        "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        "files": {
            "predictions_csv": str(output_predictions),
            "review_csv": str(output_review),
            "accepted_dir": str(output_dir) if copy_accepted else "",
            "review_dir": str(review_dir),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_prediction_rows_by_key(path: Path, *, allowed_kinds: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            row_kind = str(row.get("kind") or "").strip().lower()
            if row_kind not in allowed_kinds:
                continue
            asset_text = row.get("input_path") or row.get("rank_path") or row.get("suit_path") or ""
            if not asset_text:
                continue
            asset_path = resolve_prediction_asset_path(asset_text, Path(path).parent)
            rows[(row_kind, prediction_key_path(asset_path))] = dict(row)
    return rows


def prediction_key_path(path: Path) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(Path(path).absolute()).lower()


def build_ensemble_prediction(
    rows: list[dict[str, Any]],
    *,
    required_teachers: int,
    current_labels: list[str],
) -> tuple[dict[str, Any] | None, str]:
    if len(rows) < required_teachers:
        return None, "missing_teacher"
    if len(current_labels) > 1:
        return None, "current_label_disagrees"
    labeled_rows = [
        row
        for row in rows
        if str(row.get("teacher_label") or "").strip()
    ]
    if len(labeled_rows) < required_teachers:
        return None, "teacher_label_missing"
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for row in labeled_rows:
        label = str(row.get("teacher_label") or "").strip()
        rows_by_label.setdefault(label, []).append(row)
    sorted_votes = sorted(
        rows_by_label.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(safe_row_float(row, "teacher_score") for row in item[1]) / max(1, len(item[1])),
            item[0],
        ),
    )
    top_label, vote_rows = sorted_votes[0]
    top_count = len(vote_rows)
    tied_top = len(sorted_votes) > 1 and len(sorted_votes[1][1]) == top_count
    if top_count < required_teachers or tied_top:
        model_votes = [
            f"{str(row.get('teacher_model') or '').strip() or f'teacher{index + 1}'}={str(row.get('teacher_label') or '').strip() or '?'}"
            for index, row in enumerate(rows)
        ]
        return {
            "label": "",
            "score": 0.0,
            "margin": 0.0,
            "second_score": max((safe_row_float(row, "teacher_second_score") for row in rows), default=0.0),
            "model": "ensemble_disagree[" + ";".join(model_votes) + "]",
            "backend": "hf_embedding_probe_ensemble",
        }, "teacher_disagrees"
    models = [str(row.get("teacher_model") or "").strip() for row in vote_rows]
    return {
        "label": top_label,
        "score": min((safe_row_float(row, "teacher_score") for row in vote_rows), default=0.0),
        "margin": min((safe_row_float(row, "teacher_margin") for row in vote_rows), default=0.0),
        "second_score": max((safe_row_float(row, "teacher_second_score") for row in vote_rows), default=0.0),
        "model": "ensemble[" + ";".join(model for model in models if model) + "]",
        "backend": "hf_embedding_probe_ensemble",
    }, ""


def safe_row_float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key)
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve_prediction_asset_path(text: Any, base_dir: Path) -> Path:
    path = Path(str(text or ""))
    if path.exists():
        return path
    candidate = Path(base_dir) / path
    if candidate.exists():
        return candidate
    return path


def classify_hf_probe_glyph_path(
    path_text: Any,
    kind: str,
    *,
    probe_dir: Path,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any] | None:
    if kind not in ("rank", "suit"):
        raise ValueError("kind must be rank or suit")
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.exists():
        return None
    probe = load_probe(Path(probe_dir) / PROBE_FILENAMES[kind])
    record = LabeledCrop(kind=kind, path=path, label="")
    features, failed_indices = extract_probe_features_with_failures(
        [record],
        kind=kind,
        model_name=str(probe["metadata"]["model_name"]),
        batch_size=1,
        device=device,
        local_files_only=local_files_only,
    )
    if failed_indices or features.shape[0] < 1:
        return None
    return predict_from_probe(probe, features[0])


def apply_hf_probe_to_review(
    *,
    review_csv: Path,
    output_dir: Path,
    probe_dir: Path,
    max_rows: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any]:
    review_csv = Path(review_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "review.csv"

    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    predicted_slots = 0
    changed_slots = 0
    missing_slots = 0
    processed_slots = 0
    slot_inputs: dict[tuple[int, int], dict[str, Any]] = {}
    tasks: dict[str, list[tuple[int, int, Path]]] = {"rank": [], "suit": []}
    started_at = time.perf_counter()
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max(0, int(max_rows)):
                break
            updated = dict(row)
            rows.append(updated)
            for slot in (0, 1):
                current_card = normalize_probe_card(updated.get(f"card{slot}"))
                rank_path = resolve_review_asset_path(review_csv, updated.get(f"card{slot}_rank_path") or "")
                suit_path = resolve_review_asset_path(review_csv, updated.get(f"card{slot}_suit_path") or "")
                if not rank_path and not suit_path:
                    continue
                processed_slots += 1
                slot_inputs[(row_index, slot)] = {
                    "current_card": current_card,
                    "rank_path": rank_path,
                    "suit_path": suit_path,
                }
                if rank_path:
                    tasks["rank"].append((row_index, slot, rank_path))
                if suit_path:
                    tasks["suit"].append((row_index, slot, suit_path))

    predictions: dict[tuple[int, int, str], dict[str, Any] | None] = {}
    for kind, items in tasks.items():
        if not items:
            continue
        probe = load_probe(Path(probe_dir) / PROBE_FILENAMES[kind])
        feature_records = [LabeledCrop(kind=kind, path=path, label="") for _row_index, _slot, path in items]
        features, failed_indices = extract_probe_features_with_failures(
            feature_records,
            kind=kind,
            model_name=str(probe["metadata"]["model_name"]),
            batch_size=batch_size,
            device=device,
            local_files_only=local_files_only,
        )
        feature_cursor = 0
        for local_index, (row_index, slot, _path) in enumerate(items):
            key = (row_index, slot, kind)
            if local_index in failed_indices:
                predictions[key] = None
                continue
            predictions[key] = predict_from_probe(probe, features[feature_cursor])
            feature_cursor += 1

    for (row_index, slot), slot_input in slot_inputs.items():
        updated = rows[row_index]
        current_card = slot_input.get("current_card")
        rank_result = predictions.get((row_index, slot, "rank"))
        suit_result = predictions.get((row_index, slot, "suit"))
        rank = normalize_probe_rank((rank_result or {}).get("label"))
        suit = normalize_probe_suit((suit_result or {}).get("label"))
        if not rank or not suit:
            missing_slots += 1
            continue
        predicted_card = f"{rank}{suit}"
        predicted_slots += 1
        if current_card and current_card != predicted_card:
            changed_slots += 1
        updated[f"card{slot}"] = predicted_card
        if rank_result:
            updated[f"card{slot}_rank_confidence"] = format_float(rank_result.get("score"))
            updated[f"card{slot}_rank_margin"] = format_float(rank_result.get("margin"))
        if suit_result:
            updated[f"card{slot}_suit_confidence"] = format_float(suit_result.get("score"))
            updated[f"card{slot}_suit_margin"] = format_float(suit_result.get("margin"))
        if f"card{slot}_roi_mode" in updated:
            updated[f"card{slot}_roi_mode"] = "hf_probe"

    write_review_rows(output_csv, rows, fieldnames)
    summary = {
        "ok": True,
        "review_csv": str(review_csv),
        "output_dir": str(output_dir),
        "probe_dir": str(probe_dir),
        "rows": len(rows),
        "processed_slots": processed_slots,
        "predicted_slots": predicted_slots,
        "changed_slots": changed_slots,
        "missing_slots": missing_slots,
        "device": device,
        "batch_size": int(batch_size),
        "local_files_only": bool(local_files_only),
        "sample": {
            "max_rows": max_rows,
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "files": {
            "review_csv": str(output_csv),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def resolve_review_asset_path(review_csv: Path, text: str) -> Path | None:
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    candidate = Path(review_csv).parent / text
    if candidate.exists():
        return candidate
    return path


def write_review_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    observed = list(fieldnames)
    seen = set(observed)
    for row in rows:
        for key in row:
            if key not in seen:
                observed.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=observed, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_probe_card(value: Any) -> str | None:
    text = str(value or "").strip().replace("10", "T")
    if len(text) != 2 or "?" in text:
        return None
    rank = normalize_probe_rank(text[0])
    suit = normalize_probe_suit(text[1])
    return f"{rank}{suit}" if rank and suit else None


def normalize_probe_rank(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("10", "T")
    return text if text in set(RANK_LABELS) else None


def normalize_probe_suit(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "spade": "s",
        "heart": "h",
        "diamond": "d",
        "club": "c",
    }
    text = aliases.get(text, text)
    return text if text in set(SUIT_LABELS) else None


def collect_labeled_crops(
    *,
    input_dirs: list[Path],
    kind: str,
    labels: list[str],
    template_dir: Path | None,
    include_templates: bool,
    max_images_per_class: int | None,
) -> list[LabeledCrop]:
    valid = set(labels)
    crops: list[LabeledCrop] = []
    source_groups: list[list[LabeledCrop]] = []
    seen: set[Path] = set()
    for root in input_dirs:
        group: list[LabeledCrop] = []
        for record in collect_crop_records([Path(root)], allowed_kinds=(kind,)):
            if record.current_label not in valid:
                continue
            resolved = safe_resolve(record.path)
            if resolved in seen:
                continue
            seen.add(resolved)
            group.append(LabeledCrop(kind=kind, path=record.path, label=record.current_label))
        if group:
            source_groups.append(group)
            crops.extend(group)
    if include_templates and template_dir is not None:
        template_crops = []
        for crop in collect_template_crops(Path(template_dir), kind=kind, labels=labels):
            resolved = safe_resolve(crop.path)
            if resolved in seen:
                continue
            seen.add(resolved)
            template_crops.append(crop)
        if template_crops:
            source_groups.append(template_crops)
            crops.extend(template_crops)
    if max_images_per_class is None:
        return crops
    return limit_labeled_crops_by_source(source_groups, labels=labels, max_images_per_class=int(max_images_per_class))


def limit_labeled_crops_by_source(
    source_groups: list[list[LabeledCrop]],
    *,
    labels: list[str],
    max_images_per_class: int,
) -> list[LabeledCrop]:
    max_images_per_class = max(0, int(max_images_per_class))
    if max_images_per_class <= 0:
        return []
    by_source_label: list[dict[str, list[LabeledCrop]]] = []
    for group in source_groups:
        buckets = {label: [] for label in labels}
        for crop in group:
            if crop.label in buckets:
                buckets[crop.label].append(crop)
        by_source_label.append(buckets)

    limited: list[LabeledCrop] = []
    counts = {label: 0 for label in labels}
    for label in labels:
        round_index = 0
        while counts[label] < max_images_per_class:
            progressed = False
            for buckets in by_source_label:
                candidates = buckets.get(label) or []
                if round_index >= len(candidates):
                    continue
                limited.append(candidates[round_index])
                counts[label] += 1
                progressed = True
                if counts[label] >= max_images_per_class:
                    break
            if not progressed:
                break
            round_index += 1
    return limited


def safe_resolve(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path).absolute()


def collect_template_crops(template_dir: Path, *, kind: str, labels: list[str]) -> list[LabeledCrop]:
    valid = set(labels)
    prefix = f"{kind}_"
    crops: list[LabeledCrop] = []
    if not template_dir.exists():
        return crops
    for path in sorted(template_dir.glob(f"{prefix}*.png")):
        label = path.stem.removeprefix(prefix).split("_", 1)[0]
        if label in valid:
            crops.append(LabeledCrop(kind=kind, path=path, label=label))
    return crops


def extract_probe_features(
    records: list[LabeledCrop],
    *,
    kind: str,
    model_name: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
) -> Any:
    features, failed_indices = extract_probe_features_with_failures(
        records,
        kind=kind,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )
    if failed_indices:
        failed = ", ".join(str(records[index].path) for index in sorted(failed_indices)[:5])
        raise ValueError(f"failed to read {len(failed_indices)} {kind} crops; examples: {failed}")
    return features


def extract_probe_features_with_failures(
    records: list[LabeledCrop],
    *,
    kind: str,
    model_name: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
) -> tuple[Any, set[int]]:
    cv2, np = load_cv()
    loaded = load_vision_encoder(model_name, device=device, local_files_only=local_files_only)
    torch = loaded["torch"]
    processor = loaded["processor"]
    model = loaded["model"]
    torch_device = loaded["device"]
    features: list[Any] = []
    failed_indices: set[int] = set()
    batch_images: list[Any] = []
    batch_positions: list[int] = []

    def flush() -> None:
        if not batch_images:
            return
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items() if hasattr(value, "to")}
        with torch.no_grad():
            encoded = encode_images(model, inputs)
            encoded = encoded.detach().float().cpu().numpy()
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        encoded = encoded / np.maximum(norms, 1e-12)
        features.extend(encoded)
        batch_images.clear()
        batch_positions.clear()

    for index, record in enumerate(records):
        image = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            failed_indices.add(index)
            continue
        batch_images.append(glyph_to_pil(image, kind=kind, source_path=record.path))
        batch_positions.append(index)
        if len(batch_images) >= max(1, int(batch_size)):
            flush()
    flush()
    if features:
        return np.asarray(features, dtype="float32"), failed_indices
    return np.zeros((0, 0), dtype="float32"), failed_indices


def load_vision_encoder(model_name: str, *, device: str, local_files_only: bool) -> dict[str, Any]:
    torch = load_torch()
    resolved_device = resolve_device(torch, device)
    cache_key = (model_name, bool(local_files_only), str(resolved_device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise RuntimeError("transformers is required for HuggingFace probe models: pip install transformers") from error
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=bool(local_files_only))
    model = AutoModel.from_pretrained(model_name, local_files_only=bool(local_files_only))
    model.eval()
    model.to(resolved_device)
    loaded = {"torch": torch, "processor": processor, "model": model, "device": resolved_device}
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def encode_images(model: Any, inputs: dict[str, Any]) -> Any:
    if hasattr(model, "get_image_features"):
        return model.get_image_features(**inputs)
    outputs = model(**inputs)
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("HuggingFace vision model did not expose image features")
    return hidden[:, 0] if hidden.ndim == 3 else hidden


def stratified_split_indices(records: list[LabeledCrop], *, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_label.setdefault(record.label, []).append(index)
    train: list[int] = []
    val: list[int] = []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * max(0.0, min(0.8, float(val_split)))))
        if len(shuffled) >= 5:
            val_count = max(1, val_count)
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:] or shuffled[:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_probe_payload(
    *,
    records: list[LabeledCrop],
    features: Any,
    labels: list[str],
    kind: str,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    cv2, np = load_cv()
    del cv2
    label_to_index = {label: index for index, label in enumerate(labels)}
    label_indices = np.asarray([label_to_index[record.label] for record in records], dtype="int64")
    prototypes = np.zeros((len(labels), int(features.shape[1]) if features.ndim == 2 else 0), dtype="float32")
    for label, label_index in label_to_index.items():
        mask = label_indices == label_index
        if not bool(mask.any()):
            continue
        proto = features[mask].mean(axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        prototypes[label_index] = proto.astype("float32")
    metadata = {
        "kind": kind,
        "model_name": model_name,
        "backend": "hf_embedding_probe",
        "labels": labels,
        "temperature": float(temperature),
        "source_count": len(records),
        "label_counts": count_labeled(records, labels),
    }
    return {
        "metadata": metadata,
        "labels": labels,
        "label_indices": label_indices,
        "features": features.astype("float32"),
        "prototypes": prototypes,
        "source_paths": [str(record.path) for record in records],
    }


def predict_from_probe(probe: dict[str, Any], feature: Any) -> dict[str, Any]:
    cv2, np = load_cv()
    del cv2
    metadata = probe["metadata"]
    labels = list(probe["labels"])
    prototypes = probe["prototypes"]
    scores = prototypes @ feature.astype("float32")
    present = np.linalg.norm(prototypes, axis=1) > 0
    scores = np.where(present, scores, -1e9)
    probs = softmax(scores / max(1e-6, float(metadata.get("temperature") or 0.04)))
    order = np.argsort(-probs)
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return {
        "label": labels[best],
        "score": float(probs[best]),
        "margin": float(probs[best] - (probs[second] if second != best else 0.0)),
        "second_score": float(probs[second] if second != best else 0.0),
        "model": str(metadata.get("model_name") or ""),
        "backend": "hf_embedding_probe",
    }


def softmax(values: Any) -> Any:
    _cv2, np = load_cv()
    finite = np.asarray(values, dtype="float32")
    max_value = float(np.max(finite))
    exp = np.exp(finite - max_value)
    total = float(exp.sum())
    if not math.isfinite(total) or total <= 0:
        return np.ones_like(exp) / max(1, exp.size)
    return exp / total


def evaluate_probe(*, probe: dict[str, Any], records: list[LabeledCrop], features: Any) -> dict[str, Any]:
    if not records:
        return {"count": 0, "accuracy": None, "correct": 0}
    correct = 0
    by_label: dict[str, dict[str, int]] = {}
    for record, feature in zip(records, features):
        prediction = predict_from_probe(probe, feature)
        label_stats = by_label.setdefault(record.label, {"count": 0, "correct": 0})
        label_stats["count"] += 1
        if prediction.get("label") == record.label:
            correct += 1
            label_stats["correct"] += 1
    for stats in by_label.values():
        stats["accuracy"] = round(stats["correct"] / max(1, stats["count"]), 6)
    return {"count": len(records), "correct": correct, "accuracy": round(correct / max(1, len(records)), 6), "by_label": by_label}


def save_probe(path: Path, payload: dict[str, Any]) -> None:
    _cv2, np = load_cv()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        metadata_json=json.dumps(payload["metadata"], ensure_ascii=False),
        labels=np.asarray(payload["labels"], dtype=object),
        label_indices=payload["label_indices"],
        features=payload["features"],
        prototypes=payload["prototypes"],
        source_paths=np.asarray(payload["source_paths"], dtype=object),
    )


def load_probe(path: Path) -> dict[str, Any]:
    _cv2, np = load_cv()
    path = Path(path)
    if not path.exists():
        raise ValueError(f"probe model not found: {path}")
    data = np.load(str(path), allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"].item()))
    return {
        "metadata": metadata,
        "labels": [str(item) for item in data["labels"].tolist()],
        "label_indices": data["label_indices"],
        "features": data["features"],
        "prototypes": data["prototypes"],
        "source_paths": [str(item) for item in data["source_paths"].tolist()],
        "path": str(path),
    }


def count_labeled(records: list[LabeledCrop], labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


def probe_source_dirs(
    *,
    input_dirs: list[Path],
    template_dir: Path | None,
    include_templates: bool,
) -> list[Path]:
    source_dirs = [Path(path) for path in input_dirs]
    if include_templates and template_dir is not None:
        source_dirs.append(Path(template_dir))
    return source_dirs


def count_labeled_by_source(
    records: list[LabeledCrop],
    *,
    source_dirs: list[Path],
    labels: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        str(path): {"count": 0, "label_counts": {label: 0 for label in labels}}
        for path in source_dirs
    }
    other_key = "_other"
    for record in records:
        key = source_key_for_path(record.path, source_dirs) or other_key
        if key not in result:
            result[key] = {"count": 0, "label_counts": {label: 0 for label in labels}}
        result[key]["count"] += 1
        result[key]["label_counts"][record.label] = result[key]["label_counts"].get(record.label, 0) + 1
    return {key: value for key, value in result.items() if value.get("count") or key != other_key}


def source_key_for_path(path: Path, source_dirs: list[Path]) -> str | None:
    resolved = safe_resolve(path)
    for source_dir in source_dirs:
        source_resolved = safe_resolve(source_dir)
        try:
            resolved.relative_to(source_resolved)
            return str(source_dir)
        except ValueError:
            continue
    return None


def limit_records_total_per_kind(records: list[CropRecord], *, max_per_kind: int) -> list[CropRecord]:
    counts: dict[str, int] = {}
    limited: list[CropRecord] = []
    for record in records:
        count = counts.get(record.kind, 0)
        if count >= max_per_kind:
            continue
        counts[record.kind] = count + 1
        limited.append(record)
    return limited


def build_probe_row(
    index: int,
    record: CropRecord,
    prediction: dict[str, Any] | None,
    *,
    accepted: bool,
    reason: str,
    output_path: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "kind": record.kind,
        "input_path": str(record.path),
        "current_label": record.current_label,
        "teacher_label": str((prediction or {}).get("label") or ""),
        "teacher_score": format_float((prediction or {}).get("score")),
        "teacher_margin": format_float((prediction or {}).get("margin")),
        "teacher_second_score": format_float((prediction or {}).get("second_score")),
        "teacher_model": str((prediction or {}).get("model") or ""),
        "accepted": bool(accepted),
        "reason": reason,
        "output_path": output_path,
        "rank_path": str(record.path) if record.kind == "rank" else "",
        "suit_path": str(record.path) if record.kind == "suit" else "",
        "card_path": "",
        "final_rank": "",
        "final_suit": "",
        "note": "",
    }


def format_hf_probe_train_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"train-card-hf-probe failed: {payload.get('error')}"
    lines = [
        f"Output: {payload.get('output_dir')}",
        f"Model: rank={payload.get('rank_model')} suit={payload.get('suit_model')}",
        f"Kind: {payload.get('kind')}",
    ]
    for kind, result in (payload.get("results") or {}).items():
        val = result.get("val") or {}
        accuracy = val.get("accuracy")
        accuracy_text = "-" if accuracy is None else f"{float(accuracy):.3f}"
        lines.extend(
            [
                f"{kind}: source={result.get('source_count')} train={result.get('train_count')} val={result.get('val_count')} val_acc={accuracy_text}",
                f"{kind} sources: {format_source_counts(result.get('source_counts_by_dir') or {})}",
                f"{kind} probe: {result.get('model_path')}",
            ]
        )
    lines.append(f"Summary: {(payload.get('files') or {}).get('summary')}")
    return "\n".join(lines)


def format_source_counts(source_counts_by_dir: dict[str, Any]) -> str:
    if not source_counts_by_dir:
        return "-"
    return ", ".join(
        f"{path}={int((info or {}).get('count') or 0)}"
        for path, info in source_counts_by_dir.items()
    )


def format_hf_probe_label_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"label-card-crops-hf-probe failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Processed: {payload.get('processed', 0)}",
            f"Accepted: {payload.get('accepted', 0)}",
            f"Copied accepted: {payload.get('copied_accepted', 0)}",
            f"Review: {payload.get('review', 0)}",
            f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
            f"Probe dir: {payload.get('probe_dir')}",
            f"Accepted dir: {files.get('accepted_dir')}",
            f"Review CSV: {files.get('review_csv')}",
            f"Predictions CSV: {files.get('predictions_csv')}",
        ]
    )


def format_hf_probe_filter_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"filter-card-hf-predictions failed: {payload.get('error')}"
    files = payload.get("files") or {}
    lines = [
        f"Source predictions: {payload.get('predictions_csv')}",
        f"Processed: {payload.get('processed', 0)}",
        f"Accepted: {payload.get('accepted', 0)}",
        f"Copied accepted: {payload.get('copied_accepted', 0)}",
        f"Review: {payload.get('review', 0)}",
        f"Missing images: {payload.get('missing_images', 0)}",
        f"Counts: {json.dumps(payload.get('counts') or {}, ensure_ascii=False)}",
        f"Accepted dir: {files.get('accepted_dir')}",
        f"Review CSV: {files.get('review_csv')}",
        f"Predictions CSV: {files.get('predictions_csv')}",
    ]
    distill = payload.get("distill_runtime") or {}
    if distill:
        if distill.get("stopped"):
            lines.append(f"Runtime distill: stopped ({distill.get('reason')})")
        else:
            lines.extend(
                [
                    f"Runtime candidate: {distill.get('candidate_name')} decision={str(distill.get('decision') or '').upper()} promote={distill.get('promote')}",
                    f"Runtime model: {distill.get('model_path')}",
                    f"Runtime gate report: {(distill.get('files') or {}).get('gate_report')}",
                    f"Runtime risk queue: {(distill.get('files') or {}).get('risk_label_queue_html') or '-'}",
                ]
            )
    return "\n".join(lines)


def format_hf_probe_ensemble_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"ensemble-card-hf-predictions failed: {payload.get('error')}"
    sources = payload.get("predictions_csvs") or []
    lines = [
        f"Source predictions: {len(sources)}",
        *[f"- {path}" for path in sources[:8]],
        f"Teachers required: {payload.get('min_teachers')} / {payload.get('teacher_count')}",
        *format_hf_probe_filter_summary({**payload, "predictions_csv": "ensemble"}).splitlines()[1:],
    ]
    return "\n".join(lines)


def format_hf_probe_review_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"apply-card-hf-probe-review failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Rows: {payload.get('rows', 0)}",
            f"Processed slots: {payload.get('processed_slots', 0)}",
            f"Predicted slots: {payload.get('predicted_slots', 0)}",
            f"Changed slots: {payload.get('changed_slots', 0)}",
            f"Missing slots: {payload.get('missing_slots', 0)}",
            f"Probe dir: {payload.get('probe_dir')}",
            f"Review CSV: {files.get('review_csv')}",
            f"Summary: {files.get('summary')}",
        ]
    )


def load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for HuggingFace probe models: pip install torch") from error
    return torch
