from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .card_glyph_label_queue import prepare_card_glyph_label_queue
from .card_glyph_label_server import (
    infer_glyph_context_paths,
    load_glyph_queue_csv,
    write_glyph_queue_csv,
)
from .screen_overlay import absolute_box, draw_box, render_diagnostic_frame
from .screen_vision import append_card_sample_glyph_rows, write_card_debug_assets
from .video_vision import (
    absolute_box_from_relative,
    load_cv,
    locked_profile_hero_read_boxes,
    make_crop_info,
    profile_hero_card_boxes,
    recognize_card_crop,
    select_locked_hero_variants,
)


def replay_fixed_card_samples(
    *,
    samples_dir: Path,
    layout_profile_path: Path,
    output_dir: Path,
    old_queue_csv: Path | None = None,
    sample_prefix: str | None = None,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    sample_dirs = sorted(
        path
        for path in samples_dir.iterdir()
        if path.is_dir()
        and (not sample_prefix or path.name.startswith(sample_prefix))
        and (path / "frame.png").is_file()
        and (path / "metadata.json").is_file()
    )
    if not sample_dirs:
        raise ValueError(f"no card sample frames found: {samples_dir}")

    base_profile = json.loads(Path(layout_profile_path).read_text(encoding="utf-8-sig"))
    first_frame = cv2.imread(str(sample_dirs[0] / "frame.png"))
    if first_frame is None:
        raise ValueError(f"cannot read first replay frame: {sample_dirs[0] / 'frame.png'}")
    frame_height, frame_width = first_frame.shape[:2]
    records = [json.loads((path / "metadata.json").read_text(encoding="utf-8-sig")) for path in sample_dirs]
    raw_hero_boxes = profile_hero_card_boxes(base_profile)
    shifted_hero_boxes = locked_profile_hero_read_boxes(base_profile)
    hero_boxes = raw_hero_boxes
    board_boxes = infer_fixed_board_boxes(records, frame_width, frame_height)
    if len(board_boxes) != 5:
        configured_board_boxes = base_profile.get("board_card_boxes")
        if isinstance(configured_board_boxes, list) and len(configured_board_boxes) == 5:
            board_boxes = [dict(item) for item in configured_board_boxes if isinstance(item, dict)]
    if len(hero_boxes) != 2:
        raise ValueError("layout profile does not contain two stable hero card boxes")
    if len(board_boxes) != 5:
        raise ValueError("could not infer five stable board card boxes")

    profile = dict(base_profile)
    profile.update(
        {
            "id": f"fixed-replay-{frame_width}x{frame_height}",
            "method": "fixed-relative-card-replay",
            "hero_card_source": "manual_hero_cards",
            "hero_search_source": "fixed_relative_replay",
            "hero_card_boxes": hero_boxes,
            "board_card_boxes": board_boxes,
            "show_fixed_card_boxes": True,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_output = output_dir / "samples"
    queue_output = output_dir / "glyph_label_queue"
    queue_csv = queue_output / "glyph_label_queue.csv"
    preserved_labels = load_existing_glyph_labels(queue_csv)
    predictions_csv = output_dir / "glyph_predictions.csv"
    if predictions_csv.exists():
        predictions_csv.unlink()
    samples_output.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "fixed_layout_profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    card_count = 0
    for sample_dir, metadata in zip(sample_dirs, records):
        frame = cv2.imread(str(sample_dir / "frame.png"))
        if frame is None:
            continue
        expected = expected_card_slots(metadata)
        shifted_hero_details = fixed_details_for_slots(
            frame,
            shifted_hero_boxes,
            expected.get("hero") or [],
            source="hero",
            roi_mode="fixed_replay_hero_shifted",
        )
        raw_hero_details = fixed_details_for_slots(
            frame,
            raw_hero_boxes,
            expected.get("hero") or [],
            source="hero",
            roi_mode="fixed_replay_hero_raw",
        )
        hero_details = select_fixed_hero_details(shifted_hero_details, raw_hero_details)
        board_details = fixed_details_for_slots(
            frame,
            board_boxes,
            expected.get("board") or [],
            source="board",
            roi_mode="fixed_replay_board",
        )
        cards = {
            "hero": [str(item.get("card") or "??") for item in hero_details],
            "board": [str(item.get("card") or "??") for item in board_details],
            "hero_details": hero_details,
            "board_details": board_details,
        }
        frame_result = {"cards": cards, "timing_ms": {}}
        old_source = dict(metadata.get("source") or {})
        state = {
            "ok": True,
            "source": old_source,
            "hero": {"cards": cards["hero"]},
            "table": {
                "board": cards["board"],
                "street": street_from_board_count(len(board_details)),
                "pot_bb": None,
                "dealer_seat": None,
            },
            "confidence": {
                "cards": {
                    "hero": hero_details,
                    "board": board_details,
                }
            },
        }
        diagnostic = render_diagnostic_frame(cv2, frame, frame_result, state, profile)
        draw_fixed_board_boxes(cv2, diagnostic, board_boxes)
        written = write_card_debug_assets(
            cv2=cv2,
            frame=frame,
            frame_result=frame_result,
            state=state,
            output_dir=samples_output,
            basename=sample_dir.name,
            problem="fixed_card_replay",
            diagnostic_frame=diagnostic,
        )
        if not written:
            continue
        timestamp = float(metadata.get("timestamp_sec") or old_source.get("timestamp_sec") or 0.0)
        frame_index = int(metadata.get("frame_index") or old_source.get("frame_index") or 0)
        append_card_sample_glyph_rows(
            predictions_csv,
            written.get("saved") or [],
            sample_id=sample_dir.name,
            timestamp_sec=timestamp,
            frame_index=frame_index,
        )
        card_count += len(written.get("saved") or [])

    queue_summary = prepare_card_glyph_label_queue(
        predictions_csvs=[predictions_csv],
        output_dir=queue_output,
        max_rows=10000,
        prefill_final_label="none",
    )
    migrated = migrate_existing_glyph_labels(
        old_queue_csv=Path(old_queue_csv) if old_queue_csv else None,
        new_queue_csv=Path(queue_summary["files"]["glyph_label_queue_csv"]),
        preserved_labels=preserved_labels,
    )
    preview = choose_replay_preview(samples_output)
    summary = {
        "ok": True,
        "source_samples": len(sample_dirs),
        "replayed_cards": card_count,
        "glyph_rows": int(queue_summary.get("selected_count") or 0),
        "migrated_labels": migrated,
        "hero_boxes": hero_boxes,
        "board_boxes": board_boxes,
        "files": {
            "profile": str(profile_path),
            "predictions_csv": str(predictions_csv),
            "queue_csv": queue_summary["files"]["glyph_label_queue_csv"],
            "queue_sheet": queue_summary["files"]["glyph_label_queue_sheet"],
            "preview": str(preview) if preview else "",
        },
    }
    summary_path = output_dir / "fixed_replay_summary.json"
    summary["files"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def infer_fixed_board_boxes(
    records: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, float]]:
    values: dict[int, list[dict[str, float]]] = {index: [] for index in range(5)}
    for metadata in records:
        for item in metadata.get("saved") or []:
            if str(item.get("group") or "") != "board":
                continue
            slot = int(item.get("slot") or 0)
            box = item.get("roi_box") or {}
            width = float(box.get("width") or 0)
            height = float(box.get("height") or 0)
            if slot not in values or width < frame_width * 0.055 or height < frame_height * 0.10:
                continue
            values[slot].append(
                {
                    "x": float(box.get("x") or 0),
                    "y": float(box.get("y") or 0),
                    "width": width,
                    "height": height,
                }
            )
    inferred: list[dict[str, float]] = []
    for slot in range(5):
        slot_values = values[slot]
        if not slot_values:
            return []
        median = {
            key: statistics.median(float(item[key]) for item in slot_values)
            for key in ("x", "y", "width", "height")
        }
        inferred.append(
            {
                "x": median["x"] / max(frame_width, 1),
                "y": median["y"] / max(frame_height, 1),
                "width": median["width"] / max(frame_width, 1),
                "height": median["height"] / max(frame_height, 1),
            }
        )
    return inferred


def expected_card_slots(metadata: dict[str, Any]) -> dict[str, list[int]]:
    slots: dict[str, set[int]] = {"hero": set(), "board": set()}
    for item in metadata.get("saved") or []:
        group = str(item.get("group") or "")
        if group in slots:
            slots[group].add(int(item.get("slot") or 0))
    return {group: sorted(values) for group, values in slots.items()}


def fixed_details_for_slots(
    frame: Any,
    relative_boxes: list[dict[str, float]],
    slots: list[int],
    *,
    source: str,
    roi_mode: str,
) -> list[dict[str, Any]]:
    frame_height, frame_width = frame.shape[:2]
    details = []
    for slot in slots:
        if slot < 0 or slot >= len(relative_boxes):
            continue
        box = absolute_box_from_relative(relative_boxes[slot], frame_width, frame_height)
        crop_info = make_crop_info(frame, box["x"], box["y"], box["width"], box["height"])
        detail = recognize_card_crop(
            crop_info["crop"],
            source=source,
            index=slot,
            allow_partial_hero=source == "hero",
            return_rejected=True,
        )
        if detail is None:
            detail = {
                "card": "??",
                "rank": "?",
                "suit": "?",
                "source": source,
                "index": slot,
                "rank_confidence": 0.0,
                "rank_margin": 0.0,
                "suit_confidence": 0.0,
                "suit_margin": 0.0,
                "color": "unknown",
            }
        detail["roi_mode"] = roi_mode
        detail["roi_box"] = crop_info["box"]
        details.append(detail)
    return details


def select_fixed_hero_details(
    shifted_details: list[dict[str, Any]],
    raw_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return select_locked_hero_variants(shifted_details, raw_details)


def draw_fixed_board_boxes(cv2: Any, image: Any, boxes: list[dict[str, float]]) -> None:
    height, width = image.shape[:2]
    for slot, relative in enumerate(boxes):
        draw_box(
            cv2,
            image,
            absolute_box(relative, width, height),
            (230, 170, 50),
            f"LOCK B{slot + 1}",
            dashed=True,
            label_row=slot % 2 + 2,
        )


def migrate_existing_glyph_labels(
    *,
    old_queue_csv: Path | None,
    new_queue_csv: Path,
    preserved_labels: dict[tuple[str, str, str, str], tuple[str, str, str]] | None = None,
) -> int:
    labels = load_existing_glyph_labels(old_queue_csv)
    labels.update(preserved_labels or {})
    visual_labels = load_existing_glyph_visual_labels(old_queue_csv)
    if not labels and not visual_labels:
        return 0
    new_rows, new_fields = load_glyph_queue_csv(new_queue_csv)
    migrated = 0
    for row in new_rows:
        label = labels.get(glyph_identity(row))
        migration_note = "migrated_from_old_crop"
        if label is None and not str(row.get("final_label") or "").strip():
            label = visual_labels.get(glyph_visual_identity(row))
            migration_note = "migrated_from_identical_glyph"
        if label is None:
            continue
        row["final_label"] = label[0]
        row["ignored"] = label[2]
        notes = []
        for item in label[1].split("|"):
            item = item.strip()
            if item and item not in notes:
                notes.append(item)
        if migration_note not in notes:
            notes.append(migration_note)
        row["notes"] = " | ".join(notes)
        migrated += 1
    write_glyph_queue_csv(new_queue_csv, new_rows, new_fields)
    return migrated


def load_existing_glyph_labels(
    queue_csv: Path | None,
) -> dict[tuple[str, str, str, str], tuple[str, str, str]]:
    if queue_csv is None or not queue_csv.is_file():
        return {}
    rows, _fields = load_glyph_queue_csv(queue_csv)
    labels = {}
    for row in rows:
        final_label = str(row.get("final_label") or "").strip()
        ignored = str(row.get("ignored") or "").strip()
        if final_label or ignored:
            labels[glyph_identity(row)] = (final_label, str(row.get("notes") or ""), ignored)
    return labels


def load_existing_glyph_visual_labels(
    queue_csv: Path | None,
) -> dict[tuple[str, int, int, str], tuple[str, str, str]]:
    """Return unambiguous human labels keyed by the exact glyph pixels."""
    if queue_csv is None or not queue_csv.is_file():
        return {}
    rows, _fields = load_glyph_queue_csv(queue_csv)
    candidates: dict[tuple[str, int, int, str], list[tuple[str, str, str]]] = {}
    for row in rows:
        final_label = str(row.get("final_label") or "").strip()
        identity = glyph_visual_identity(row)
        if not final_label or identity is None:
            continue
        candidates.setdefault(identity, []).append(
            (final_label, str(row.get("notes") or ""), "")
        )
    labels = {}
    for identity, choices in candidates.items():
        final_labels = {choice[0] for choice in choices}
        if len(final_labels) == 1:
            labels[identity] = choices[-1]
    return labels


def glyph_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    context = infer_glyph_context_paths(row)
    return (
        str(context.get("sample_id") or ""),
        str(context.get("group") or ""),
        str(context.get("slot") or ""),
        str(row.get("kind") or ""),
    )


def glyph_visual_identity(row: dict[str, Any]) -> tuple[str, int, int, str] | None:
    """Fingerprint a rendered glyph so duplicate replay crops share a label."""
    input_path = Path(str(row.get("input_path") or ""))
    if not input_path.is_file():
        return None
    cv2, _np = load_cv()
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    height, width = image.shape[:2]
    return str(row.get("kind") or ""), int(height), int(width), digest


def choose_replay_preview(samples_output: Path) -> Path | None:
    candidates = sorted(samples_output.glob("*/diagnostic_overlay.png"))
    for path in candidates:
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if len(metadata.get("board") or []) >= 5 and len(metadata.get("hero_cards") or []) >= 2:
            return path
    return candidates[0] if candidates else None


def street_from_board_count(count: int) -> str:
    if count >= 5:
        return "river"
    if count == 4:
        return "turn"
    if count >= 3:
        return "flop"
    return "preflop"


def format_fixed_replay_summary(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Source samples: {payload.get('source_samples')}",
            f"Replayed cards: {payload.get('replayed_cards')}",
            f"Glyph rows: {payload.get('glyph_rows')}",
            f"Migrated labels: {payload.get('migrated_labels')}",
            f"Preview: {files.get('preview')}",
            f"Queue CSV: {files.get('queue_csv')}",
        ]
    )
