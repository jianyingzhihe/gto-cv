from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .card_glyph_label_queue import prepare_card_glyph_label_queue
from .card_label_queue import prepare_card_label_queue
from .card_review_export import (
    count_rows,
    format_card_review_summary,
    write_review_csv,
    write_review_markdown,
    write_review_sheet,
)


def collect_card_debug_review(
    *,
    input_dir: Path,
    output_dir: Path,
    include_fallback: bool = True,
    max_rows: int | None = None,
    max_sheet_rows: int = 160,
    prepare_label_queue: bool = False,
    queue_output_dir: Path | None = None,
    queue_max_rows: int = 80,
    copy_queue_assets: bool = True,
    prepare_glyph_label_queue: bool = False,
    glyph_queue_output_dir: Path | None = None,
    glyph_queue_max_rows: int = 160,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(input_dir.rglob("metadata.json"))
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        if max_rows is not None and len(rows) >= max(0, int(max_rows)):
            break
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            skipped.append({"metadata": str(metadata_path), "reason": f"read_failed:{error}"})
            continue
        row = row_from_card_debug_metadata(metadata_path, metadata, include_fallback=include_fallback)
        if row is None:
            skipped.append({"metadata": str(metadata_path), "reason": "no_hero_card_assets"})
            continue
        rows.append(row)

    review_csv_path = output_dir / "review.csv"
    review_md_path = output_dir / "review.md"
    sheet_path = output_dir / "review_sheet.jpg"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.jsonl"
    runbook_path = output_dir / "runbook.md"
    queue_dir = Path(queue_output_dir) if queue_output_dir else output_dir / "label_queue"
    glyph_queue_dir = Path(glyph_queue_output_dir) if glyph_queue_output_dir else output_dir / "glyph_label_queue"
    retrain_dir = output_dir / "label_retrain"
    glyph_apply_dir = output_dir / "glyph_label_applied"

    write_review_csv(review_csv_path, rows)
    write_review_markdown(review_md_path, rows)
    write_review_sheet(sheet_path, rows[: max(0, int(max_sheet_rows))])
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    queue_payload = None
    if prepare_label_queue:
        queue_payload = prepare_card_label_queue(
            review_csvs=[review_csv_path],
            output_dir=queue_dir,
            max_rows=queue_max_rows,
            copy_assets=copy_queue_assets,
        )

    glyph_queue_payload = None
    if prepare_glyph_label_queue:
        glyph_queue_payload = prepare_card_glyph_label_queue(
            predictions_csvs=[],
            review_csvs=[review_csv_path],
            output_dir=glyph_queue_dir,
            max_rows=glyph_queue_max_rows,
            copy_assets=copy_queue_assets,
        )

    summary = {
        "ok": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "metadata_files": len(metadata_paths),
        "rows": len(rows),
        "skipped": len(skipped),
        "counts": count_rows(rows),
        "label_queue": queue_payload,
        "glyph_label_queue": glyph_queue_payload,
        "commands": {
            "prepare_label_queue": (
                f'python gto.py prepare-card-label-queue --review-csv "{review_csv_path}" '
                f'--output-dir "{queue_dir}" --max-rows {int(queue_max_rows)} --format text'
            ),
            "serve_label_queue": (
                f'python gto.py serve-card-label-queue --queue-csv "{queue_dir / "label_queue.csv"}" --open-browser'
            ),
            "audit_label_queue": (
                f'python gto.py audit-card-label-queue --queue-csv "{queue_dir / "label_queue.csv"}" '
                f'--output-dir "{queue_dir}" --format text'
            ),
            "retrain_label_queue": (
                f'python gto.py retrain-card-label-queue --queue-csv "{queue_dir / "label_queue.csv"}" '
                f'--output-dir "{retrain_dir}" --format text'
            ),
            "prepare_glyph_label_queue": (
                f'python gto.py prepare-card-glyph-label-queue --review-csv "{review_csv_path}" '
                f'--output-dir "{glyph_queue_dir}" --max-rows {int(glyph_queue_max_rows)} --format text'
            ),
            "apply_glyph_label_queue": (
                f'python gto.py apply-card-glyph-label-queue --queue-csv "{glyph_queue_dir / "glyph_label_queue.csv"}" '
                f'--output-dir "{glyph_apply_dir}" --format text'
            ),
        },
        "files": {
            "manifest": str(manifest_path),
            "review_csv": str(review_csv_path),
            "review_md": str(review_md_path),
            "review_sheet": str(sheet_path),
            "runbook": str(runbook_path),
            "summary": str(summary_path),
        },
        "skipped_rows": skipped[:100],
    }
    runbook_path.write_text(format_card_debug_review_runbook(summary), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def row_from_card_debug_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    include_fallback: bool,
) -> dict[str, Any] | None:
    hero_assets = hero_debug_assets(metadata, include_fallback=include_fallback)
    if not hero_assets:
        return None
    cards = [str(card) for card in (metadata.get("hero_cards") or []) if str(card or "").strip()]
    problem = str(metadata.get("problem") or "card_debug")
    source = metadata.get("source") or {}
    row: dict[str, Any] = {
        "video": source.get("path") or metadata.get("video") or str(metadata_path.parent),
        "timestamp_sec": metadata.get("timestamp_sec", ""),
        "frame_index": metadata.get("frame_index", ""),
        "class": problem,
        "review_reason": problem,
        "raw_hero_cards": " ".join(cards),
        "stabilized_hero_cards": " ".join(cards),
        "board": " ".join(str(card) for card in (metadata.get("board") or [])),
        "street": "",
        "dealer": "",
        "hero_position": "",
        "hero_turn": "",
        "table_frame_path": metadata.get("frame", ""),
        "final_card0": "",
        "final_card1": "",
        "notes": f"card_debug={metadata_path}",
    }
    for slot in range(2):
        asset = hero_assets.get(slot) or {}
        row.update(
            {
                f"card{slot}": asset.get("card", ""),
                f"card{slot}_rank_confidence": asset.get("rank_confidence", ""),
                f"card{slot}_rank_margin": asset.get("rank_margin", ""),
                f"card{slot}_suit_confidence": asset.get("suit_confidence", ""),
                f"card{slot}_suit_margin": asset.get("suit_margin", ""),
                f"card{slot}_roi_mode": asset.get("roi_mode", ""),
                f"card{slot}_card_path": asset.get("card_path", ""),
                f"card{slot}_rank_path": asset.get("rank_path", ""),
                f"card{slot}_suit_path": asset.get("suit_path", ""),
            }
        )
    return row


def hero_debug_assets(metadata: dict[str, Any], *, include_fallback: bool) -> dict[int, dict[str, Any]]:
    assets: dict[int, dict[str, Any]] = {}
    for item in metadata.get("saved") or []:
        if str(item.get("group") or "") != "hero":
            continue
        slot = int(item.get("slot") or 0)
        assets[slot] = dict(item)
    if include_fallback:
        for item in metadata.get("fallback") or []:
            if str(item.get("group") or "") != "hero":
                continue
            slot = int(item.get("slot") or 0)
            assets.setdefault(slot, dict(item))
    return assets


def format_card_debug_review_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"collect-card-debug-review failed: {payload.get('error')}"
    summary = format_card_review_summary(
        {
            "ok": True,
            "sample": {"rows": payload.get("rows")},
            "counts": payload.get("counts") or {},
            "files": payload.get("files") or {},
        }
    )
    runbook = (payload.get("files") or {}).get("runbook")
    label_queue = payload.get("label_queue") or {}
    queue_files = label_queue.get("files") or {}
    glyph_queue = payload.get("glyph_label_queue") or {}
    glyph_files = glyph_queue.get("files") or {}
    extras = []
    if queue_files.get("label_queue_csv"):
        extras.append(f"Queue CSV: {queue_files.get('label_queue_csv')}")
    if queue_files.get("label_queue_html"):
        extras.append(f"Queue HTML: {queue_files.get('label_queue_html')}")
    if glyph_files.get("glyph_label_queue_csv"):
        extras.append(f"Glyph queue CSV: {glyph_files.get('glyph_label_queue_csv')}")
    if glyph_files.get("glyph_label_queue_html"):
        extras.append(f"Glyph queue HTML: {glyph_files.get('glyph_label_queue_html')}")
    if runbook:
        extras.append(f"Runbook: {runbook}")
    return summary + ("\n" + "\n".join(extras) if extras else "")


def format_card_debug_review_runbook(payload: dict[str, Any]) -> str:
    commands = payload.get("commands") or {}
    files = payload.get("files") or {}
    lines = [
        "# Live Card Debug Review",
        "",
        f"- Input: `{payload.get('input_dir')}`",
        f"- Rows: `{payload.get('rows')}`",
        f"- Counts: `{json.dumps(payload.get('counts') or {}, ensure_ascii=False)}`",
        f"- Review CSV: `{files.get('review_csv')}`",
        f"- Review sheet: `{files.get('review_sheet')}`",
    ]
    label_queue = payload.get("label_queue") or {}
    if label_queue:
        queue_files = label_queue.get("files") or {}
        lines.extend(
            [
                f"- Label queue CSV: `{queue_files.get('label_queue_csv')}`",
                f"- Label queue HTML: `{queue_files.get('label_queue_html')}`",
            ]
        )
    glyph_queue = payload.get("glyph_label_queue") or {}
    if glyph_queue:
        glyph_files = glyph_queue.get("files") or {}
        lines.extend(
            [
                f"- Glyph label queue CSV: `{glyph_files.get('glyph_label_queue_csv')}`",
                f"- Glyph label queue HTML: `{glyph_files.get('glyph_label_queue_html')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
        ]
    )
    if label_queue:
        lines.extend(
            [
                "The label queue has already been built for this run. Rebuild it only if you collect more debug samples.",
                "",
            ]
        )
    lines.extend(
        [
            "1. Build the browser label queue:",
            "",
            "```powershell",
            str(commands.get("prepare_label_queue") or ""),
            "```",
            "",
            "2. Open the label UI and fill `final_card0` / `final_card1`:",
            "",
            "```powershell",
            str(commands.get("serve_label_queue") or ""),
            "```",
            "",
            "3. Check whether the queue is complete:",
            "",
            "```powershell",
            str(commands.get("audit_label_queue") or ""),
            "```",
            "",
            "4. Train, validate, and gate a candidate KNN model:",
            "",
            "```powershell",
            str(commands.get("retrain_label_queue") or ""),
            "```",
            "",
            "Do not promote a candidate unless the retrain gate says `promote`.",
            "",
            "## Split Rank/Suit Glyph Queue",
            "",
            "Use this when the problem is a single rank or suit crop rather than the whole two-card hand.",
            "",
            "```powershell",
            str(commands.get("prepare_glyph_label_queue") or ""),
            "```",
            "",
            "After filling `final_label`, apply the glyph labels:",
            "",
            "```powershell",
            str(commands.get("apply_glyph_label_queue") or ""),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
