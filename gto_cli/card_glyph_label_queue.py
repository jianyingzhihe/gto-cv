from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any

from .card_deep_model import RANK_LABELS, SUIT_LABELS
from .card_glyph_export import export_rank_glyph_image, safe_label, safe_stem
from .card_hf_probe import resolve_prediction_asset_path
from .card_teacher_label import format_float


GLYPH_LABEL_QUEUE_COLUMNS = [
    "label_id",
    "priority",
    "reason",
    "source_csv",
    "source_row",
    "kind",
    "input_path",
    "asset_path",
    "current_label",
    "current_confidence",
    "current_margin",
    "teacher_label",
    "teacher_score",
    "teacher_margin",
    "teacher_model",
    "final_label",
    "ignored",
    "notes",
]


def prepare_card_glyph_label_queue(
    *,
    predictions_csvs: list[Path],
    review_csvs: list[Path] | None = None,
    output_dir: Path,
    max_rows: int = 200,
    allowed_reasons: list[str] | None = None,
    include_accepted: bool = False,
    prefill_final_label: str = "none",
    copy_assets: bool = True,
    render_contact_sheet: bool = True,
) -> dict[str, Any]:
    if prefill_final_label not in {"none", "current", "teacher"}:
        raise ValueError("prefill_final_label must be one of: none, current, teacher")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    if copy_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)

    allowed = {str(reason).strip() for reason in (allowed_reasons or []) if str(reason).strip()}
    candidates: list[dict[str, Any]] = []
    for predictions_csv in [Path(path) for path in predictions_csvs]:
        candidates.extend(
            collect_glyph_label_candidates(
                predictions_csv,
                allowed_reasons=allowed,
                include_accepted=include_accepted,
            )
        )
    for review_csv in [Path(path) for path in (review_csvs or [])]:
        candidates.extend(
            collect_review_glyph_label_candidates(
                review_csv,
                allowed_reasons=allowed,
                include_accepted=include_accepted,
            )
        )
    if not predictions_csvs and not review_csvs:
        raise ValueError("provide at least one predictions_csv or review_csv")
    deduped = dedupe_glyph_candidates(candidates)
    selected = sorted(deduped, key=lambda row: float(row.get("priority") or 0.0), reverse=True)
    if max_rows is not None and int(max_rows) >= 0:
        selected = selected[: int(max_rows)]

    for index, row in enumerate(selected, start=1):
        row["label_id"] = f"G{index:04d}"
        if prefill_final_label != "none":
            row["final_label"] = suggested_final_label(row, mode=prefill_final_label)
        if copy_assets:
            attach_glyph_asset(row, assets_dir)

    queue_csv = output_dir / "glyph_label_queue.csv"
    queue_md = output_dir / "glyph_label_queue.md"
    queue_html = output_dir / "glyph_label_queue.html"
    write_glyph_queue_csv(queue_csv, selected)
    contact_sheet = (
        render_glyph_label_queue_contact_sheet(
            selected,
            output_path=output_dir / "glyph_label_queue_sheet.jpg",
            base_dir=output_dir,
        )
        if render_contact_sheet
        else {"ok": False, "skipped": True, "reason": "disabled"}
    )
    queue_md.write_text(format_glyph_label_queue_markdown(selected, queue_csv, queue_html, contact_sheet), encoding="utf-8")
    queue_html.write_text(format_glyph_label_queue_html(selected, queue_csv), encoding="utf-8")
    summary = {
        "ok": True,
        "input_csvs": [str(path) for path in predictions_csvs],
        "input_review_csvs": [str(path) for path in (review_csvs or [])],
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "deduped_count": len(deduped),
        "selected_count": len(selected),
        "include_accepted": bool(include_accepted),
        "prefill_final_label": prefill_final_label,
        "prefilled_count": sum(1 for row in selected if str(row.get("final_label") or "").strip()),
        "allowed_reasons": sorted(allowed),
        "copy_assets": bool(copy_assets),
        "reason_counts": count_reasons(selected),
        "kind_counts": count_kinds(selected),
        "contact_sheet": contact_sheet,
        "files": {
            "glyph_label_queue_csv": str(queue_csv),
            "glyph_label_queue_md": str(queue_md),
            "glyph_label_queue_html": str(queue_html),
            "glyph_label_queue_sheet": str(contact_sheet.get("path") or ""),
            "assets_dir": str(assets_dir) if copy_assets else "",
        },
    }
    (output_dir / "glyph_label_queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def suggested_final_label(row: dict[str, Any], *, mode: str) -> str:
    kind = str(row.get("kind") or "").strip().lower()
    source_key = "current_label" if mode == "current" else "teacher_label"
    return normalize_glyph_label(row.get(source_key), kind)


def collect_glyph_label_candidates(
    predictions_csv: Path,
    *,
    allowed_reasons: set[str],
    include_accepted: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(predictions_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, source_row in enumerate(reader):
            kind = str(source_row.get("kind") or "").strip().lower()
            if kind not in {"rank", "suit"}:
                continue
            reason = str(source_row.get("reason") or "").strip()
            if allowed_reasons and reason not in allowed_reasons:
                continue
            if not include_accepted and reason == "accepted":
                continue
            path_text = source_row.get("input_path") or source_row.get("rank_path") or source_row.get("suit_path") or ""
            if not path_text:
                continue
            input_path = resolve_prediction_asset_path(path_text, Path(predictions_csv).parent)
            rows.append(normalize_glyph_prediction_row(Path(predictions_csv), index, source_row, input_path))
    return rows


def collect_review_glyph_label_candidates(
    review_csv: Path,
    *,
    allowed_reasons: set[str],
    include_accepted: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(review_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, source_row in enumerate(reader):
            reason = str(source_row.get("review_reason") or source_row.get("class") or "review_row").strip()
            if allowed_reasons and reason not in allowed_reasons:
                continue
            if not include_accepted and reason in {"ok", "accepted"}:
                continue
            for slot in (0, 1):
                card = str(source_row.get(f"card{slot}") or "").strip()
                for kind in ("rank", "suit"):
                    path_text = source_row.get(f"card{slot}_{kind}_path") or ""
                    if not path_text:
                        continue
                    input_path = resolve_prediction_asset_path(path_text, Path(review_csv).parent)
                    current_label = current_label_from_card(card, kind)
                    rows.append(
                        normalize_review_glyph_row(
                            review_csv=Path(review_csv),
                            source_row=index,
                            slot=slot,
                            kind=kind,
                            input_path=input_path,
                            current_label=current_label,
                            reason=reason,
                            card=card,
                        )
                    )
    return rows


def current_label_from_card(card: str, kind: str) -> str:
    text = str(card or "").strip()
    if kind == "rank":
        if text.startswith("10"):
            return "T"
        return normalize_glyph_label(text[:1], "rank") if text else ""
    if kind == "suit":
        if len(text) >= 2:
            return normalize_glyph_label(text[-1:], "suit")
    return ""


def normalize_review_glyph_row(
    *,
    review_csv: Path,
    source_row: int,
    slot: int,
    kind: str,
    input_path: Path,
    current_label: str,
    reason: str,
    card: str,
) -> dict[str, Any]:
    return {
        "label_id": "",
        "priority": score_glyph_reason(reason),
        "reason": reason,
        "source_csv": str(review_csv),
        "source_row": f"{source_row}:card{slot}_{kind}",
        "kind": kind,
        "input_path": str(input_path),
        "asset_path": "",
        "current_label": current_label,
        "current_confidence": "",
        "current_margin": "",
        "teacher_label": "",
        "teacher_score": "",
        "teacher_margin": "",
        "teacher_model": "review_csv",
        "final_label": "",
        "notes": f"review_card={card or '-'}; slot={slot}; source={review_csv}",
    }


def normalize_glyph_prediction_row(predictions_csv: Path, index: int, row: dict[str, Any], input_path: Path) -> dict[str, Any]:
    reason = str(row.get("reason") or "").strip()
    current_label = str(row.get("current_label") or "").strip()
    teacher_label = str(row.get("teacher_label") or "").strip()
    return {
        "label_id": "",
        "priority": score_glyph_reason(reason),
        "reason": reason,
        "source_csv": str(predictions_csv),
        "source_row": index,
        "kind": str(row.get("kind") or "").strip().lower(),
        "input_path": str(input_path),
        "asset_path": "",
        "current_label": current_label,
        "current_confidence": format_float(row.get("confidence")),
        "current_margin": format_float(row.get("margin")),
        "teacher_label": teacher_label,
        "teacher_score": format_float(row.get("teacher_score")),
        "teacher_margin": format_float(row.get("teacher_margin")),
        "teacher_model": str(row.get("teacher_model") or ""),
        "final_label": "",
        "notes": f"current={current_label or '-'}; teacher={teacher_label or '-'}; model={row.get('teacher_model') or '-'}",
    }


def score_glyph_reason(reason: str) -> float:
    text = str(reason or "").lower()
    score = 10.0
    if "teacher_disagrees" in text:
        score += 100
    if "current_disagrees" in text:
        score += 80
    if "low_score" in text:
        score += 55
    if "low_margin" in text:
        score += 35
    if "image_missing" in text:
        score += 20
    if text == "accepted":
        score = 5
    return round(score, 4)


def dedupe_glyph_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('kind')}|{safe_resolved_text(row.get('input_path'))}"
        current = by_key.get(key)
        if current is None or float(row.get("priority") or 0.0) > float(current.get("priority") or 0.0):
            by_key[key] = row
    return list(by_key.values())


def safe_resolved_text(value: Any) -> str:
    path = Path(str(value or ""))
    try:
        return str(path.resolve()).lower()
    except OSError:
        return str(path.absolute()).lower()


def attach_glyph_asset(row: dict[str, Any], assets_dir: Path) -> None:
    source = Path(str(row.get("input_path") or ""))
    if not source.exists():
        return
    destination = assets_dir / safe_filename(f"{row.get('label_id')}_{row.get('kind')}_{source.name}")
    if str(row.get("kind") or "").lower() == "rank":
        card_path = source.with_name(source.name.replace("_rank.", "_card."))
        if card_path.exists():
            try:
                from .video_vision import load_cv

                cv2, _np = load_cv()
                card_crop = cv2.imread(str(card_path))
                if card_crop is not None:
                    card_source = "board" if source.name.startswith("board_") else "hero"
                    rank_image = export_rank_glyph_image(card_crop, card_source)
                    if cv2.imwrite(str(destination), rank_image):
                        row["asset_path"] = str(destination)
                        return
            except (OSError, ValueError):
                pass
    try:
        shutil.copy2(source, destination)
        row["asset_path"] = str(destination)
    except OSError:
        row["asset_path"] = ""


def apply_card_glyph_label_queue(*, queue_csv: Path, output_dir: Path) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_glyph_queue_rows(queue_csv)
    copied = 0
    skipped = 0
    invalid: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "").strip().lower()
        final_label = normalize_glyph_label(row.get("final_label"), kind)
        if not final_label:
            skipped += 1
            raw = str(row.get("final_label") or "").strip()
            if raw:
                invalid.append({"label_id": row.get("label_id"), "kind": kind, "value": raw})
            continue
        source = resolve_existing_queue_path(row.get("asset_path"), queue_csv.parent) or resolve_existing_queue_path(row.get("input_path"), queue_csv.parent)
        if source is None:
            skipped += 1
            missing.append({"label_id": row.get("label_id"), "path": row.get("input_path") or row.get("asset_path") or ""})
            continue
        destination = output_dir / kind / final_label / f"glyph_{safe_label(str(row.get('label_id') or 'row'))}_{safe_stem(source.stem)}{source.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
        label_key = f"{kind}:{final_label}"
        label_counts[label_key] = label_counts.get(label_key, 0) + 1
    summary = {
        "ok": True,
        "queue_csv": str(queue_csv),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "copied": copied,
        "skipped": skipped,
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "label_counts": dict(sorted(label_counts.items())),
        "files": {
            "dataset_dir": str(output_dir),
            "summary": str(output_dir / "glyph_label_apply_summary.json"),
        },
    }
    (output_dir / "glyph_label_apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def normalize_glyph_label(value: Any, kind: str) -> str:
    text = str(value or "").strip()
    if kind == "rank":
        text = text.upper().replace("10", "T")
        return text if text in set(RANK_LABELS) else ""
    if kind == "suit":
        text = text.lower()
        return text if text in set(SUIT_LABELS) else ""
    return ""


def read_glyph_queue_rows(queue_csv: Path) -> list[dict[str, Any]]:
    with Path(queue_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_glyph_queue_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_glyph_label_queue_contact_sheet(rows: list[dict[str, Any]], *, output_path: Path, base_dir: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return {"ok": False, "skipped": True, "reason": "pillow_missing"}
    if not rows:
        return {"ok": False, "skipped": True, "reason": "no_rows"}
    cell_w, cell_h = 360, 150
    cols = 2
    shown = rows[:80]
    rows_count = (len(shown) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, max(1, rows_count) * cell_h + 36), (28, 26, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Glyph label queue - rows={len(rows)} shown={len(shown)}", fill=(245, 245, 245), font=font)
    for index, row in enumerate(shown):
        col = index % cols
        line = index // cols
        x = col * cell_w + 8
        y = 36 + line * cell_h + 8
        draw.rectangle((x, y, x + cell_w - 16, y + cell_h - 12), outline=(95, 88, 78))
        image_path = resolve_existing_queue_path(row.get("asset_path"), base_dir) or resolve_existing_queue_path(row.get("input_path"), base_dir)
        if image_path:
            try:
                crop = Image.open(image_path).convert("RGB")
                crop.thumbnail((96, 96))
                sheet.paste(crop, (x + 8, y + 26))
            except Exception:
                draw.text((x + 12, y + 60), "image error", fill=(235, 95, 95), font=font)
        draw.text((x + 8, y + 8), str(row.get("label_id") or ""), fill=(255, 218, 96), font=font)
        text = (
            f"{row.get('kind')} cur={row.get('current_label') or '-'} "
            f"teacher={row.get('teacher_label') or '-'} final={row.get('final_label') or '-'}\n"
            f"{row.get('reason') or ''}\n"
            f"{row.get('teacher_model') or ''}"
        )
        draw.multiline_text((x + 116, y + 28), text[:180], fill=(235, 235, 225), font=font, spacing=3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)
    return {"ok": True, "path": str(output_path), "rows_rendered": len(shown)}


def format_glyph_label_queue_markdown(rows: list[dict[str, Any]], queue_csv: Path, queue_html: Path, contact_sheet: dict[str, Any]) -> str:
    lines = [
        "# Card Glyph Label Queue",
        "",
        f"- Rows: `{len(rows)}`",
        f"- CSV: `{queue_csv}`",
        f"- HTML: `{queue_html}`",
        f"- Contact sheet: `{contact_sheet.get('path') or ''}`",
        "",
        "Fill or correct `final_label` with a rank (`A K Q J T 9 ... 2`) or suit (`s h d c`), then apply:",
        "",
        "```powershell",
        f'python gto.py apply-card-glyph-label-queue --queue-csv "{queue_csv}" --output-dir "video_frames\\card_glyph_label_applied"',
        "```",
        "",
        "| ID | Kind | Current | Teacher | Reason | Final |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows[:120]:
        lines.append(
            f"| {row.get('label_id')} | {row.get('kind')} | {row.get('current_label') or '-'} | "
            f"{row.get('teacher_label') or '-'} | {row.get('reason') or '-'} | {row.get('final_label') or ''} |"
        )
    return "\n".join(lines)


def format_glyph_label_queue_html(rows: list[dict[str, Any]], queue_csv: Path) -> str:
    cards = []
    for row in rows:
        img = html.escape(str(row.get("asset_path") or row.get("input_path") or ""))
        cards.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('label_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('kind') or ''))}</td>"
            f"<td><img src='{img}' style='max-width:96px;max-height:96px;background:white'></td>"
            f"<td>{html.escape(str(row.get('current_label') or '-'))}</td>"
            f"<td>{html.escape(str(row.get('teacher_label') or '-'))}</td>"
            f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
            f"<td>{html.escape(str(row.get('teacher_model') or ''))}</td>"
            f"<td>{html.escape(str(row.get('final_label') or ''))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Card Glyph Label Queue</title>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#1f1b18;color:#eee}table{border-collapse:collapse;width:100%}td,th{border:1px solid #54483f;padding:6px;vertical-align:top}th{background:#352d27}</style>",
            "<h1>Card Glyph Label Queue</h1>",
            f"<p>CSV: <code>{html.escape(str(queue_csv))}</code></p>",
            "<table><thead><tr><th>ID</th><th>Kind</th><th>Image</th><th>Current</th><th>Teacher</th><th>Reason</th><th>Model</th><th>Final</th></tr></thead><tbody>",
            *cards,
            "</tbody></table>",
        ]
    )


def resolve_existing_queue_path(value: Any, base_dir: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    candidate = Path(base_dir) / path
    if candidate.exists():
        return candidate
    return None


def count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def count_kinds(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


def format_card_glyph_label_queue_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"prepare-card-glyph-label-queue failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Candidates: {payload.get('candidate_count')}",
            f"Selected: {payload.get('selected_count')}",
            f"Kinds: {json.dumps(payload.get('kind_counts') or {}, ensure_ascii=False)}",
            f"Reasons: {json.dumps(payload.get('reason_counts') or {}, ensure_ascii=False)}",
            f"Queue CSV: {files.get('glyph_label_queue_csv')}",
            f"HTML: {files.get('glyph_label_queue_html')}",
            f"Contact sheet: {files.get('glyph_label_queue_sheet')}",
        ]
    )


def format_card_glyph_label_apply_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"apply-card-glyph-label-queue failed: {payload.get('error')}"
    return "\n".join(
        [
            f"Rows: {payload.get('rows')}",
            f"Copied: {payload.get('copied')}",
            f"Skipped: {payload.get('skipped')}",
            f"Invalid: {payload.get('invalid_count')}",
            f"Missing: {payload.get('missing_count')}",
            f"Dataset: {(payload.get('files') or {}).get('dataset_dir')}",
            f"Counts: {json.dumps(payload.get('label_counts') or {}, ensure_ascii=False)}",
        ]
    )
