from __future__ import annotations

import csv
import html
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any


LABEL_QUEUE_COLUMNS = [
    "label_id",
    "priority",
    "reason",
    "source_csv",
    "source_row",
    "video",
    "timestamp_sec",
    "frame_index",
    "class",
    "street",
    "dealer",
    "hero_position",
    "hero_turn",
    "raw_hero_cards",
    "stabilized_hero_cards",
    "card0",
    "card0_consensus",
    "card0_card_path",
    "card0_rank_path",
    "card0_suit_path",
    "card1",
    "card1_consensus",
    "card1_card_path",
    "card1_rank_path",
    "card1_suit_path",
    "table_frame_path",
    "asset_table",
    "asset_card0",
    "asset_rank0",
    "asset_suit0",
    "asset_card1",
    "asset_rank1",
    "asset_suit1",
    "final_card0",
    "final_card1",
    "notes",
]


def prepare_card_label_queue(
    *,
    review_csvs: list[Path],
    output_dir: Path,
    max_rows: int = 120,
    include_ok: bool = False,
    include_completed: bool = False,
    copy_assets: bool = True,
    render_contact_sheet: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    if copy_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for review_csv in [Path(path) for path in review_csvs]:
        candidates.extend(
            collect_label_candidates(
                review_csv,
                include_ok=include_ok,
                include_completed=include_completed,
            )
        )
    deduped = dedupe_candidates(candidates)
    selected = sorted(deduped, key=lambda row: float(row.get("priority") or 0.0), reverse=True)
    if max_rows is not None and int(max_rows) >= 0:
        selected = selected[: int(max_rows)]

    for index, row in enumerate(selected, start=1):
        row["label_id"] = f"L{index:04d}"
        if copy_assets:
            attach_assets(row, assets_dir)

    queue_csv = output_dir / "label_queue.csv"
    queue_md = output_dir / "label_queue.md"
    queue_html = output_dir / "label_queue.html"
    write_queue_csv(queue_csv, selected)
    contact_sheet = (
        render_label_queue_contact_sheet(
            selected,
            output_path=output_dir / "label_queue_sheet.jpg",
            base_dir=output_dir,
        )
        if render_contact_sheet
        else {"ok": False, "skipped": True, "reason": "disabled"}
    )
    queue_md.write_text(format_label_queue_markdown(selected, queue_csv, queue_html, contact_sheet), encoding="utf-8")
    queue_html.write_text(format_label_queue_html(selected, queue_csv), encoding="utf-8")
    summary = {
        "ok": True,
        "input_csvs": [str(path) for path in review_csvs],
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "deduped_count": len(deduped),
        "selected_count": len(selected),
        "include_ok": bool(include_ok),
        "include_completed": bool(include_completed),
        "copy_assets": bool(copy_assets),
        "contact_sheet": contact_sheet,
        "reason_counts": count_reasons(selected),
        "files": {
            "label_queue_csv": str(queue_csv),
            "label_queue_md": str(queue_md),
            "label_queue_html": str(queue_html),
            "label_queue_sheet": str(contact_sheet.get("path") or ""),
            "assets_dir": str(assets_dir) if copy_assets else "",
        },
    }
    (output_dir / "label_queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_card_diff_label_queue(
    *,
    diff_csv: Path,
    output_dir: Path,
    max_rows: int = 80,
    risk_only: bool = True,
    include_same: bool = False,
    prefer_candidate_assets: bool = True,
    copy_assets: bool = True,
    render_contact_sheet: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    if copy_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_diff_label_candidates(
        Path(diff_csv),
        risk_only=risk_only,
        include_same=include_same,
        prefer_candidate_assets=prefer_candidate_assets,
    )
    deduped = dedupe_candidates(candidates)
    selected = sorted(deduped, key=lambda row: float(row.get("priority") or 0.0), reverse=True)
    if max_rows is not None and int(max_rows) >= 0:
        selected = selected[: int(max_rows)]

    for index, row in enumerate(selected, start=1):
        row["label_id"] = f"D{index:04d}"
        if copy_assets:
            attach_assets(row, assets_dir)

    queue_csv = output_dir / "label_queue.csv"
    queue_md = output_dir / "label_queue.md"
    queue_html = output_dir / "label_queue.html"
    write_queue_csv(queue_csv, selected)
    contact_sheet = (
        render_label_queue_contact_sheet(
            selected,
            output_path=output_dir / "label_queue_sheet.jpg",
            base_dir=output_dir,
        )
        if render_contact_sheet
        else {"ok": False, "skipped": True, "reason": "disabled"}
    )
    queue_md.write_text(format_label_queue_markdown(selected, queue_csv, queue_html, contact_sheet), encoding="utf-8")
    queue_html.write_text(format_label_queue_html(selected, queue_csv), encoding="utf-8")
    summary = {
        "ok": True,
        "diff_csv": str(diff_csv),
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "deduped_count": len(deduped),
        "selected_count": len(selected),
        "risk_only": bool(risk_only),
        "include_same": bool(include_same),
        "asset_preference": "candidate" if prefer_candidate_assets else "baseline",
        "copy_assets": bool(copy_assets),
        "contact_sheet": contact_sheet,
        "reason_counts": count_reasons(selected),
        "files": {
            "label_queue_csv": str(queue_csv),
            "label_queue_md": str(queue_md),
            "label_queue_html": str(queue_html),
            "label_queue_sheet": str(contact_sheet.get("path") or ""),
            "assets_dir": str(assets_dir) if copy_assets else "",
        },
    }
    (output_dir / "label_queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit_card_label_queue(
    *,
    queue_csv: Path,
    output_dir: Path | None = None,
    applied_output_dir: Path | None = None,
    render_contact_sheet: bool = True,
) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    if output_dir is None:
        output_dir = queue_csv.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if applied_output_dir is None:
        applied_output_dir = output_dir / "applied"

    rows = read_queue_rows(queue_csv)
    invalid_labels: list[dict[str, Any]] = []
    missing_assets: list[dict[str, Any]] = []
    unlabeled_rows: list[str] = []
    label_matches = {
        "current": 0,
        "consensus": 0,
        "other": 0,
        "blank": 0,
    }
    total_slots = 0
    labeled_slots = 0
    card_counts: dict[str, int] = {}

    for row in rows:
        row_has_slot = False
        row_has_label = False
        for asset_key in (
            "asset_table",
            "asset_card0",
            "asset_rank0",
            "asset_suit0",
            "asset_card1",
            "asset_rank1",
            "asset_suit1",
        ):
            path_text = str(row.get(asset_key) or "").strip()
            if path_text and not resolve_existing_path(path_text, queue_csv.parent):
                missing_assets.append(
                    {
                        "label_id": row.get("label_id", ""),
                        "field": asset_key,
                        "path": path_text,
                    }
                )
        for slot in (0, 1):
            current = clean_card(row.get(f"card{slot}"))
            consensus = clean_card(row.get(f"card{slot}_consensus"))
            final_card = clean_card(row.get(f"final_card{slot}"))
            has_slot = bool(
                current
                or consensus
                or row.get(f"card{slot}_card_path")
                or row.get(f"asset_card{slot}")
                or row.get(f"card{slot}_rank_path")
                or row.get(f"card{slot}_suit_path")
            )
            if not has_slot:
                continue
            row_has_slot = True
            total_slots += 1
            raw_final = str(row.get(f"final_card{slot}") or "").strip()
            if raw_final and not final_card:
                invalid_labels.append(
                    {
                        "label_id": row.get("label_id", ""),
                        "slot": slot,
                        "value": raw_final,
                    }
                )
                label_matches["blank"] += 1
                continue
            if not final_card:
                label_matches["blank"] += 1
                continue
            labeled_slots += 1
            row_has_label = True
            card_counts[final_card] = card_counts.get(final_card, 0) + 1
            if current and final_card == current:
                label_matches["current"] += 1
            elif consensus and final_card == consensus:
                label_matches["consensus"] += 1
            else:
                label_matches["other"] += 1
        if row_has_slot and not row_has_label:
            unlabeled_rows.append(str(row.get("label_id") or ""))

    labeled_rows = len([row for row in rows if clean_card(row.get("final_card0")) or clean_card(row.get("final_card1"))])
    completion = (float(labeled_slots) / float(total_slots)) if total_slots else 1.0
    apply_command = f'python gto.py apply-card-review --review-csv "{queue_csv}" --output-dir "{applied_output_dir}"'
    contact_sheet = (
        render_label_queue_contact_sheet(
            rows,
            output_path=output_dir / "label_queue_sheet.jpg",
            base_dir=queue_csv.parent,
        )
        if render_contact_sheet
        else {"ok": False, "skipped": True, "reason": "disabled"}
    )
    summary = {
        "ok": True,
        "queue_csv": str(queue_csv),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": len(unlabeled_rows),
        "unlabeled_row_ids": unlabeled_rows[:50],
        "total_slots": total_slots,
        "labeled_slots": labeled_slots,
        "unlabeled_slots": max(0, total_slots - labeled_slots),
        "completion": round(completion, 6),
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels[:50],
        "missing_asset_count": len(missing_assets),
        "missing_assets": missing_assets[:50],
        "label_matches": label_matches,
        "final_card_counts": dict(sorted(card_counts.items(), key=lambda item: (-item[1], item[0]))),
        "contact_sheet": contact_sheet,
        "ready_to_apply": bool(labeled_slots > 0 and not invalid_labels and not missing_assets),
        "ready_to_retrain": bool(labeled_slots == total_slots and total_slots > 0 and not invalid_labels and not missing_assets),
        "commands": {
            "apply_review": apply_command,
        },
        "files": {
            "audit_json": str(output_dir / "label_queue_audit.json"),
            "audit_md": str(output_dir / "label_queue_audit.md"),
            "label_queue_sheet": str(contact_sheet.get("path") or ""),
        },
    }
    (output_dir / "label_queue_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "label_queue_audit.md").write_text(format_card_label_queue_audit_markdown(summary), encoding="utf-8")
    return summary


def read_queue_rows(queue_csv: Path) -> list[dict[str, str]]:
    with Path(queue_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader]


def collect_diff_label_candidates(
    diff_csv: Path,
    *,
    risk_only: bool,
    include_same: bool,
    prefer_candidate_assets: bool,
) -> list[dict[str, Any]]:
    rows = []
    with Path(diff_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, source_row in enumerate(reader):
            status = str(source_row.get("status") or "")
            risk = truthy_text(source_row.get("risk"))
            baseline_card = clean_card(source_row.get("baseline_card"))
            candidate_card = clean_card(source_row.get("candidate_card"))
            if risk_only and not risk:
                continue
            if not include_same and status.startswith("same"):
                continue
            if not baseline_card and not candidate_card:
                continue
            row = normalize_diff_row(
                Path(diff_csv),
                index,
                source_row,
                prefer_candidate_assets=prefer_candidate_assets,
            )
            if not any(row.get(key) for key in ("card0_card_path", "card0_rank_path", "card0_suit_path", "table_frame_path")):
                continue
            rows.append(row)
    return rows


def normalize_diff_row(
    diff_csv: Path,
    index: int,
    row: dict[str, Any],
    *,
    prefer_candidate_assets: bool = True,
) -> dict[str, Any]:
    status = str(row.get("status") or "")
    risk_reason = str(row.get("risk_reason") or "")
    baseline_card = clean_card(row.get("baseline_card"))
    candidate_card = clean_card(row.get("candidate_card"))
    slot = str(row.get("slot") or "")
    reason = ";".join(part for part in ["diff", status, risk_reason, f"slot={slot}", f"{baseline_card or '-'}->{candidate_card or '-'}"] if part)
    notes = (
        f"original_slot={slot}; baseline={baseline_card or '-'}; candidate={candidate_card or '-'}; "
        f"baseline_rank_conf={row.get('baseline_rank_confidence') or '-'}; "
        f"candidate_rank_conf={row.get('candidate_rank_confidence') or '-'}; "
        f"baseline_suit_conf={row.get('baseline_suit_confidence') or '-'}; "
        f"candidate_suit_conf={row.get('candidate_suit_confidence') or '-'}"
    )
    first_prefix = "candidate" if prefer_candidate_assets else "baseline"
    second_prefix = "baseline" if prefer_candidate_assets else "candidate"

    def choose_asset(suffix: str) -> Any:
        return row.get(f"{first_prefix}_{suffix}") or row.get(f"{second_prefix}_{suffix}") or ""

    return {
        "label_id": "",
        "priority": score_diff_reason(status=status, risk_reason=risk_reason, risk=truthy_text(row.get("risk"))),
        "reason": reason,
        "source_csv": str(diff_csv),
        "source_row": index,
        "video": row.get("video", ""),
        "timestamp_sec": row.get("timestamp_sec", ""),
        "frame_index": row.get("frame_index", ""),
        "class": "diff_risk" if truthy_text(row.get("risk")) else "diff_review",
        "street": "",
        "dealer": "",
        "hero_position": "",
        "hero_turn": "",
        "raw_hero_cards": "",
        "stabilized_hero_cards": "",
        "card0": baseline_card,
        "card0_consensus": candidate_card,
        "card0_card_path": choose_asset("card_path"),
        "card0_rank_path": choose_asset("rank_path"),
        "card0_suit_path": choose_asset("suit_path"),
        "card1": "",
        "card1_consensus": "",
        "card1_card_path": "",
        "card1_rank_path": "",
        "card1_suit_path": "",
        "table_frame_path": choose_asset("table_frame_path"),
        "asset_table": "",
        "asset_card0": "",
        "asset_rank0": "",
        "asset_suit0": "",
        "asset_card1": "",
        "asset_rank1": "",
        "asset_suit1": "",
        "final_card0": "",
        "final_card1": "",
        "notes": notes,
    }


def score_diff_reason(*, status: str, risk_reason: str, risk: bool) -> float:
    score = 50.0 if risk else 20.0
    text = f"{status};{risk_reason}".lower()
    if "regression" in text:
        score += 50
    if "changed_high_confidence" in text:
        score += 35
    if "downgraded" in text:
        score += 25
    if "candidate_lost" in text or "missing" in text:
        score += 20
    if "changed" in text:
        score += 10
    return round(score, 4)


def truthy_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def collect_label_candidates(
    review_csv: Path,
    *,
    include_ok: bool,
    include_completed: bool,
) -> list[dict[str, Any]]:
    rows = []
    with Path(review_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, source_row in enumerate(reader):
            row = normalize_review_row(Path(review_csv), index, source_row)
            if not include_completed and (clean_card(row.get("final_card0")) or clean_card(row.get("final_card1"))):
                continue
            reason = str(row.get("reason") or "")
            if not include_ok and reason in {"", "ok"}:
                continue
            if not any(row.get(key) for key in ("card0_card_path", "card1_card_path", "table_frame_path")):
                continue
            rows.append(row)
    return rows


def normalize_review_row(review_csv: Path, index: int, row: dict[str, Any]) -> dict[str, Any]:
    reason = str(row.get("audit_reason") or row.get("review_reason") or "")
    priority = parse_float(row.get("audit_priority"))
    if priority is None:
        priority = score_reason(reason, row.get("class"))
    output = {
        "label_id": "",
        "priority": round(float(priority), 4),
        "reason": reason,
        "source_csv": str(review_csv),
        "source_row": index,
        "video": row.get("video", ""),
        "timestamp_sec": row.get("timestamp_sec", ""),
        "frame_index": row.get("frame_index", ""),
        "class": row.get("class", ""),
        "street": row.get("street", ""),
        "dealer": row.get("dealer", ""),
        "hero_position": row.get("hero_position", ""),
        "hero_turn": row.get("hero_turn", ""),
        "raw_hero_cards": row.get("raw_hero_cards", ""),
        "stabilized_hero_cards": row.get("stabilized_hero_cards", ""),
        "card0": row.get("card0", ""),
        "card0_consensus": row.get("card0_consensus", row.get("card0", "")),
        "card0_card_path": row.get("card0_card_path", ""),
        "card0_rank_path": row.get("card0_rank_path", ""),
        "card0_suit_path": row.get("card0_suit_path", ""),
        "card1": row.get("card1", ""),
        "card1_consensus": row.get("card1_consensus", row.get("card1", "")),
        "card1_card_path": row.get("card1_card_path", ""),
        "card1_rank_path": row.get("card1_rank_path", ""),
        "card1_suit_path": row.get("card1_suit_path", ""),
        "table_frame_path": row.get("table_frame_path", ""),
        "asset_table": "",
        "asset_card0": "",
        "asset_rank0": "",
        "asset_suit0": "",
        "asset_card1": "",
        "asset_rank1": "",
        "asset_suit1": "",
        "final_card0": row.get("final_card0", ""),
        "final_card1": row.get("final_card1", ""),
        "notes": row.get("notes", ""),
    }
    return output


def score_reason(reason: str, row_class: Any) -> float:
    text = f"{reason};{row_class}".lower()
    score = 0.0
    if "obstructed" in text:
        score += 95
    if "consensus_card_disagrees" in text:
        score += 90
    if "rank_low" in text or "rank_current_low_confidence" in text:
        score += 65
    if "suit_low" in text or "suit_current_low_confidence" in text:
        score += 55
    if "open_suit" in text:
        score += 40
    if "incomplete" in text or "missed" in text:
        score += 75
    if "ok" in text and score == 0:
        score = 10
    return score


def dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = dedupe_key(row)
        current = by_key.get(key)
        if current is None or float(row.get("priority") or 0.0) > float(current.get("priority") or 0.0):
            by_key[key] = row
    return list(by_key.values())


def dedupe_key(row: dict[str, Any]) -> str:
    paths = [str(row.get(key) or "") for key in ("card0_card_path", "card1_card_path", "table_frame_path")]
    if any(paths):
        return "|".join(paths)
    return "|".join(str(row.get(key) or "") for key in ("video", "timestamp_sec", "frame_index"))


def attach_assets(row: dict[str, Any], assets_dir: Path) -> None:
    mapping = {
        "asset_table": "table_frame_path",
        "asset_card0": "card0_card_path",
        "asset_rank0": "card0_rank_path",
        "asset_suit0": "card0_suit_path",
        "asset_card1": "card1_card_path",
        "asset_rank1": "card1_rank_path",
        "asset_suit1": "card1_suit_path",
    }
    for asset_key, source_key in mapping.items():
        source = resolve_path(row.get(source_key))
        if source is None or not source.exists():
            continue
        stem = safe_filename(f"{row.get('label_id')}_{asset_key}_{source.name}")
        destination = assets_dir / stem
        try:
            shutil.copy2(source, destination)
            row[asset_key] = str(destination)
        except OSError:
            row[asset_key] = ""


def resolve_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


def resolve_existing_path(path_text: str, base_dir: Path) -> Path | None:
    path_text = str(path_text or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_queue_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_QUEUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_label_queue_markdown(
    rows: list[dict[str, Any]],
    queue_csv: Path,
    queue_html: Path,
    contact_sheet: dict[str, Any] | None = None,
) -> str:
    sheet_path = str((contact_sheet or {}).get("path") or "")
    lines = [
        "# Card Manual Label Queue",
        "",
        f"- Queue CSV: `{queue_csv}`",
        f"- HTML review: `{queue_html}`",
        f"- Contact sheet: `{sheet_path}`" if sheet_path else "- Contact sheet: `not generated`",
        f"- Rows: `{len(rows)}`",
        "",
        "Fill `final_card0` and `final_card1` in `label_queue.csv`, then run:",
        "",
        "```powershell",
        f'python gto.py apply-card-review --review-csv "{queue_csv}" --output-dir "video_frames\\card_label_queue_applied"',
        "```",
        "",
        "| ID | Priority | Reason | Current | Consensus | Table | Card0 | Card1 |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        table = md_image(row.get("asset_table") or row.get("table_frame_path"), "table")
        card0 = md_image(row.get("asset_card0") or row.get("card0_card_path"), "card0")
        card1 = md_image(row.get("asset_card1") or row.get("card1_card_path"), "card1")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("label_id")),
                    str(row.get("priority")),
                    truncate(str(row.get("reason") or "-"), 80),
                    f"{row.get('card0') or '-'} {row.get('card1') or '-'}",
                    f"{row.get('card0_consensus') or '-'} {row.get('card1_consensus') or '-'}",
                    table,
                    card0,
                    card1,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_label_queue_html(rows: list[dict[str, Any]], queue_csv: Path) -> str:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Card Manual Label Queue</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:18px;background:#171717;color:#eee}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #444;padding:6px;vertical-align:top}",
        "th{position:sticky;top:0;background:#252525}",
        "img.table{max-width:360px} img.card{max-width:92px} img.glyph{max-width:42px;image-rendering:pixelated}",
        ".reason{max-width:360px;white-space:normal}.muted{color:#aaa}.label{color:#ffdf62;font-weight:bold}",
        "</style></head><body>",
        "<h1>Card Manual Label Queue</h1>",
        f"<p>Fill <code>final_card0</code>/<code>final_card1</code> in <code>{html.escape(str(queue_csv))}</code>, then run <code>apply-card-review</code>.</p>",
        "<table>",
        "<tr><th>ID</th><th>Priority</th><th>Reason</th><th>Context</th><th>Table</th><th>Card 0</th><th>Card 1</th><th>Fill</th></tr>",
    ]
    for row in rows:
        parts.append("<tr>")
        parts.append(f"<td class='label'>{html.escape(str(row.get('label_id') or ''))}</td>")
        parts.append(f"<td>{html.escape(str(row.get('priority') or ''))}</td>")
        parts.append(f"<td class='reason'>{html.escape(str(row.get('reason') or '-'))}</td>")
        context = "<br>".join(
            html.escape(str(item))
            for item in [
                Path(str(row.get("video") or "")).name,
                f"t={row.get('timestamp_sec')} frame={row.get('frame_index')}",
                f"{row.get('street') or '-'} dealer={row.get('dealer') or '-'} hero={row.get('hero_position') or '-'} turn={row.get('hero_turn') or '-'}",
                f"raw={row.get('raw_hero_cards') or '-'} stable={row.get('stabilized_hero_cards') or '-'}",
            ]
        )
        parts.append(f"<td class='muted'>{context}</td>")
        parts.append(f"<td>{html_img(row.get('asset_table') or row.get('table_frame_path'), 'table')}</td>")
        parts.append(format_card_cell(row, 0))
        parts.append(format_card_cell(row, 1))
        parts.append(
            "<td>"
            f"final_card0: <b>{html.escape(str(row.get('final_card0') or ''))}</b><br>"
            f"final_card1: <b>{html.escape(str(row.get('final_card1') or ''))}</b><br>"
            f"notes: {html.escape(str(row.get('notes') or ''))}"
            "</td>"
        )
        parts.append("</tr>")
    parts.extend(["</table>", "</body></html>"])
    return "\n".join(parts)


def render_label_queue_contact_sheet(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    base_dir: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return {
            "ok": False,
            "skipped": True,
            "reason": "pillow_unavailable",
            "path": str(output_path),
            "rows": len(rows),
            "rendered_images": 0,
        }

    selected = list(rows if max_rows is None else rows[: max(0, int(max_rows))])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1560
    header_height = 34
    row_height = 228
    height = header_height + max(1, len(selected)) * row_height
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = load_sheet_font(ImageFont, size=17)
    small_font = load_sheet_font(ImageFont, size=13)
    title_font = load_sheet_font(ImageFont, size=18)

    rendered_images = 0
    image_errors: list[dict[str, str]] = []
    draw.rectangle((0, 0, width, header_height), fill=(32, 32, 32))
    draw.text((10, 7), f"Card label queue contact sheet - rows={len(selected)}", fill=(245, 245, 245), font=title_font)

    for index, row in enumerate(selected):
        y = header_height + index * row_height
        fill = (255, 255, 255) if index % 2 == 0 else (238, 238, 238)
        draw.rectangle((0, y, width, y + row_height - 1), fill=fill)
        draw.line((0, y, width, y), fill=(205, 205, 205))
        draw_queue_row_text(draw, row, x=10, y=y + 8, width_chars=45, font=small_font)

        specs = [
            ((330, y + 28, 286, 162), ("asset_table", "table_frame_path"), "table"),
            ((646, y + 28, 116, 162), ("asset_card0", "card0_card_path"), "card0"),
            ((792, y + 38, 82, 82), ("asset_rank0", "card0_rank_path"), "rank0"),
            ((898, y + 38, 82, 82), ("asset_suit0", "card0_suit_path"), "suit0"),
            ((1024, y + 28, 116, 162), ("asset_card1", "card1_card_path"), "card1"),
            ((1170, y + 38, 82, 82), ("asset_rank1", "card1_rank_path"), "rank1"),
            ((1276, y + 38, 82, 82), ("asset_suit1", "card1_suit_path"), "suit1"),
        ]
        for box, keys, label in specs:
            path = first_existing_path(row, keys, base_dir)
            draw_image_label(draw, box, label, font=small_font)
            if not path:
                continue
            pasted = paste_sheet_image(canvas, path, box, Image, image_errors)
            rendered_images += 1 if pasted else 0
        draw_queue_row_notes(draw, row, x=1378, y=y + 28, width_chars=24, font=small_font)

    try:
        canvas.save(output_path, quality=92)
    except OSError as error:
        return {
            "ok": False,
            "skipped": False,
            "reason": f"save_failed:{error}",
            "path": str(output_path),
            "rows": len(selected),
            "rendered_images": rendered_images,
            "image_errors": image_errors[:20],
        }
    return {
        "ok": True,
        "path": str(output_path),
        "rows": len(selected),
        "rendered_images": rendered_images,
        "image_error_count": len(image_errors),
        "image_errors": image_errors[:20],
    }


def load_sheet_font(image_font: Any, *, size: int) -> Any:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return image_font.truetype(name, size=size)
        except OSError:
            continue
    return image_font.load_default()


def draw_queue_row_text(draw: Any, row: dict[str, Any], *, x: int, y: int, width_chars: int, font: Any) -> None:
    lines = [
        f"{row.get('label_id') or '-'}  {Path(str(row.get('video') or '')).name}",
        f"t={row.get('timestamp_sec') or '-'}  frame={row.get('frame_index') or '-'}  priority={row.get('priority') or '-'}",
        f"current={row.get('card0') or '-'} {row.get('card1') or '-'}",
        f"candidate={row.get('card0_consensus') or '-'} {row.get('card1_consensus') or '-'}",
        f"final={row.get('final_card0') or '?'} {row.get('final_card1') or '?'}",
    ]
    reason = str(row.get("reason") or "")
    if reason:
        lines.extend(textwrap.wrap(f"reason={reason}", width=width_chars)[:4])
    draw.multiline_text((x, y), "\n".join(lines), fill=(28, 28, 28), font=font, spacing=3)


def draw_queue_row_notes(draw: Any, row: dict[str, Any], *, x: int, y: int, width_chars: int, font: Any) -> None:
    notes = str(row.get("notes") or "")
    lines = textwrap.wrap(notes, width=width_chars)[:9] if notes else ["no notes"]
    draw.multiline_text((x, y), "\n".join(lines), fill=(55, 55, 55), font=font, spacing=3)


def draw_image_label(draw: Any, box: tuple[int, int, int, int], label: str, *, font: Any) -> None:
    x, y, width, height = box
    draw.rectangle((x, y, x + width, y + height), outline=(210, 210, 210), width=1)
    draw.text((x + 4, y - 18), label, fill=(95, 95, 95), font=font)


def first_existing_path(row: dict[str, Any], keys: tuple[str, ...], base_dir: Path) -> Path | None:
    for key in keys:
        path = resolve_existing_path(str(row.get(key) or ""), base_dir)
        if path:
            return path
    return None


def paste_sheet_image(canvas: Any, path: Path, box: tuple[int, int, int, int], image_module: Any, errors: list[dict[str, str]]) -> bool:
    x, y, width, height = box
    try:
        with image_module.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((width - 8, height - 8), resample=get_resample_filter(image_module))
            px = x + max(4, (width - image.width) // 2)
            py = y + max(4, (height - image.height) // 2)
            canvas.paste(image, (px, py))
        return True
    except (OSError, ValueError) as error:
        errors.append({"path": str(path), "error": str(error)})
        return False


def get_resample_filter(image_module: Any) -> Any:
    resampling = getattr(image_module, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return getattr(image_module, "LANCZOS", 1)


def format_card_cell(row: dict[str, Any], slot: int) -> str:
    card = html.escape(str(row.get(f"card{slot}") or "-"))
    consensus = html.escape(str(row.get(f"card{slot}_consensus") or "-"))
    card_img = html_img(row.get(f"asset_card{slot}") or row.get(f"card{slot}_card_path"), "card")
    rank_img = html_img(row.get(f"asset_rank{slot}") or row.get(f"card{slot}_rank_path"), "glyph")
    suit_img = html_img(row.get(f"asset_suit{slot}") or row.get(f"card{slot}_suit_path"), "glyph")
    return f"<td><div>cur={card} cons={consensus}</div>{card_img}<br>{rank_img} {suit_img}</td>"


def html_img(path_text: Any, css_class: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    return f"<img class='{css_class}' src='{html.escape(path_to_uri(text))}'>"


def md_image(path_text: Any, alt: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "-"
    return f"![{alt}]({text.replace(chr(92), '/')})"


def path_to_uri(text: str) -> str:
    return Path(text).as_posix()


def count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("reason") or "-").split(";"):
            key = reason or "-"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def clean_card(value: Any) -> str:
    text = str(value or "").strip().replace("10", "T")
    if len(text) != 2 or "?" in text:
        return ""
    return text


def parse_float(value: Any) -> float | None:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return float(text)
    except ValueError:
        return None


def truncate(text: str, length: int) -> str:
    return text if len(text) <= length else text[: max(0, length - 3)] + "..."


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def format_card_label_queue_summary(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    contact_sheet = payload.get("contact_sheet") or {}
    return "\n".join(
        [
            f"Candidates: {payload.get('candidate_count')}",
            f"Deduped: {payload.get('deduped_count')}",
            f"Selected: {payload.get('selected_count')}",
            f"Asset preference: {payload.get('asset_preference') or '-'}",
            f"Reasons: {json.dumps(payload.get('reason_counts') or {}, ensure_ascii=False)}",
            f"Queue CSV: {files.get('label_queue_csv')}",
            f"HTML: {files.get('label_queue_html')}",
            f"Markdown: {files.get('label_queue_md')}",
            f"Contact sheet: {contact_sheet.get('path') or files.get('label_queue_sheet') or ''}",
        ]
    )


def format_card_label_queue_audit_summary(payload: dict[str, Any]) -> str:
    commands = payload.get("commands") or {}
    contact_sheet = payload.get("contact_sheet") or {}
    return "\n".join(
        [
            f"Rows: {payload.get('row_count')}",
            f"Labeled rows: {payload.get('labeled_rows')} / {payload.get('row_count')}",
            f"Labeled slots: {payload.get('labeled_slots')} / {payload.get('total_slots')}",
            f"Completion: {float(payload.get('completion') or 0.0):.1%}",
            f"Invalid labels: {payload.get('invalid_label_count')}",
            f"Missing assets: {payload.get('missing_asset_count')}",
            f"Matches: {json.dumps(payload.get('label_matches') or {}, ensure_ascii=False)}",
            f"Ready to apply: {payload.get('ready_to_apply')}",
            f"Ready to retrain: {payload.get('ready_to_retrain')}",
            f"Contact sheet: {contact_sheet.get('path') or ((payload.get('files') or {}).get('label_queue_sheet')) or ''}",
            f"Apply command: {commands.get('apply_review')}",
        ]
    )


def format_card_label_queue_audit_markdown(payload: dict[str, Any]) -> str:
    commands = payload.get("commands") or {}
    lines = [
        "# Card Label Queue Audit",
        "",
        f"- Queue CSV: `{payload.get('queue_csv')}`",
        f"- Rows: `{payload.get('row_count')}`",
        f"- Labeled rows: `{payload.get('labeled_rows')}`",
        f"- Labeled slots: `{payload.get('labeled_slots')} / {payload.get('total_slots')}`",
        f"- Completion: `{float(payload.get('completion') or 0.0):.1%}`",
        f"- Invalid labels: `{payload.get('invalid_label_count')}`",
        f"- Missing assets: `{payload.get('missing_asset_count')}`",
        f"- Contact sheet: `{((payload.get('contact_sheet') or {}).get('path')) or ((payload.get('files') or {}).get('label_queue_sheet')) or ''}`",
        f"- Ready to apply: `{payload.get('ready_to_apply')}`",
        f"- Ready to retrain: `{payload.get('ready_to_retrain')}`",
        "",
        "## Label Matches",
        "",
        "```json",
        json.dumps(payload.get("label_matches") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Apply",
        "",
        "```powershell",
        str(commands.get("apply_review") or ""),
        "```",
    ]
    if payload.get("unlabeled_row_ids"):
        lines.extend(["", "## Unlabeled Rows", "", ", ".join(payload.get("unlabeled_row_ids") or [])])
    if payload.get("invalid_labels"):
        lines.extend(["", "## Invalid Labels", "", "```json", json.dumps(payload.get("invalid_labels"), ensure_ascii=False, indent=2), "```"])
    if payload.get("missing_assets"):
        lines.extend(["", "## Missing Assets", "", "```json", json.dumps(payload.get("missing_assets"), ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"
