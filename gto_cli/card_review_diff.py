from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .card_benchmark import normalize_card


DIFF_COLUMNS = [
    "status",
    "risk",
    "risk_reason",
    "video",
    "timestamp_sec",
    "frame_index",
    "slot",
    "truth_card",
    "baseline_card",
    "candidate_card",
    "baseline_ok",
    "candidate_ok",
    "baseline_reason",
    "candidate_reason",
    "baseline_rank_confidence",
    "baseline_rank_margin",
    "baseline_suit_confidence",
    "baseline_suit_margin",
    "candidate_rank_confidence",
    "candidate_rank_margin",
    "candidate_suit_confidence",
    "candidate_suit_margin",
    "baseline_card_path",
    "candidate_card_path",
    "baseline_rank_path",
    "candidate_rank_path",
    "baseline_suit_path",
    "candidate_suit_path",
    "baseline_table_frame_path",
    "candidate_table_frame_path",
]


def diff_card_review(
    *,
    baseline_csv: Path,
    candidate_csv: Path,
    output_dir: Path,
    risky_baseline_reasons: tuple[str, ...] = ("ok",),
    max_rows: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_review_csv(Path(baseline_csv))
    candidate_rows = read_review_csv(Path(candidate_csv))
    baseline_by_key = {review_row_key(row): row for row in baseline_rows}
    candidate_by_key = {review_row_key(row): row for row in candidate_rows}

    matched_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    missing_in_candidate = sorted(set(baseline_by_key) - set(candidate_by_key))
    extra_in_candidate = sorted(set(candidate_by_key) - set(baseline_by_key))

    rows: list[dict[str, Any]] = []
    for key in matched_keys:
        baseline_row = baseline_by_key[key]
        candidate_row = candidate_by_key[key]
        for slot in (0, 1):
            diff_row = compare_slot(
                baseline_row,
                candidate_row,
                slot=slot,
                risky_baseline_reasons=risky_baseline_reasons,
            )
            if diff_row:
                rows.append(diff_row)
                if max_rows is not None and len(rows) >= max(0, int(max_rows)):
                    break
        if max_rows is not None and len(rows) >= max(0, int(max_rows)):
            break

    summary = {
        "ok": True,
        "baseline_csv": str(baseline_csv),
        "candidate_csv": str(candidate_csv),
        "output_dir": str(output_dir),
        "baseline_row_count": len(baseline_rows),
        "candidate_row_count": len(candidate_rows),
        "matched_row_count": len(matched_keys),
        "missing_in_candidate_count": len(missing_in_candidate),
        "extra_in_candidate_count": len(extra_in_candidate),
        "risky_baseline_reasons": list(risky_baseline_reasons),
        "counts": summarize_diff_rows(rows),
        "changed_examples": select_examples(rows, include_safe=True),
        "risky_examples": select_examples([row for row in rows if truthy(row.get("risk"))]),
        "missing_in_candidate_sample": [key_to_dict(key) for key in missing_in_candidate[:20]],
        "extra_in_candidate_sample": [key_to_dict(key) for key in extra_in_candidate[:20]],
        "sample": {
            "max_rows": max_rows,
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "files": {
            "summary": str(output_dir / "card_review_diff_summary.json"),
            "rows_csv": str(output_dir / "card_review_diff_rows.csv"),
            "report_md": str(output_dir / "card_review_diff_report.md"),
        },
    }
    write_diff_rows(output_dir / "card_review_diff_rows.csv", rows)
    (output_dir / "card_review_diff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "card_review_diff_report.md").write_text(
        format_card_review_diff_markdown(summary),
        encoding="utf-8",
    )
    return summary


def read_review_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def review_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    video = Path(str(row.get("video") or "")).name
    timestamp = normalize_timestamp(row.get("timestamp_sec"))
    frame_index = normalize_int_text(row.get("frame_index"))
    return (video, timestamp, frame_index)


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


def compare_slot(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    *,
    slot: int,
    risky_baseline_reasons: tuple[str, ...],
) -> dict[str, Any] | None:
    baseline_card = normalize_card(baseline_row.get(f"card{slot}"))
    candidate_card = normalize_card(candidate_row.get(f"card{slot}"))
    truth_card = normalize_card(candidate_row.get(f"final_card{slot}")) or normalize_card(baseline_row.get(f"final_card{slot}"))
    if not baseline_card and not candidate_card and not truth_card:
        return None

    baseline_ok = bool(truth_card and baseline_card == truth_card)
    candidate_ok = bool(truth_card and candidate_card == truth_card)
    baseline_reason = str(baseline_row.get("review_reason") or "")
    candidate_reason = str(candidate_row.get("review_reason") or "")
    status = classify_status(
        baseline_card=baseline_card,
        candidate_card=candidate_card,
        truth_card=truth_card,
        baseline_ok=baseline_ok,
        candidate_ok=candidate_ok,
        baseline_reason=baseline_reason,
        candidate_reason=candidate_reason,
        risky_baseline_reasons=risky_baseline_reasons,
    )
    risk, risk_reason = classify_risk(
        status=status,
        baseline_card=baseline_card,
        candidate_card=candidate_card,
        baseline_reason=baseline_reason,
        candidate_reason=candidate_reason,
        risky_baseline_reasons=risky_baseline_reasons,
    )
    return {
        "status": status,
        "risk": risk,
        "risk_reason": risk_reason,
        "video": Path(str(baseline_row.get("video") or candidate_row.get("video") or "")).name,
        "timestamp_sec": normalize_timestamp(baseline_row.get("timestamp_sec") or candidate_row.get("timestamp_sec")),
        "frame_index": normalize_int_text(baseline_row.get("frame_index") or candidate_row.get("frame_index")),
        "slot": slot,
        "truth_card": truth_card or "",
        "baseline_card": baseline_card or "",
        "candidate_card": candidate_card or "",
        "baseline_ok": baseline_ok,
        "candidate_ok": candidate_ok,
        "baseline_reason": baseline_reason,
        "candidate_reason": candidate_reason,
        "baseline_rank_confidence": baseline_row.get(f"card{slot}_rank_confidence") or "",
        "baseline_rank_margin": baseline_row.get(f"card{slot}_rank_margin") or "",
        "baseline_suit_confidence": baseline_row.get(f"card{slot}_suit_confidence") or "",
        "baseline_suit_margin": baseline_row.get(f"card{slot}_suit_margin") or "",
        "candidate_rank_confidence": candidate_row.get(f"card{slot}_rank_confidence") or "",
        "candidate_rank_margin": candidate_row.get(f"card{slot}_rank_margin") or "",
        "candidate_suit_confidence": candidate_row.get(f"card{slot}_suit_confidence") or "",
        "candidate_suit_margin": candidate_row.get(f"card{slot}_suit_margin") or "",
        "baseline_card_path": baseline_row.get(f"card{slot}_card_path") or "",
        "candidate_card_path": candidate_row.get(f"card{slot}_card_path") or "",
        "baseline_rank_path": baseline_row.get(f"card{slot}_rank_path") or "",
        "candidate_rank_path": candidate_row.get(f"card{slot}_rank_path") or "",
        "baseline_suit_path": baseline_row.get(f"card{slot}_suit_path") or "",
        "candidate_suit_path": candidate_row.get(f"card{slot}_suit_path") or "",
        "baseline_table_frame_path": baseline_row.get("table_frame_path") or "",
        "candidate_table_frame_path": candidate_row.get("table_frame_path") or "",
    }


def classify_status(
    *,
    baseline_card: str | None,
    candidate_card: str | None,
    truth_card: str | None,
    baseline_ok: bool,
    candidate_ok: bool,
    baseline_reason: str,
    candidate_reason: str,
    risky_baseline_reasons: tuple[str, ...],
) -> str:
    if baseline_card == candidate_card:
        if truth_card and baseline_ok and candidate_ok:
            return "same_correct"
        if truth_card and not baseline_ok and not candidate_ok:
            return "same_both_wrong"
        if not baseline_card and candidate_card:
            return "same_empty"
        return "same"
    if truth_card:
        if baseline_ok and not candidate_ok:
            return "regression"
        if not baseline_ok and candidate_ok:
            return "improved"
        if not baseline_ok and not candidate_ok:
            return "changed_both_wrong"
    if baseline_card and not candidate_card:
        return "candidate_missing"
    if candidate_card and not baseline_card:
        return "candidate_filled"
    if baseline_reason in risky_baseline_reasons and baseline_card:
        return "risky_unverified_change"
    if baseline_reason != candidate_reason:
        return "reason_changed"
    return "changed_unverified"


def classify_risk(
    *,
    status: str,
    baseline_card: str | None,
    candidate_card: str | None,
    baseline_reason: str,
    candidate_reason: str,
    risky_baseline_reasons: tuple[str, ...],
) -> tuple[bool, str]:
    if status in {"same_correct", "improved"}:
        return False, ""
    if status == "regression":
        return True, "manual_truth_regression"
    if status == "changed_both_wrong":
        return True, "manual_truth_both_wrong"
    if status == "same_both_wrong":
        return True, "manual_truth_same_wrong"
    if status == "candidate_missing" and baseline_card:
        return True, "candidate_lost_card"
    if status == "risky_unverified_change":
        return True, "changed_high_confidence_baseline_without_truth"
    if baseline_reason in risky_baseline_reasons and baseline_card != candidate_card:
        return True, "changed_high_confidence_baseline"
    if baseline_reason in risky_baseline_reasons and candidate_reason not in risky_baseline_reasons:
        return True, "downgraded_review_reason"
    return False, ""


def summarize_diff_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row.get("baseline_card") != row.get("candidate_card")]
    risk = [row for row in rows if truthy(row.get("risk"))]
    truth_rows = [row for row in rows if row.get("truth_card")]
    return {
        "slot_count": len(rows),
        "truth_count": len(truth_rows),
        "changed_count": len(changed),
        "risk_count": len(risk),
        "status": count_values(row.get("status") for row in rows),
        "risk_reason": count_values(row.get("risk_reason") for row in risk),
        "baseline_reason": count_values(row.get("baseline_reason") for row in rows),
        "candidate_reason": count_values(row.get("candidate_reason") for row in rows),
    }


def select_examples(rows: list[dict[str, Any]], *, include_safe: bool = False, limit: int = 20) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if not include_safe and not truthy(row.get("risk")):
            continue
        if include_safe and row.get("baseline_card") == row.get("candidate_card"):
            continue
        selected.append({key: row.get(key) for key in DIFF_COLUMNS[:14]})
        if len(selected) >= limit:
            break
    return selected


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "-")
        counts[key] = counts.get(key, 0) + 1
    return counts


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def key_to_dict(key: tuple[str, str, str]) -> dict[str, str]:
    return {"video": key[0], "timestamp_sec": key[1], "frame_index": key[2]}


def write_diff_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIFF_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_card_review_diff_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"Baseline: {payload.get('baseline_csv')}",
        f"Candidate: {payload.get('candidate_csv')}",
        f"Matched rows: {payload.get('matched_row_count')} / baseline {payload.get('baseline_row_count')} / candidate {payload.get('candidate_row_count')}",
        f"Slots: {counts.get('slot_count')} | changed={counts.get('changed_count')} | risk={counts.get('risk_count')} | truth={counts.get('truth_count')}",
        f"Status: {json.dumps(counts.get('status') or {}, ensure_ascii=False)}",
        f"Risk reasons: {json.dumps(counts.get('risk_reason') or {}, ensure_ascii=False)}",
        f"Rows CSV: {(payload.get('files') or {}).get('rows_csv')}",
        f"Report: {(payload.get('files') or {}).get('report_md')}",
    ]
    risky = payload.get("risky_examples") or []
    if risky:
        lines.extend(["", "Risky examples:"])
        for row in risky[:8]:
            lines.append(
                f"- {row.get('video')} t={row.get('timestamp_sec')} slot={row.get('slot')}: "
                f"{row.get('baseline_card') or '-'} -> {row.get('candidate_card') or '-'} "
                f"({row.get('risk_reason')})"
            )
    return "\n".join(lines)


def format_card_review_diff_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    lines = [
        "# Card Review Diff",
        "",
        f"- Baseline: `{summary.get('baseline_csv')}`",
        f"- Candidate: `{summary.get('candidate_csv')}`",
        f"- Matched rows: `{summary.get('matched_row_count')}`",
        f"- Missing in candidate: `{summary.get('missing_in_candidate_count')}`",
        f"- Extra in candidate: `{summary.get('extra_in_candidate_count')}`",
        f"- Slots: `{counts.get('slot_count')}`",
        f"- Changed: `{counts.get('changed_count')}`",
        f"- Risk: `{counts.get('risk_count')}`",
        f"- Truth slots: `{counts.get('truth_count')}`",
        "",
        "## Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted((counts.get("status") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Risk Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted((counts.get("risk_reason") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Risky Examples", ""])
    risky = summary.get("risky_examples") or []
    if not risky:
        lines.append("No risky examples detected.")
    else:
        lines.extend(["| Video | Time | Slot | Truth | Baseline | Candidate | Status | Risk |", "|---|---:|---:|---|---|---|---|---|"])
        for row in risky:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("video") or ""),
                        str(row.get("timestamp_sec") or ""),
                        str(row.get("slot") or ""),
                        str(row.get("truth_card") or ""),
                        str(row.get("baseline_card") or ""),
                        str(row.get("candidate_card") or ""),
                        str(row.get("status") or ""),
                        str(row.get("risk_reason") or ""),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Changed Examples", ""])
    changed = summary.get("changed_examples") or []
    if not changed:
        lines.append("No card changes detected.")
    else:
        lines.extend(["| Video | Time | Slot | Truth | Baseline | Candidate | Status | Risk |", "|---|---:|---:|---|---|---|---|---|"])
        for row in changed:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("video") or ""),
                        str(row.get("timestamp_sec") or ""),
                        str(row.get("slot") or ""),
                        str(row.get("truth_card") or ""),
                        str(row.get("baseline_card") or ""),
                        str(row.get("candidate_card") or ""),
                        str(row.get("status") or ""),
                        str(row.get("risk_reason") or ""),
                    ]
                )
                + " |"
            )
    return "\n".join(lines).rstrip() + "\n"
