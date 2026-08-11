from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


GROUP_COLUMNS = [
    "group_id",
    "count",
    "risk_reason",
    "transition",
    "baseline_card",
    "candidate_card",
    "rank_transition",
    "suit_transition",
    "videos",
    "times",
    "slots",
    "avg_baseline_rank_conf",
    "avg_candidate_rank_conf",
    "avg_baseline_suit_conf",
    "avg_candidate_suit_conf",
    "example_card_path",
    "example_rank_path",
    "example_suit_path",
    "example_table_frame_path",
    "recommended_action",
]


def summarize_card_diff_risks(
    *,
    diff_csv: Path,
    output_dir: Path,
    risk_only: bool = True,
    include_same: bool = True,
    max_examples: int = 8,
) -> dict[str, Any]:
    diff_csv = Path(diff_csv)
    if not diff_csv.is_file():
        raise ValueError(f"diff CSV not found: {diff_csv}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(diff_csv)
    selected = select_rows(rows, risk_only=risk_only, include_same=include_same)
    groups = build_groups(selected)
    groups = sorted(groups, key=group_sort_key)
    for index, group in enumerate(groups, start=1):
        group["group_id"] = f"G{index:04d}"

    payload = {
        "ok": True,
        "diff_csv": str(diff_csv),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "selected_count": len(selected),
        "risk_count": sum(1 for row in rows if truthy(row.get("risk"))),
        "changed_count": sum(1 for row in rows if normalized_card(row.get("baseline_card")) != normalized_card(row.get("candidate_card"))),
        "risk_only": bool(risk_only),
        "include_same": bool(include_same),
        "reason_counts": count_values(row.get("risk_reason") for row in selected),
        "transition_counts": count_values(transition_text(row) for row in selected),
        "video_counts": count_values(Path(str(row.get("video") or "")).name for row in selected),
        "groups": groups,
        "examples": selected[: max(0, int(max_examples))],
        "files": {
            "summary_json": str(output_dir / "card_diff_risk_summary.json"),
            "groups_csv": str(output_dir / "card_diff_risk_groups.csv"),
            "report_md": str(output_dir / "card_diff_risk_summary.md"),
        },
    }
    write_group_csv(output_dir / "card_diff_risk_groups.csv", groups)
    (output_dir / "card_diff_risk_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "card_diff_risk_summary.md").write_text(format_diff_risk_markdown(payload), encoding="utf-8")
    return payload


def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_rows(rows: list[dict[str, Any]], *, risk_only: bool, include_same: bool) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        baseline = normalized_card(row.get("baseline_card"))
        candidate = normalized_card(row.get("candidate_card"))
        if risk_only and not truthy(row.get("risk")):
            continue
        if not include_same and baseline == candidate:
            continue
        selected.append(row)
    return selected


def build_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("risk_reason") or "-"),
            normalized_card(row.get("baseline_card")) or "-",
            normalized_card(row.get("candidate_card")) or "-",
        )
        grouped.setdefault(key, []).append(row)
    return [build_group(risk_reason, baseline, candidate, items) for (risk_reason, baseline, candidate), items in grouped.items()]


def build_group(risk_reason: str, baseline: str, candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    example = rows[0] if rows else {}
    videos = sorted({Path(str(row.get("video") or "")).name for row in rows if row.get("video")})
    times = sorted({str(row.get("timestamp_sec") or "") for row in rows if row.get("timestamp_sec")}, key=timestamp_sort_key)
    slots = sorted({str(row.get("slot") or "") for row in rows if row.get("slot") != ""})
    return {
        "group_id": "",
        "count": len(rows),
        "risk_reason": risk_reason,
        "transition": f"{baseline}->{candidate}",
        "baseline_card": "" if baseline == "-" else baseline,
        "candidate_card": "" if candidate == "-" else candidate,
        "rank_transition": f"{card_rank(baseline)}->{card_rank(candidate)}",
        "suit_transition": f"{card_suit(baseline)}->{card_suit(candidate)}",
        "videos": "; ".join(videos),
        "times": "; ".join(times[:10]),
        "slots": "; ".join(slots),
        "avg_baseline_rank_conf": format_float(avg_float(row.get("baseline_rank_confidence") for row in rows)),
        "avg_candidate_rank_conf": format_float(avg_float(row.get("candidate_rank_confidence") for row in rows)),
        "avg_baseline_suit_conf": format_float(avg_float(row.get("baseline_suit_confidence") for row in rows)),
        "avg_candidate_suit_conf": format_float(avg_float(row.get("candidate_suit_confidence") for row in rows)),
        "example_card_path": str(example.get("baseline_card_path") or example.get("candidate_card_path") or ""),
        "example_rank_path": str(example.get("baseline_rank_path") or example.get("candidate_rank_path") or ""),
        "example_suit_path": str(example.get("baseline_suit_path") or example.get("candidate_suit_path") or ""),
        "example_table_frame_path": str(example.get("baseline_table_frame_path") or example.get("candidate_table_frame_path") or ""),
        "recommended_action": recommend_action(risk_reason=risk_reason, baseline=baseline, candidate=candidate),
    }


def group_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    action_rank = {
        "manual_label_card_change": 0,
        "manual_label_missing_or_roi": 1,
        "review_confidence_downgrade": 2,
    }.get(str(group.get("recommended_action") or ""), 9)
    return (action_rank, -int(group.get("count") or 0), str(group.get("transition") or ""))


def recommend_action(*, risk_reason: str, baseline: str, candidate: str) -> str:
    if risk_reason == "candidate_lost_card" or candidate == "-":
        return "manual_label_missing_or_roi"
    if baseline != candidate:
        return "manual_label_card_change"
    return "review_confidence_downgrade"


def write_group_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GROUP_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for group in groups:
            writer.writerow(group)


def format_diff_risk_summary(payload: dict[str, Any]) -> str:
    files = payload.get("files") or {}
    groups = payload.get("groups") or []
    lines = [
        f"Rows: {payload.get('row_count')}",
        f"Selected: {payload.get('selected_count')}",
        f"Risk: {payload.get('risk_count')}",
        f"Changed: {payload.get('changed_count')}",
        f"Groups: {len(groups)}",
        f"Reasons: {json.dumps(payload.get('reason_counts') or {}, ensure_ascii=False)}",
        f"Groups CSV: {files.get('groups_csv')}",
        f"Report: {files.get('report_md')}",
    ]
    if groups:
        lines.extend(["", "Top groups:"])
        for group in groups[:8]:
            lines.append(
                f"- {group.get('group_id')} {group.get('transition')} x{group.get('count')} "
                f"{group.get('risk_reason')} [{group.get('recommended_action')}]"
            )
    return "\n".join(lines)


def format_diff_risk_markdown(payload: dict[str, Any]) -> str:
    groups = payload.get("groups") or []
    lines = [
        "# Card Diff Risk Summary",
        "",
        f"- Diff CSV: `{payload.get('diff_csv')}`",
        f"- Rows: `{payload.get('row_count')}`",
        f"- Selected: `{payload.get('selected_count')}`",
        f"- Risk rows: `{payload.get('risk_count')}`",
        f"- Changed rows: `{payload.get('changed_count')}`",
        "",
        "## Reason Counts",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted((payload.get("reason_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Groups",
            "",
            "| Group | Count | Transition | Rank | Suit | Reason | Action | Videos | Times |",
            "|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for group in groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group.get("group_id") or ""),
                    str(group.get("count") or ""),
                    str(group.get("transition") or ""),
                    str(group.get("rank_transition") or ""),
                    str(group.get("suit_transition") or ""),
                    str(group.get("risk_reason") or ""),
                    str(group.get("recommended_action") or ""),
                    str(group.get("videos") or ""),
                    str(group.get("times") or ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    if not groups:
        lines.append("No selected risk groups.")
    else:
        manual_groups = [group for group in groups if str(group.get("recommended_action")) == "manual_label_card_change"]
        missing_groups = [group for group in groups if str(group.get("recommended_action")) == "manual_label_missing_or_roi"]
        downgrade_groups = [group for group in groups if str(group.get("recommended_action")) == "review_confidence_downgrade"]
        if manual_groups:
            lines.append(f"- Label the true card for `{len(manual_groups)}` card-change groups before using this candidate.")
        if missing_groups:
            lines.append(f"- Inspect ROI/card presence for `{len(missing_groups)}` missing-card groups.")
        if downgrade_groups:
            lines.append(f"- Check `{len(downgrade_groups)}` confidence-only downgrade groups; these may not require new labels if the card is unchanged.")
    return "\n".join(lines).rstrip() + "\n"


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


def normalized_card(value: Any) -> str:
    text = str(value or "").strip().replace("10", "T")
    return text if len(text) == 2 and "?" not in text else ""


def transition_text(row: dict[str, Any]) -> str:
    return f"{normalized_card(row.get('baseline_card')) or '-'}->{normalized_card(row.get('candidate_card')) or '-'}"


def card_rank(card: str) -> str:
    return card[0] if len(card) == 2 and card != "-" else "-"


def card_suit(card: str) -> str:
    return card[1] if len(card) == 2 and card != "-" else "-"


def avg_float(values: Any) -> float | None:
    parsed = [value for value in (safe_float(item) for item in values) if value is not None]
    return sum(parsed) / len(parsed) if parsed else None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or value == "-":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def timestamp_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))
