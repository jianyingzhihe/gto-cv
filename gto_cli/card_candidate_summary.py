from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CANDIDATE_COLUMNS = [
    "rank",
    "candidate_name",
    "decision",
    "promote",
    "evaluator",
    "card_acc",
    "rank_acc",
    "suit_acc",
    "diff_slots",
    "diff_changed",
    "diff_risk",
    "missing_rows",
    "real_problem",
    "board_bad",
    "median_ms",
    "p90_ms",
    "has_validation",
    "failed_check_count",
    "failed_checks",
    "summary_path",
    "report_path",
]


def summarize_card_candidates(
    *,
    gate_paths: list[Path] | None = None,
    search_dir: Path = Path("video_frames"),
    output_dir: Path = Path("video_frames") / "card_candidate_summary",
    keep_duplicates: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = collect_gate_summaries(gate_paths=gate_paths, search_dir=search_dir)
    rows = [candidate_row(summary) for summary in summaries]
    if not keep_duplicates:
        rows = dedupe_candidate_rows(rows)
    rows = sorted(rows, key=candidate_sort_key)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    csv_path = output_dir / "card_candidate_summary.csv"
    md_path = output_dir / "card_candidate_summary.md"
    json_path = output_dir / "card_candidate_summary.json"
    write_candidate_csv(csv_path, rows)
    payload = {
        "ok": True,
        "search_dir": str(search_dir),
        "output_dir": str(output_dir),
        "candidate_count": len(rows),
        "gate_summary_count": len(summaries),
        "keep_duplicates": bool(keep_duplicates),
        "promote_count": sum(1 for row in rows if truthy(row.get("promote"))),
        "best_candidate": rows[0] if rows else None,
        "rows": rows,
        "files": {
            "summary_json": str(json_path),
            "summary_csv": str(csv_path),
            "summary_md": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(format_candidate_summary_markdown(payload), encoding="utf-8")
    return payload


def dedupe_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("candidate_name") or ""), str(row.get("evaluator") or ""))
        current = by_key.get(key)
        if current is None or candidate_sort_key(row) < candidate_sort_key(current):
            by_key[key] = row
    return list(by_key.values())


def collect_gate_summaries(*, gate_paths: list[Path] | None, search_dir: Path) -> list[dict[str, Any]]:
    if gate_paths:
        paths = [Path(path) for path in gate_paths]
    else:
        paths = sorted(Path(search_dir).rglob("card_model_gate_summary.json"))
    summaries = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        with path.open("r", encoding="utf-8-sig") as stream:
            summary = json.load(stream)
        summary["_summary_path"] = str(path)
        summaries.append(summary)
    return summaries


def candidate_row(summary: dict[str, Any]) -> dict[str, Any]:
    evaluator_name = str(summary.get("candidate_evaluator") or "")
    evaluator = ((summary.get("benchmark") or {}).get("evaluators") or {}).get(evaluator_name) or {}
    diff = summary.get("diff") or {}
    diff_counts = diff.get("counts") or {}
    validation = summary.get("candidate_validation") or {}
    timing = validation.get("timing_ms") or {}
    checks = summary.get("checks") or []
    failed_checks = [str(check.get("name") or "") for check in checks if not check.get("pass")]
    files = summary.get("files") or {}
    return {
        "rank": "",
        "candidate_name": summary.get("candidate_name") or Path(str(summary.get("output_dir") or "")).name,
        "decision": summary.get("decision") or "",
        "promote": bool(summary.get("promote")),
        "evaluator": evaluator_name,
        "card_acc": optional_float(evaluator.get("card_acc")),
        "rank_acc": optional_float(evaluator.get("rank_acc")),
        "suit_acc": optional_float(evaluator.get("suit_acc")),
        "diff_slots": optional_int(diff_counts.get("slot_count")),
        "diff_changed": optional_int(diff_counts.get("changed_count")),
        "diff_risk": optional_int(diff_counts.get("risk_count")),
        "missing_rows": optional_int(diff.get("missing_in_candidate_count")),
        "real_problem": optional_int(validation.get("real_problem_count")),
        "board_bad": optional_int(validation.get("board_bad_count")),
        "median_ms": optional_float(timing.get("median")),
        "p90_ms": optional_float(timing.get("p90")),
        "has_validation": bool(validation),
        "failed_check_count": len(failed_checks),
        "failed_checks": ",".join(failed_checks),
        "summary_path": summary.get("_summary_path") or files.get("summary") or "",
        "report_path": files.get("report_md") or "",
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if truthy(row.get("promote")) else 1,
        0 if truthy(row.get("has_validation")) else 1,
        optional_int(row.get("failed_check_count"), default=10**9),
        -optional_float(row.get("card_acc"), default=-1.0),
        -optional_float(row.get("rank_acc"), default=-1.0),
        -optional_float(row.get("suit_acc"), default=-1.0),
        optional_int(row.get("diff_risk"), default=10**9),
        optional_int(row.get("missing_rows"), default=10**9),
        optional_float(row.get("median_ms"), default=10**9),
        str(row.get("candidate_name") or ""),
    )


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_candidate_summary(payload: dict[str, Any]) -> str:
    best = payload.get("best_candidate") or {}
    lines = [
        f"Candidates: {payload.get('candidate_count')}",
        f"Promotable: {payload.get('promote_count')}",
    ]
    if best:
        lines.append(
            "Best: "
            f"{best.get('candidate_name')} decision={best.get('decision')} "
            f"card={format_metric(best.get('card_acc'))} risk={best.get('diff_risk')} "
            f"median={format_metric(best.get('median_ms'))}ms"
        )
    files = payload.get("files") or {}
    lines.extend(
        [
            f"CSV: {files.get('summary_csv')}",
            f"Markdown: {files.get('summary_md')}",
            f"JSON: {files.get('summary_json')}",
        ]
    )
    return "\n".join(lines)


def format_candidate_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Card Candidate Summary",
        "",
        f"- Candidates: `{payload.get('candidate_count')}`",
        f"- Promotable: `{payload.get('promote_count')}`",
        "",
        "| Rank | Candidate | Decision | Card | Rank Acc | Suit Acc | Risk | Changed | Missing | Real Problem | Median ms | Validation | Failed | Failed Checks |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank") or ""),
                    str(row.get("candidate_name") or ""),
                    str(row.get("decision") or ""),
                    format_metric(row.get("card_acc")),
                    format_metric(row.get("rank_acc")),
                    format_metric(row.get("suit_acc")),
                    format_metric(row.get("diff_risk")),
                    format_metric(row.get("diff_changed")),
                    format_metric(row.get("missing_rows")),
                    format_metric(row.get("real_problem")),
                    format_metric(row.get("median_ms")),
                    "yes" if truthy(row.get("has_validation")) else "no",
                    format_metric(row.get("failed_check_count")),
                    str(row.get("failed_checks") or "-"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def optional_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "promote"}


def format_metric(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
