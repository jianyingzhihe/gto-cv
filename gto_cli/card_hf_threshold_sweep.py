from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .card_deep_model import RANK_LABELS, SUIT_LABELS


DEFAULT_RANK_SCORE_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
DEFAULT_RANK_MARGIN_THRESHOLDS = (0.02, 0.04, 0.08, 0.12)
DEFAULT_SUIT_SCORE_THRESHOLDS = (0.65,)
DEFAULT_SUIT_MARGIN_THRESHOLDS = (0.06,)


def sweep_hf_prediction_thresholds(
    *,
    predictions_csv: Path,
    output_dir: Path,
    rank_score_thresholds: list[float] | None = None,
    rank_margin_thresholds: list[float] | None = None,
    suit_score_thresholds: list[float] | None = None,
    suit_margin_thresholds: list[float] | None = None,
    require_current_agreement: bool = True,
) -> dict[str, Any]:
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.is_file():
        raise ValueError(f"predictions CSV not found: {predictions_csv}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prediction_rows(predictions_csv)
    rank_scores = normalize_thresholds(rank_score_thresholds, DEFAULT_RANK_SCORE_THRESHOLDS)
    rank_margins = normalize_thresholds(rank_margin_thresholds, DEFAULT_RANK_MARGIN_THRESHOLDS)
    suit_scores = normalize_thresholds(suit_score_thresholds, DEFAULT_SUIT_SCORE_THRESHOLDS)
    suit_margins = normalize_thresholds(suit_margin_thresholds, DEFAULT_SUIT_MARGIN_THRESHOLDS)

    sweep_rows: list[dict[str, Any]] = []
    for rank_score in rank_scores:
        for rank_margin in rank_margins:
            for suit_score in suit_scores:
                for suit_margin in suit_margins:
                    sweep_rows.append(
                        evaluate_threshold_row(
                            rows,
                            rank_score=rank_score,
                            rank_margin=rank_margin,
                            suit_score=suit_score,
                            suit_margin=suit_margin,
                            require_current_agreement=require_current_agreement,
                        )
                    )

    recommended = choose_recommended_threshold(sweep_rows)
    files = {
        "summary": str(output_dir / "threshold_sweep_summary.json"),
        "csv": str(output_dir / "threshold_sweep.csv"),
        "report_md": str(output_dir / "threshold_sweep.md"),
    }
    payload = {
        "ok": True,
        "predictions_csv": str(predictions_csv),
        "output_dir": str(output_dir),
        "input_rows": len(rows),
        "rank_score_thresholds": rank_scores,
        "rank_margin_thresholds": rank_margins,
        "suit_score_thresholds": suit_scores,
        "suit_margin_thresholds": suit_margins,
        "require_current_agreement": bool(require_current_agreement),
        "recommended": recommended,
        "rows": sweep_rows,
        "files": files,
    }
    write_sweep_csv(Path(files["csv"]), sweep_rows)
    Path(files["summary"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(files["report_md"]).write_text(format_threshold_sweep_markdown(payload), encoding="utf-8")
    return payload


def read_prediction_rows(predictions_csv: Path) -> list[dict[str, Any]]:
    with Path(predictions_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def normalize_thresholds(values: list[float] | None, defaults: tuple[float, ...]) -> list[float]:
    source = defaults if values is None or not values else tuple(values)
    return sorted({round(float(value), 6) for value in source})


def evaluate_threshold_row(
    rows: list[dict[str, Any]],
    *,
    rank_score: float,
    rank_margin: float,
    suit_score: float,
    suit_margin: float,
    require_current_agreement: bool,
) -> dict[str, Any]:
    accepted_rank_labels: set[str] = set()
    accepted_suit_labels: set[str] = set()
    rank_accepted = 0
    suit_accepted = 0
    rank_processed = 0
    suit_processed = 0
    agreement_rejected = 0
    threshold_rejected = 0
    for row in rows:
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in {"rank", "suit"}:
            continue
        current_label = str(row.get("current_label") or "").strip()
        teacher_label = str(row.get("teacher_label") or "").strip()
        score = safe_float(row.get("teacher_score"))
        margin = safe_float(row.get("teacher_margin"))
        agree = current_label == teacher_label
        if kind == "rank":
            rank_processed += 1
            passes_threshold = score >= rank_score and margin >= rank_margin
        else:
            suit_processed += 1
            passes_threshold = score >= suit_score and margin >= suit_margin
        accepted = passes_threshold and (agree or not require_current_agreement)
        if accepted:
            if kind == "rank":
                rank_accepted += 1
                accepted_rank_labels.add(current_label)
            else:
                suit_accepted += 1
                accepted_suit_labels.add(current_label)
        elif require_current_agreement and not agree:
            agreement_rejected += 1
        else:
            threshold_rejected += 1

    missing_rank = [label for label in RANK_LABELS if label not in accepted_rank_labels]
    missing_suit = [label for label in SUIT_LABELS if label not in accepted_suit_labels]
    accepted = rank_accepted + suit_accepted
    processed = rank_processed + suit_processed
    return {
        "rank_score_threshold": rank_score,
        "rank_margin_threshold": rank_margin,
        "suit_score_threshold": suit_score,
        "suit_margin_threshold": suit_margin,
        "processed": processed,
        "accepted": accepted,
        "review": processed - accepted,
        "rank_processed": rank_processed,
        "rank_accepted": rank_accepted,
        "suit_processed": suit_processed,
        "suit_accepted": suit_accepted,
        "rank_accept_rate": round(rank_accepted / rank_processed, 6) if rank_processed else 0.0,
        "suit_accept_rate": round(suit_accepted / suit_processed, 6) if suit_processed else 0.0,
        "accepted_rank_labels": "".join(label for label in RANK_LABELS if label in accepted_rank_labels),
        "accepted_suit_labels": "".join(label for label in SUIT_LABELS if label in accepted_suit_labels),
        "missing_rank_labels": ",".join(missing_rank),
        "missing_suit_labels": ",".join(missing_suit),
        "rank_labels_complete": not missing_rank,
        "suit_labels_complete": not missing_suit,
        "labels_complete": not missing_rank and not missing_suit,
        "agreement_rejected": agreement_rejected,
        "threshold_rejected": threshold_rejected,
    }


def choose_recommended_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [row for row in rows if row.get("labels_complete")]
    if not complete:
        return None
    return sorted(
        complete,
        key=lambda row: (
            float(row.get("rank_score_threshold") or 0),
            float(row.get("suit_score_threshold") or 0),
            float(row.get("rank_margin_threshold") or 0),
            float(row.get("suit_margin_threshold") or 0),
            int(row.get("accepted") or 0),
        ),
        reverse=True,
    )[0]


def write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank_score_threshold",
        "rank_margin_threshold",
        "suit_score_threshold",
        "suit_margin_threshold",
        "processed",
        "accepted",
        "review",
        "rank_processed",
        "rank_accepted",
        "suit_processed",
        "suit_accepted",
        "rank_accept_rate",
        "suit_accept_rate",
        "accepted_rank_labels",
        "accepted_suit_labels",
        "missing_rank_labels",
        "missing_suit_labels",
        "rank_labels_complete",
        "suit_labels_complete",
        "labels_complete",
        "agreement_rejected",
        "threshold_rejected",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def format_threshold_sweep_markdown(payload: dict[str, Any]) -> str:
    recommended = payload.get("recommended") or {}
    rows = sorted(
        payload.get("rows") or [],
        key=lambda row: (bool(row.get("labels_complete")), int(row.get("accepted") or 0)),
        reverse=True,
    )
    lines = [
        "# HF Prediction Threshold Sweep",
        "",
        f"- Predictions: `{payload.get('predictions_csv')}`",
        f"- Rows: `{payload.get('input_rows')}`",
        f"- Require current agreement: `{payload.get('require_current_agreement')}`",
        "",
    ]
    if recommended:
        lines.extend(
            [
                "## Recommended",
                "",
                f"- Rank score: `{recommended.get('rank_score_threshold')}`",
                f"- Rank margin: `{recommended.get('rank_margin_threshold')}`",
                f"- Suit score: `{recommended.get('suit_score_threshold')}`",
                f"- Suit margin: `{recommended.get('suit_margin_threshold')}`",
                f"- Accepted: `{recommended.get('accepted')}`",
                f"- Rank accepted: `{recommended.get('rank_accepted')}` / `{recommended.get('rank_processed')}`",
                f"- Suit accepted: `{recommended.get('suit_accepted')}` / `{recommended.get('suit_processed')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Rows",
            "",
            "| Rank Score | Rank Margin | Suit Score | Suit Margin | Accepted | Rank | Suit | Missing Rank |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows[:20]:
        lines.append(
            "| "
            f"{row.get('rank_score_threshold')} | "
            f"{row.get('rank_margin_threshold')} | "
            f"{row.get('suit_score_threshold')} | "
            f"{row.get('suit_margin_threshold')} | "
            f"{row.get('accepted')} | "
            f"{row.get('rank_accepted')}/{row.get('rank_processed')} | "
            f"{row.get('suit_accepted')}/{row.get('suit_processed')} | "
            f"{row.get('missing_rank_labels') or '-'} |"
        )
    return "\n".join(lines) + "\n"


def format_threshold_sweep_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"sweep-card-hf-thresholds failed: {payload.get('error')}"
    recommended = payload.get("recommended") or {}
    if recommended:
        rec = (
            f"recommended rank_score={recommended.get('rank_score_threshold')} "
            f"rank_margin={recommended.get('rank_margin_threshold')} "
            f"suit_score={recommended.get('suit_score_threshold')} "
            f"suit_margin={recommended.get('suit_margin_threshold')} "
            f"accepted={recommended.get('accepted')}"
        )
    else:
        rec = "recommended: none with complete rank/suit coverage"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Predictions: {payload.get('predictions_csv')}",
            f"Rows: {payload.get('input_rows')}",
            rec,
            f"CSV: {files.get('csv')}",
            f"Report: {files.get('report_md')}",
        ]
    )


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
