from __future__ import annotations

import csv
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .card_classifier import DEFAULT_MODEL_PATH, classify_rank_glyph, classify_suit_glyph
from .card_deep_model import RANK_LABELS, SUIT_LABELS, classify_deep_glyph
from .card_hf_probe import classify_hf_probe_glyph_path
from .video_vision import load_cv, recognize_card_crop


EVALUATOR_PREFIXES = ("current_csv", "runtime", "knn", "deep", "hf_probe")


def benchmark_card_review(
    *,
    review_csvs: list[Path],
    output_dir: Path,
    deep_model_dir: Path | None = None,
    deep_rank_model_dir: Path | None = None,
    deep_suit_model_dir: Path | None = None,
    knn_model_path: Path = DEFAULT_MODEL_PATH,
    hf_probe_dir: Path | None = None,
    hf_probe_device: str = "auto",
    hf_probe_local_files_only: bool = False,
    include_ok_pseudo: bool = False,
    allowed_pseudo_reasons: tuple[str, ...] = ("ok",),
    run_runtime: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_review_samples(
        [Path(path) for path in review_csvs],
        include_ok_pseudo=include_ok_pseudo,
        allowed_pseudo_reasons=allowed_pseudo_reasons,
        max_samples=max_samples,
    )
    if not samples:
        raise ValueError("no benchmark samples found; fill final_card0/final_card1 or use --include-ok-pseudo")

    rows = []
    for sample in samples:
        rows.append(
            evaluate_sample(
                sample,
                deep_model_dir=deep_model_dir,
                deep_rank_model_dir=deep_rank_model_dir,
                deep_suit_model_dir=deep_suit_model_dir,
                knn_model_path=knn_model_path,
                hf_probe_dir=hf_probe_dir,
                hf_probe_device=hf_probe_device,
                hf_probe_local_files_only=hf_probe_local_files_only,
                run_runtime=run_runtime,
            )
        )

    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "review_csvs": [str(path) for path in review_csvs],
        "sample_count": len(rows),
        "truth_sources": count_values(row.get("truth_source") for row in rows),
        "allowed_pseudo_reasons": list(allowed_pseudo_reasons),
        "include_ok_pseudo": bool(include_ok_pseudo),
        "run_runtime": bool(run_runtime),
        "deep_model_dir": str(deep_model_dir) if deep_model_dir else None,
        "deep_rank_model_dir": str(deep_rank_model_dir or deep_model_dir) if (deep_rank_model_dir or deep_model_dir) else None,
        "deep_suit_model_dir": str(deep_suit_model_dir or deep_model_dir) if (deep_suit_model_dir or deep_model_dir) else None,
        "knn_model_path": str(knn_model_path),
        "hf_probe_dir": str(hf_probe_dir) if hf_probe_dir else None,
        "hf_probe_device": hf_probe_device,
        "hf_probe_local_files_only": bool(hf_probe_local_files_only),
        "evaluators": evaluator_summaries(rows),
        "confusions": evaluator_confusions(rows),
        "sample_rows": rows,
        "sample": {
            "max_samples": max_samples,
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
        },
        "files": {
            "summary": str(output_dir / "card_benchmark_summary.json"),
            "rows_csv": str(output_dir / "card_benchmark_rows.csv"),
            "report_md": str(output_dir / "card_benchmark_report.md"),
        },
    }
    write_rows_csv(output_dir / "card_benchmark_rows.csv", rows)
    (output_dir / "card_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "card_benchmark_report.md").write_text(format_card_benchmark_markdown(summary), encoding="utf-8")
    return summary


def collect_review_samples(
    review_csvs: list[Path],
    *,
    include_ok_pseudo: bool,
    allowed_pseudo_reasons: tuple[str, ...],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    samples_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str, str, int]] = []
    allowed = set(allowed_pseudo_reasons)
    for review_csv in review_csvs:
        with Path(review_csv).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_index, row in enumerate(reader):
                for slot in (0, 1):
                    final_card = normalize_card(row.get(f"final_card{slot}"))
                    current_card = normalize_card(row.get(f"card{slot}"))
                    truth_card = None
                    truth_source = None
                    if final_card:
                        truth_card = final_card
                        truth_source = "manual"
                    elif include_ok_pseudo and str(row.get("review_reason") or "") in allowed and current_card:
                        truth_card = current_card
                        truth_source = f"pseudo:{row.get('review_reason')}"
                    if not truth_card:
                        continue
                    sample_slot = parse_original_slot(row, default=slot) if final_card else slot
                    card_path = resolve_review_path(review_csv, row.get(f"card{slot}_card_path") or "")
                    rank_path = resolve_review_path(review_csv, row.get(f"card{slot}_rank_path") or "")
                    suit_path = resolve_review_path(review_csv, row.get(f"card{slot}_suit_path") or "")
                    sample = {
                        "review_csv": str(review_csv),
                        "row_index": row_index,
                        "slot": sample_slot,
                        "video": row.get("video") or "",
                        "timestamp_sec": row.get("timestamp_sec") or "",
                        "frame_index": row.get("frame_index") or "",
                        "review_reason": row.get("review_reason") or "",
                        "truth_card": truth_card,
                        "truth_source": truth_source,
                        "current_card": current_card or "",
                        "card_path": str(card_path) if card_path else "",
                        "rank_path": str(rank_path) if rank_path else "",
                        "suit_path": str(suit_path) if suit_path else "",
                    }
                    key = sample_key(sample)
                    previous = samples_by_key.get(key)
                    if previous is None:
                        ordered_keys.append(key)
                        samples_by_key[key] = sample
                    elif sample_priority(sample) > sample_priority(previous):
                        samples_by_key[key] = sample
    samples = [samples_by_key[key] for key in ordered_keys]
    if max_samples is not None:
        return samples[: max(0, int(max_samples))]
    return samples


def sample_key(sample: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        Path(str(sample.get("video") or "")).name,
        normalize_timestamp(sample.get("timestamp_sec")),
        normalize_int_text(sample.get("frame_index")),
        int(sample.get("slot") or 0),
    )


def sample_priority(sample: dict[str, Any]) -> tuple[int, int]:
    manual = 1 if sample.get("truth_source") == "manual" else 0
    has_assets = 1 if sample.get("card_path") and sample.get("rank_path") and sample.get("suit_path") else 0
    return (manual, has_assets)


def parse_original_slot(row: dict[str, Any], *, default: int) -> int:
    text = f"{row.get('notes') or ''};{row.get('reason') or ''}"
    match = re.search(r"(?:original_slot|slot)\s*=\s*([01])", text)
    if match:
        return int(match.group(1))
    return int(default)


def evaluate_sample(
    sample: dict[str, Any],
    *,
    deep_model_dir: Path | None,
    deep_rank_model_dir: Path | None,
    deep_suit_model_dir: Path | None,
    knn_model_path: Path,
    hf_probe_dir: Path | None,
    hf_probe_device: str,
    hf_probe_local_files_only: bool,
    run_runtime: bool,
) -> dict[str, Any]:
    expected_card = str(sample["truth_card"])
    expected_rank = expected_card[0]
    expected_suit = expected_card[1]
    row = dict(sample)
    row.update({"expected_rank": expected_rank, "expected_suit": expected_suit})
    add_card_result(row, "current_csv", normalize_card(sample.get("current_card")))

    if run_runtime and sample.get("card_path"):
        runtime_card = recognize_runtime_card(
            Path(str(sample["card_path"])),
            int(sample.get("slot") or 0),
            knn_model_path=knn_model_path,
        )
        add_card_result(row, "runtime", runtime_card)

    knn_rank_result = classify_knn_glyph_path(sample.get("rank_path"), "rank", knn_model_path)
    knn_suit_result = classify_knn_glyph_path(sample.get("suit_path"), "suit", knn_model_path)
    knn_rank = normalize_rank(knn_rank_result.get("label") if knn_rank_result else None)
    knn_suit = normalize_suit(knn_suit_result.get("label") if knn_suit_result else None)
    add_card_result(row, "knn", f"{knn_rank}{knn_suit}" if knn_rank and knn_suit else None)
    if knn_rank_result:
        row["knn_rank_score"] = round(float(knn_rank_result.get("score") or 0.0), 6)
        row["knn_rank_margin"] = round(float(knn_rank_result.get("margin") or 0.0), 6)
    if knn_suit_result:
        row["knn_suit_score"] = round(float(knn_suit_result.get("score") or 0.0), 6)
        row["knn_suit_margin"] = round(float(knn_suit_result.get("margin") or 0.0), 6)

    rank_model_dir = deep_rank_model_dir or deep_model_dir
    suit_model_dir = deep_suit_model_dir or deep_model_dir
    deep_rank_result = classify_glyph_path(sample.get("rank_path"), "rank", rank_model_dir)
    deep_suit_result = classify_glyph_path(sample.get("suit_path"), "suit", suit_model_dir)
    deep_rank = normalize_rank(deep_rank_result.get("label") if deep_rank_result else None)
    deep_suit = normalize_suit(deep_suit_result.get("label") if deep_suit_result else None)
    deep_card = f"{deep_rank}{deep_suit}" if deep_rank and deep_suit else None
    add_card_result(row, "deep", deep_card)
    if deep_rank_result:
        row["deep_rank_score"] = round(float(deep_rank_result.get("score") or 0.0), 6)
        row["deep_rank_margin"] = round(float(deep_rank_result.get("margin") or 0.0), 6)
        row["deep_rank_arch"] = deep_rank_result.get("arch")
    if deep_suit_result:
        row["deep_suit_score"] = round(float(deep_suit_result.get("score") or 0.0), 6)
        row["deep_suit_margin"] = round(float(deep_suit_result.get("margin") or 0.0), 6)
        row["deep_suit_arch"] = deep_suit_result.get("arch")

    hf_rank_result = classify_hf_probe_path(
        sample.get("rank_path"),
        "rank",
        hf_probe_dir,
        device=hf_probe_device,
        local_files_only=hf_probe_local_files_only,
    )
    hf_suit_result = classify_hf_probe_path(
        sample.get("suit_path"),
        "suit",
        hf_probe_dir,
        device=hf_probe_device,
        local_files_only=hf_probe_local_files_only,
    )
    hf_rank = normalize_rank(hf_rank_result.get("label") if hf_rank_result else None)
    hf_suit = normalize_suit(hf_suit_result.get("label") if hf_suit_result else None)
    add_card_result(row, "hf_probe", f"{hf_rank}{hf_suit}" if hf_rank and hf_suit else None)
    if hf_rank_result:
        row["hf_probe_rank_score"] = round(float(hf_rank_result.get("score") or 0.0), 6)
        row["hf_probe_rank_margin"] = round(float(hf_rank_result.get("margin") or 0.0), 6)
        row["hf_probe_rank_model"] = hf_rank_result.get("model")
    if hf_suit_result:
        row["hf_probe_suit_score"] = round(float(hf_suit_result.get("score") or 0.0), 6)
        row["hf_probe_suit_margin"] = round(float(hf_suit_result.get("margin") or 0.0), 6)
        row["hf_probe_suit_model"] = hf_suit_result.get("model")
    return row


def recognize_runtime_card(card_path: Path, slot: int, *, knn_model_path: Path | None = None) -> str | None:
    cv2, _np = load_cv()
    path = Path(card_path)
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    with temporary_card_knn_model(knn_model_path):
        detail = recognize_card_crop(image, source="hero", index=slot)
    if not detail:
        return None
    return normalize_card(detail.get("card"))


@contextmanager
def temporary_card_knn_model(model_path: Path | None) -> Any:
    if model_path is None:
        yield
        return
    old_value = os.environ.get("GTO_CARD_KNN_MODEL")
    os.environ["GTO_CARD_KNN_MODEL"] = str(model_path)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("GTO_CARD_KNN_MODEL", None)
        else:
            os.environ["GTO_CARD_KNN_MODEL"] = old_value


def classify_glyph_path(path_text: Any, kind: str, model_dir: Path | None) -> dict[str, Any] | None:
    if model_dir is None or not path_text:
        return None
    cv2, _np = load_cv()
    path = Path(str(path_text))
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return classify_deep_glyph(image, kind, model_dir=Path(model_dir))


def classify_knn_glyph_path(path_text: Any, kind: str, model_path: Path) -> dict[str, Any] | None:
    if not path_text:
        return None
    cv2, _np = load_cv()
    path = Path(str(path_text))
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if kind == "rank":
        return classify_rank_glyph(image, model_path=Path(model_path))
    if kind == "suit":
        return classify_suit_glyph(image, model_path=Path(model_path))
    raise ValueError("kind must be rank or suit")


def classify_hf_probe_path(
    path_text: Any,
    kind: str,
    probe_dir: Path | None,
    *,
    device: str,
    local_files_only: bool,
) -> dict[str, Any] | None:
    if probe_dir is None or not path_text:
        return None
    return classify_hf_probe_glyph_path(
        path_text,
        kind,
        probe_dir=Path(probe_dir),
        device=device,
        local_files_only=local_files_only,
    )


def add_card_result(row: dict[str, Any], prefix: str, card: str | None) -> None:
    expected_rank = str(row.get("expected_rank") or "")
    expected_suit = str(row.get("expected_suit") or "")
    normalized = normalize_card(card)
    row[f"{prefix}_card"] = normalized or ""
    row[f"{prefix}_rank"] = normalized[0] if normalized else ""
    row[f"{prefix}_suit"] = normalized[1] if normalized else ""
    row[f"{prefix}_card_ok"] = bool(normalized and normalized == row.get("truth_card"))
    row[f"{prefix}_rank_ok"] = bool(normalized and normalized[0] == expected_rank)
    row[f"{prefix}_suit_ok"] = bool(normalized and normalized[1] == expected_suit)


def evaluator_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluators = {}
    for prefix in EVALUATOR_PREFIXES:
        present = [row for row in rows if row.get(f"{prefix}_card")]
        if not present:
            continue
        evaluators[prefix] = {
            "count": len(present),
            "card_acc": accuracy(present, f"{prefix}_card_ok"),
            "rank_acc": accuracy(present, f"{prefix}_rank_ok"),
            "suit_acc": accuracy(present, f"{prefix}_suit_ok"),
            "truth_sources": count_values(row.get("truth_source") for row in present),
        }
    return evaluators


def evaluator_confusions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for prefix in EVALUATOR_PREFIXES:
        rank_confusions = count_confusions(rows, expected_key="expected_rank", predicted_key=f"{prefix}_rank")
        suit_confusions = count_confusions(rows, expected_key="expected_suit", predicted_key=f"{prefix}_suit")
        card_confusions = count_confusions(rows, expected_key="truth_card", predicted_key=f"{prefix}_card")
        if rank_confusions or suit_confusions or card_confusions:
            output[prefix] = {
                "rank": rank_confusions[:20],
                "suit": suit_confusions[:20],
                "card": card_confusions[:20],
            }
    return output


def count_confusions(rows: list[dict[str, Any]], *, expected_key: str, predicted_key: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        expected = str(row.get(expected_key) or "")
        predicted = str(row.get(predicted_key) or "")
        if not expected or not predicted or expected == predicted:
            continue
        key = (expected, predicted)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def accuracy(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 6)


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "-")
        counts[key] = counts.get(key, 0) + 1
    return counts


def resolve_review_path(review_csv: Path, text: str) -> Path | None:
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    candidate = Path(review_csv).parent / text
    if candidate.exists():
        return candidate
    return path


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


def normalize_card(card: Any) -> str | None:
    text = str(card or "").strip()
    if not text or "?" in text:
        return None
    text = text.replace("10", "T")
    if len(text) != 2:
        return None
    rank = normalize_rank(text[0])
    suit = normalize_suit(text[1])
    return f"{rank}{suit}" if rank and suit else None


def normalize_rank(rank: Any) -> str | None:
    text = str(rank or "").strip().upper().replace("10", "T")
    return text if text in set(RANK_LABELS) else None


def normalize_suit(suit: Any) -> str | None:
    text = str(suit or "").strip().lower()
    aliases = {
        "♠": "s",
        "♥": "h",
        "♦": "d",
        "♣": "c",
        "spade": "s",
        "heart": "h",
        "diamond": "d",
        "club": "c",
    }
    text = aliases.get(text, text)
    return text if text in set(SUIT_LABELS) else None


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_csv",
        "row_index",
        "slot",
        "video",
        "timestamp_sec",
        "frame_index",
        "review_reason",
        "truth_source",
        "truth_card",
        "expected_rank",
        "expected_suit",
        "current_card",
        "current_csv_card",
        "current_csv_card_ok",
        "current_csv_rank_ok",
        "current_csv_suit_ok",
        "runtime_card",
        "runtime_card_ok",
        "runtime_rank_ok",
        "runtime_suit_ok",
        "knn_card",
        "knn_card_ok",
        "knn_rank_ok",
        "knn_suit_ok",
        "knn_rank_score",
        "knn_rank_margin",
        "knn_suit_score",
        "knn_suit_margin",
        "deep_card",
        "deep_card_ok",
        "deep_rank_ok",
        "deep_suit_ok",
        "deep_rank_score",
        "deep_rank_margin",
        "deep_suit_score",
        "deep_suit_margin",
        "hf_probe_card",
        "hf_probe_card_ok",
        "hf_probe_rank_ok",
        "hf_probe_suit_ok",
        "hf_probe_rank_score",
        "hf_probe_rank_margin",
        "hf_probe_suit_score",
        "hf_probe_suit_margin",
        "hf_probe_rank_model",
        "hf_probe_suit_model",
        "card_path",
        "rank_path",
        "suit_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_card_benchmark_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"Samples: {payload.get('sample_count')}",
        f"Truth sources: {json.dumps(payload.get('truth_sources') or {}, ensure_ascii=False)}",
        f"Output: {payload.get('output_dir')}",
        f"Rows CSV: {(payload.get('files') or {}).get('rows_csv')}",
        f"Report: {(payload.get('files') or {}).get('report_md')}",
        "",
        "Evaluators:",
    ]
    for name, item in (payload.get("evaluators") or {}).items():
        lines.append(
            f"- {name}: n={item.get('count')} card={item.get('card_acc')} "
            f"rank={item.get('rank_acc')} suit={item.get('suit_acc')}"
        )
    return "\n".join(lines)


def format_card_benchmark_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Card Recognition Benchmark",
        "",
        f"- Samples: `{summary.get('sample_count')}`",
        f"- Truth sources: `{json.dumps(summary.get('truth_sources') or {}, ensure_ascii=False)}`",
        f"- Include pseudo truth: `{summary.get('include_ok_pseudo')}`",
        f"- Rows CSV: `{(summary.get('files') or {}).get('rows_csv')}`",
        "",
        "| Evaluator | Count | Card Acc | Rank Acc | Suit Acc | Truth Sources |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in (summary.get("evaluators") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(item.get("count")),
                    str(item.get("card_acc")),
                    str(item.get("rank_acc")),
                    str(item.get("suit_acc")),
                    f"`{json.dumps(item.get('truth_sources') or {}, ensure_ascii=False)}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Confusions", ""])
    confusions = summary.get("confusions") or {}
    for evaluator, groups in confusions.items():
        lines.append(f"### {evaluator}")
        for kind in ("card", "rank", "suit"):
            items = groups.get(kind) or []
            if not items:
                continue
            compact = ", ".join(f"{item['expected']}->{item['predicted']} x{item['count']}" for item in items[:8])
            lines.append(f"- {kind}: {compact}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
