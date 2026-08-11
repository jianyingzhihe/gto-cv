from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .card_classifier import classify_rank_glyph, classify_suit_glyph
from .card_deep_model import classify_deep_glyph, warm_deep_card_models
from .card_glyph_export import safe_label, safe_stem


def audit_card_glyphs(
    *,
    manifest_path: Path,
    output_dir: Path,
    teacher_model_dir: Path | None = None,
    teacher_rank_model_dir: Path | None = None,
    teacher_suit_model_dir: Path | None = None,
    realtime_model_dir: Path | None = None,
    realtime_rank_model_dir: Path | None = None,
    realtime_suit_model_dir: Path | None = None,
    max_review: int = 240,
    rank_confidence_threshold: float = 0.82,
    suit_confidence_threshold: float = 0.55,
    temporal_window_frames: int = 120,
    temporal_min_support: int = 2,
    copy_accepted: bool = True,
) -> dict[str, Any]:
    cv2, _np = load_cv()
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review"
    accepted_dir = output_dir / "accepted"
    review_dir.mkdir(parents=True, exist_ok=True)
    if copy_accepted:
        accepted_dir.mkdir(parents=True, exist_ok=True)
    teacher_rank_dir = teacher_rank_model_dir or teacher_model_dir
    teacher_suit_dir = teacher_suit_model_dir or teacher_model_dir
    realtime_rank_dir = realtime_rank_model_dir or realtime_model_dir
    realtime_suit_dir = realtime_suit_model_dir or realtime_model_dir
    for model_dir in sorted(
        {Path(path) for path in (teacher_rank_dir, teacher_suit_dir, realtime_rank_dir, realtime_suit_dir) if path},
        key=lambda path: str(path),
    ):
        warm_deep_card_models(model_dir)

    records = read_jsonl(manifest_path)
    audited = []
    for index, record in enumerate(records):
        rank_eval = evaluate_glyph(
            cv2=cv2,
            kind="rank",
            image_path=Path(record.get("rank_path") or ""),
            current_label=str(record.get("rank") or ""),
            current_confidence=record.get("rank_confidence"),
            current_margin=record.get("rank_margin"),
            confidence_threshold=rank_confidence_threshold,
            teacher_model_dir=teacher_rank_dir,
            realtime_model_dir=realtime_rank_dir,
        )
        suit_eval = evaluate_glyph(
            cv2=cv2,
            kind="suit",
            image_path=Path(record.get("suit_path") or ""),
            current_label=str(record.get("suit") or ""),
            current_confidence=record.get("suit_confidence"),
            current_margin=record.get("suit_margin"),
            confidence_threshold=suit_confidence_threshold,
            teacher_model_dir=teacher_suit_dir,
            realtime_model_dir=realtime_suit_dir,
        )
        priority = max(rank_eval["priority"], suit_eval["priority"])
        needs_review = rank_eval["needs_review"] or suit_eval["needs_review"]
        consensus_card = f"{rank_eval.get('consensus_label') or record.get('rank') or '?'}{suit_eval.get('consensus_label') or record.get('suit') or '?'}"
        audit_record = {
            "index": index,
            "priority": priority,
            "needs_review": needs_review,
            "video": record.get("video", ""),
            "frame_index": record.get("frame_index", ""),
            "timestamp_sec": record.get("timestamp_sec", ""),
            "source": record.get("source", ""),
            "card_index": record.get("card_index", ""),
            "roi_mode": record.get("roi_mode", ""),
            "card": record.get("card", ""),
            "consensus_card": consensus_card,
            "rank": rank_eval,
            "suit": suit_eval,
            "rank_path": record.get("rank_path", ""),
            "suit_path": record.get("suit_path", ""),
            "card_path": record.get("card_path", ""),
        }
        audited.append(audit_record)

    apply_temporal_consensus(
        audited,
        window_frames=max(0, int(temporal_window_frames)),
        min_support=max(1, int(temporal_min_support)),
    )
    accepted_count = 0
    for item in audited:
        item["priority"] = max(float(item["rank"].get("priority") or 0.0), float(item["suit"].get("priority") or 0.0))
        item["needs_review"] = bool(item["rank"].get("needs_review") or item["suit"].get("needs_review"))
        item["consensus_card"] = (
            f"{item['rank'].get('consensus_label') or item['rank'].get('current_label') or '?'}"
            f"{item['suit'].get('consensus_label') or item['suit'].get('current_label') or '?'}"
        )
        if copy_accepted and not item["needs_review"]:
            copied = copy_consensus_images(item, accepted_dir)
            accepted_count += 1 if copied else 0

    review_records = sorted((item for item in audited if item["needs_review"]), key=lambda item: item["priority"], reverse=True)
    review_records = review_records[: max(0, int(max_review))]
    for item in review_records:
        copy_review_images(item, review_dir)

    all_csv = output_dir / "audit.csv"
    review_csv = output_dir / "review.csv"
    write_audit_csv(all_csv, audited)
    write_audit_csv(review_csv, review_records, include_final_columns=True)
    markdown_path = output_dir / "review.md"
    write_review_markdown(markdown_path, review_records)
    sheet_path = output_dir / "review_sheet.jpg"
    write_review_sheet(sheet_path, review_records)
    summary = {
        "ok": True,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "teacher_rank_model_dir": str(teacher_rank_dir) if teacher_rank_dir else "",
        "teacher_suit_model_dir": str(teacher_suit_dir) if teacher_suit_dir else "",
        "realtime_rank_model_dir": str(realtime_rank_dir) if realtime_rank_dir else "",
        "realtime_suit_model_dir": str(realtime_suit_dir) if realtime_suit_dir else "",
        "audited": len(audited),
        "needs_review": len([item for item in audited if item["needs_review"]]),
        "review_limited_to": len(review_records),
        "accepted_copied": accepted_count,
        "temporal_window_frames": int(temporal_window_frames),
        "temporal_min_support": int(temporal_min_support),
        "files": {
            "audit_csv": str(all_csv),
            "review_csv": str(review_csv),
            "review_md": str(markdown_path),
            "review_sheet": str(sheet_path),
            "review_dir": str(review_dir),
            "accepted_dir": str(accepted_dir) if copy_accepted else "",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def apply_card_review(*, review_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_rank = 0
    copied_suit = 0
    copied_card = 0
    skipped = []
    with Path(review_csv).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for slot in (0, 1):
                final_card = clean_final_card(row.get(f"final_card{slot}"))
                if not final_card:
                    continue
                rank, suit = final_card[0], final_card[1]
                rank_src = Path(row.get(f"card{slot}_rank_path") or "")
                suit_src = Path(row.get(f"card{slot}_suit_path") or "")
                card_src = Path(row.get(f"card{slot}_card_path") or "")
                if is_existing_file(rank_src):
                    dst = output_dir / "rank" / rank / rank_src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rank_src, dst)
                    copied_rank += 1
                else:
                    skipped.append({"path": str(rank_src), "reason": f"card{slot}_rank_missing"})
                if is_existing_file(suit_src):
                    dst = output_dir / "suit" / suit / suit_src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(suit_src, dst)
                    copied_suit += 1
                else:
                    skipped.append({"path": str(suit_src), "reason": f"card{slot}_suit_missing"})
                if is_existing_file(card_src):
                    dst = output_dir / "card" / final_card / card_src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(card_src, dst)
                    copied_card += 1
                else:
                    skipped.append({"path": str(card_src), "reason": f"card{slot}_card_missing"})
            final_rank = clean_final_label(row.get("final_rank"))
            final_suit = clean_final_label(row.get("final_suit"))
            if final_rank:
                src = Path(row.get("rank_path") or "")
                if is_existing_file(src):
                    dst = output_dir / "rank" / final_rank / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_rank += 1
                else:
                    skipped.append({"path": str(src), "reason": "rank_missing"})
            if final_suit:
                src = Path(row.get("suit_path") or "")
                if is_existing_file(src):
                    dst = output_dir / "suit" / final_suit / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_suit += 1
                else:
                    skipped.append({"path": str(src), "reason": "suit_missing"})
    summary = {
        "ok": True,
        "review_csv": str(review_csv),
        "output_dir": str(output_dir),
        "copied_rank": copied_rank,
        "copied_suit": copied_suit,
        "copied_card": copied_card,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
    }
    (output_dir / "applied_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def is_existing_file(path: Path) -> bool:
    return bool(str(path)) and path.exists() and path.is_file()


def evaluate_glyph(
    *,
    cv2: Any,
    kind: str,
    image_path: Path,
    current_label: str,
    current_confidence: Any,
    current_margin: Any,
    confidence_threshold: float,
    teacher_model_dir: Path | None,
    realtime_model_dir: Path | None,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    result: dict[str, Any] = {
        "kind": kind,
        "current_label": normalize_label(current_label),
        "current_confidence": safe_float(current_confidence),
        "current_margin": safe_float(current_margin),
        "image_path": str(image_path),
        "knn": None,
        "teacher": None,
        "realtime": None,
        "temporal": None,
        "consensus_label": "",
        "needs_review": True,
        "priority": 0.0,
        "reasons": [],
        "confidence_threshold": float(confidence_threshold),
    }
    if image is None:
        result["reasons"].append("image_missing")
        result["priority"] = 100.0
        return result
    if kind == "rank":
        result["knn"] = classify_rank_glyph(image)
        if teacher_model_dir:
            result["teacher"] = classify_deep_glyph(image, "rank", model_dir=teacher_model_dir)
        if realtime_model_dir:
            result["realtime"] = classify_deep_glyph(image, "rank", model_dir=realtime_model_dir)
    else:
        allowed = ("h", "d") if result["current_label"] in ("h", "d") else ("s", "c") if result["current_label"] in ("s", "c") else None
        result["knn"] = classify_suit_glyph(image, allowed=allowed)
        if teacher_model_dir:
            result["teacher"] = classify_deep_glyph(image, "suit", model_dir=teacher_model_dir, allowed=allowed)
        if realtime_model_dir:
            result["realtime"] = classify_deep_glyph(image, "suit", model_dir=realtime_model_dir, allowed=allowed)
    labels = prediction_labels(result)
    current = result["current_label"]
    consensus = majority_label(labels)
    result["consensus_label"] = consensus or current

    reasons = []
    if not current or current == "?":
        reasons.append("current_unknown")
    if result["current_confidence"] is None or result["current_confidence"] < confidence_threshold:
        reasons.append("current_low_confidence")
    if consensus and current and consensus != current:
        reasons.append("consensus_disagrees_current")
    if high_confidence_disagreement(result, current):
        reasons.append("high_confidence_model_disagreement")
    teacher = result.get("teacher") or {}
    teacher_low = bool(teacher and safe_float(teacher.get("score")) is not None and safe_float(teacher.get("score")) < 0.45)
    if teacher_low:
        result["teacher_low_score"] = True
    result["reasons"] = reasons
    result["needs_review"] = bool(reasons)
    result["priority"] = priority_score(result, labels, reasons, confidence_threshold, teacher_low=teacher_low)
    return result


def apply_temporal_consensus(
    rows: list[dict[str, Any]],
    *,
    window_frames: int,
    min_support: int,
) -> None:
    if window_frames <= 0 or min_support <= 0:
        return
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in rows:
        key = (
            str(item.get("video") or ""),
            str(item.get("source") or ""),
            str(item.get("card_index") if item.get("card_index") not in (None, "") else "0"),
        )
        groups.setdefault(key, []).append(item)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: safe_int(item.get("frame_index")))
        for item in group_rows:
            frame_index = safe_int(item.get("frame_index"))
            neighbors = [
                other
                for other in group_rows
                if abs(safe_int(other.get("frame_index")) - frame_index) <= window_frames
            ]
            for kind in ("rank", "suit"):
                vote = temporal_vote(
                    neighbors,
                    kind=kind,
                    min_support=min_support,
                )
                if not vote:
                    continue
                merge_temporal_vote(item[kind], vote)


def temporal_vote(rows: list[dict[str, Any]], *, kind: str, min_support: int) -> dict[str, Any] | None:
    counts: dict[str, float] = {}
    examples: dict[str, list[int]] = {}
    for item in rows:
        evaluation = item.get(kind) or {}
        label = evaluation.get("current_label")
        confidence = safe_float(evaluation.get("current_confidence")) or 0.0
        threshold = safe_float(evaluation.get("confidence_threshold")) or 0.0
        margin = safe_float(evaluation.get("current_margin")) or 0.0
        if not label or label == "?" or label == "_unknown":
            continue
        if confidence < threshold:
            continue
        weight = 1.0 + min(1.0, max(0.0, margin))
        counts[str(label)] = counts.get(str(label), 0.0) + weight
        examples.setdefault(str(label), []).append(safe_int(item.get("frame_index")))
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    label, weight = ordered[0]
    total = sum(counts.values())
    support = len(examples.get(label) or [])
    ratio = weight / total if total else 0.0
    if support < min_support or ratio < 0.62:
        return None
    return {
        "label": label,
        "support": support,
        "ratio": round(ratio, 4),
        "weight": round(weight, 4),
        "total_weight": round(total, 4),
        "frames": examples.get(label, [])[:8],
    }


def merge_temporal_vote(evaluation: dict[str, Any], vote: dict[str, Any]) -> None:
    temporal_label = str(vote.get("label") or "")
    if not temporal_label:
        return
    evaluation["temporal"] = vote
    current = str(evaluation.get("current_label") or "")
    consensus = str(evaluation.get("consensus_label") or current or "")
    reasons = list(evaluation.get("reasons") or [])
    if current and current != "?" and temporal_label != current:
        if "temporal_disagrees_current" not in reasons:
            reasons.append("temporal_disagrees_current")
    elif temporal_label == consensus or temporal_label == current:
        for weak_reason in ("current_low_confidence",):
            if weak_reason in reasons:
                reasons.remove(weak_reason)
    if (not consensus or consensus == "?") and temporal_label:
        evaluation["consensus_label"] = temporal_label
    labels = prediction_labels(evaluation)
    threshold = safe_float(evaluation.get("confidence_threshold")) or 0.0
    teacher_low = bool(evaluation.get("teacher_low_score"))
    evaluation["reasons"] = reasons
    evaluation["needs_review"] = bool(reasons)
    evaluation["priority"] = priority_score(evaluation, labels, reasons, threshold, teacher_low=teacher_low)


def prediction_labels(result: dict[str, Any]) -> list[str]:
    labels = []
    current = result.get("current_label")
    if current and current != "?":
        labels.append(str(current))
    for key in ("knn", "teacher", "realtime", "temporal"):
        pred = result.get(key) or {}
        label = pred.get("label")
        if label:
            labels.append(str(label))
    return labels


def majority_label(labels: list[str]) -> str:
    if not labels:
        return ""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (item[1], labels.count(item[0])), reverse=True)
    return ordered[0][0]


def disagreement_count(labels: list[str]) -> int:
    return len(set(label for label in labels if label))


def high_confidence_disagreement(result: dict[str, Any], current: str) -> bool:
    if not current or current == "?":
        return False
    for key in ("knn", "teacher", "realtime"):
        pred = result.get(key) or {}
        label = pred.get("label")
        score = safe_float(pred.get("score")) or 0.0
        margin = safe_float(pred.get("margin")) or 0.0
        if label and label != current and score >= 0.72 and margin >= 0.12:
            return True
    return False


def priority_score(
    result: dict[str, Any],
    labels: list[str],
    reasons: list[str],
    confidence_threshold: float,
    *,
    teacher_low: bool = False,
) -> float:
    score = float(len(reasons)) * 10.0 + float(disagreement_count(labels)) * 3.0
    confidence = result.get("current_confidence")
    if confidence is None:
        score += 8.0
    else:
        score += max(0.0, confidence_threshold - float(confidence)) * 20.0
    teacher = result.get("teacher") or {}
    teacher_margin = safe_float(teacher.get("margin"))
    if teacher_margin is not None:
        score += max(0.0, 0.20 - teacher_margin) * 10.0
    if teacher_low:
        score += 2.0
    return round(score, 4)


def copy_review_images(item: dict[str, Any], review_dir: Path) -> None:
    base = review_basename(item)
    for kind in ("rank", "suit"):
        src = Path(item.get(f"{kind}_path") or "")
        if src.exists():
            dst = review_dir / kind / f"{base}_{kind}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    card_src = Path(item.get("card_path") or "")
    if card_src.exists():
        dst = review_dir / "card" / f"{base}_{card_src.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(card_src, dst)


def copy_consensus_images(item: dict[str, Any], accepted_dir: Path) -> bool:
    rank_label = item["rank"].get("consensus_label") or item["rank"].get("current_label")
    suit_label = item["suit"].get("consensus_label") or item["suit"].get("current_label")
    copied = False
    for kind, label in (("rank", rank_label), ("suit", suit_label)):
        if not label or label == "?":
            continue
        src = Path(item.get(f"{kind}_path") or "")
        if not src.exists():
            continue
        dst = accepted_dir / kind / safe_label(str(label)) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied = True
    return copied


def review_basename(item: dict[str, Any]) -> str:
    card = safe_label(str(item.get("card") or "unknown"))
    consensus = safe_label(str(item.get("consensus_card") or "unknown"))
    source = safe_label(str(item.get("source") or "src"))
    frame = safe_label(str(item.get("frame_index") or "f"))
    priority = int(round(float(item.get("priority") or 0.0) * 10))
    return f"p{priority:04d}_{source}_f{frame}_{card}_to_{consensus}"


def write_audit_csv(path: Path, rows: list[dict[str, Any]], include_final_columns: bool = False) -> None:
    fields = [
        "index",
        "priority",
        "needs_review",
        "video",
        "frame_index",
        "timestamp_sec",
        "source",
        "card_index",
        "roi_mode",
        "card",
        "consensus_card",
        "rank_current",
        "rank_knn",
        "rank_teacher",
        "rank_realtime",
        "rank_temporal",
        "rank_reasons",
        "suit_current",
        "suit_knn",
        "suit_teacher",
        "suit_realtime",
        "suit_temporal",
        "suit_reasons",
        "rank_path",
        "suit_path",
        "card_path",
    ]
    if include_final_columns:
        fields.extend(["final_rank", "final_suit", "note"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            row = flatten_audit_row(item)
            if include_final_columns:
                row["final_rank"] = ""
                row["final_suit"] = ""
                row["note"] = ""
            writer.writerow(row)


def flatten_audit_row(item: dict[str, Any]) -> dict[str, Any]:
    rank = item.get("rank") or {}
    suit = item.get("suit") or {}
    return {
        "index": item.get("index", ""),
        "priority": item.get("priority", ""),
        "needs_review": item.get("needs_review", ""),
        "video": item.get("video", ""),
        "frame_index": item.get("frame_index", ""),
        "timestamp_sec": item.get("timestamp_sec", ""),
        "source": item.get("source", ""),
        "card_index": item.get("card_index", ""),
        "roi_mode": item.get("roi_mode", ""),
        "card": item.get("card", ""),
        "consensus_card": item.get("consensus_card", ""),
        "rank_current": rank.get("current_label", ""),
        "rank_knn": prediction_summary(rank.get("knn")),
        "rank_teacher": prediction_summary(rank.get("teacher")),
        "rank_realtime": prediction_summary(rank.get("realtime")),
        "rank_temporal": prediction_summary(rank.get("temporal")),
        "rank_reasons": ";".join(rank.get("reasons") or []),
        "suit_current": suit.get("current_label", ""),
        "suit_knn": prediction_summary(suit.get("knn")),
        "suit_teacher": prediction_summary(suit.get("teacher")),
        "suit_realtime": prediction_summary(suit.get("realtime")),
        "suit_temporal": prediction_summary(suit.get("temporal")),
        "suit_reasons": ";".join(suit.get("reasons") or []),
        "rank_path": item.get("rank_path", ""),
        "suit_path": item.get("suit_path", ""),
        "card_path": item.get("card_path", ""),
    }


def prediction_summary(prediction: dict[str, Any] | None) -> str:
    if not prediction:
        return ""
    label = prediction.get("label", "")
    if "ratio" in prediction or "support" in prediction:
        ratio = safe_float(prediction.get("ratio"))
        support = prediction.get("support", "")
        return f"{label}:{(ratio or 0.0):.3f}/n{support}"
    score = safe_float(prediction.get("score"))
    margin = safe_float(prediction.get("margin"))
    if score is None:
        return str(label)
    return f"{label}:{score:.3f}/{(margin or 0.0):.3f}"


def write_review_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Card Glyph Review",
        "",
        "Fill `final_rank` and `final_suit` in `review.csv`, then run `apply-card-review`.",
        "",
        "| priority | card -> consensus | reasons | card crop | rank glyph | suit glyph |",
        "|---:|---|---|---|---|---|",
    ]
    for item in rows[:200]:
        rank_reasons = ",".join(item["rank"].get("reasons") or [])
        suit_reasons = ",".join(item["suit"].get("reasons") or [])
        reasons = f"rank:{rank_reasons}<br>suit:{suit_reasons}"
        card = image_md(item.get("card_path"))
        rank = image_md(item.get("rank_path"))
        suit = image_md(item.get("suit_path"))
        lines.append(
            f"| {item.get('priority')} | {item.get('card')} -> {item.get('consensus_card')} | {reasons} | {card} | {rank} | {suit} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_sheet(path: Path, rows: list[dict[str, Any]], limit: int = 80) -> None:
    cv2, np = load_cv()
    rows = rows[: max(0, int(limit))]
    if not rows:
        canvas = np.full((120, 600, 3), 245, np.uint8)
        cv2.putText(canvas, "No review rows", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        cv2.imwrite(str(path), canvas)
        return
    row_h = 185
    col_w = 620
    cols = 2
    sheet_rows = (len(rows) + cols - 1) // cols
    canvas = np.full((sheet_rows * row_h, cols * col_w, 3), 245, np.uint8)
    for idx, item in enumerate(rows):
        col = idx % cols
        row = idx // cols
        x0 = col * col_w
        y0 = row * row_h
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + col_w - 6, y0 + row_h - 6), (210, 210, 210), 1)
        card = load_resized_bgr(item.get("card_path"), (90, 120))
        rank = load_resized_bgr(item.get("rank_path"), (54, 70))
        suit = load_resized_bgr(item.get("suit_path"), (48, 48))
        paste(canvas, card, x0 + 10, y0 + 20)
        paste(canvas, rank, x0 + 112, y0 + 20)
        paste(canvas, suit, x0 + 175, y0 + 32)
        text_lines = [
            f"p={item.get('priority')}  {item.get('card')} -> {item.get('consensus_card')}",
            f"rank {prediction_summary(item['rank'].get('knn'))} | T {prediction_summary(item['rank'].get('teacher'))}",
            f"rtime {prediction_summary(item['rank'].get('realtime'))} | temp {prediction_summary(item['rank'].get('temporal'))}",
            f"suit {prediction_summary(item['suit'].get('knn'))} | T {prediction_summary(item['suit'].get('teacher'))}",
            f"stime {prediction_summary(item['suit'].get('realtime'))} | temp {prediction_summary(item['suit'].get('temporal'))}",
            f"R: {','.join(item['rank'].get('reasons') or [])}",
            f"S: {','.join(item['suit'].get('reasons') or [])}",
        ]
        for line_idx, text in enumerate(text_lines):
            cv2.putText(
                canvas,
                text[:62],
                (x0 + 235, y0 + 28 + line_idx * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def load_resized_bgr(path_value: Any, size: tuple[int, int]) -> Any:
    cv2, np = load_cv()
    width, height = size
    image = cv2.imread(str(path_value), cv2.IMREAD_COLOR) if path_value else None
    if image is None:
        return np.full((height, width, 3), 235, np.uint8)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def paste(canvas: Any, image: Any, x: int, y: int) -> None:
    h, w = image.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    x2 = min(canvas_w, x + w)
    y2 = min(canvas_h, y + h)
    if x2 <= x or y2 <= y:
        return
    canvas[y:y2, x:x2] = image[: y2 - y, : x2 - x]


def image_md(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value)).resolve()
    return f"![]({path.as_posix()})"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_label(value: str) -> str:
    value = str(value or "").strip()
    return "" if value in ("_unknown", "unknown", "None") else value


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def clean_final_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in ("?", "_unknown", "skip", "SKIP"):
        return ""
    return safe_label(text)


def clean_final_card(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in ("?", "_unknown", "unknown", "skip", "SKIP"):
        return ""
    suit_map = {
        "♠": "s",
        "♤": "s",
        "黑桃": "s",
        "♥": "h",
        "♡": "h",
        "红桃": "h",
        "♦": "d",
        "♢": "d",
        "方块": "d",
        "♣": "c",
        "♧": "c",
        "梅花": "c",
    }
    for raw, normalized in suit_map.items():
        text = text.replace(raw, normalized)
    text = text.replace("10", "T").replace(" ", "").replace("-", "")
    if len(text) < 2:
        return ""
    rank = text[0].upper()
    suit = text[1].lower()
    if rank not in set("AKQJT98765432") or suit not in set("shdc"):
        return ""
    return f"{rank}{suit}"


def format_glyph_audit_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"audit-card-glyphs failed: {payload.get('error')}"
    files = payload.get("files") or {}
    return "\n".join(
        [
            f"Audited: {payload.get('audited', 0)}",
            f"Needs review: {payload.get('needs_review', 0)}",
            f"Review rows: {payload.get('review_limited_to', 0)}",
            f"Accepted copied: {payload.get('accepted_copied', 0)}",
            f"Review CSV: {files.get('review_csv')}",
            f"Review MD: {files.get('review_md')}",
            f"Accepted dir: {files.get('accepted_dir')}",
        ]
    )


def format_apply_review_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"apply-card-review failed: {payload.get('error')}"
    return "\n".join(
        [
            f"Copied rank: {payload.get('copied_rank', 0)}",
            f"Copied suit: {payload.get('copied_suit', 0)}",
            f"Copied card: {payload.get('copied_card', 0)}",
            f"Output: {payload.get('output_dir')}",
            f"Skipped: {payload.get('skipped_count', 0)}",
        ]
    )


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
