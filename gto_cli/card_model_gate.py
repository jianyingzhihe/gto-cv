from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .card_benchmark import benchmark_card_review
from .card_classifier import DEFAULT_MODEL_PATH
from .card_review_diff import diff_card_review


def gate_card_model(
    *,
    benchmark_review_csvs: list[Path],
    baseline_review_csv: Path,
    candidate_review_csv: Path,
    output_dir: Path,
    candidate_name: str = "candidate",
    candidate_evaluator: str = "knn",
    knn_model_path: Path | None = None,
    deep_model_dir: Path | None = None,
    deep_rank_model_dir: Path | None = None,
    deep_suit_model_dir: Path | None = None,
    hf_probe_dir: Path | None = None,
    hf_probe_device: str = "auto",
    hf_probe_local_files_only: bool = False,
    candidate_validation_summary_json: Path | None = None,
    baseline_validation_summary_json: Path | None = None,
    include_ok_pseudo: bool = False,
    allowed_pseudo_reasons: tuple[str, ...] = ("ok",),
    run_runtime: bool = True,
    max_benchmark_samples: int | None = None,
    max_diff_rows: int | None = None,
    max_risk: int = 0,
    require_validation: bool = False,
    max_real_problem: int = 0,
    max_board_bad: int = 0,
    max_median_ms: float | None = None,
    max_p90_ms: float | None = None,
    max_median_regression_ms: float | None = None,
    max_p90_regression_ms: float | None = None,
    min_candidate_card_acc: float = 0.999,
    min_candidate_rank_acc: float = 0.999,
    min_candidate_suit_acc: float = 0.999,
    require_no_missing_rows: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = output_dir / "benchmark"
    diff_dir = output_dir / "review_diff"
    benchmark = benchmark_card_review(
        review_csvs=[Path(path) for path in benchmark_review_csvs],
        output_dir=benchmark_dir,
        deep_model_dir=deep_model_dir,
        deep_rank_model_dir=deep_rank_model_dir,
        deep_suit_model_dir=deep_suit_model_dir,
        knn_model_path=knn_model_path or DEFAULT_MODEL_PATH,
        hf_probe_dir=hf_probe_dir,
        hf_probe_device=hf_probe_device,
        hf_probe_local_files_only=hf_probe_local_files_only,
        include_ok_pseudo=include_ok_pseudo,
        allowed_pseudo_reasons=allowed_pseudo_reasons,
        run_runtime=run_runtime,
        max_samples=max_benchmark_samples,
    )
    diff = diff_card_review(
        baseline_csv=Path(baseline_review_csv),
        candidate_csv=Path(candidate_review_csv),
        output_dir=diff_dir,
        risky_baseline_reasons=allowed_pseudo_reasons,
        max_rows=max_diff_rows,
    )
    candidate_validation = load_validation_summary(candidate_validation_summary_json)
    baseline_validation = load_validation_summary(baseline_validation_summary_json)

    checks = build_gate_checks(
        benchmark=benchmark,
        diff=diff,
        candidate_validation=candidate_validation,
        baseline_validation=baseline_validation,
        max_risk=max_risk,
        require_validation=require_validation,
        max_real_problem=max_real_problem,
        max_board_bad=max_board_bad,
        max_median_ms=max_median_ms,
        max_p90_ms=max_p90_ms,
        max_median_regression_ms=max_median_regression_ms,
        max_p90_regression_ms=max_p90_regression_ms,
        min_candidate_card_acc=min_candidate_card_acc,
        min_candidate_rank_acc=min_candidate_rank_acc,
        min_candidate_suit_acc=min_candidate_suit_acc,
        require_no_missing_rows=require_no_missing_rows,
        candidate_evaluator=candidate_evaluator,
    )
    promote = all(bool(check.get("pass")) for check in checks)
    summary = {
        "ok": True,
        "promote": promote,
        "decision": "promote" if promote else "reject",
        "candidate_name": candidate_name,
        "candidate_evaluator": candidate_evaluator,
        "output_dir": str(output_dir),
        "benchmark_review_csvs": [str(path) for path in benchmark_review_csvs],
        "baseline_review_csv": str(baseline_review_csv),
        "candidate_review_csv": str(candidate_review_csv),
        "knn_model_path": str(knn_model_path) if knn_model_path else None,
        "deep_model_dir": str(deep_model_dir) if deep_model_dir else None,
        "hf_probe_dir": str(hf_probe_dir) if hf_probe_dir else None,
        "hf_probe_device": hf_probe_device,
        "hf_probe_local_files_only": bool(hf_probe_local_files_only),
        "candidate_validation_summary_json": str(candidate_validation_summary_json) if candidate_validation_summary_json else None,
        "baseline_validation_summary_json": str(baseline_validation_summary_json) if baseline_validation_summary_json else None,
        "checks": checks,
        "benchmark": compact_benchmark(benchmark),
        "diff": compact_diff(diff),
        "candidate_validation": compact_validation(candidate_validation),
        "baseline_validation": compact_validation(baseline_validation),
        "sample": {
            "wall_time_sec": round(float(time.perf_counter() - started_at), 3),
            "max_benchmark_samples": max_benchmark_samples,
            "max_diff_rows": max_diff_rows,
        },
        "files": {
            "summary": str(output_dir / "card_model_gate_summary.json"),
            "report_md": str(output_dir / "card_model_gate_report.md"),
            "benchmark_summary": (benchmark.get("files") or {}).get("summary"),
            "benchmark_rows_csv": (benchmark.get("files") or {}).get("rows_csv"),
            "diff_summary": (diff.get("files") or {}).get("summary"),
            "diff_rows_csv": (diff.get("files") or {}).get("rows_csv"),
        },
    }
    (output_dir / "card_model_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "card_model_gate_report.md").write_text(format_card_model_gate_markdown(summary), encoding="utf-8")
    return summary


def build_gate_checks(
    *,
    benchmark: dict[str, Any],
    diff: dict[str, Any],
    candidate_validation: dict[str, Any] | None,
    baseline_validation: dict[str, Any] | None,
    max_risk: int,
    require_validation: bool,
    max_real_problem: int,
    max_board_bad: int,
    max_median_ms: float | None,
    max_p90_ms: float | None,
    max_median_regression_ms: float | None,
    max_p90_regression_ms: float | None,
    min_candidate_card_acc: float,
    min_candidate_rank_acc: float,
    min_candidate_suit_acc: float,
    require_no_missing_rows: bool,
    candidate_evaluator: str,
) -> list[dict[str, Any]]:
    evaluator_name, evaluator = select_candidate_evaluator(benchmark, candidate_evaluator)
    diff_counts = diff.get("counts") or {}
    risk_count = int(diff_counts.get("risk_count") or 0)
    missing_count = int(diff.get("missing_in_candidate_count") or 0)
    checks = [
        {
            "name": "benchmark_candidate_present",
            "pass": evaluator is not None,
            "actual": evaluator_name or "",
            "required": "selected evaluator present",
        },
        {
            "name": "benchmark_card_acc",
            "pass": bool(evaluator and float(evaluator.get("card_acc") or 0.0) >= float(min_candidate_card_acc)),
            "actual": float((evaluator or {}).get("card_acc") or 0.0),
            "required": float(min_candidate_card_acc),
        },
        {
            "name": "benchmark_rank_acc",
            "pass": bool(evaluator and float(evaluator.get("rank_acc") or 0.0) >= float(min_candidate_rank_acc)),
            "actual": float((evaluator or {}).get("rank_acc") or 0.0),
            "required": float(min_candidate_rank_acc),
        },
        {
            "name": "benchmark_suit_acc",
            "pass": bool(evaluator and float(evaluator.get("suit_acc") or 0.0) >= float(min_candidate_suit_acc)),
            "actual": float((evaluator or {}).get("suit_acc") or 0.0),
            "required": float(min_candidate_suit_acc),
        },
        {
            "name": "review_diff_risk",
            "pass": risk_count <= int(max_risk),
            "actual": risk_count,
            "required": int(max_risk),
        },
        {
            "name": "review_diff_missing_rows",
            "pass": (missing_count == 0) if require_no_missing_rows else True,
            "actual": missing_count,
            "required": 0 if require_no_missing_rows else "ignored",
        },
    ]
    checks.extend(
        build_validation_checks(
            candidate_validation=candidate_validation,
            baseline_validation=baseline_validation,
            require_validation=require_validation,
            max_real_problem=max_real_problem,
            max_board_bad=max_board_bad,
            max_median_ms=max_median_ms,
            max_p90_ms=max_p90_ms,
            max_median_regression_ms=max_median_regression_ms,
            max_p90_regression_ms=max_p90_regression_ms,
        )
    )
    return checks


def build_validation_checks(
    *,
    candidate_validation: dict[str, Any] | None,
    baseline_validation: dict[str, Any] | None,
    require_validation: bool,
    max_real_problem: int,
    max_board_bad: int,
    max_median_ms: float | None,
    max_p90_ms: float | None,
    max_median_regression_ms: float | None,
    max_p90_regression_ms: float | None,
) -> list[dict[str, Any]]:
    if not candidate_validation:
        if not require_validation:
            return []
        return [
            {
                "name": "validation_summary_present",
                "pass": False,
                "actual": "",
                "required": "candidate validate-cv summary",
            }
        ]

    timing = candidate_validation.get("timing_ms") or {}
    baseline_timing = (baseline_validation or {}).get("timing_ms") or {}
    card_health = candidate_validation.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    checks = [
        {
            "name": "validation_ok",
            "pass": bool(candidate_validation.get("ok")),
            "actual": bool(candidate_validation.get("ok")),
            "required": True,
        },
        {
            "name": "validation_real_problem_count",
            "pass": int(candidate_validation.get("real_problem_count") or 0) <= int(max_real_problem),
            "actual": int(candidate_validation.get("real_problem_count") or 0),
            "required": int(max_real_problem),
        },
        {
            "name": "validation_board_bad_count",
            "pass": int(candidate_validation.get("board_bad_count") or 0) <= int(max_board_bad),
            "actual": int(candidate_validation.get("board_bad_count") or 0),
            "required": int(max_board_bad),
        },
    ]
    if card_health:
        issue_count = card_issue_count(card_health)
        checks.extend(
            [
                {
                    "name": "validation_hero_incomplete_or_missed",
                    "pass": int(hero_health.get("incomplete_or_missed_frames") or 0) <= int(max_real_problem),
                    "actual": int(hero_health.get("incomplete_or_missed_frames") or 0),
                    "required": int(max_real_problem),
                },
                {
                    "name": "validation_hero_turn_blocked",
                    "pass": int(hero_health.get("turn_blocked_frames") or 0) <= int(max_real_problem),
                    "actual": int(hero_health.get("turn_blocked_frames") or 0),
                    "required": int(max_real_problem),
                },
                {
                    "name": "validation_board_health_bad_frames",
                    "pass": int(board_health.get("bad_frames") or 0) <= int(max_board_bad),
                    "actual": int(board_health.get("bad_frames") or 0),
                    "required": int(max_board_bad),
                },
                {
                    "name": "validation_card_issue_count",
                    "pass": issue_count <= int(max_real_problem) + int(max_board_bad),
                    "actual": issue_count,
                    "required": int(max_real_problem) + int(max_board_bad),
                },
            ]
        )
    median = optional_float(timing.get("median"))
    p90 = optional_float(timing.get("p90"))
    baseline_median = optional_float(baseline_timing.get("median"))
    baseline_p90 = optional_float(baseline_timing.get("p90"))
    if max_median_ms is not None:
        checks.append(
            {
                "name": "validation_median_ms",
                "pass": median is not None and median <= float(max_median_ms),
                "actual": median,
                "required": float(max_median_ms),
            }
        )
    if max_p90_ms is not None:
        checks.append(
            {
                "name": "validation_p90_ms",
                "pass": p90 is not None and p90 <= float(max_p90_ms),
                "actual": p90,
                "required": float(max_p90_ms),
            }
        )
    if max_median_regression_ms is not None:
        required = baseline_median + float(max_median_regression_ms) if baseline_median is not None else None
        checks.append(
            {
                "name": "validation_median_regression_ms",
                "pass": median is not None and required is not None and median <= required,
                "actual": median,
                "required": required,
            }
        )
    if max_p90_regression_ms is not None:
        required = baseline_p90 + float(max_p90_regression_ms) if baseline_p90 is not None else None
        checks.append(
            {
                "name": "validation_p90_regression_ms",
                "pass": p90 is not None and required is not None and p90 <= required,
                "actual": p90,
                "required": required,
            }
        )
    return checks


def card_issue_count(card_health: dict[str, Any]) -> int:
    return sum(int(value or 0) for value in (card_health.get("issue_counts") or {}).values())


def select_candidate_evaluator(benchmark: dict[str, Any], candidate_evaluator: str) -> tuple[str | None, dict[str, Any] | None]:
    evaluators = benchmark.get("evaluators") or {}
    candidate_evaluator = str(candidate_evaluator or "").strip().lower()
    if candidate_evaluator in evaluators:
        return candidate_evaluator, evaluators.get(candidate_evaluator)
    return candidate_evaluator or None, None


def compact_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": payload.get("sample_count"),
        "truth_sources": payload.get("truth_sources"),
        "evaluators": payload.get("evaluators"),
        "confusions": payload.get("confusions"),
        "files": payload.get("files"),
    }


def compact_diff(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "matched_row_count": payload.get("matched_row_count"),
        "missing_in_candidate_count": payload.get("missing_in_candidate_count"),
        "extra_in_candidate_count": payload.get("extra_in_candidate_count"),
        "counts": payload.get("counts"),
        "risky_examples": payload.get("risky_examples"),
        "changed_examples": payload.get("changed_examples"),
        "files": payload.get("files"),
    }


def load_validation_summary(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def compact_validation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "ok": payload.get("ok"),
        "video_count": payload.get("video_count"),
        "counts": payload.get("counts"),
        "real_problem_count": payload.get("real_problem_count"),
        "board_bad_count": payload.get("board_bad_count"),
        "card_health": payload.get("card_health") or {},
        "timing_ms": payload.get("timing_ms"),
        "sample": payload.get("sample"),
        "files": payload.get("files"),
    }


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_card_model_gate_summary(payload: dict[str, Any]) -> str:
    benchmark = payload.get("benchmark") or {}
    diff = payload.get("diff") or {}
    validation = payload.get("candidate_validation") or {}
    diff_counts = diff.get("counts") or {}
    timing = validation.get("timing_ms") or {}
    card_health = validation.get("card_health") or {}
    hero_health = card_health.get("hero") or {}
    board_health = card_health.get("board") or {}
    lines = [
        f"Candidate: {payload.get('candidate_name')}",
        f"Evaluator: {payload.get('candidate_evaluator')}",
        f"Decision: {str(payload.get('decision')).upper()}",
        f"Promote: {payload.get('promote')}",
        f"Benchmark samples: {benchmark.get('sample_count')}",
        f"Diff slots: {diff_counts.get('slot_count')} | changed={diff_counts.get('changed_count')} | risk={diff_counts.get('risk_count')}",
        f"Validation: real_problem={validation.get('real_problem_count', '-')} board_bad={validation.get('board_bad_count', '-')} median={timing.get('median', '-')}ms p90={timing.get('p90', '-')}ms",
        (
            "Card health: "
            f"hero_complete={hero_health.get('complete_frames', '-')} "
            f"hero_incomplete_or_missed={hero_health.get('incomplete_or_missed_frames', '-')} "
            f"hero_turn_blocked={hero_health.get('turn_blocked_frames', '-')} "
            f"board_bad={board_health.get('bad_frames', '-')} "
            f"issues={card_issue_count(card_health) if card_health else '-'}"
        ),
        "",
        "Checks:",
    ]
    for check in payload.get("checks") or []:
        mark = "PASS" if check.get("pass") else "FAIL"
        lines.append(f"- {mark} {check.get('name')}: actual={check.get('actual')} required={check.get('required')}")
    files = payload.get("files") or {}
    lines.extend(
        [
            "",
            f"Report: {files.get('report_md')}",
            f"Summary: {files.get('summary')}",
            f"Benchmark rows: {files.get('benchmark_rows_csv')}",
            f"Diff rows: {files.get('diff_rows_csv')}",
        ]
    )
    risky = diff.get("risky_examples") or []
    if risky:
        lines.extend(["", "Risky examples:"])
        for row in risky[:8]:
            lines.append(
                f"- {row.get('video')} t={row.get('timestamp_sec')} slot={row.get('slot')}: "
                f"{row.get('baseline_card') or '-'} -> {row.get('candidate_card') or '-'}"
            )
    return "\n".join(lines)


def format_card_model_gate_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Card Model Promotion Gate",
        "",
        f"- Candidate: `{summary.get('candidate_name')}`",
        f"- Evaluator: `{summary.get('candidate_evaluator')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Promote: `{summary.get('promote')}`",
        f"- Output: `{summary.get('output_dir')}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Actual | Required |",
        "|---|---|---:|---:|",
    ]
    for check in summary.get("checks") or []:
        result = "PASS" if check.get("pass") else "FAIL"
        lines.append(f"| {check.get('name')} | {result} | {check.get('actual')} | {check.get('required')} |")

    benchmark = summary.get("benchmark") or {}
    lines.extend(["", "## Benchmark", "", "| Evaluator | Count | Card | Rank | Suit |", "|---|---:|---:|---:|---:|"])
    for name, item in (benchmark.get("evaluators") or {}).items():
        lines.append(
            f"| {name} | {item.get('count')} | {item.get('card_acc')} | {item.get('rank_acc')} | {item.get('suit_acc')} |"
        )

    validation = summary.get("candidate_validation")
    baseline_validation = summary.get("baseline_validation")
    if validation:
        timing = validation.get("timing_ms") or {}
        baseline_timing = (baseline_validation or {}).get("timing_ms") or {}
        card_health = validation.get("card_health") or {}
        baseline_card_health = (baseline_validation or {}).get("card_health") or {}
        hero_health = card_health.get("hero") or {}
        baseline_hero_health = baseline_card_health.get("hero") or {}
        board_health = card_health.get("board") or {}
        baseline_board_health = baseline_card_health.get("board") or {}
        lines.extend(
            [
                "",
                "## CV Validation",
                "",
                "| Metric | Candidate | Baseline |",
                "|---|---:|---:|",
                f"| real_problem_count | {validation.get('real_problem_count')} | {(baseline_validation or {}).get('real_problem_count', '')} |",
                f"| board_bad_count | {validation.get('board_bad_count')} | {(baseline_validation or {}).get('board_bad_count', '')} |",
                f"| hero_incomplete_or_missed | {hero_health.get('incomplete_or_missed_frames', '')} | {baseline_hero_health.get('incomplete_or_missed_frames', '')} |",
                f"| hero_turn_blocked | {hero_health.get('turn_blocked_frames', '')} | {baseline_hero_health.get('turn_blocked_frames', '')} |",
                f"| board_health_bad_frames | {board_health.get('bad_frames', '')} | {baseline_board_health.get('bad_frames', '')} |",
                f"| card_issue_count | {card_issue_count(card_health) if card_health else ''} | {card_issue_count(baseline_card_health) if baseline_card_health else ''} |",
                f"| median_ms | {timing.get('median')} | {baseline_timing.get('median', '')} |",
                f"| p90_ms | {timing.get('p90')} | {baseline_timing.get('p90', '')} |",
                f"| max_ms | {timing.get('max')} | {baseline_timing.get('max', '')} |",
            ]
        )

    diff = summary.get("diff") or {}
    diff_counts = diff.get("counts") or {}
    lines.extend(
        [
            "",
            "## Review Diff",
            "",
            f"- Matched rows: `{diff.get('matched_row_count')}`",
            f"- Missing rows: `{diff.get('missing_in_candidate_count')}`",
            f"- Extra rows: `{diff.get('extra_in_candidate_count')}`",
            f"- Slots: `{diff_counts.get('slot_count')}`",
            f"- Changed: `{diff_counts.get('changed_count')}`",
            f"- Risk: `{diff_counts.get('risk_count')}`",
            "",
            "## Risky Examples",
            "",
        ]
    )
    risky = diff.get("risky_examples") or []
    if not risky:
        lines.append("No risky examples detected.")
    else:
        lines.extend(["| Video | Time | Slot | Baseline | Candidate | Risk |", "|---|---:|---:|---|---|---|"])
        for row in risky:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("video") or ""),
                        str(row.get("timestamp_sec") or ""),
                        str(row.get("slot") or ""),
                        str(row.get("baseline_card") or ""),
                        str(row.get("candidate_card") or ""),
                        str(row.get("risk_reason") or ""),
                    ]
                )
                + " |"
            )
    return "\n".join(lines).rstrip() + "\n"
