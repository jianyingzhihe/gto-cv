from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summarize_glyph_queue(queue_csv: Path) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    with queue_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        all_rows = [dict(row) for row in csv.DictReader(stream)]
    rows = [row for row in all_rows if not glyph_row_is_ignored(row)]

    groups: dict[str, dict[str, int]] = {}
    mismatches = []
    for kind in ("rank", "suit"):
        for group in ("hero", "board"):
            selected = [
                row
                for row in rows
                if row.get("kind") == kind
                and is_board_row(row) == (group == "board")
            ]
            labeled = [row for row in selected if str(row.get("final_label") or "").strip()]
            correct = [
                row
                for row in labeled
                if normalized_label(row.get("current_label")) == normalized_label(row.get("final_label"))
            ]
            groups[f"{kind}_{group}"] = {
                "total": len(selected),
                "labeled": len(labeled),
                "correct": len(correct),
                "errors": len(labeled) - len(correct),
            }

    for row in rows:
        final_label = normalized_label(row.get("final_label"))
        if not final_label or normalized_label(row.get("current_label")) == final_label:
            continue
        input_path = Path(str(row.get("input_path") or ""))
        mismatches.append(
            {
                "label_id": str(row.get("label_id") or ""),
                "kind": str(row.get("kind") or ""),
                "group": "board" if is_board_row(row) else "hero",
                "current": normalized_label(row.get("current_label")) or "?",
                "truth": final_label,
                "confidence": str(row.get("current_confidence") or ""),
                "margin": str(row.get("current_margin") or ""),
                "sample": input_path.parent.name,
                "input_path": str(input_path),
                "reason": str(row.get("reason") or ""),
            }
        )

    return {
        "rows": len(rows),
        "ignored": len(all_rows) - len(rows),
        "raw_rows": len(all_rows),
        "labeled": sum(bool(normalized_label(row.get("final_label"))) for row in rows),
        "unlabeled": sum(not bool(normalized_label(row.get("final_label"))) for row in rows),
        "groups": groups,
        "mismatches": mismatches,
    }


def is_board_row(row: dict[str, Any]) -> bool:
    return "board_" in Path(str(row.get("input_path") or "")).name


def glyph_row_is_ignored(row: dict[str, Any]) -> bool:
    return str(row.get("ignored") or "").strip().lower() in {"1", "true", "yes", "ignored"}


def normalized_label(value: Any) -> str:
    return str(value or "").strip()


def render_error_report(
    *,
    ledger: dict[str, Any],
    queue_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    output_path: Path,
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.now().astimezone()
    expected_samples = int((ledger.get("dataset") or {}).get("expected_samples") or 0)
    replayed_samples = int(replay_summary.get("source_samples") or 0)
    error_count = len(queue_summary.get("mismatches") or [])
    fully_labeled = int(queue_summary.get("labeled") or 0) == int(queue_summary.get("rows") or 0)
    replay_complete = expected_samples > 0 and replayed_samples == expected_samples
    tests_text = str((ledger.get("last_regression") or {}).get("tests") or "")
    tests_pass = "passed" in tests_text.lower()
    gate_pass = fully_labeled and error_count == 0 and replay_complete and tests_pass

    lines = [
        "# 扑克 CV 识别错误账本",
        "",
        f"- 生成时间：`{generated_at.isoformat(timespec='seconds')}`",
        f"- 数据集：`{(ledger.get('dataset') or {}).get('name') or '-'}`",
        f"- 版本闸门：`{'通过' if gate_pass else '阻塞'}`",
        "",
        "## 零错误验收闸门",
        "",
        checklist(fully_labeled, f"人工校对完成：{queue_summary['labeled']}/{queue_summary['rows']} 个字形"),
        checklist(error_count == 0, f"已校对样本中的错误数：{error_count}"),
        checklist(replay_complete, f"全帧重放：{replayed_samples}/{expected_samples} 个源帧"),
        checklist(tests_pass, f"自动化测试：{tests_text or '未记录'}"),
        "",
        "> 只有全部字形完成校对、已校对结果零错误、最后一次修复后重新跑完全部帧，",
        "> 并且自动化测试通过，才允许把这一版标记为通过。",
        "",
        "## 当前校对状态",
        "",
        "| 组件 | 已校对 | 正确 | 错误 | 总数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("rank_hero", "手牌数字/字母"),
        ("suit_hero", "手牌花色"),
        ("rank_board", "公共牌数字/字母"),
        ("suit_board", "公共牌花色"),
    ):
        item = (queue_summary.get("groups") or {}).get(key) or {}
        lines.append(
            f"| {label} | {item.get('labeled', 0)} | {item.get('correct', 0)} | "
            f"{item.get('errors', 0)} | {item.get('total', 0)} |"
        )

    lines.extend(["", "## 已校对但仍错误的样本", ""])
    mismatches = queue_summary.get("mismatches") or []
    if mismatches:
        lines.extend(
            [
                "| ID | 区域 | 当前输出 | 人工真值 | 置信度 / margin | 样本 | 原因 |",
                "|---|---|---:|---:|---|---|---|",
            ]
        )
        for row in mismatches:
            sample_link = markdown_link(
                row["sample"],
                Path(row["input_path"]),
                output_path=output_path,
            )
            lines.append(
                f"| {row['label_id']} | {row['group']} {row['kind']} | `{row['current']}` | "
                f"`{row['truth']}` | `{row['confidence']} / {row['margin']}` | "
                f"{sample_link} | `{row['reason']}` |"
            )
    else:
        lines.append("已校对样本中没有错误。")

    lines.extend(["", "## 错误案例", ""])
    for case in ledger.get("cases") or []:
        status = "已修复" if str(case.get("status") or "open").lower() == "fixed" else "未解决"
        lines.extend(
            [
                f"### {case.get('id')} - {case.get('title')} [{status}]",
                "",
                f"- 区域：`{case.get('area') or '-'}`",
                f"- 样本：`{case.get('sample') or '-'}`",
                f"- 人工真值：`{case.get('truth') or '-'}`",
                f"- 修复前输出：`{case.get('previous_output') or '-'}`",
                f"- 根因：{case.get('root_cause') or '-'}",
                f"- 修复：{case.get('correction') or '-'}",
                f"- 回归证据：{case.get('regression') or '-'}",
                "",
            ]
        )
        for label, key in (("修复前", "before_image"), ("修复后", "after_image"), ("修复后字形", "after_glyph")):
            image_path = str(case.get(key) or "").strip()
            if image_path:
                lines.extend(
                    [
                        f"**{label}**",
                        "",
                        f"![{case.get('id')} {label}]({relative_asset(image_path, output_path)})",
                        "",
                    ]
                )

    replay_command = str((ledger.get("commands") or {}).get("full_replay") or "")
    report_command = str((ledger.get("commands") or {}).get("update_report") or "")
    review_url = str((ledger.get("commands") or {}).get("review_url") or "")
    lines.extend(
        [
            "## 每轮迭代流程",
            "",
            "1. 在校对页面填写真实数字、字母或花色。",
            "2. 在 `cases.json` 新增或更新一个案例，记录例图、错误输出和人工真值。",
            "3. 判断故障阶段：牌桌/槽位定位、裁剪、二值化、分类器、时序稳定或状态门控。",
            "4. 修复时同时加入针对该例的回归测试。",
            "5. 先运行聚焦测试，确认该例修复且相邻案例没有退化。",
            "6. 重新识别全部源帧，并按物理样本和槽位保留已有人工标签。",
            "7. 根据新队列和重放摘要重新生成本报告。",
            "8. 继续人工校对；发现错误后重复以上流程，直到零错误闸门通过。",
            "",
            "## 命令",
            "",
            "全帧重新识别：",
            "",
            "```powershell",
            replay_command,
            "```",
            "",
            "更新本报告：",
            "",
            "```powershell",
            report_command,
            "```",
            "",
            f"人工校对页面：{review_url or '-'}",
            "",
        ]
    )
    return "\n".join(lines)


def checklist(done: bool, text: str) -> str:
    return f"- [{'x' if done else ' '}] {text}"


def relative_asset(path_text: str, output_path: Path) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.relpath(path, output_path.parent)).as_posix()


def markdown_link(label: str, path: Path, *, output_path: Path) -> str:
    return f"[{label}]({relative_asset(str(path), output_path)})"


def build_report(
    *,
    ledger_path: Path,
    queue_csv: Path,
    replay_summary_json: Path,
    output_path: Path,
) -> dict[str, Any]:
    ledger = load_json(ledger_path)
    queue_summary = summarize_glyph_queue(queue_csv)
    replay_summary = load_json(replay_summary_json)
    report = render_error_report(
        ledger=ledger,
        queue_summary=queue_summary,
        replay_summary=replay_summary,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "ok": True,
        "output": str(output_path),
        "rows": queue_summary["rows"],
        "labeled": queue_summary["labeled"],
        "mismatches": len(queue_summary["mismatches"]),
        "source_samples": replay_summary.get("source_samples"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the persistent poker CV error ledger report.")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(
        ledger_path=args.ledger,
        queue_csv=args.queue_csv,
        replay_summary_json=args.replay_summary,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
