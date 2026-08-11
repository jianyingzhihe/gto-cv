from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_candidate_summary import summarize_card_candidates


class CardCandidateSummaryTest(unittest.TestCase):
    def test_summary_ranks_low_risk_candidate_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good" / "card_model_gate_summary.json"
            bad = root / "bad" / "card_model_gate_summary.json"
            good.parent.mkdir()
            bad.parent.mkdir()
            write_gate_summary(good, name="good", risk=0, card_acc=1.0, promote=True)
            write_gate_summary(bad, name="bad", risk=3, card_acc=1.0, promote=False)

            payload = summarize_card_candidates(search_dir=root, output_dir=root / "summary")

            self.assertEqual(payload["candidate_count"], 2)
            self.assertEqual(payload["promote_count"], 1)
            self.assertEqual(payload["rows"][0]["candidate_name"], "good")
            self.assertTrue(Path(payload["files"]["summary_csv"]).exists())


def write_gate_summary(path: Path, *, name: str, risk: int, card_acc: float, promote: bool) -> None:
    payload = {
        "candidate_name": name,
        "candidate_evaluator": "knn",
        "decision": "promote" if promote else "reject",
        "promote": promote,
        "benchmark": {
            "evaluators": {
                "knn": {
                    "card_acc": card_acc,
                    "rank_acc": 1.0,
                    "suit_acc": 1.0,
                }
            }
        },
        "diff": {
            "missing_in_candidate_count": 0,
            "counts": {
                "slot_count": 10,
                "changed_count": risk,
                "risk_count": risk,
            },
        },
        "candidate_validation": {
            "real_problem_count": 0,
            "board_bad_count": 0,
            "timing_ms": {"median": 100.0, "p90": 200.0},
        },
        "checks": [{"name": "review_diff_risk", "pass": risk == 0}],
        "files": {"report_md": str(path.with_suffix(".md"))},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
