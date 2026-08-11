import unittest

from gto_cli.card_review_diff import classify_risk, classify_status


class CardReviewDiffTest(unittest.TestCase):
    def test_improved_with_truth_is_not_risk(self) -> None:
        risk, reason = classify_risk(
            status="improved",
            baseline_card="6h",
            candidate_card="4h",
            baseline_reason="ok",
            candidate_reason="ok",
            risky_baseline_reasons=("ok",),
        )

        self.assertFalse(risk)
        self.assertEqual(reason, "")

    def test_same_correct_with_truth_is_not_downgrade_risk(self) -> None:
        risk, reason = classify_risk(
            status="same_correct",
            baseline_card="Jd",
            candidate_card="Jd",
            baseline_reason="ok",
            candidate_reason="rank_low",
            risky_baseline_reasons=("ok",),
        )

        self.assertFalse(risk)
        self.assertEqual(reason, "")

    def test_same_both_wrong_with_truth_is_risk(self) -> None:
        status = classify_status(
            baseline_card="3h",
            candidate_card="3h",
            truth_card="8h",
            baseline_ok=False,
            candidate_ok=False,
            baseline_reason="ok",
            candidate_reason="ok",
            risky_baseline_reasons=("ok",),
        )
        risk, reason = classify_risk(
            status=status,
            baseline_card="3h",
            candidate_card="3h",
            baseline_reason="ok",
            candidate_reason="ok",
            risky_baseline_reasons=("ok",),
        )

        self.assertEqual(status, "same_both_wrong")
        self.assertTrue(risk)
        self.assertEqual(reason, "manual_truth_same_wrong")


if __name__ == "__main__":
    unittest.main()
