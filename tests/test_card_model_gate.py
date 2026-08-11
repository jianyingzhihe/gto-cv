import unittest

from gto_cli.card_model_gate import build_validation_checks, compact_validation, format_card_model_gate_summary


class CardModelGateTest(unittest.TestCase):
    def test_validation_latency_limits_fail_slow_candidate(self) -> None:
        checks = build_validation_checks(
            candidate_validation={
                "ok": True,
                "real_problem_count": 0,
                "board_bad_count": 0,
                "timing_ms": {"median": 962.8, "p90": 1967.1},
            },
            baseline_validation={
                "ok": True,
                "real_problem_count": 0,
                "board_bad_count": 0,
                "timing_ms": {"median": 189.6, "p90": 665.0},
            },
            require_validation=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300.0,
            max_p90_ms=900.0,
            max_median_regression_ms=None,
            max_p90_regression_ms=None,
        )

        checks_by_name = {check["name"]: check for check in checks}
        self.assertFalse(checks_by_name["validation_median_ms"]["pass"])
        self.assertFalse(checks_by_name["validation_p90_ms"]["pass"])

    def test_validation_latency_limits_pass_current_baseline_speed(self) -> None:
        checks = build_validation_checks(
            candidate_validation={
                "ok": True,
                "real_problem_count": 0,
                "board_bad_count": 0,
                "timing_ms": {"median": 189.6, "p90": 665.0},
            },
            baseline_validation=None,
            require_validation=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300.0,
            max_p90_ms=900.0,
            max_median_regression_ms=None,
            max_p90_regression_ms=None,
        )

        checks_by_name = {check["name"]: check for check in checks}
        self.assertTrue(checks_by_name["validation_median_ms"]["pass"])
        self.assertTrue(checks_by_name["validation_p90_ms"]["pass"])

    def test_validation_card_health_fails_hidden_card_issues(self) -> None:
        checks = build_validation_checks(
            candidate_validation={
                "ok": True,
                "real_problem_count": 0,
                "board_bad_count": 0,
                "timing_ms": {"median": 40, "p90": 80},
                "card_health": {
                    "hero": {
                        "complete_frames": 9,
                        "incomplete_or_missed_frames": 1,
                        "turn_blocked_frames": 1,
                    },
                    "board": {"bad_frames": 0},
                    "issue_counts": {"hero_card_unknown": 1, "hero_turn_cards_not_ready": 1},
                },
            },
            baseline_validation=None,
            require_validation=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300.0,
            max_p90_ms=900.0,
            max_median_regression_ms=None,
            max_p90_regression_ms=None,
        )

        checks_by_name = {check["name"]: check for check in checks}
        self.assertFalse(checks_by_name["validation_hero_incomplete_or_missed"]["pass"])
        self.assertFalse(checks_by_name["validation_hero_turn_blocked"]["pass"])
        self.assertFalse(checks_by_name["validation_card_issue_count"]["pass"])

    def test_validation_card_health_passes_clean_summary_and_is_reported(self) -> None:
        validation = {
            "ok": True,
            "real_problem_count": 0,
            "board_bad_count": 0,
            "timing_ms": {"median": 40, "p90": 80},
            "card_health": {
                "hero": {
                    "complete_frames": 9,
                    "incomplete_or_missed_frames": 0,
                    "turn_blocked_frames": 0,
                },
                "board": {"bad_frames": 0},
                "issue_counts": {},
            },
        }
        checks = build_validation_checks(
            candidate_validation=validation,
            baseline_validation=None,
            require_validation=False,
            max_real_problem=0,
            max_board_bad=0,
            max_median_ms=300.0,
            max_p90_ms=900.0,
            max_median_regression_ms=None,
            max_p90_regression_ms=None,
        )
        payload = {
            "candidate_name": "candidate",
            "candidate_evaluator": "knn",
            "decision": "promote",
            "promote": True,
            "benchmark": {"sample_count": 1},
            "diff": {"counts": {}},
            "candidate_validation": compact_validation(validation),
            "checks": checks,
            "files": {},
        }
        checks_by_name = {check["name"]: check for check in checks}
        text = format_card_model_gate_summary(payload)

        self.assertTrue(checks_by_name["validation_card_issue_count"]["pass"])
        self.assertEqual(payload["candidate_validation"]["card_health"]["hero"]["complete_frames"], 9)
        self.assertIn("Card health: hero_complete=9", text)


if __name__ == "__main__":
    unittest.main()
