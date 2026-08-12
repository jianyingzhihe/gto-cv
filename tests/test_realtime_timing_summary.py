from __future__ import annotations

import unittest

from gto_cli.live_vision import event_source_timing_summary


class RealtimeTimingSummaryTest(unittest.TestCase):
    def test_event_source_timing_summary_counts_cache_and_components(self) -> None:
        events = [
            {
                "ok": True,
                "source": {
                    "analysis_ms": 10.0,
                    "cards_hint_used": False,
                    "card_cache_hit": False,
                    "ocr_mode": "action_only_skipped",
                    "cv_timing_ms": {"cards_ms": 7.0, "ocr_ms": 0.0},
                    "screen_timing_ms": {"action_controls_ocr_ms": 5.0},
                },
            },
            {
                "ok": True,
                "source": {
                    "analysis_ms": 4.0,
                    "cards_hint_used": True,
                    "card_cache_hit": True,
                    "ocr_mode": "action_only_used",
                    "cv_timing_ms": {"cards_ms": 1.0, "ocr_ms": 3.0},
                    "screen_timing_ms": {"action_controls_ocr_ms": 1.0},
                },
            },
            {"ok": False, "source": {"analysis_ms": 999.0}},
        ]

        summary = event_source_timing_summary(events)

        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["cards_hint_used"], 1)
        self.assertEqual(summary["card_cache_hit"], 1)
        self.assertEqual(summary["ocr_modes"], {"action_only_skipped": 1, "action_only_used": 1})
        self.assertEqual(summary["analysis_ms"]["count"], 2)
        self.assertEqual(summary["cv_timing_ms"]["cards_ms"]["median"], 4.0)
        self.assertEqual(summary["screen_timing_ms"]["action_controls_ocr_ms"]["median"], 3.0)


if __name__ == "__main__":
    unittest.main()
