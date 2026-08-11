from __future__ import annotations

import unittest

from gto_cli.screen_vision import auto_bbox_preserves_hero_cards, should_accept_bbox_refresh


class AutoBboxRefreshTest(unittest.TestCase):
    def test_rejects_strong_anchor_shrink_from_full_window(self) -> None:
        current = {"left": 115, "top": 155, "width": 1545, "height": 1092}
        smaller = {"left": 224, "top": 178, "width": 1354, "height": 957}
        search = {"left": 0, "top": 0, "width": 1920, "height": 1200}
        detection = {"method": "dealer-button-anchor", "dealer_confidence": 0.98}

        accepted, reason = should_accept_bbox_refresh(current, smaller, search, detection)

        self.assertFalse(accepted)
        self.assertEqual(reason, "strong_anchor_shrink")

    def test_accepts_strong_anchor_when_not_shrinking(self) -> None:
        current = {"left": 224, "top": 178, "width": 1354, "height": 957}
        similar = {"left": 226, "top": 180, "width": 1354, "height": 957}
        search = {"left": 0, "top": 0, "width": 1920, "height": 1200}
        detection = {"method": "dealer-button-anchor", "dealer_confidence": 0.98}

        accepted, reason = should_accept_bbox_refresh(current, similar, search, detection)

        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_rejects_auto_inner_bbox_that_cuts_hero_card_footer(self) -> None:
        capture = {"left": 246, "top": 312, "width": 1371, "height": 960}
        too_shallow = {"left": 370, "top": 479, "width": 1125, "height": 614}

        accepted, reason = auto_bbox_preserves_hero_cards(too_shallow, capture)

        self.assertFalse(accepted)
        self.assertEqual(reason, "inner_table_cuts_hero_cards")

    def test_accepts_inner_bbox_that_keeps_hero_card_footer(self) -> None:
        capture = {"left": 246, "top": 312, "width": 1371, "height": 960}
        card_safe = {"left": 342, "top": 332, "width": 1227, "height": 867}

        accepted, reason = auto_bbox_preserves_hero_cards(card_safe, capture)

        self.assertTrue(accepted)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
