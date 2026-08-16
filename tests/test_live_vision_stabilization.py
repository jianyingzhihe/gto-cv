from __future__ import annotations

import unittest
from pathlib import Path

from gto_cli.live_vision import build_realtime_state, stabilize_hero_cards


def make_state(
    cards: list[str],
    timestamp: float,
    *,
    board: list[str] | None = None,
    street: str = "preflop",
    pot_bb: float | None = 3.4,
    position: str = "HJ",
    dealer_seat: str = "left",
    has_cards: bool = True,
    status: str = "active_or_showdown",
    rank_confidence: float = 0.90,
    suit_confidence: float = 0.95,
) -> dict:
    return {
        "ok": True,
        "source": {"timestamp_sec": timestamp},
        "hero": {
            "cards": list(cards),
            "has_cards": has_cards,
            "status": status,
            "seat": "bottom_hero",
            "position": position,
            "gto_position": position,
        },
        "table": {
            "board": list(board or []),
            "street": street,
            "pot_bb": pot_bb,
            "dealer_position": "BTN",
            "dealer_seat": dealer_seat,
        },
        "confidence": {
            "cards": {
                "hero": [
                    {
                        "card": card,
                        "rank_confidence": rank_confidence,
                        "suit_confidence": suit_confidence,
                    }
                    for card in cards
                ]
            }
        },
    }


class LiveVisionCardStabilizationTest(unittest.TestCase):
    def test_locks_two_matching_reads_against_later_complete_misread(self) -> None:
        cache = None
        for timestamp in (1.0, 2.0):
            state = make_state(["Qs", "5h"], timestamp)
            cache = stabilize_hero_cards(state, cache)

        wrong = make_state(["8s", "5h"], 3.0)
        cache = stabilize_hero_cards(wrong, cache)

        self.assertEqual(wrong["hero"]["cards"], ["Qs", "5h"])
        self.assertTrue(wrong["source"]["hero_cards_stabilized"])
        self.assertEqual(wrong["source"]["hero_cards_raw"], ["8s", "5h"])
        self.assertTrue(cache["confirmed"])

    def test_keeps_locked_hole_cards_when_board_appears(self) -> None:
        cache = None
        for timestamp in (10.0, 11.0):
            state = make_state(["Ah", "Js"], timestamp)
            cache = stabilize_hero_cards(state, cache)

        flop_misread = make_state(["4h", "Js"], 12.0, board=["9d", "Tc", "2h"])
        stabilize_hero_cards(flop_misread, cache)

        self.assertEqual(flop_misread["hero"]["cards"], ["Ah", "Js"])
        self.assertTrue(flop_misread["source"]["hero_cards_stabilized"])

    def test_empty_between_hands_unlocks_even_when_position_repeats(self) -> None:
        cache = None
        for timestamp in (20.0, 21.0):
            state = make_state(["7h", "7d"], timestamp)
            cache = stabilize_hero_cards(state, cache)

        empty = make_state([], 22.0, has_cards=False, status="folded_or_empty")
        cache = stabilize_hero_cards(empty, cache)
        first_new = make_state(["Ac", "9s"], 23.0)
        cache = stabilize_hero_cards(first_new, cache)
        second_new = make_state(["Ac", "9s"], 24.0)
        cache = stabilize_hero_cards(second_new, cache)

        self.assertEqual(first_new["hero"]["cards"], ["Ac", "9s"])
        self.assertEqual(second_new["hero"]["cards"], ["Ac", "9s"])
        self.assertEqual(cache["cards"], ["Ac", "9s"])
        self.assertTrue(cache["confirmed"])

    def test_position_change_starts_new_hand_without_waiting_for_old_cache(self) -> None:
        cache = None
        for timestamp in (30.0, 31.0):
            state = make_state(["Kd", "Jd"], timestamp, position="BB")
            cache = stabilize_hero_cards(state, cache)

        new_hand = make_state(["9d", "4d"], 32.0, position="UTG")
        cache = stabilize_hero_cards(new_hand, cache)

        self.assertEqual(new_hand["hero"]["cards"], ["9d", "4d"])
        self.assertEqual(cache["cards"], ["9d", "4d"])
        self.assertFalse(cache["confirmed"])

    def test_keeps_first_candidate_during_preconfirmation_tie(self) -> None:
        cache = None
        first = make_state(["Qd", "4h"], 40.0)
        cache = stabilize_hero_cards(first, cache)

        one_frame_misread = make_state(["8d", "4h"], 41.0)
        cache = stabilize_hero_cards(one_frame_misread, cache)
        final = make_state(["Qd", "4h"], 42.0)
        cache = stabilize_hero_cards(final, cache)

        self.assertEqual(one_frame_misread["hero"]["cards"], ["Qd", "4h"])
        self.assertEqual(one_frame_misread["source"]["hero_cards_raw"], ["8d", "4h"])
        self.assertEqual(
            one_frame_misread["source"]["hero_cards_stabilization_reason"],
            "candidate_hysteresis",
        )
        self.assertEqual(final["hero"]["cards"], ["Qd", "4h"])
        self.assertTrue(cache["confirmed"])

    def test_postflop_to_preflop_reset_unlocks_same_position_and_dealer(self) -> None:
        cache = None
        for timestamp in (50.0, 51.0):
            state = make_state(["As", "8s"], timestamp)
            cache = stabilize_hero_cards(state, cache)
        flop = make_state(
            ["As", "8s"],
            52.0,
            board=["Ah", "5d", "9h"],
            street="flop",
            pot_bb=14.2,
        )
        cache = stabilize_hero_cards(flop, cache)

        next_hand = make_state(["9h", "8d"], 53.0)
        cache = stabilize_hero_cards(next_hand, cache)

        self.assertEqual(next_hand["hero"]["cards"], ["9h", "8d"])
        self.assertEqual(cache["cards"], ["9h", "8d"])
        self.assertFalse(cache["confirmed"])

    def test_new_hand_partial_card_does_not_fill_from_postflop_cache(self) -> None:
        cache = None
        for timestamp in (60.0, 61.0):
            state = make_state(["As", "8s"], timestamp)
            cache = stabilize_hero_cards(state, cache)
        flop = make_state(
            ["As", "8s"],
            62.0,
            board=["Ah", "5d", "9h"],
            street="flop",
            pot_bb=14.2,
        )
        cache = stabilize_hero_cards(flop, cache)

        partial = make_state(["K?"], 63.0)
        cache = stabilize_hero_cards(partial, cache)

        self.assertEqual(partial["hero"]["cards"], ["K?"])
        self.assertIsNone(cache)

    def test_opening_pot_and_dealer_move_unlocks_preflop_only_hand(self) -> None:
        cache = None
        for timestamp in (70.0, 71.0):
            state = make_state(["Ad", "Jh"], timestamp, pot_bb=9.0)
            cache = stabilize_hero_cards(state, cache)

        next_hand = make_state(
            ["5c", "2h"],
            72.0,
            pot_bb=3.4,
            dealer_seat="top_left",
        )
        cache = stabilize_hero_cards(next_hand, cache)

        self.assertEqual(next_hand["hero"]["cards"], ["5c", "2h"])
        self.assertEqual(cache["cards"], ["5c", "2h"])
        self.assertFalse(cache["confirmed"])


def test_visible_check_without_call_overrides_stale_bet_difference() -> None:
    state = build_realtime_state(
        {
            "hero": {
                "seat_index": 0,
                "seat": "bottom_hero",
                "position": "BTN",
                "cards": ["As", "Kd"],
                "has_cards": True,
                "bet_bb": 13.6,
            },
            "dealer": {},
            "cards": {"board": ["2c", "7d", "Th"], "hero_details": [], "board_details": []},
            "pot": {"amount_bb": 170.0},
            "seats": [
                {"index": 0, "name": "bottom_hero", "bet_bb": 13.6},
                {"index": 1, "name": "opponent", "bet_bb": 185.0},
            ],
            "action_controls": {
                "visible": True,
                "actions": ["check", "bet"],
                "red_button_regions": [{"x": 1, "y": 1, "width": 10, "height": 10}],
            },
        },
        Path("capture.mp4"),
        timestamp_sec=1.0,
        frame_index=1,
        sample_index=1,
    )

    assert state["table"]["to_call_bb"] == 0.0


if __name__ == "__main__":
    unittest.main()
