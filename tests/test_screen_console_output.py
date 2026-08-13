from __future__ import annotations

import copy
import unittest

from gto_cli.screen_vision import format_screen_advice_line, screen_advice_console_signature


def make_state(*, ready: bool, reason: str = "") -> dict:
    advice = {
        "ready": ready,
        "should_act": ready,
    }
    if ready:
        advice.update(
            {
                "action": "3bet",
                "amount_bb": 8.5,
                "summary": "3BET 8.5 BB (3bet 80% / call 20%)",
            }
        )
    else:
        advice["reason"] = reason
    return {
        "ok": True,
        "source": {"timestamp_sec": 12.345, "analysis_ms": 418.2},
        "event": {"index": 7},
        "table": {
            "street": "preflop",
            "pot_bb": 3.4,
            "to_call_bb": 2.0,
            "board": [],
        },
        "hero": {
            "position": "CO",
            "gto_position": "CO",
            "cards": ["Ah", "Kd"],
            "status": "active_or_showdown",
        },
        "hero_turn": {
            "is_turn": ready,
            "reason": "red_buttons_and_action_text" if ready else "action_controls_not_visible",
        },
        "gto_advice": advice,
        "bets": [{"seat": "left", "amount_bb": 2.0}],
        "seats": [{"seat": "left", "position": "BB"}],
    }


class ScreenConsoleOutputTest(unittest.TestCase):
    def test_ready_line_focuses_on_advice_and_minimal_context(self) -> None:
        line = format_screen_advice_line(make_state(ready=True))

        self.assertIn("ADVICE | 3BET 8.5 BB", line)
        self.assertIn("hero=CO AhKd", line)
        self.assertIn("pot=3.4BB call=2BB", line)
        self.assertNotIn("bets=", line)
        self.assertNotIn("seats=", line)
        self.assertNotIn("dealer=", line)
        self.assertNotIn("turn_reason=", line)

    def test_wait_line_exposes_only_the_blocking_reason(self) -> None:
        line = format_screen_advice_line(
            make_state(ready=False, reason="hero_action_controls_not_visible")
        )

        self.assertIn("WATCH | wait=hero_action_controls_not_visible", line)
        self.assertIn("hero=CO AhKd", line)

    def test_turn_line_prefers_the_visible_call_button_amount(self) -> None:
        state = make_state(ready=False, reason="preflop_scenario_not_supported")
        state["hero_turn"] = {"is_turn": True, "reason": "red_buttons_and_action_text"}
        state["action_controls"] = {"visible": True, "actions": ["fold", "call"], "call_amount_bb": 48.7}
        state["table"]["to_call_bb"] = 0.0

        line = format_screen_advice_line(state)

        self.assertIn("call=48.7BB", line)

    def test_signature_ignores_bet_noise_but_tracks_advice_context(self) -> None:
        state = make_state(ready=False, reason="hero_action_controls_not_visible")
        original = screen_advice_console_signature(state)

        bet_change = copy.deepcopy(state)
        bet_change["bets"][0]["amount_bb"] = 9.4
        self.assertEqual(original, screen_advice_console_signature(bet_change))

        preflop_bet_change = copy.deepcopy(state)
        preflop_bet_change["seats"][0]["bet_bb"] = 2.2
        self.assertNotEqual(original, screen_advice_console_signature(preflop_bet_change))

        card_change = copy.deepcopy(state)
        card_change["hero"]["cards"] = ["Qs", "Jh"]
        self.assertNotEqual(original, screen_advice_console_signature(card_change))

        advice_change = make_state(ready=True)
        self.assertNotEqual(original, screen_advice_console_signature(advice_change))

    def test_preflop_line_shows_position_history_and_visible_bets(self) -> None:
        state = make_state(ready=True)
        state["hero"].update({"position": "UTG+1", "gto_position": "UTG", "preflop_action_order": 2})
        state["gto_advice"]["preflop_context"] = {
            "scenario": "vs_open",
            "raw_position": "UTG+1",
            "solver_position": "UTG",
            "preflop_action_order": 2,
            "actions_before_hero": [{"position": "UTG", "action": "raise", "amount_bb": 2.0}],
        }
        state["seats"] = [
            {"seat": "bottom_left", "position": "SB", "bet_bb": 0.4},
            {"seat": "left", "position": "BB", "bet_bb": 1.0},
            {"seat": "top_left", "position": "UTG", "bet_bb": 2.0},
        ]

        line = format_screen_advice_line(state)

        self.assertIn("PF=vs_open", line)
        self.assertIn("pos=UTG+1->UTG", line)
        self.assertIn("order=#2", line)
        self.assertIn("history=UTG R2BB", line)
        self.assertIn("visible=SB:0.4BB,BB:1BB,UTG:2BB", line)


if __name__ == "__main__":
    unittest.main()
