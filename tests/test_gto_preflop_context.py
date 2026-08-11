from __future__ import annotations

from gto_cli.advisor import advise_state
from gto_cli.cli import format_text
from gto_cli.cv_advisor import build_gto_advice
from gto_cli.preflop_context import build_preflop_context


def test_unopened_history_is_rfi_even_when_blinds_create_a_to_call_value() -> None:
    context = build_preflop_context(
        {
            "hero": {"position": "UTG+1", "gto_position": "UTG"},
            "table": {"to_call_bb": 2},
            "preflop": {"action_history": [{"position": "UTG", "action": "fold"}]},
        }
    )

    assert context["status"] == "unopened"
    assert context["scenario"] == "rfi"
    assert context["solver_position"] == "UTG"
    assert context["supported"]


def test_action_history_distinguishes_open_and_three_bet() -> None:
    facing_open = build_preflop_context(
        {
            "hero": {"position": "CO"},
            "table": {"to_call_bb": 2.5},
            "preflop": {"action_history": [{"position": "HJ", "action": "raise", "amount_bb": 2.5}]},
        }
    )
    facing_three_bet = build_preflop_context(
        {
            "hero": {"position": "CO"},
            "table": {"to_call_bb": 7.5},
            "preflop": {
                "action_history": [
                    {"position": "CO", "action": "raise", "is_hero": True},
                    {"position": "BTN", "action": "3bet", "amount_bb": 9},
                ]
            },
        }
    )

    assert facing_open["scenario"] == "vs_open"
    assert facing_open["aggressor_position"] == "HJ"
    assert facing_three_bet["scenario"] == "vs_3bet"
    assert facing_three_bet["aggressor_position"] == "BTN"


def test_missing_action_history_returns_wait_not_a_false_fold() -> None:
    result = advise_state(
        {
            "hero": {"cards": ["7s", "2d"], "position": "UTG+1", "gto_position": "UTG"},
            "table": {"pot_bb": 3.4, "to_call_bb": 2, "effective_stack_bb": 100},
        }
    )

    assert result["ok"]
    assert result["decision"]["primary_action"] == "wait"
    assert result["preflop_context"]["status"] == "unknown"
    assert result["preflop_context"]["scenario"] is None


def test_cv_adapter_does_not_infer_open_from_visible_blinds() -> None:
    advice = build_gto_advice(
        {
            "ok": True,
            "action_controls": {"visible": True, "call_amount_bb": 2},
            "hero": {"cards": ["7s", "2d"], "position": "UTG+1", "gto_position": "UTG"},
            "table": {"street": "preflop", "pot_bb": 3.4, "to_call_bb": 2, "board": []},
            "bets": [
                {"seat": "bottom_left", "amount_bb": 0.4},
                {"seat": "left", "amount_bb": 1},
                {"seat": "top_left", "amount_bb": 2},
            ],
        }
    )

    assert not advice["ready"]
    assert advice["reason"] == "preflop_context_incomplete"
    assert advice["preflop_context"]["status"] == "unknown"


def test_text_output_uses_direct_preflop_action_names() -> None:
    result = advise_state(
        {
            "hero": {"cards": ["As", "Ks"], "position": "UTG+1", "gto_position": "UTG"},
            "table": {"pot_bb": 1.5, "to_call_bb": 0, "effective_stack_bb": 100},
            "preflop": {"action_history": [{"position": "UTG", "action": "fold"}]},
        }
    )

    text = format_text(result)
    assert "UTG+1 -> 策略桶 UTG" in text
    assert "前位无人加注" in text
    assert "OPEN RAISE TO 2.2 BB" in text
