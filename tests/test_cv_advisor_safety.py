from __future__ import annotations

from gto_cli.cv_advisor import action_to_call, build_gto_advice, build_size_mix, format_advice_summary


def postflop_state(*, pot_bb: float | None) -> dict:
    return {
        "ok": True,
        "action_controls": {
            "visible": True,
            "actions": ["fold", "call", "raise"],
            "call_amount_bb": 210.0,
            "raise_amount_bb": 15.8,
        },
        "hero": {"cards": ["As", "Kd"], "position": "CO", "gto_position": "CO"},
        "table": {
            "street": "flop",
            "pot_bb": pot_bb,
            "to_call_bb": 5.6,
            "board": ["2c", "7d", "Th"],
        },
    }


def test_postflop_waits_without_pot_instead_of_emitting_a_capped_raise() -> None:
    advice = build_gto_advice(postflop_state(pot_bb=None), effective_stack_bb=100)

    assert not advice["ready"]
    assert advice["reason"] == "pot_amount_unavailable"


def test_call_amount_larger_than_effective_stack_is_rejected() -> None:
    amount, source = action_to_call(
        {"call_amount_bb": 210.0},
        {"to_call_bb": 5.6},
        effective_stack_bb=100,
    )

    assert amount == 5.6
    assert source == "table_bets_rejected_control_amount"


def test_postflop_advice_uses_visible_bets_when_control_amount_is_a_stack() -> None:
    advice = build_gto_advice(postflop_state(pot_bb=12.0), effective_stack_bb=100)

    assert advice["ready"]
    assert advice["input"]["table"]["to_call_bb"] == 5.6
    assert advice["input"]["action"]["call_amount_source"] == "table_bets_rejected_control_amount"


def test_fast_fold_only_does_not_request_normal_gto_advice() -> None:
    state = postflop_state(pot_bb=12.0)
    state["hero_turn"] = {"is_turn": False, "reason": "fast_fold_only"}
    state["action_controls"] = {"visible": True, "actions": ["fold"]}

    advice = build_gto_advice(state)

    assert not advice["ready"]
    assert advice["reason"] == "hero_turn_not_confirmed"
    assert advice["hero_turn_reason"] == "fast_fold_only"


def test_all_in_choice_is_recognized_but_not_sent_to_normal_gto_advice() -> None:
    state = postflop_state(pot_bb=147.0)
    state["hero_turn"] = {"is_turn": True, "reason": "red_buttons_and_action_text"}
    state["action_controls"] = {"visible": True, "actions": ["fold", "all_in"]}

    advice = build_gto_advice(state)

    assert not advice["ready"]
    assert advice["should_act"] is False
    assert advice["reason"] == "all_in_action_not_supported"


def test_displayed_sizing_uses_pot_percentages_not_bb_amounts() -> None:
    size_mix = build_size_mix(
        decision={"mix": {"raise": 100.0}},
        result={"mode": "postflop"},
        pot_bb=12.0,
        to_call_bb=0.0,
        effective_stack_bb=100.0,
    )
    summary = format_advice_summary("raise", 9.0, {"raise": 100.0}, size_mix)

    assert "% POT" in summary
    assert "BB" not in summary
    assert "BB" not in size_mix["summary"]
