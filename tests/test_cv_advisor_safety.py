from __future__ import annotations

from gto_cli import cv_advisor
from gto_cli.cv_advisor import (
    action_to_call,
    available_action_mix,
    build_gto_advice,
    build_size_mix,
    format_advice_summary,
    weighted_random_action,
)


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


def test_check_without_call_forces_zero_to_call() -> None:
    amount, source = action_to_call(
        {"actions": ["check", "bet"], "call_amount_bb": None},
        {"to_call_bb": 171.4},
        effective_stack_bb=100,
    )

    assert amount == 0.0
    assert source == "visible_check_without_call"


def test_advice_waits_when_primary_action_is_not_a_visible_button(monkeypatch) -> None:
    state = postflop_state(pot_bb=13.6)
    state["action_controls"] = {"visible": True, "actions": ["check", "bet"]}
    state["table"]["to_call_bb"] = 171.4
    monkeypatch.setattr(
        cv_advisor,
        "advise_state",
        lambda *_args, **_kwargs: {
            "ok": True,
            "mode": "postflop",
            "decision": {
                "primary_action": "fold",
                "mix": {"fold": 100.0},
                "recommended_size_bb": None,
                "confidence": "high",
            },
        },
    )

    advice = build_gto_advice(state, effective_stack_bb=100)

    assert not advice["ready"]
    assert advice["reason"] == "advice_action_not_available"
    assert advice["actions"] == ["bet", "check"]


def test_weighted_random_action_uses_all_positive_weights() -> None:
    mix = {"fold": 30.0, "call": 40.0, "raise": 30.0}

    assert weighted_random_action(mix, 0.00) == "fold"
    assert weighted_random_action(mix, 0.29) == "fold"
    assert weighted_random_action(mix, 0.30) == "call"
    assert weighted_random_action(mix, 0.69) == "call"
    assert weighted_random_action(mix, 0.70) == "raise"
    assert weighted_random_action(mix, 0.99) == "raise"


def test_unavailable_mixed_action_is_filtered_and_remaining_weights_are_normalized() -> None:
    available, excluded = available_action_mix(
        {"raise": 40.0, "call": 55.0, "fold": 5.0},
        {"call", "fold"},
    )

    assert available == {"call": 91.6667, "fold": 8.3333}
    assert excluded == ["raise"]


def test_all_in_call_fold_panel_samples_only_visible_actions(monkeypatch) -> None:
    state = postflop_state(pot_bb=51.6)
    state["action_controls"] = {"visible": True, "actions": ["call", "fold"]}
    state["table"]["to_call_bb"] = 6.1
    monkeypatch.setattr(cv_advisor, "stable_random_value", lambda _context: 0.50)
    monkeypatch.setattr(
        cv_advisor,
        "advise_state",
        lambda *_args, **_kwargs: {
            "ok": True,
            "mode": "postflop",
            "decision": {
                "primary_action": "call",
                "mix": {"raise": 40.0, "call": 55.0, "fold": 5.0},
                "recommended_size_bb": None,
                "confidence": "high",
            },
        },
    )

    advice = build_gto_advice(state, effective_stack_bb=100)

    assert advice["ready"]
    assert advice["action"] == "call"
    assert advice["available_mix"] == {"call": 91.6667, "fold": 8.3333}
    assert advice["selection"]["excluded_unavailable_actions"] == ["raise"]


def test_gto_advice_uses_weighted_sample_and_keeps_it_stable(monkeypatch) -> None:
    state = postflop_state(pot_bb=13.6)
    state["action_controls"] = {"visible": True, "actions": ["check", "bet"]}
    state["table"]["to_call_bb"] = 0.0
    monkeypatch.setattr(cv_advisor, "stable_random_value", lambda _context: 0.05)
    monkeypatch.setattr(
        cv_advisor,
        "advise_state",
        lambda *_args, **_kwargs: {
            "ok": True,
            "mode": "postflop",
            "decision": {
                "primary_action": "check",
                "mix": {"bet": 10.0, "check": 90.0},
                "recommended_size_bb": None,
                "confidence": "high",
            },
        },
    )

    first = build_gto_advice(state, effective_stack_bb=100)
    second = build_gto_advice(state, effective_stack_bb=100)

    assert first["ready"]
    assert first["action"] == "bet"
    assert first["model_primary_action"] == "check"
    assert first["selection"]["method"] == "weighted_random"
    assert first["selection"]["roll_pct"] == 5.0
    assert first["amount_bb"] == 4.49
    assert second["action"] == first["action"]
