from __future__ import annotations

from copy import deepcopy

from gto_cli.cv_advisor import build_gto_advice
from gto_cli.preflop_tracker import PreflopActionTracker


def state_for_hero_turn(*, hero_position: str = "HJ", hero_order: int = 4, hero_bet: float | None = None) -> dict:
    seats = [
        seat("utg", "UTG", 1, "folded_or_empty", None),
        seat("utg1", "UTG+1", 2, "active_or_showdown", 2.2),
        seat("lj", "LJ", 3, "folded_or_empty", None),
        seat("bottom_hero", hero_position, hero_order, "active_or_showdown", hero_bet),
        seat("co", "CO", 5, "active_or_showdown", None),
        seat("btn", "BTN", 6, "active_or_showdown", None),
        seat("sb", "SB", 7, "active_or_showdown", 0.4),
        seat("bb", "BB", 8, "active_or_showdown", 1.0),
    ]
    return {
        "ok": True,
        "table": {"street": "preflop", "board": [], "dealer_seat": "btn", "pot_bb": 3.6, "to_call_bb": 2.2},
        "hero": {
            "seat": "bottom_hero",
            "position": hero_position,
            "gto_position": "HJ",
            "preflop_action_order": hero_order,
            "cards": ["As", "Kd"],
            "bet_bb": hero_bet,
        },
        "seats": seats,
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 2.2},
    }


def seat(name: str, position: str, order: int, status: str, bet: float | None) -> dict:
    return {
        "seat": name,
        "position": position,
        "preflop_action_order": order,
        "status": status,
        "bet_bb": bet,
    }


def test_tracker_attaches_visible_open_before_hero_and_unlocks_advice() -> None:
    state = state_for_hero_turn()
    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert history[-1]["action"] == "hero_to_act"
    history = history[:-1]
    assert [item["action"] for item in history] == ["fold", "raise", "fold"]
    assert history[1]["position"] == "UTG+1"
    assert state["preflop"]["history_source"] == "cv_temporal_preflop_tracker"

    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "vs_open"
    assert advice["preflop_context"]["source"] == "cv_temporal_preflop_tracker"


def test_tracker_attaches_unopened_history_when_all_earlier_seats_folded() -> None:
    state = state_for_hero_turn()
    state["seats"][1]["bet_bb"] = None
    state["seats"][1]["status"] = "folded_or_empty"
    state["table"]["to_call_bb"] = 1.0
    state["action_controls"]["call_amount_bb"] = 1.0

    PreflopActionTracker().update(state)

    assert [item["action"] for item in state["preflop"]["action_history"][:-1]] == ["fold", "fold", "fold"]
    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "rfi"


def test_tracker_refuses_to_sync_after_hero_has_already_invested() -> None:
    state = state_for_hero_turn(hero_bet=6.0)

    PreflopActionTracker().update(state)

    assert "preflop" not in state
    assert state["preflop_tracker"]["reason"] == "hero_already_invested_before_sync"
    advice = build_gto_advice(state)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_context_incomplete"


def test_tracker_reports_visible_four_bet_depth_without_inventing_history() -> None:
    state = {
        "ok": True,
        "table": {
            "street": "preflop",
            "board": [],
            "pot_bb": 157.0,
            "to_call_bb": 94.1,
            "blind_structure": {
                "kind": "three_blind",
                "posts_bb": {"SB": 0.4, "BB": 1.0, "THIRD_BLIND": 2.0},
            },
        },
        "hero": {
            "seat": "bottom_hero",
            "position": "BTN",
            "gto_position": "BTN",
            "preflop_action_order": 5,
            "cards": ["Ac", "Js"],
            "bet_bb": 10.9,
        },
        "seats": [
            seat("top", "UTG", 1, "active_or_showdown", 2.0),
            seat("top_right", "LJ", 2, "folded_or_empty", None),
            seat("right", "HJ", 3, "folded_or_empty", None),
            seat("bottom_right", "CO", 4, "folded_or_empty", None),
            seat("bottom_hero", "BTN", 5, "active_or_showdown", 10.9),
            seat("bottom_left", "SB", 6, "active_or_showdown", 0.4),
            seat("left", "BB", 7, "active_or_showdown", 38.0),
            seat("top_left", "THIRD_BLIND", 8, "active_or_showdown", 105.0),
        ],
        "action_controls": {
            "visible": True,
            "actions": ["fold", "call", "raise"],
            "call_amount_bb": 94.4,
            "call_amount_evidence": "button_row_ocr",
        },
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    assert "preflop" not in state
    assert state["preflop_tracker"]["reason"] == "four_bet_or_more_visible_levels"
    advice = build_gto_advice(state)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_scenario_not_supported"


def test_tracker_does_not_guess_when_only_two_raised_levels_are_visible() -> None:
    state = state_for_hero_turn(hero_position="CO", hero_order=4, hero_bet=11.3)
    state["seats"] = [
        seat("utg", "UTG", 1, "active_or_showdown", 2.0),
        seat("utg1", "UTG+1", 2, "folded_or_empty", None),
        seat("hj", "HJ", 3, "active_or_showdown", 2.0),
        seat("bottom_hero", "CO", 4, "active_or_showdown", 11.3),
        seat("btn", "BTN", 5, "active_or_showdown", 30.0),
        seat("sb", "SB", 6, "active_or_showdown", 0.4),
        seat("bb", "BB", 7, "active_or_showdown", 1.0),
    ]
    state["table"].update({"pot_bb": 46.7, "to_call_bb": 18.7})
    state["table"]["blind_structure"] = {
        "kind": "three_blind",
        "posts_bb": {"SB": 0.4, "BB": 1.0, "THIRD_BLIND": 2.0},
    }
    state["action_controls"].update({"call_amount_bb": 18.7, "call_amount_evidence": "button_row_ocr"})
    state["hero_turn"] = {"is_turn": True}

    PreflopActionTracker().update(state)

    assert state["preflop_tracker"]["reason"] == "hero_already_invested_before_sync"
    assert build_gto_advice(state)["reason"] == "preflop_context_incomplete"


def test_tracker_recovers_utg_open_facing_later_three_bet_when_pot_reconciles() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 9.0, "to_call_bb": 3.6},
        "confidence": {"pot_ocr": 0.81},
        "hero": {
            "seat": "bottom_hero",
            "position": "UTG",
            "gto_position": "UTG",
            "preflop_action_order": 1,
            "cards": ["As", "2c"],
            "bet_bb": 2.0,
        },
        "seats": [
            seat("bottom_hero", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "active_or_showdown", None),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "active_or_showdown", 5.6),
            seat("btn", "BTN", 6, "folded_or_empty", None),
            seat("sb", "SB", 7, "active_or_showdown", 0.4),
            seat("bb", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 3.6},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == [
        "raise",
        "fold",
        "fold",
        "fold",
        "3bet",
        "fold",
        "fold",
        "fold",
    ]
    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "vs_3bet"


def test_tracker_recovers_utg_open_three_bet_and_one_exact_caller_when_pot_reconciles() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 14.6, "to_call_bb": 3.6},
        "confidence": {"pot_ocr": 0.85},
        "hero": {
            "seat": "bottom_hero",
            "position": "UTG",
            "gto_position": "UTG",
            "preflop_action_order": 1,
            "cards": ["4c", "2h"],
            "bet_bb": 2.0,
        },
        "seats": [
            seat("bottom_hero", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "active_or_showdown", None),
            seat("lj", "LJ", 3, "active_or_showdown", None),
            seat("hj", "HJ", 4, "folded_or_empty", 5.6),
            seat("co", "CO", 5, "folded_or_empty", 5.6),
            seat("btn", "BTN", 6, "folded_or_empty", None),
            seat("sb", "SB", 7, "folded_or_empty", 0.4),
            seat("bb", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 3.6},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == [
        "raise",
        "fold",
        "fold",
        "3bet",
        "call",
        "fold",
        "fold",
        "fold",
    ]
    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "vs_3bet"


def test_tracker_recovers_utg_open_then_caller_then_three_bet_when_pot_reconciles() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 12.4, "to_call_bb": 5.0},
        "confidence": {"pot_ocr": 0.82},
        "hero": {
            "seat": "bottom_hero",
            "position": "UTG",
            "gto_position": "UTG",
            "preflop_action_order": 1,
            "cards": ["5s", "3s"],
            "bet_bb": 2.0,
        },
        "seats": [
            seat("bottom_hero", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "active_or_showdown", 2.0),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", 7.0),
            seat("co", "CO", 5, "folded_or_empty", None),
            seat("btn", "BTN", 6, "folded_or_empty", None),
            seat("sb", "SB", 7, "folded_or_empty", 0.4),
            seat("bb", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call"], "call_amount_bb": 5.0},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == [
        "raise",
        "call",
        "fold",
        "3bet",
        "fold",
        "fold",
        "fold",
        "fold",
    ]
    assert build_gto_advice(state)["scenario"] == "vs_3bet"


def test_tracker_recovers_utg_open_then_later_four_bet_when_pot_reconciles() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 28.6, "to_call_bb": 16.8},
        "confidence": {"pot_ocr": 0.82},
        "hero": {
            "seat": "bottom_hero",
            "position": "UTG",
            "gto_position": "UTG",
            "preflop_action_order": 1,
            "cards": ["7c", "4d"],
            "bet_bb": 2.0,
        },
        "seats": [
            seat("bottom_hero", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "folded_or_empty", None),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "folded_or_empty", None),
            seat("btn", "BTN", 6, "folded_or_empty", 7.4),
            seat("sb", "SB", 7, "folded_or_empty", 0.4),
            seat("bb", "BB", 8, "folded_or_empty", 18.8),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 16.8},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == [
        "raise",
        "fold",
        "fold",
        "fold",
        "fold",
        "3bet",
        "fold",
        "4bet",
    ]
    advice = build_gto_advice(state)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_scenario_not_supported"


def test_tracker_rejects_visible_bets_when_blinds_are_attached_to_wrong_seats() -> None:
    state = state_for_hero_turn()
    for item in state["seats"]:
        if item["position"] in {"SB", "BB"}:
            item["bet_bb"] = None
    state["seats"][2]["bet_bb"] = 0.4
    state["seats"][4]["bet_bb"] = 1.0

    PreflopActionTracker().update(state)

    assert "preflop" not in state
    assert state["preflop_tracker"]["reason"] == "blind_posts_unconfirmed"


def test_tracker_recovers_open_when_only_big_blind_chip_is_unreadable() -> None:
    state = state_for_hero_turn()
    state["seats"][-1]["bet_bb"] = None

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == ["fold", "raise", "fold"]
    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "vs_open"


def test_tracker_uses_pot_to_imply_one_missing_forced_blind() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 5.4, "to_call_bb": 2.0},
        "confidence": {"pot_ocr": 0.84},
        "hero": {
            "seat": "bottom_hero",
            "position": "BTN",
            "gto_position": "BTN",
            "preflop_action_order": 6,
            "cards": ["Ah", "7c"],
            "bet_bb": None,
        },
        "seats": [
            seat("utg", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "folded_or_empty", None),
            seat("lj", "LJ", 3, "active_or_showdown", 2.0),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "folded_or_empty", None),
            seat("bottom_hero", "BTN", 6, "active_or_showdown", None),
            seat("sb", "SB", 7, "active_or_showdown", 0.4),
            seat("bb", "BB", 8, "active_or_showdown", None),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 0.4},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    assert "preflop_tracker" not in state
    assert [item["action"] for item in state["preflop"]["action_history"][:-1]] == [
        "raise",
        "fold",
        "call",
        "fold",
        "fold",
    ]


def test_tracker_prefers_table_call_when_control_ocr_reads_hero_blind() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 10.0, "to_call_bb": 5.0},
        "confidence": {"pot_ocr": 0.82},
        "hero": {
            "seat": "bottom_hero",
            "position": "UTG",
            "gto_position": "UTG",
            "preflop_action_order": 1,
            "cards": ["9d", "6c"],
            "bet_bb": 2.0,
        },
        "seats": [
            seat("bottom_hero", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "folded_or_empty", None),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "folded_or_empty", None),
            seat("btn", "BTN", 6, "folded_or_empty", None),
            seat("sb", "SB", 7, "active_or_showdown", 7.0),
            seat("bb", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 0.4},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    assert "preflop_tracker" not in state
    assert build_gto_advice(state)["scenario"] == "vs_3bet"


def test_tracker_uses_reconciled_pot_to_ignore_false_active_avatar_before_hero() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 3.4, "to_call_bb": 1.6},
        "confidence": {"pot_ocr": 0.87},
        "hero": {
            "seat": "bottom_hero",
            "position": "SB",
            "gto_position": "SB",
            "preflop_action_order": 7,
            "cards": ["Ac", "Jh"],
            "bet_bb": 0.4,
        },
        "seats": [
            seat("utg", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "folded_or_empty", None),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "folded_or_empty", None),
            seat("btn", "BTN", 6, "active_or_showdown", None),
            seat("bottom_hero", "SB", 7, "active_or_showdown", 0.4),
            seat("bb", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 1.6},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    assert [item["action"] for item in state["preflop"]["action_history"][:-1]] == [
        "raise",
        "fold",
        "fold",
        "fold",
        "fold",
        "fold",
    ]
    assert build_gto_advice(state)["ready"]


def test_tracker_does_not_fill_false_active_avatar_when_pot_has_unseen_money() -> None:
    state = state_for_hero_turn(hero_position="SB", hero_order=7, hero_bet=0.4)
    state["table"]["pot_bb"] = 5.4
    state["confidence"] = {"pot_ocr": 0.87}
    state["seats"] = [
        seat("utg", "UTG", 1, "active_or_showdown", 2.0),
        seat("utg1", "UTG+1", 2, "folded_or_empty", None),
        seat("lj", "LJ", 3, "folded_or_empty", None),
        seat("hj", "HJ", 4, "folded_or_empty", None),
        seat("co", "CO", 5, "folded_or_empty", None),
        seat("btn", "BTN", 6, "active_or_showdown", None),
        seat("bottom_hero", "SB", 7, "active_or_showdown", 0.4),
        seat("bb", "BB", 8, "active_or_showdown", 1.0),
    ]

    PreflopActionTracker().update(state)

    assert state["preflop_tracker"]["reason"] == "prior_seat_unresolved:BTN"


def test_tracker_uses_reconciled_pot_for_false_active_small_blind_before_big_blind() -> None:
    state = {
        "ok": True,
        "table": {"street": "preflop", "board": [], "pot_bb": 9.0, "to_call_bb": 4.6},
        "confidence": {"pot_ocr": 0.87},
        "hero": {
            "seat": "bottom_hero",
            "position": "BB",
            "gto_position": "BB",
            "preflop_action_order": 8,
            "cards": ["Jd", "3d"],
            "bet_bb": 1.0,
        },
        "seats": [
            seat("utg", "UTG", 1, "active_or_showdown", 2.0),
            seat("utg1", "UTG+1", 2, "folded_or_empty", None),
            seat("lj", "LJ", 3, "folded_or_empty", None),
            seat("hj", "HJ", 4, "folded_or_empty", None),
            seat("co", "CO", 5, "active_or_showdown", 5.6),
            seat("btn", "BTN", 6, "folded_or_empty", None),
            seat("sb", "SB", 7, "active_or_showdown", 0.4),
            seat("bottom_hero", "BB", 8, "active_or_showdown", 1.0),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 4.6},
        "hero_turn": {"is_turn": True},
    }

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == ["raise", "fold", "fold", "fold", "3bet", "fold", "fold"]
    assert build_gto_advice(state)["scenario"] == "vs_3bet"


def test_tracker_records_hero_raise_then_later_three_bet() -> None:
    tracker = PreflopActionTracker()
    first = state_for_hero_turn(hero_position="UTG", hero_order=1)
    first["seats"] = [
        seat("bottom_hero", "UTG", 1, "active_or_showdown", None),
        seat("co", "CO", 5, "active_or_showdown", None),
        seat("btn", "BTN", 6, "active_or_showdown", None),
        seat("sb", "SB", 7, "active_or_showdown", 0.4),
        seat("bb", "BB", 8, "active_or_showdown", 1.0),
    ]
    tracker.update(first)
    assert [item["action"] for item in first["preflop"]["action_history"]] == ["hero_to_act"]

    after_raise = deepcopy(first)
    after_raise["hero"]["bet_bb"] = 2.2
    after_raise["seats"][0]["bet_bb"] = 2.2
    after_raise["action_controls"] = {"visible": False}
    tracker.update(after_raise)

    facing_three_bet = deepcopy(after_raise)
    facing_three_bet["action_controls"] = {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 6.8}
    facing_three_bet["table"]["to_call_bb"] = 6.8
    facing_three_bet["seats"][1]["bet_bb"] = 9.0
    tracker.update(facing_three_bet)

    assert [item["action"] for item in facing_three_bet["preflop"]["action_history"][:-1]] == ["raise", "3bet"]
    advice = build_gto_advice(facing_three_bet)
    assert advice["ready"]
    assert advice["scenario"] == "vs_3bet"


def test_tracker_rejects_same_seat_call_then_raise_without_an_intervening_raise() -> None:
    tracker = PreflopActionTracker()
    first = state_for_hero_turn(hero_position="BTN", hero_order=6)
    first["hero"].update({"position": "BTN", "gto_position": "BTN", "preflop_action_order": 6})
    first["seats"] = [
        seat("utg", "UTG", 1, "active_or_showdown", 2.0),
        seat("utg1", "UTG+1", 2, "folded_or_empty", None),
        seat("lj", "LJ", 3, "folded_or_empty", None),
        seat("hj", "HJ", 4, "folded_or_empty", None),
        seat("co", "CO", 5, "folded_or_empty", None),
        seat("bottom_hero", "BTN", 6, "active_or_showdown", None),
        seat("sb", "SB", 7, "active_or_showdown", 0.4),
        seat("bb", "BB", 8, "active_or_showdown", 1.0),
    ]
    first["table"].update({"pot_bb": 3.4, "to_call_bb": 2.0, "dealer_seat": "bottom_hero"})
    first["action_controls"] = {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 2.0}
    first["hero_turn"] = {"is_turn": True}
    tracker.update(first)

    after_hero_three_bet = deepcopy(first)
    after_hero_three_bet["hero"]["bet_bb"] = 6.1
    after_hero_three_bet["seats"][5]["bet_bb"] = 6.1
    after_hero_three_bet["seats"][7]["bet_bb"] = 5.0
    after_hero_three_bet["action_controls"] = {"visible": False}
    after_hero_three_bet["hero_turn"] = {"is_turn": False}
    tracker.update(after_hero_three_bet)
    assert [event["action"] for event in after_hero_three_bet["preflop"]["action_history"]] == [
        "raise",
        "fold",
        "fold",
        "fold",
        "fold",
        "3bet",
        "call",
    ]

    contradictory_raise = deepcopy(after_hero_three_bet)
    contradictory_raise["seats"][7]["bet_bb"] = 52.7
    contradictory_raise["action_controls"] = {"visible": True, "actions": ["fold", "call"], "call_amount_bb": 48.7}
    contradictory_raise["hero_turn"] = {"is_turn": True}
    tracker.update(contradictory_raise)

    assert "preflop" not in contradictory_raise
    assert contradictory_raise["preflop_tracker"]["reason"] == (
        "contradictory_bet_transition:bb:call_to_4bet_without_intervening_raise"
    )
    advice = build_gto_advice(contradictory_raise)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_context_incomplete"


def test_tracker_reconstructs_all_in_four_bet_from_pot_change_and_call_button() -> None:
    tracker = PreflopActionTracker()
    first = state_for_hero_turn(hero_position="BTN", hero_order=6)
    first["hero"].update({"position": "BTN", "gto_position": "BTN", "preflop_action_order": 6})
    first["seats"] = [
        seat("utg", "UTG", 1, "active_or_showdown", 2.0),
        seat("utg1", "UTG+1", 2, "folded_or_empty", None),
        seat("lj", "LJ", 3, "folded_or_empty", None),
        seat("hj", "HJ", 4, "folded_or_empty", None),
        seat("co", "CO", 5, "folded_or_empty", None),
        seat("bottom_hero", "BTN", 6, "active_or_showdown", None),
        seat("sb", "SB", 7, "folded_or_empty", 0.4),
        seat("bb", "BB", 8, "folded_or_empty", 1.0),
    ]
    first["table"].update({"pot_bb": 3.4, "to_call_bb": 2.0, "dealer_seat": "bottom_hero"})
    first["action_controls"] = {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 2.0}
    first["hero_turn"] = {"is_turn": True}
    tracker.update(first)

    after_hero_three_bet = deepcopy(first)
    after_hero_three_bet["hero"]["bet_bb"] = 6.1
    after_hero_three_bet["seats"][5]["bet_bb"] = 6.1
    after_hero_three_bet["table"]["pot_bb"] = 9.5
    after_hero_three_bet["action_controls"] = {"visible": False}
    after_hero_three_bet["hero_turn"] = {"is_turn": False}
    tracker.update(after_hero_three_bet)

    facing_all_in = deepcopy(after_hero_three_bet)
    facing_all_in["table"]["pot_bb"] = 62.2
    facing_all_in["table"]["to_call_bb"] = 0.0
    facing_all_in["seats"][0]["has_cards"] = True
    facing_all_in["action_controls"] = {"visible": True, "actions": ["fold", "call"], "call_amount_bb": 48.7}
    facing_all_in["hero_turn"] = {"is_turn": True}
    tracker.update(facing_all_in)

    history = facing_all_in["preflop"]["action_history"]
    assert [(event.get("seat"), event["action"], event.get("amount_bb")) for event in history[-3:]] == [
        ("bottom_hero", "3bet", 6.1),
        ("utg", "4bet", 54.8),
        (None, "hero_to_act", None),
    ]
    advice = build_gto_advice(facing_all_in)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_scenario_not_supported"


def test_tracker_marks_cold_call_facing_squeeze_as_unsupported() -> None:
    tracker = PreflopActionTracker()
    state = state_for_hero_turn(hero_position="UTG+1", hero_order=2)
    state["hero"].update(
        {
            "position": "UTG+1",
            "gto_position": "UTG",
            "preflop_action_order": 2,
            "bet_bb": 2.0,
        }
    )
    state["seats"] = [
        seat("utg", "UTG", 1, "active_or_showdown", 2.0),
        seat("bottom_hero", "UTG+1", 2, "active_or_showdown", 2.0),
        seat("lj", "LJ", 3, "folded_or_empty", None),
        seat("hj", "HJ", 4, "folded_or_empty", None),
        seat("co", "CO", 5, "folded_or_empty", None),
        seat("btn", "BTN", 6, "active_or_showdown", 5.7),
        seat("sb", "SB", 7, "folded_or_empty", 0.4),
        seat("bb", "BB", 8, "folded_or_empty", 1.0),
    ]
    state["table"].update({"pot_bb": 11.1, "to_call_bb": 3.7, "dealer_seat": "btn"})
    state["action_controls"] = {
        "visible": True,
        "actions": ["fold", "call", "raise"],
        "call_amount_bb": 3.7,
    }
    state["hero_turn"] = {"is_turn": True}

    tracker.update(state)

    history = state["preflop"]["action_history"]
    assert [(event.get("position"), event["action"]) for event in history[:2]] == [
        ("UTG", "raise"),
        ("UTG+1", "call"),
    ]
    advice = build_gto_advice(state)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_scenario_not_supported"
    assert advice["preflop_context"]["status"] == "cold_call_facing_squeeze"


def test_tracker_keeps_confirmed_layout_when_a_weak_dealer_match_rotates_positions() -> None:
    tracker = PreflopActionTracker()
    first = {
        "ok": True,
        "source": {"dealer_button_cached": False},
        "confidence": {"dealer_button": 0.95, "pot_ocr": 0.86},
        "table": {"street": "preflop", "board": [], "dealer_seat": "bottom_left", "dealer_seat_index": 1, "dealer_position": "BTN", "pot_bb": 21.6, "to_call_bb": 13.4},
        "hero": {"seat": "bottom_hero", "position": "CO", "gto_position": "CO", "preflop_action_order": 5, "cards": ["9d", "7s"], "bet_bb": None},
        "seats": [
            seat("bottom_hero", "CO", 5, "active_or_showdown", None),
            seat("bottom_left", "BTN", 6, "active_or_showdown", None),
            seat("left", "SB", 7, "active_or_showdown", 0.4),
            seat("top_left", "BB", 8, "active_or_showdown", 1.0),
            seat("top", "UTG", 1, "active_or_showdown", 2.0),
            seat("top_right", "UTG+1", 2, "active_or_showdown", 4.8),
            seat("right", "LJ", 3, "active_or_showdown", 13.4),
            seat("bottom_right", "HJ", 4, "folded_or_empty", None),
        ],
        "action_controls": {"visible": True, "actions": ["fold", "call", "raise"], "call_amount_bb": 13.4},
        "hero_turn": {"is_turn": True},
    }

    tracker.update(first)
    assert build_gto_advice(first)["reason"] == "preflop_scenario_not_supported"

    weak_match = deepcopy(first)
    weak_match["source"] = {"dealer_button_cached": False}
    weak_match["confidence"]["dealer_button"] = 0.70
    weak_match["table"].update({"dealer_seat": "top_left", "dealer_seat_index": 3, "dealer_position": "BTN"})
    wrong_positions = {
        "bottom_hero": ("LJ", "HJ", 3),
        "bottom_left": ("HJ", "HJ", 4),
        "left": ("CO", "CO", 5),
        "top_left": ("BTN", "BTN", 6),
        "top": ("SB", "SB", 7),
        "top_right": ("BB", "BB", 8),
        "right": ("UTG", "UTG", 1),
        "bottom_right": ("UTG+1", "UTG", 2),
    }
    for item in weak_match["seats"]:
        position, gto_position, order = wrong_positions[item["seat"]]
        item.update({"position": position, "gto_position": gto_position, "preflop_action_order": order})
        if item["seat"] == "bottom_hero":
            weak_match["hero"].update({"position": position, "gto_position": gto_position, "preflop_action_order": order})

    tracker.update(weak_match)

    assert weak_match["source"]["seat_layout_stabilized"] is True
    assert weak_match["table"]["dealer_seat"] == "bottom_left"
    assert weak_match["hero"]["position"] == "CO"
    assert [event["action"] for event in weak_match["preflop"]["action_history"][:-1]] == [
        "raise",
        "3bet",
        "4bet",
        "fold",
    ]
    assert build_gto_advice(weak_match)["reason"] == "preflop_scenario_not_supported"
