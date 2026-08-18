from __future__ import annotations

from unittest.mock import patch

import numpy as np

from gto_cli.cv_advisor import build_gto_advice
from gto_cli.preflop_tracker import PreflopActionTracker
from gto_cli.video_vision import analyze_video_frame, detect_blind_structure, three_blind_action_order_number


def seat(name: str, position: str, order: int, status: str, bet: float | None) -> dict:
    return {
        "seat": name,
        "position": position,
        "gto_position": "BB" if position == "THIRD_BLIND" else position,
        "preflop_action_order": order,
        "status": status,
        "bet_bb": bet,
    }


def three_blind_state(*, hero_position: str = "SB", hero_order: int = 6) -> dict:
    seats = [
        seat("utg", "UTG", 1, "folded_or_empty", None),
        seat("lj", "LJ", 2, "folded_or_empty", None),
        seat("hj", "HJ", 3, "folded_or_empty", None),
        seat("co", "CO", 4, "folded_or_empty", None),
        seat("btn", "BTN", 5, "folded_or_empty", None),
        seat("bottom_hero", hero_position, hero_order, "active_or_showdown", 0.4),
        seat("bb", "BB", 7, "active_or_showdown", 1.0),
        seat("third", "THIRD_BLIND", 8, "active_or_showdown", 2.0),
    ]
    return {
        "ok": True,
        "table": {
            "street": "preflop",
            "board": [],
            "dealer_seat": "btn",
            "pot_bb": 3.4,
            "to_call_bb": 1.6,
            "blind_structure": {
                "kind": "three_blind",
                "posts_bb": {"SB": 0.4, "BB": 1.0, "THIRD_BLIND": 2.0},
            },
        },
        "confidence": {"pot_ocr": 0.90},
        "hero": {
            "seat": "bottom_hero",
            "position": hero_position,
            "gto_position": "SB",
            "preflop_action_order": hero_order,
            "cards": ["As", "Qd"],
            "bet_bb": 0.4,
        },
        "seats": seats,
        "action_controls": {
            "visible": True,
            "actions": ["fold", "call", "raise"],
            "call_amount_bb": 1.6,
        },
        "hero_turn": {"is_turn": True},
    }


def test_table_title_detects_three_forced_blinds() -> None:
    result = detect_blind_structure(
        [
            (
                [[10, 5], [250, 5], [250, 28], [10, 28]],
                "Fast-8160 - 0.20/0.50/1 - table",
                "0.91",
            )
        ],
        frame_height=756,
    )

    assert result["kind"] == "three_blind"
    assert result["posts_bb"] == {"SB": 0.4, "BB": 1.0, "THIRD_BLIND": 2.0}


def test_cached_three_blind_structure_survives_title_ocr_miss() -> None:
    hint = {
        "kind": "three_blind",
        "source": "table_title_ocr",
        "posts_bb": {"SB": 0.4, "BB": 1.0, "THIRD_BLIND": 2.0},
    }
    dealer = {"center": (50.0, 50.0), "confidence": 0.99}
    cards = {"hero": [], "board": []}
    with (
        patch("gto_cli.video_vision.build_seats") as build_seats,
        patch("gto_cli.video_vision.nearest_seat_index", return_value=0),
        patch("gto_cli.video_vision.detect_pot", return_value={}),
        patch("gto_cli.video_vision.detect_bets", return_value={}),
        patch("gto_cli.video_vision.detect_action_controls", return_value={}),
        patch("gto_cli.video_vision.detect_card_statuses", return_value={}),
    ):
        build_seats.return_value = [
            {"index": index, "name": f"seat_{index}"} for index in range(8)
        ]
        result = analyze_video_frame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            template=None,
            ocr_result_hint=[],
            dealer_button_hint=dealer,
            cards_hint=cards,
            blind_structure_hint=hint,
        )

    assert result["blind_structure"]["kind"] == "three_blind"
    assert result["blind_structure"]["source"] == "cached_table_title"
    assert result["seats"][3]["position"] == "THIRD_BLIND"


def test_three_blind_action_order_starts_after_third_blind() -> None:
    assert [three_blind_action_order_number(offset, 8) for offset in range(8)] == [5, 6, 7, 8, 1, 2, 3, 4]


def test_forced_third_blind_is_not_recorded_as_an_open() -> None:
    state = three_blind_state()

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == ["fold", "fold", "fold", "fold", "fold"]
    advice = build_gto_advice(state)
    assert advice["ready"]
    assert advice["scenario"] == "rfi"


def test_missing_third_blind_is_implied_by_reconciled_pot() -> None:
    state = three_blind_state()
    state["table"]["pot_bb"] = 9.0
    state["table"]["to_call_bb"] = 5.2
    state["action_controls"]["call_amount_bb"] = 5.2
    state["seats"][-1]["bet_bb"] = None
    state["seats"][2].update({"status": "active_or_showdown", "bet_bb": 5.6})

    PreflopActionTracker().update(state)

    history = state["preflop"]["action_history"]
    assert [item["action"] for item in history[:-1]] == ["fold", "fold", "raise", "fold", "fold"]
    assert build_gto_advice(state)["scenario"] == "vs_open"


def test_third_blind_check_option_waits_for_a_supported_strategy() -> None:
    state = three_blind_state(hero_position="THIRD_BLIND", hero_order=8)
    state["hero"].update({"gto_position": "BB", "bet_bb": 2.0})
    state["seats"][5] = seat("sb", "SB", 6, "folded_or_empty", 0.4)
    state["seats"][7] = seat("bottom_hero", "THIRD_BLIND", 8, "active_or_showdown", 2.0)
    state["action_controls"] = {"visible": True, "actions": ["check", "raise"]}
    state["table"]["to_call_bb"] = 0.0

    PreflopActionTracker().update(state)

    assert state["preflop_tracker"]["reason"] == "three_blind_option_not_supported"
    advice = build_gto_advice(state)
    assert not advice["ready"]
    assert advice["reason"] == "preflop_scenario_not_supported"
