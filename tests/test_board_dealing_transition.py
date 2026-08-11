from gto_cli.cv_advisor import build_gto_advice
from gto_cli.live_vision import build_realtime_state


def test_postflop_buttons_with_empty_board_are_treated_as_board_dealing() -> None:
    state = build_realtime_state(
        {
            "hero": {
                "cards": ["Kh", "Qd"],
                "seat_index": 0,
                "seat": "bottom_hero",
                "position": "SB",
                "gto_position": "SB",
                "has_cards": True,
            },
            "dealer": {"seat_index": 7, "seat": "bottom_right", "position": "BTN"},
            "cards": {"board": []},
            "pot": {"amount_bb": 13.5},
            "seats": [],
            "action_controls": {
                "visible": True,
                "actions": ["fold", "check", "bet"],
                "red_button_regions": [{"x": 1, "y": 1, "width": 1, "height": 1}],
            },
        },
        video_path="screen://monitor/1",
        timestamp_sec=0.0,
        frame_index=0,
        sample_index=0,
    )

    assert state["table"]["street"] == "flop"
    assert state["table"]["board_pending"] is True
    assert build_gto_advice(state)["reason"] == "board_cards_incomplete"
