import numpy as np

from gto_cli import video_vision
from gto_cli.video_vision import nearest_bet_text_seat


def test_current_wpt_blind_texts_map_to_their_physical_seats() -> None:
    shape = (736, 1120, 3)

    small_blind = nearest_bet_text_seat({"x": 858, "y": 193, "width": 103, "height": 37}, shape, 8)
    big_blind = nearest_bet_text_seat({"x": 936, "y": 333, "width": 79, "height": 33}, shape, 8)
    under_the_gun = nearest_bet_text_seat({"x": 738, "y": 461, "width": 80, "height": 34}, shape, 8)

    assert small_blind[0] == 5
    assert big_blind[0] == 6
    assert under_the_gun[0] == 7
    assert max(small_blind[1], big_blind[1], under_the_gun[1]) < 35


def test_distant_edge_text_is_not_assigned_as_a_bet() -> None:
    seat_index, distance = nearest_bet_text_seat({"x": 1, "y": 376, "width": 82, "height": 33}, (736, 1120, 3), 8)

    assert seat_index is None
    assert distance > 92


def test_chip_near_text_cannot_override_the_text_seat(monkeypatch) -> None:
    monkeypatch.setattr(
        video_vision,
        "detect_red_chips",
        lambda _frame: [{"x": 208.0, "y": 234.5, "area": 20.0, "circularity": 0.9}],
    )
    frame = np.zeros((819, 1238, 3), dtype=np.uint8)
    seats = [{"name": f"seat_{index}"} for index in range(8)]
    ocr = [
        (
            [[235, 217], [348, 217], [348, 253], [235, 253]],
            "0.4 BB",
            0.95,
        )
    ]

    bets = video_vision.detect_bets(frame, seats, ocr)

    assert bets[3]["amount_bb"] == 0.4


def test_stack_panel_amount_is_never_treated_as_a_bet_even_with_a_nearby_chip(monkeypatch) -> None:
    monkeypatch.setattr(
        video_vision,
        "detect_red_chips",
        lambda _frame: [{"x": 193.0, "y": 331.0, "area": 20.0, "circularity": 0.9}],
    )
    frame = np.zeros((1081, 1549, 3), dtype=np.uint8)
    seats = [{"name": f"seat_{index}"} for index in range(8)]
    ocr = [
        (
            [[195, 324], [332, 324], [332, 359], [195, 359]],
            "52.7 BB",
            0.81,
        )
    ]

    bets = video_vision.detect_bets(frame, seats, ocr)

    assert bets == {}
