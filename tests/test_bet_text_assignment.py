import numpy as np

from gto_cli import video_vision
from gto_cli.video_vision import nearest_bet_text_seat, repair_bet_amount


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


def test_scaled_live_bet_texts_keep_their_seat_assignment() -> None:
    shape = (1140, 1652, 3)

    top_right = nearest_bet_text_seat({"x": 1151, "y": 411, "width": 149, "height": 53}, shape, 8)
    right = nearest_bet_text_seat({"x": 1216, "y": 571, "width": 150, "height": 55}, shape, 8)

    assert top_right[0] == 5
    assert right[0] == 6


def test_chip_glyph_prefix_is_removed_only_when_amount_exceeds_pot() -> None:
    assert repair_bet_amount(91.0, "91 BB", 5.4) == 1.0
    assert repair_bet_amount(80.4, "80.4 BB", 5.4) == 0.4
    assert repair_bet_amount(4.8, "4.8 BB", 8.2) == 4.8


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


def test_chipless_big_blind_close_to_bet_anchor_survives_stack_overlap(monkeypatch) -> None:
    monkeypatch.setattr(video_vision, "detect_red_chips", lambda _frame: [])
    frame = np.zeros((736, 1120, 3), dtype=np.uint8)
    seats = [{"name": f"seat_{index}"} for index in range(8)]
    ocr = [
        (
            [[98, 372], [195, 372], [195, 403], [98, 403]],
            "1 BB",
            0.91,
        )
    ]

    bets = video_vision.detect_bets(frame, seats, ocr)

    assert bets[2]["amount_bb"] == 1.0


def test_repaired_player_stack_is_not_rescued_as_a_blind(monkeypatch) -> None:
    monkeypatch.setattr(video_vision, "detect_red_chips", lambda _frame: [])
    frame = np.zeros((736, 1120, 3), dtype=np.uint8)
    seats = [{"name": f"seat_{index}"} for index in range(8)]
    ocr = [
        (
            [[98, 372], [195, 372], [195, 403], [98, 403]],
            "81 BB",
            0.94,
        )
    ]

    bets = video_vision.detect_bets(
        frame,
        seats,
        ocr,
        pot={"amount_bb": 5.4, "box": {"x": 460, "y": 250, "width": 120, "height": 40}},
    )

    assert bets == {}


def test_chipless_small_stack_far_from_bet_anchor_stays_rejected(monkeypatch) -> None:
    monkeypatch.setattr(video_vision, "detect_red_chips", lambda _frame: [])
    frame = np.zeros((736, 1120, 3), dtype=np.uint8)
    seats = [{"name": f"seat_{index}"} for index in range(8)]
    ocr = [
        (
            [[140, 580], [220, 580], [220, 608], [140, 608]],
            "1 BB",
            0.97,
        )
    ]

    bets = video_vision.detect_bets(frame, seats, ocr)

    assert bets == {}
