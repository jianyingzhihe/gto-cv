from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from gto_cli import video_vision


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "card_regressions" / "20260817_batch2"


@pytest.mark.parametrize(
    ("filename", "source", "expected"),
    [
        ("board_qh_card.png", "board", "Qh"),
        ("board_3h_card.png", "board", "3h"),
        ("hero_6d_card.png", "hero", "6d"),
        ("hero_qd_card.png", "hero", "Qd"),
    ],
)
def test_second_cross_device_batch_cards(filename: str, source: str, expected: str) -> None:
    crop = cv2.imread(str(FIXTURE_DIR / filename))

    detail = video_vision.recognize_card_crop(
        crop,
        source=source,
        index=0,
        return_rejected=True,
    )

    assert detail is not None
    assert detail["card"] == expected


def test_red_suit_component_excludes_card_border() -> None:
    crop = cv2.imread(str(FIXTURE_DIR / "hero_6d_card.png"))

    glyph = video_vision.normalized_hero_red_suit_component(crop, (42, 42))
    prediction = video_vision.classify_suit_glyph(
        glyph,
        allowed=("h", "d"),
        model_path=video_vision.HERO_SUIT_MODEL_PATH,
    )

    assert glyph is not None
    assert prediction is not None
    assert prediction["label"] == "d"
    assert prediction["margin"] >= 0.04
