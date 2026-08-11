from __future__ import annotations

import unittest
from pathlib import Path

from gto_cli.card_classifier import parse_card_label


class CardLabelParseTest(unittest.TestCase):
    def assert_card(self, path_text: str, expected: tuple[str, str]) -> None:
        self.assertEqual(parse_card_label(Path(path_text)), expected)

    def test_compact_rank_suit(self) -> None:
        self.assert_card(r"dataset/AS.png", ("A", "s"))
        self.assert_card(r"dataset/10h.jpg", ("T", "h"))

    def test_compact_suit_rank(self) -> None:
        self.assert_card(r"dataset/SA.png", ("A", "s"))
        self.assert_card(r"dataset/h10.jpg", ("T", "h"))

    def test_bracket_labels(self) -> None:
        self.assert_card(r"dataset/card_[D5].png", ("5", "d"))
        self.assert_card(r"dataset/card_[5D].png", ("5", "d"))

    def test_word_labels(self) -> None:
        self.assert_card(r"dataset/ace_of_spades/card.png", ("A", "s"))
        self.assert_card(r"dataset/spades_ace/card.png", ("A", "s"))

    def test_unicode_suits(self) -> None:
        self.assert_card("dataset/A♠.png", ("A", "s"))
        self.assert_card("dataset/Q♥.png", ("Q", "h"))
        self.assert_card("dataset/7♦.png", ("7", "d"))
        self.assert_card("dataset/2♣.png", ("2", "c"))


if __name__ == "__main__":
    unittest.main()
