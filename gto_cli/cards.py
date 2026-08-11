from __future__ import annotations

from itertools import combinations
from typing import Iterable

RANKS_LOW = "23456789TJQKA"
RANKS_HIGH = "AKQJT98765432"
SUITS = "shdc"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANKS_LOW)}
CATEGORY_NAMES = {
    8: "straight_flush",
    7: "four_of_a_kind",
    6: "full_house",
    5: "flush",
    4: "straight",
    3: "three_of_a_kind",
    2: "two_pair",
    1: "one_pair",
    0: "high_card",
}


def normalize_card(card: str) -> str:
    token = (
        card.strip()
        .replace("10", "T")
        .replace("♠", "s")
        .replace("♥", "h")
        .replace("♦", "d")
        .replace("♣", "c")
    )
    if len(token) != 2:
        raise ValueError(f"invalid card: {card!r}")

    rank = token[0].upper()
    suit = token[1].lower()
    if rank not in RANK_VALUE or suit not in SUITS:
        raise ValueError(f"invalid card: {card!r}")
    return f"{rank}{suit}"


def parse_cards(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [token for token in value.replace(",", " ").replace(";", " ").split() if token]
    else:
        tokens = list(value)
    cards = [normalize_card(token) for token in tokens]
    ensure_unique(cards)
    return cards


def ensure_unique(cards: Iterable[str]) -> None:
    cards = list(cards)
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards in state")


def build_deck(exclude: Iterable[str] = ()) -> list[str]:
    blocked = set(exclude)
    return [f"{rank}{suit}" for rank in RANKS_LOW for suit in SUITS if f"{rank}{suit}" not in blocked]


def hand_code_from_cards(cards: Iterable[str]) -> str:
    first, second = parse_cards(cards)
    a_rank, a_suit = first[0], first[1]
    b_rank, b_suit = second[0], second[1]
    if RANK_VALUE[b_rank] > RANK_VALUE[a_rank]:
        a_rank, b_rank = b_rank, a_rank
        a_suit, b_suit = b_suit, a_suit
    if a_rank == b_rank:
        return f"{a_rank}{b_rank}"
    suffix = "s" if a_suit == b_suit else "o"
    return f"{a_rank}{b_rank}{suffix}"


def expand_hand_code(hand_code: str) -> list[tuple[str, str]]:
    code = hand_code.strip().upper()
    if len(code) not in (2, 3):
        raise ValueError(f"invalid hand code: {hand_code!r}")
    high = code[0]
    low = code[1]
    suitedness = "pair" if len(code) == 2 else code[2].lower()
    if high not in RANK_VALUE or low not in RANK_VALUE:
        raise ValueError(f"invalid hand code: {hand_code!r}")

    if suitedness == "pair":
        if high != low:
            raise ValueError(f"invalid pair hand code: {hand_code!r}")
        return [(f"{high}{a}", f"{low}{b}") for a, b in combinations(SUITS, 2)]
    if suitedness == "s":
        return [(f"{high}{suit}", f"{low}{suit}") for suit in SUITS]
    if suitedness == "o":
        return [
            (f"{high}{high_suit}", f"{low}{low_suit}")
            for high_suit in SUITS
            for low_suit in SUITS
            if high_suit != low_suit
        ]
    raise ValueError(f"invalid hand suitedness: {hand_code!r}")


def evaluate_7(cards: Iterable[str]) -> int:
    parsed = parse_cards(cards)
    if len(parsed) < 5 or len(parsed) > 7:
        raise ValueError("evaluate_7 expects 5 to 7 cards")
    best = 0
    for combo in combinations(parsed, 5):
        best = max(best, evaluate_5(combo))
    return best


def evaluate_5(cards: Iterable[str]) -> int:
    parsed = parse_cards(cards)
    if len(parsed) != 5:
        raise ValueError("evaluate_5 expects exactly 5 cards")

    values = sorted((RANK_VALUE[card[0]] for card in parsed), reverse=True)
    suits = [card[1] for card in parsed]
    flush = len(set(suits)) == 1
    straight = straight_high(values)
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    if flush and straight:
        return encode_score(8, [straight])
    if groups[0][1] == 4:
        kicker = max(value for value, count in groups if count == 1)
        return encode_score(7, [groups[0][0], kicker])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return encode_score(6, [groups[0][0], groups[1][0]])
    if flush:
        return encode_score(5, values)
    if straight:
        return encode_score(4, [straight])
    if groups[0][1] == 3:
        kickers = [value for value, count in groups if count == 1]
        return encode_score(3, [groups[0][0], *kickers])
    if groups[0][1] == 2 and groups[1][1] == 2:
        pairs = [value for value, count in groups if count == 2]
        kicker = max(value for value, count in groups if count == 1)
        return encode_score(2, [*pairs, kicker])
    if groups[0][1] == 2:
        kickers = [value for value, count in groups if count == 1]
        return encode_score(1, [groups[0][0], *kickers])
    return encode_score(0, values)


def score_category(score: int) -> int:
    return score // (15**6)


def score_category_name(score: int) -> str:
    return CATEGORY_NAMES.get(score_category(score), "unknown")


def straight_high(values: Iterable[int]) -> int:
    unique = sorted(set(values), reverse=True)
    if 14 in unique:
        unique.append(1)
    for index in range(len(unique) - 4):
        window = unique[index : index + 5]
        if window[0] - window[4] == 4:
            return 5 if window[0] == 1 else window[0]
    return 0


def encode_score(category: int, kickers: Iterable[int]) -> int:
    score = category * 15**6
    for index, kicker in enumerate(kickers):
        score += kicker * 15 ** (5 - index)
    return score
