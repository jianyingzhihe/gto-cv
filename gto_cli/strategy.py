from __future__ import annotations

from dataclasses import dataclass

from .cards import RANK_VALUE, RANKS_HIGH, SUITS, expand_hand_code

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
SCENARIO_ALIASES = {
    "rfi": "rfi",
    "open": "rfi",
    "first_in": "rfi",
    "vsopen": "vs_open",
    "vs_open": "vs_open",
    "vs-open": "vs_open",
    "call_open": "vs_open",
    "vs3bet": "vs_3bet",
    "vs_3bet": "vs_3bet",
    "vs-3bet": "vs_3bet",
    "facing_3bet": "vs_3bet",
}
AGGRESSIVE_LABEL = {
    "rfi": "raise",
    "vs_open": "3bet",
    "vs_3bet": "4bet",
}
PASSIVE_LABEL = {
    "rfi": "limp",
    "vs_open": "call",
    "vs_3bet": "call",
}


@dataclass(frozen=True)
class HandInfo:
    high: str
    low: str
    kind: str
    high_value: int
    low_value: int
    pair: bool


def all_hand_codes() -> list[str]:
    hands: list[str] = []
    for row_index, row_rank in enumerate(RANKS_HIGH):
        for col_index, col_rank in enumerate(RANKS_HIGH):
            if row_index == col_index:
                hands.append(f"{row_rank}{col_rank}")
            elif row_index < col_index:
                hands.append(f"{row_rank}{col_rank}s")
            else:
                hands.append(f"{col_rank}{row_rank}o")
    return hands


ALL_HANDS = all_hand_codes()


def normalize_position(position: str | None) -> str:
    value = (position or "BTN").upper()
    if value not in POSITIONS:
        raise ValueError(f"unknown position: {position!r}")
    return value


def normalize_scenario(scenario: str | None) -> str:
    value = (scenario or "rfi").strip().lower()
    if value not in SCENARIO_ALIASES:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return SCENARIO_ALIASES[value]


def parse_hand_code(hand_code: str) -> HandInfo:
    code = hand_code.strip().upper()
    if len(code) not in (2, 3):
        raise ValueError(f"invalid hand code: {hand_code!r}")
    high = code[0]
    low = code[1]
    kind = "pair" if len(code) == 2 else code[2].lower()
    if high not in RANK_VALUE or low not in RANK_VALUE:
        raise ValueError(f"invalid hand code: {hand_code!r}")
    if kind == "pair" and high != low:
        raise ValueError(f"invalid pair hand code: {hand_code!r}")
    if kind not in ("pair", "s", "o"):
        raise ValueError(f"invalid hand code: {hand_code!r}")
    return HandInfo(high, low, kind, RANK_VALUE[high], RANK_VALUE[low], high == low)


def combo_count(hand_code: str) -> int:
    return len(expand_hand_code(hand_code))


def hand_score(hand_code: str) -> float:
    info = parse_hand_code(hand_code)
    if info.pair:
        return 54 + (info.high_value - 2) * 3.8

    suited = info.kind == "s"
    gap = info.high_value - info.low_value - 1
    if info.high == "A" and info.low_value <= 5:
        gap = max(0, 5 - info.low_value)

    score = (info.high_value - 2) * 3.05 + (info.low_value - 2) * 1.75
    score += 6.7 if suited else 0
    score += 3.6 if info.high == "A" else 0
    score += 5.3 if info.high_value >= 10 and info.low_value >= 10 else 0
    score += 4.1 if gap == 0 else 2.3 if gap == 1 else 0.8 if gap == 2 else 0
    score -= min(gap, 5) * (1.25 if suited else 2.15)
    score += 4.2 if suited and info.high == "A" and info.low_value <= 5 else 0
    score += 2.3 if suited and info.high_value <= 10 and gap <= 1 else 0
    return clamp(score, 0, 100)


def preflop_mix(
    hand_code: str,
    position: str = "BTN",
    scenario: str = "rfi",
    stack_bb: float = 100,
) -> dict[str, int]:
    position = normalize_position(position)
    scenario = normalize_scenario(scenario)
    score = hand_score(hand_code)
    info = parse_hand_code(hand_code)
    stack_shift = (stack_bb - 100) / 25

    if scenario == "rfi":
        threshold = {
            "UTG": 55,
            "HJ": 49,
            "CO": 43,
            "BTN": 35,
            "SB": 39,
            "BB": 66,
        }[position]
        deep_penalty = max(0, stack_shift) * 0.7
        short_boost = min(0, stack_shift) * -1.2
        aggressive = smooth_frequency(score, threshold + deep_penalty - short_boost, 15)
        passive = 0
        if position == "SB" and not info.pair and threshold - 9 < score < threshold + 5:
            passive = min(28, round((100 - aggressive) * 0.28))
        aggressive = max(0, aggressive - passive)
        return normalize_mix(aggressive, passive)

    if scenario == "vs_open":
        continue_threshold = {
            "UTG": 62,
            "HJ": 57,
            "CO": 52,
            "BTN": 46,
            "SB": 50,
            "BB": 41,
        }[position]
        aggression_threshold = {
            "UTG": 74,
            "HJ": 69,
            "CO": 64,
            "BTN": 58,
            "SB": 61,
            "BB": 56,
        }[position]
        continue_freq = smooth_frequency(score, continue_threshold + stack_shift * 0.7, 17)
        aggressive = smooth_frequency(score, aggression_threshold + stack_shift * 1.2, 14)
        if info.kind == "s" and not info.pair:
            aggressive += 5
        if info.pair and info.high_value <= 7 and stack_bb > 80:
            aggressive -= 12
        aggressive = min(continue_freq, clamp(aggressive, 0, 100))
        return normalize_mix(aggressive, max(0, continue_freq - aggressive))

    continue_threshold = {
        "UTG": 57,
        "HJ": 54,
        "CO": 50,
        "BTN": 46,
        "SB": 48,
        "BB": 43,
    }[position]
    aggression_threshold = {
        "UTG": 77,
        "HJ": 74,
        "CO": 70,
        "BTN": 66,
        "SB": 68,
        "BB": 64,
    }[position]
    continue_freq = smooth_frequency(score, continue_threshold + stack_shift, 14)
    aggressive = smooth_frequency(score, aggression_threshold - min(0, stack_shift) * 2, 11)
    if info.pair and info.high_value >= 12:
        aggressive += 8
    if info.high == "A" and info.low_value <= 5 and info.kind == "s":
        aggressive += 7
    if stack_bb > 120:
        aggressive -= 7
    aggressive = min(continue_freq, clamp(aggressive, 0, 100))
    return normalize_mix(aggressive, max(0, continue_freq - aggressive))


def preflop_decision(
    hand_code: str,
    position: str = "BTN",
    scenario: str = "rfi",
    stack_bb: float = 100,
) -> dict[str, object]:
    scenario = normalize_scenario(scenario)
    position = normalize_position(position)
    mix = preflop_mix(hand_code, position, scenario, stack_bb)
    aggressive_label = AGGRESSIVE_LABEL[scenario]
    passive_label = PASSIVE_LABEL[scenario]
    action_mix = {
        aggressive_label: mix["aggressive"],
        passive_label: mix["passive"],
        "fold": mix["fold"],
    }
    primary = max(action_mix, key=action_mix.get)
    if primary == "mix" and scenario == "rfi":
        primary = "raise" if action_mix["raise"] >= 25 else "fold"

    return {
        "primary_action": primary,
        "mix": action_mix,
        "raw_mix": mix,
        "aggressive_label": aggressive_label,
        "passive_label": passive_label,
        "recommended_size_bb": preflop_size(position, scenario, stack_bb, primary),
        "aggressive_size_bb": preflop_size(position, scenario, stack_bb, aggressive_label),
        "range_frequency": range_frequency(position, scenario, stack_bb),
    }


def range_frequency(position: str, scenario: str, stack_bb: float) -> float:
    total = 0
    playable = 0.0
    for hand_code in ALL_HANDS:
        combos = combo_count(hand_code)
        mix = preflop_mix(hand_code, position, scenario, stack_bb)
        total += combos
        playable += combos * (mix["aggressive"] + mix["passive"]) / 100
    return round((playable / total) * 100, 2)


def preflop_size(position: str, scenario: str, stack_bb: float, action: str) -> float | None:
    if action == "fold" or action == "call":
        return None
    if scenario == "rfi":
        return 3.0 if position == "SB" else 2.2
    if scenario == "vs_open":
        return 8.5 if position in ("SB", "BB") else 7.5
    if scenario == "vs_3bet":
        if stack_bb <= 35:
            return round(stack_bb, 1)
        return 22.0
    return None


def villain_profile_weight(profile: str, hand_code: str, context: dict[str, object] | None = None) -> float:
    profile = (profile or "standard").lower()
    score = hand_score(hand_code)
    if profile == "tight":
        return max(0, min(100, (score - 50) * 4)) if score >= 58 else 0
    if profile == "wide":
        return max(0, min(100, (score - 28) * 2.6)) if score >= 34 else 0
    if profile == "current" and context:
        mix = preflop_mix(
            hand_code,
            str(context.get("position", "BTN")),
            str(context.get("scenario", "rfi")),
            float(context.get("stack_bb", 100)),
        )
        return mix["aggressive"] + mix["passive"]
    return max(0, min(100, (score - 38) * 3)) if score >= 45 else 0


def smooth_frequency(score: float, threshold: float, width: float) -> int:
    raw = ((score - threshold + width) / (width * 2)) * 100
    return round(clamp(raw, 0, 100) / 5) * 5


def normalize_mix(aggressive: float, passive: float) -> dict[str, int]:
    aggressive = int(clamp(round(aggressive), 0, 100))
    passive = int(clamp(round(passive), 0, 100 - aggressive))
    return {
        "aggressive": aggressive,
        "passive": passive,
        "fold": 100 - aggressive - passive,
    }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
