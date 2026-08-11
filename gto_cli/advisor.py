from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .cards import (
    build_deck,
    ensure_unique,
    evaluate_7,
    expand_hand_code,
    hand_code_from_cards,
    parse_cards,
    score_category,
    score_category_name,
)
from .strategy import (
    ALL_HANDS,
    preflop_decision,
    normalize_position,
    villain_profile_weight,
)
from .preflop_context import build_preflop_context


@dataclass
class NormalizedState:
    hero_cards: list[str]
    board_cards: list[str]
    position: str
    raw_position: str | None
    preflop_action_order: int | None
    preflop_context: dict[str, Any]
    scenario: str
    pot_bb: float
    to_call_bb: float
    effective_stack_bb: float
    villain_profile: str
    seed: int | None


def advise_state(state: dict[str, Any], iterations: int = 1200) -> dict[str, Any]:
    normalized = normalize_state(state)
    if not normalized.board_cards:
        return advise_preflop(normalized)
    return advise_postflop(normalized, iterations=iterations)


def normalize_state(state: dict[str, Any]) -> NormalizedState:
    hero = state.get("hero", {}) if isinstance(state.get("hero"), dict) else {}
    table = state.get("table", {}) if isinstance(state.get("table"), dict) else {}
    action = state.get("action", {}) if isinstance(state.get("action"), dict) else {}
    villain = state.get("villain", {}) if isinstance(state.get("villain"), dict) else {}

    hero_cards = parse_cards(
        state.get("hero_cards")
        or hero.get("cards")
        or state.get("cards")
    )
    board_cards = parse_cards(
        state.get("board")
        or state.get("board_cards")
        or table.get("board")
        or []
    )
    if len(hero_cards) != 2:
        raise ValueError("state must include exactly two hero cards")
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError("board must contain 0, 3, 4, or 5 cards")
    ensure_unique([*hero_cards, *board_cards])

    preflop_context = build_preflop_context(state)
    raw_position = preflop_context.get("raw_position")
    position = normalize_position(preflop_context.get("solver_position") or "BTN")
    scenario = preflop_context.get("scenario") or "rfi"
    pot_bb = to_float(state.get("pot_bb") or table.get("pot_bb") or action.get("pot_bb"), 0)
    to_call_bb = to_float(
        state.get("to_call_bb")
        or action.get("to_call_bb")
        or action.get("facing_bet_bb")
        or table.get("to_call_bb"),
        0,
    )
    effective_stack_bb = to_float(
        state.get("effective_stack_bb")
        or table.get("effective_stack_bb")
        or hero.get("stack_bb")
        or state.get("stack_bb"),
        100,
    )
    villain_profile = str(
        state.get("villain_profile")
        or villain.get("profile")
        or villain.get("range")
        or "standard"
    )
    seed_value = state.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    return NormalizedState(
        hero_cards=hero_cards,
        board_cards=board_cards,
        position=position,
        raw_position=raw_position,
        preflop_action_order=preflop_context.get("preflop_action_order"),
        preflop_context=preflop_context,
        scenario=scenario,
        pot_bb=pot_bb,
        to_call_bb=to_call_bb,
        effective_stack_bb=effective_stack_bb,
        villain_profile=villain_profile,
        seed=seed,
    )


def advise_preflop(state: NormalizedState) -> dict[str, Any]:
    hand_code = hand_code_from_cards(state.hero_cards)
    context = state.preflop_context
    if not context.get("supported"):
        needs = ", ".join(context.get("needs") or ["preflop context"])
        return {
            "ok": True,
            "mode": "preflop",
            "model": "preflop_context_gate_v1",
            "input_summary": summarize_state(state),
            "preflop_context": context,
            "decision": {
                "primary_action": "wait",
                "mix": {},
                "recommended_size_bb": None,
                "confidence": "unknown",
                "reason": "preflop_context_incomplete",
            },
            "metrics": {"hand_code": hand_code},
            "reasons": [
                f"Preflop action history is incomplete: {needs}.",
                "A positive to-call amount is not treated as proof of an open raise.",
            ],
            "warning": live_play_warning(),
        }
    decision = preflop_decision(
        hand_code=hand_code,
        position=state.position,
        scenario=state.scenario,
        stack_bb=state.effective_stack_bb,
    )
    required = required_equity(state.pot_bb, state.to_call_bb)
    reasons = [
        f"翻前手牌 {hand_code}，位置 {state.position}，场景 {state.scenario}。",
        f"当前范围继续频率约 {decision['range_frequency']}%。",
    ]
    if state.to_call_bb > 0:
        reasons.append(f"底池赔率要求权益约 {required:.1f}%。")
    if decision["recommended_size_bb"]:
        reasons.append(f"建议尺度约 {decision['recommended_size_bb']}BB。")

    return {
        "ok": True,
        "mode": "preflop",
        "model": "heuristic_gto_like_v1",
        "input_summary": summarize_state(state),
        "preflop_context": context,
        "decision": decision,
        "metrics": {
            "hand_code": hand_code,
            "required_equity_pct": round(required, 2),
            "pot_odds_pct": round(required, 2),
        },
        "reasons": reasons,
        "warning": live_play_warning(),
    }


def advise_postflop(state: NormalizedState, iterations: int = 1200) -> dict[str, Any]:
    rng = random.Random(state.seed)
    equity = estimate_equity(state, iterations=iterations, rng=rng)
    required = required_equity(state.pot_bb, state.to_call_bb)
    hero_score = evaluate_7([*state.hero_cards, *state.board_cards])
    category = score_category(hero_score)
    category_name = score_category_name(hero_score)
    decision = postflop_decision(
        equity_pct=equity["equity_pct"],
        required_equity_pct=required,
        category=category,
        to_call_bb=state.to_call_bb,
        pot_bb=state.pot_bb,
        effective_stack_bb=state.effective_stack_bb,
    )
    reasons = [
        f"当前牌力类别：{category_name}。",
        f"模拟权益约 {equity['equity_pct']:.1f}%（{equity['iterations']} 次）。",
    ]
    if state.to_call_bb > 0:
        reasons.append(f"跟注需要约 {required:.1f}% 权益。")
    else:
        reasons.append("当前无人下注，决策在 bet/check 间选择。")
    reasons.extend(decision.pop("reason_hints"))

    return {
        "ok": True,
        "mode": "postflop",
        "model": "equity_heuristic_v1",
        "input_summary": summarize_state(state),
        "preflop_context": state.preflop_context,
        "decision": decision,
        "metrics": {
            **equity,
            "required_equity_pct": round(required, 2),
            "pot_odds_pct": round(required, 2),
            "hand_category": category_name,
        },
        "reasons": reasons,
        "warning": live_play_warning(),
    }


def estimate_equity(
    state: NormalizedState,
    iterations: int = 1200,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    rng = rng or random.Random()
    iterations = max(100, int(iterations))
    known = [*state.hero_cards, *state.board_cards]
    villains = villain_combos(state, known)
    if not villains:
        raise ValueError("villain range is empty after blockers")

    wins = 0
    ties = 0
    losses = 0
    total_weight = sum(item["weight"] for item in villains)

    for _ in range(iterations):
        villain = choose_weighted(villains, total_weight, rng)
        blocked = set([*known, *villain])
        deck = build_deck(blocked)
        runout = [*state.board_cards, *rng.sample(deck, 5 - len(state.board_cards))]
        hero_score = evaluate_7([*state.hero_cards, *runout])
        villain_score = evaluate_7([*villain, *runout])
        if hero_score > villain_score:
            wins += 1
        elif hero_score == villain_score:
            ties += 1
        else:
            losses += 1

    equity_pct = ((wins + ties * 0.5) / iterations) * 100
    return {
        "equity_pct": round(equity_pct, 2),
        "iterations": iterations,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "villain_combo_count": len(villains),
    }


def villain_combos(state: NormalizedState, blocked_cards: list[str]) -> list[dict[str, Any]]:
    blocked = set(blocked_cards)
    context = {
        "position": state.position,
        "scenario": state.scenario,
        "stack_bb": state.effective_stack_bb,
    }
    combos: list[dict[str, Any]] = []
    for hand_code in ALL_HANDS:
        weight = villain_profile_weight(state.villain_profile, hand_code, context)
        if weight <= 0:
            continue
        for combo in expand_hand_code(hand_code):
            if combo[0] in blocked or combo[1] in blocked:
                continue
            combos.append({"cards": combo, "weight": weight})
    return combos


def choose_weighted(
    items: list[dict[str, Any]],
    total_weight: float,
    rng: random.Random,
) -> tuple[str, str]:
    roll = rng.random() * total_weight
    for item in items:
        roll -= float(item["weight"])
        if roll <= 0:
            return item["cards"]
    return items[-1]["cards"]


def postflop_decision(
    equity_pct: float,
    required_equity_pct: float,
    category: int,
    to_call_bb: float,
    pot_bb: float,
    effective_stack_bb: float,
) -> dict[str, Any]:
    if to_call_bb > 0:
        edge = equity_pct - required_equity_pct
        if category >= 5 or equity_pct >= 72:
            mix = {"raise": 70, "call": 25, "fold": 5}
            hints = ["强牌或高权益，优先价值加注。"]
        elif edge >= 18:
            mix = {"raise": 40, "call": 55, "fold": 5}
            hints = ["权益明显超过底池赔率，可以继续并保留部分加注。"]
        elif edge >= 5:
            mix = {"raise": 12, "call": 76, "fold": 12}
            hints = ["权益高于跟注门槛，主线跟注。"]
        elif edge >= -3:
            mix = {"raise": 4, "call": 42, "fold": 54}
            hints = ["权益接近门槛，边缘点偏谨慎。"]
        else:
            mix = {"raise": 0, "call": 12, "fold": 88}
            hints = ["权益不足以支撑跟注，主线弃牌。"]
        primary = max(mix, key=mix.get)
        return {
            "primary_action": primary,
            "mix": mix,
            "recommended_size_bb": postflop_size(primary, pot_bb, to_call_bb, effective_stack_bb),
            "confidence": confidence_from_mix(mix),
            "reason_hints": hints,
        }

    if category >= 5 or equity_pct >= 68:
        mix = {"bet": 72, "check": 28}
        hints = ["强牌或高权益，优先下注获取价值。"]
    elif equity_pct >= 52:
        mix = {"bet": 45, "check": 55}
        hints = ["中高权益，适合混合下注和过牌。"]
    elif equity_pct >= 38:
        mix = {"bet": 24, "check": 76}
        hints = ["权益一般，主线控制底池。"]
    else:
        mix = {"bet": 10, "check": 90}
        hints = ["权益偏低，主线过牌。"]
    primary = max(mix, key=mix.get)
    return {
        "primary_action": primary,
        "mix": mix,
        "recommended_size_bb": postflop_size(primary, pot_bb, to_call_bb, effective_stack_bb),
        "confidence": confidence_from_mix(mix),
        "reason_hints": hints,
    }


def postflop_size(action: str, pot_bb: float, to_call_bb: float, effective_stack_bb: float) -> float | None:
    if action in ("fold", "call", "check"):
        return None
    if action == "raise":
        size = max(to_call_bb * 3, pot_bb * 0.75)
    else:
        size = pot_bb * 0.66
    return round(min(size, effective_stack_bb), 2)


def confidence_from_mix(mix: dict[str, int]) -> str:
    values = sorted(mix.values(), reverse=True)
    if values[0] >= 75:
        return "high"
    if values[0] - values[1] >= 25:
        return "medium"
    return "mixed"


def required_equity(pot_bb: float, to_call_bb: float) -> float:
    if to_call_bb <= 0:
        return 0.0
    return (to_call_bb / (pot_bb + to_call_bb)) * 100


def summarize_state(state: NormalizedState) -> dict[str, Any]:
    return {
        "hero_cards": state.hero_cards,
        "board_cards": state.board_cards,
        "position": state.position,
        "raw_position": state.raw_position,
        "preflop_action_order": state.preflop_action_order,
        "scenario": state.scenario,
        "pot_bb": state.pot_bb,
        "to_call_bb": state.to_call_bb,
        "effective_stack_bb": state.effective_stack_bb,
        "villain_profile": state.villain_profile,
    }


def to_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def live_play_warning() -> str:
    return "Use for study, review, simulations, or private games where everyone agrees to tool assistance."
