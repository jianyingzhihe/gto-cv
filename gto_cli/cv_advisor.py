from __future__ import annotations

from typing import Any

from .advisor import advise_state


def attach_gto_advice(
    state: dict[str, Any],
    iterations: int = 600,
    effective_stack_bb: float = 100.0,
    villain_profile: str = "standard",
) -> dict[str, Any]:
    state["gto_advice"] = build_gto_advice(
        state,
        iterations=iterations,
        effective_stack_bb=effective_stack_bb,
        villain_profile=villain_profile,
    )
    return state


def build_gto_advice(
    state: dict[str, Any],
    iterations: int = 600,
    effective_stack_bb: float = 100.0,
    villain_profile: str = "standard",
) -> dict[str, Any]:
    if not state.get("ok"):
        return not_ready("state_not_ok")

    action_controls = state.get("action_controls") or {}
    if not action_controls.get("visible"):
        return not_ready("hero_action_controls_not_visible")
    hero_turn = state.get("hero_turn")
    if isinstance(hero_turn, dict) and not hero_turn.get("is_turn"):
        return not_ready(
            "hero_turn_not_confirmed",
            hero_turn_reason=hero_turn.get("reason"),
            actions=list(action_controls.get("actions") or []),
        )

    actions = {str(action).lower() for action in list(action_controls.get("actions") or [])}
    if "all_in" in actions:
        return not_ready(
            "all_in_action_not_supported",
            actions=sorted(actions),
            summary="WAIT: Hero can act, but the visible choice includes all-in; this strategy model does not advise all-in decisions.",
        )

    hero = state.get("hero") or {}
    table = state.get("table") or {}
    cards = list(hero.get("cards") or [])
    if not complete_cards(cards, expected=2):
        return not_ready("hero_cards_incomplete", hero_cards=cards)

    board = list(table.get("board") or [])
    if table.get("board_pending"):
        return not_ready(
            "board_cards_incomplete",
            board=board,
            summary="WAIT: community cards are still being dealt; wait for a stable board before advising.",
        )
    if board and not complete_cards(board, expected=len(board)):
        return not_ready("board_cards_incomplete", board=board)

    street = str(table.get("street") or street_from_board(board)).lower()
    raw_pot_bb = as_float(table.get("pot_bb"), None)
    if street != "preflop" and (raw_pot_bb is None or raw_pot_bb <= 0):
        return not_ready(
            "pot_amount_unavailable",
            input={
                "street": street,
                "pot_bb": raw_pot_bb,
                "table_to_call_bb": as_float(table.get("to_call_bb"), 0.0),
            },
            summary="WAIT: postflop pot amount is required before sizing a bet or raise.",
        )

    pot_bb = raw_pot_bb if raw_pot_bb is not None else 0.0
    to_call_bb, call_amount_source = action_to_call(
        action_controls,
        table,
        effective_stack_bb=effective_stack_bb,
    )
    advisor_state = {
        "hero": {
            "cards": cards,
            "position": hero.get("position"),
            "gto_position": hero.get("gto_position"),
            "preflop_action_order": hero.get("preflop_action_order"),
            "stack_bb": effective_stack_bb,
        },
        "table": {
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb,
            "effective_stack_bb": effective_stack_bb,
            "board": board,
        },
        "action": {
            "street": street,
            "call_amount_source": call_amount_source,
        },
        "villain": {"profile": villain_profile},
    }
    attach_explicit_preflop_context(state, advisor_state)
    try:
        result = advise_state(advisor_state, iterations=iterations)
    except Exception as error:
        return not_ready("advisor_error", error=str(error), advisor_state=advisor_state)

    decision = result.get("decision") or {}
    primary = decision.get("primary_action")
    preflop_context = result.get("preflop_context") or {}
    if result.get("mode") == "preflop" and primary == "wait":
        status = str(preflop_context.get("status") or "")
        unsupported = status in {"four_bet_or_more", "limped_pot", "unsupported_scenario"}
        return not_ready(
            "preflop_scenario_not_supported" if unsupported else "preflop_context_incomplete",
            preflop_context=preflop_context,
            preflop_tracker=state.get("preflop_tracker"),
            input=advisor_state,
            summary=(
                "WAIT: the preflop action sequence is recognized, but this strategy model does not cover it."
                if unsupported
                else "WAIT: preflop action history is required before GTO can classify RFI/vs-open/vs-3bet."
            ),
        )
    visible_primary = strategy_action_to_visible_action(primary)
    if visible_primary not in actions:
        return not_ready(
            "advice_action_not_available",
            action=primary,
            required_visible_action=visible_primary,
            actions=sorted(actions),
            input=advisor_state,
            result=result,
            summary="WAIT: the strategy action is not available in the visible Hero controls.",
        )
    unavailable_mix = sorted(
        {
            strategy_action_to_visible_action(action)
            for action, frequency in (decision.get("mix") or {}).items()
            if as_float(frequency, 0.0) > 0
            and strategy_action_to_visible_action(action) not in actions
        }
    )
    if unavailable_mix:
        return not_ready(
            "advice_action_space_mismatch",
            action=primary,
            unavailable_actions=unavailable_mix,
            actions=sorted(actions),
            input=advisor_state,
            result=result,
            summary="WAIT: the strategy action space does not match the visible Hero controls.",
        )
    amount = recommended_amount_bb(primary, decision, to_call_bb)
    size_mix = build_size_mix(
        decision=decision,
        result=result,
        pot_bb=pot_bb,
        to_call_bb=to_call_bb,
        effective_stack_bb=effective_stack_bb,
    )
    return {
        "ready": True,
        "should_act": True,
        "reason": "hero_action_controls_visible",
        "action": primary,
        "amount_bb": amount,
        "target_bet_bb": amount,
        "mix": decision.get("mix") or {},
        "size_mix": size_mix,
        "confidence": decision.get("confidence"),
        "scenario": preflop_context.get("scenario"),
        "preflop_context": preflop_context,
        "input": advisor_state,
        "result": result,
        "summary": format_advice_summary(primary, amount, decision.get("mix") or {}, size_mix),
    }


def not_ready(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"ready": False, "should_act": False, "reason": reason}
    payload.update(extra)
    return payload


def complete_cards(cards: list[Any], expected: int) -> bool:
    if len(cards) != expected:
        return False
    for card in cards:
        text = str(card)
        if len(text) != 2 or "?" in text:
            return False
    return True


def action_to_call(
    action_controls: dict[str, Any],
    table: dict[str, Any],
    *,
    effective_stack_bb: float,
) -> tuple[float, str]:
    """Choose a plausible call amount without allowing stack OCR to leak in."""

    actions = {str(action).lower() for action in action_controls.get("actions") or []}
    if "check" in actions and "call" not in actions:
        return 0.0, "visible_check_without_call"
    table_call = as_float(table.get("to_call_bb"), 0.0)
    call_amount = as_float(action_controls.get("call_amount_bb"), None)
    if call_amount is None:
        return table_call, "table_bets"
    if call_amount < 0 or call_amount > effective_stack_bb + 0.15:
        return table_call, "table_bets_rejected_control_amount"
    if table_call > 0 and call_amount > max(table_call * 4.0, table_call + 8.0):
        return table_call, "table_bets_rejected_control_mismatch"
    return call_amount, "action_controls"


def strategy_action_to_visible_action(action: Any) -> str:
    """Map strategy labels to the corresponding client button label."""

    normalized = str(action or "").lower()
    if normalized in {"open", "raise", "3bet", "4bet"}:
        return "raise"
    if normalized == "limp":
        return "call"
    return normalized


def attach_explicit_preflop_context(source_state: dict[str, Any], advisor_state: dict[str, Any]) -> None:
    """Pass declared upstream facts through without deriving a scenario from CV bets."""

    source_action = source_state.get("action") or {}
    source_preflop = source_state.get("preflop") or {}
    history = source_state.get("preflop_action_history")
    if history is None and isinstance(source_preflop, dict):
        history = source_preflop.get("action_history")
    if history is None and isinstance(source_action, dict):
        history = source_action.get("history")
    if isinstance(history, list):
        advisor_state["preflop"] = {
            "action_history": history,
            "history_source": source_preflop.get("history_source") if isinstance(source_preflop, dict) else None,
        }

    declared_scenario = source_state.get("gto_scenario")
    if declared_scenario is None and isinstance(source_preflop, dict):
        declared_scenario = source_preflop.get("scenario")
    if declared_scenario is not None:
        advisor_state["action"]["scenario"] = declared_scenario


def recommended_amount_bb(action: str | None, decision: dict[str, Any], to_call_bb: float) -> float | None:
    if action in ("fold", "check"):
        return 0.0
    if action == "call":
        return round(to_call_bb, 2)
    size = decision.get("recommended_size_bb")
    if size is None:
        return None
    return round(float(size), 2)


def build_size_mix(
    decision: dict[str, Any],
    result: dict[str, Any],
    pot_bb: float,
    to_call_bb: float,
    effective_stack_bb: float,
) -> dict[str, Any]:
    action_mix = {str(key): as_float(value, 0.0) for key, value in (decision.get("mix") or {}).items()}
    mode = str(result.get("mode") or "")
    items: list[dict[str, Any]] = []
    if mode == "preflop":
        size = as_float(decision.get("recommended_size_bb"), None)
        for action, frequency in action_mix.items():
            if frequency <= 0:
                continue
            item: dict[str, Any] = {
                "action": action,
                "frequency_pct": round(frequency, 1),
                "label": action,
            }
            if action in {"open", "raise", "3bet", "4bet", "all_in"} and size is not None:
                item["amount_bb"] = round(size, 2)
                apply_pot_percentage_label(item, pot_bb)
            elif action == "call":
                item["amount_bb"] = round(to_call_bb, 2)
                apply_pot_percentage_label(item, pot_bb)
            items.append(item)
        return finalize_size_mix(items, pot_bb, mode)

    bet_frequency = action_mix.get("bet", 0.0)
    if bet_frequency > 0 and pot_bb > 0:
        for frequency, pot_fraction in split_frequency(bet_frequency, bet_size_weights(bet_frequency)):
            amount = min(round(pot_bb * pot_fraction, 2), effective_stack_bb)
            items.append(
                {
                    "action": "bet",
                    "frequency_pct": frequency,
                    "amount_bb": amount,
                    "pot_pct": round(pot_fraction * 100, 1),
                    "label": f"bet {format_number(pot_fraction * 100)}% pot",
                }
            )

    raise_frequency = action_mix.get("raise", 0.0)
    if raise_frequency > 0:
        for frequency, pot_fraction, multiplier in split_raise_frequency(raise_frequency):
            amount = recommended_raise_amount(pot_bb, to_call_bb, effective_stack_bb, pot_fraction, multiplier)
            item = {
                "action": "raise",
                "frequency_pct": frequency,
                "amount_bb": amount,
                "label": "raise",
            }
            apply_pot_percentage_label(item, pot_bb)
            items.append(item)

    for action in ("check", "call", "fold"):
        frequency = action_mix.get(action, 0.0)
        if frequency <= 0:
            continue
        item = {"action": action, "frequency_pct": round(frequency, 1), "label": action}
        if action == "call":
            item["amount_bb"] = round(to_call_bb, 2)
            apply_pot_percentage_label(item, pot_bb)
        items.append(item)
    return finalize_size_mix(items, pot_bb, mode)


def bet_size_weights(bet_frequency: float) -> list[tuple[float, float]]:
    if bet_frequency >= 65:
        return [(0.33, 0.30), (0.66, 0.50), (1.00, 0.20)]
    if bet_frequency >= 35:
        return [(0.33, 0.55), (0.66, 0.45)]
    return [(0.33, 1.00)]


def split_raise_frequency(raise_frequency: float) -> list[tuple[float, float, float]]:
    weights = [(0.55, 2.5, 0.35), (0.75, 3.0, 0.50), (1.10, 4.0, 0.15)]
    return [
        (frequency, pot_fraction, multiplier)
        for frequency, (pot_fraction, multiplier) in zip(
            split_total(raise_frequency, [weight for _pot, _multiplier, weight in weights]),
            [(pot, multiplier) for pot, multiplier, _weight in weights],
        )
        if frequency > 0
    ]


def split_frequency(total_frequency: float, weights: list[tuple[float, float]]) -> list[tuple[float, float]]:
    frequencies = split_total(total_frequency, [weight for _fraction, weight in weights])
    return [
        (frequency, pot_fraction)
        for frequency, (pot_fraction, _weight) in zip(frequencies, weights)
        if frequency > 0
    ]


def split_total(total: float, weights: list[float]) -> list[float]:
    if total <= 0 or not weights:
        return []
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [round(total, 1)]
    raw = [total * weight / weight_sum for weight in weights]
    rounded = [round(value, 1) for value in raw]
    delta = round(total - sum(rounded), 1)
    if rounded:
        rounded[-1] = round(rounded[-1] + delta, 1)
    return rounded


def recommended_raise_amount(
    pot_bb: float,
    to_call_bb: float,
    effective_stack_bb: float,
    pot_fraction: float,
    multiplier: float,
) -> float:
    size = max(to_call_bb * multiplier, pot_bb * pot_fraction)
    return round(min(size, effective_stack_bb), 2)


def finalize_size_mix(items: list[dict[str, Any]], pot_bb: float, mode: str) -> dict[str, Any]:
    items = [item for item in items if float(item.get("frequency_pct") or 0.0) > 0]
    return {
        "mode": mode or "unknown",
        "pot_bb": round(pot_bb, 2),
        "items": items,
        "summary": format_size_mix(items),
        "note": "heuristic sizing mix; not an exact solver tree",
    }


def format_advice_summary(
    action: str | None,
    amount: float | None,
    mix: dict[str, Any],
    size_mix: dict[str, Any] | None = None,
) -> str:
    if not action:
        return "no advice"
    amount_text = display_action_size(action, size_mix)
    mix_text = " / ".join(f"{key} {value}%" for key, value in mix.items())
    size_text = (size_mix or {}).get("summary")
    if size_text:
        return f"{action.upper()} {amount_text}  ({mix_text})  sizes: {size_text}"
    return f"{action.upper()} {amount_text}  ({mix_text})"


def format_size_mix(items: list[dict[str, Any]]) -> str:
    parts = []
    for item in items:
        frequency = format_number(item.get("frequency_pct"))
        label = str(item.get("label") or item.get("action") or "-")
        parts.append(f"{label} {frequency}%")
    return " / ".join(parts)


def apply_pot_percentage_label(item: dict[str, Any], pot_bb: float) -> None:
    amount = as_float(item.get("amount_bb"), None)
    action = str(item.get("action") or "action")
    if amount is None or pot_bb <= 0:
        item["label"] = action
        return
    pot_pct = round((amount / pot_bb) * 100, 1)
    item["pot_pct"] = pot_pct
    item["label"] = f"{action} {format_number(pot_pct)}% pot"


def display_action_size(action: str, size_mix: dict[str, Any] | None) -> str:
    if action in {"fold", "check"}:
        return ""
    candidates = [
        item
        for item in list((size_mix or {}).get("items") or [])
        if str(item.get("action") or "") == action and item.get("pot_pct") is not None
    ]
    if not candidates:
        return ""
    item = max(candidates, key=lambda candidate: float(candidate.get("frequency_pct") or 0.0))
    return f"{format_number(item['pot_pct'])}% POT"


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def street_from_board(board: list[Any]) -> str:
    if len(board) >= 5:
        return "river"
    if len(board) == 4:
        return "turn"
    if len(board) == 3:
        return "flop"
    return "preflop"


def as_float(value: Any, default: float | None) -> float:
    if value is None or value == "":
        return default  # type: ignore[return-value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default  # type: ignore[return-value]
