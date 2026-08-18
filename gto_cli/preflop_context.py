from __future__ import annotations

from typing import Any

from .strategy import normalize_scenario


POSITION_BUCKETS = {
    "BTN/SB": "SB",
    "THIRD_BLIND": "BB",
    "UTG+1": "UTG",
    "UTG+2": "UTG",
    "MP": "UTG",
    "LJ": "HJ",
}
VALID_SOLVER_POSITIONS = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
RAISE_ACTIONS = {"raise", "open", "3bet", "4bet", "5bet", "all_in", "allin", "jam"}
CALL_ACTIONS = {"call", "limp"}
HERO_TO_ACT_MARKERS = {"hero_to_act", "to_act", "hero_turn", "current_hero_turn"}


def build_preflop_context(state: dict[str, Any]) -> dict[str, Any]:
    """Build a pure, auditable preflop decision context from explicit inputs.

    A positive amount to call is deliberately not used to guess that someone
    opened. Blind postings can create the same number, so a scenario must come
    from a declared scenario or from an ordered action history.
    """

    hero = as_dict(state.get("hero"))
    action = as_dict(state.get("action"))
    preflop = as_dict(state.get("preflop"))
    table = as_dict(state.get("table"))

    raw_position = first_text(
        state.get("raw_position"),
        hero.get("raw_position"),
        hero.get("position"),
        state.get("position"),
    )
    solver_position = normalize_solver_position(
        first_text(state.get("gto_position"), hero.get("gto_position"), raw_position)
    )
    action_order = first_int(
        state.get("preflop_action_order"),
        hero.get("preflop_action_order"),
        preflop.get("hero_action_order"),
    )
    to_call_bb = first_float(
        state.get("to_call_bb"),
        action.get("to_call_bb"),
        action.get("facing_bet_bb"),
        table.get("to_call_bb"),
    )
    history, history_available, history_source = action_history_from_state(state, action, preflop)
    actions_before_hero = normalized_actions_before_hero(history)

    explicit_scenario = first_text(
        state.get("scenario"),
        action.get("scenario"),
        preflop.get("scenario"),
    )
    if explicit_scenario:
        try:
            scenario = normalize_scenario(explicit_scenario)
        except ValueError:
            return context_payload(
                raw_position=raw_position,
                solver_position=solver_position,
                action_order=action_order,
                to_call_bb=to_call_bb,
                history_available=history_available,
                actions_before_hero=actions_before_hero,
                status="unsupported_scenario",
                scenario=None,
                source="explicit_scenario",
                needs=["action.scenario must be rfi, vs_open, or vs_3bet"],
            )
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=history_available,
            actions_before_hero=actions_before_hero,
            status=status_from_scenario(scenario),
            scenario=scenario,
            source="explicit_scenario",
        )

    if not history_available:
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=False,
            actions_before_hero=[],
            status="unknown",
            scenario=None,
            source="missing_history",
            needs=["preflop.action_history or action.scenario"],
        )

    raises = [event for event in actions_before_hero if event["action"] in RAISE_ACTIONS]
    calls_or_limps = [event for event in actions_before_hero if event["action"] in CALL_ACTIONS]
    hero_calls = [event for event in calls_or_limps if event.get("is_hero")]
    if hero_calls and any(event["index"] > hero_calls[-1]["index"] for event in raises):
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=True,
            actions_before_hero=actions_before_hero,
            status="cold_call_facing_squeeze",
            scenario=None,
            source=history_source,
            needs=["cold-call facing a later raise strategy is not configured"],
        )
    if not raises and calls_or_limps:
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=True,
            actions_before_hero=actions_before_hero,
            status="limped_pot",
            scenario=None,
            source=history_source,
            needs=["limped pot strategy is not configured"],
        )

    if len(raises) == 0:
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=True,
            actions_before_hero=actions_before_hero,
            status="unopened",
            scenario="rfi",
            source=history_source,
        )
    if len(raises) == 1:
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=True,
            actions_before_hero=actions_before_hero,
            status="facing_open",
            scenario="vs_open",
            source=history_source,
        )
    if len(raises) == 2:
        return context_payload(
            raw_position=raw_position,
            solver_position=solver_position,
            action_order=action_order,
            to_call_bb=to_call_bb,
            history_available=True,
            actions_before_hero=actions_before_hero,
            status="facing_3bet",
            scenario="vs_3bet",
            source=history_source,
        )
    return context_payload(
        raw_position=raw_position,
        solver_position=solver_position,
        action_order=action_order,
        to_call_bb=to_call_bb,
        history_available=True,
        actions_before_hero=actions_before_hero,
        status="four_bet_or_more",
        scenario=None,
        source=history_source,
        needs=["4bet+ strategy is not configured"],
    )


def action_history_from_state(
    state: dict[str, Any],
    action: dict[str, Any],
    preflop: dict[str, Any],
) -> tuple[list[Any], bool, str]:
    candidates = (
        (state.get("preflop_action_history"), "declared_action_history"),
        (preflop.get("action_history"), str(preflop.get("history_source") or "action_history")),
        (action.get("history"), "declared_action_history"),
        (state.get("action_history"), "declared_action_history"),
    )
    for candidate, source in candidates:
        if isinstance(candidate, list):
            return candidate, True, source
    return [], False, "missing_history"


def normalized_actions_before_hero(history: list[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, raw_event in enumerate(history, start=1):
        event = as_dict(raw_event)
        action = normalize_action(event.get("action") or event.get("type"))
        if action in HERO_TO_ACT_MARKERS or event.get("is_current_hero_turn") is True:
            break
        if event.get("before_hero") is False:
            break
        if not action or action in {"post_sb", "post_bb", "post_ante", "blind", "ante"}:
            continue
        events.append(
            {
                "index": index,
                "position": first_text(event.get("position"), event.get("actor"), event.get("seat")),
                "action": action,
                "amount_bb": first_float(event.get("amount_bb"), event.get("size_bb"), event.get("amount")),
                "is_hero": bool(event.get("is_hero")),
            }
        )
    return events


def context_payload(
    *,
    raw_position: str | None,
    solver_position: str | None,
    action_order: int | None,
    to_call_bb: float | None,
    history_available: bool,
    actions_before_hero: list[dict[str, Any]],
    status: str,
    scenario: str | None,
    source: str,
    needs: list[str] | None = None,
) -> dict[str, Any]:
    raise_events = [event for event in actions_before_hero if event["action"] in RAISE_ACTIONS]
    needs = list(needs or [])
    if not solver_position:
        needs.append("hero.position or hero.gto_position")
    if scenario in {"vs_open", "vs_3bet"} and (to_call_bb is None or to_call_bb <= 0):
        needs.append("table.to_call_bb")
    return {
        "status": status,
        "scenario": scenario,
        "source": source,
        "supported": bool(scenario and solver_position and not needs),
        "raw_position": raw_position,
        "solver_position": solver_position,
        "preflop_action_order": action_order,
        "to_call_bb": to_call_bb,
        "history_available": history_available,
        "actions_before_hero": actions_before_hero,
        "raise_count_before_hero": len(raise_events),
        "aggressor_position": raise_events[-1]["position"] if raise_events else None,
        "aggressor_amount_bb": raise_events[-1]["amount_bb"] if raise_events else None,
        "needs": needs,
    }


def normalize_solver_position(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().upper()
    if text in VALID_SOLVER_POSITIONS:
        return text
    return POSITION_BUCKETS.get(text)


def status_from_scenario(scenario: str) -> str:
    return {
        "rfi": "unopened",
        "vs_open": "facing_open",
        "vs_3bet": "facing_3bet",
    }[scenario]


def normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "reraise": "3bet",
        "three_bet": "3bet",
        "four_bet": "4bet",
        "allin": "all_in",
    }
    return aliases.get(text, text)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def first_int(*values: Any) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
