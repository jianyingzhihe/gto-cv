from __future__ import annotations

"""Conservative temporal preflop context for live CV states.

The advisor deliberately does not turn a single ambiguous bet into a poker
scenario.  This module lives on the CV side of that boundary: it records only
pre-hero actions that can be reconstructed from known seats, blind postings,
and later per-seat investment changes.  The raw advisor remains usable without
it and keeps rejecting unknown histories.
"""

from copy import deepcopy
from typing import Any


EPSILON_BB = 0.15
TRUSTED_DEALER_CONFIDENCE = 0.85


class PreflopActionTracker:
    """Track one observed hand and attach a CV-derived action history.

    A tracker may synchronize on a Hero's first decision when every earlier
    seat is visually accounted for.  When it starts after Hero has invested,
    it normally refuses to guess unseen actions.  The sole exception is a
    pot-reconciled UTG open facing one later three-bet, optionally followed by
    callers at that exact amount, where visible investments and the call button
    prove the complete action sequence.
    """

    def __init__(self) -> None:
        self._hand_key: tuple[Any, ...] | None = None
        self._history: list[dict[str, Any]] | None = None
        self._previous_bets: dict[str, float] = {}
        self._previous_statuses: dict[str, str] = {}
        self._previous_pot_bb: float | None = None
        self._pending_contradictory_transition = False
        self._folded_seats: set[str] = set()
        self._previous_street: str | None = None
        self._sequence = 0
        self._sync_reason = "not_observed"
        self._trusted_seat_layout: dict[str, Any] | None = None

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        """Attach ``preflop.action_history`` when the CV evidence is enough."""

        table = as_dict(state.get("table"))
        street = str(table.get("street") or "").lower()
        if street != "preflop" or list(table.get("board") or []):
            self._previous_street = street
            return state

        # Stabilize a weak dealer match before it participates in the hand
        # boundary key. A confirmed dealer change is still allowed to start a
        # new hand immediately.
        self._stabilize_seat_layout(state)
        hand_key = observed_hand_key(state)
        if hand_key is not None and (
            self._hand_key is None
            or hand_key != self._hand_key
            or self._previous_street not in {None, "preflop"}
        ):
            self._reset(hand_key)
            self._stabilize_seat_layout(state)

        preflop = as_dict(state.get("preflop"))
        explicit_history = preflop.get("action_history")
        if isinstance(explicit_history, list) and preflop.get("history_source") != "cv_temporal_preflop_tracker":
            self._history = deepcopy(explicit_history)
            self._remember_observation(state)
            self._previous_street = street
            return state

        if self._history is None:
            inferred, self._sync_reason = infer_visible_prehero_history_with_reason(state)
            if inferred is not None:
                self._history = inferred
                self._sequence = len(inferred)
                self._folded_seats = {
                    str(item.get("seat") or "") for item in inferred if item.get("action") == "fold" and item.get("seat")
                }
        else:
            self._record_transitions(state)

        if self._history is not None and not self._pending_contradictory_transition:
            visible_history = deepcopy(self._history)
            hero_turn_value = state.get("hero_turn")
            is_hero_turn = (
                bool(as_dict(hero_turn_value).get("is_turn"))
                if isinstance(hero_turn_value, dict)
                else bool(as_dict(state.get("action_controls")).get("visible"))
            )
            if is_hero_turn:
                visible_history.append(
                    {
                        "index": len(visible_history) + 1,
                        "position": as_dict(state.get("hero")).get("position"),
                        "action": "hero_to_act",
                        "is_current_hero_turn": True,
                        "source": "cv_temporal_preflop_tracker",
                    }
                )
            state["preflop"] = {
                "action_history": visible_history,
                "history_source": "cv_temporal_preflop_tracker",
                "history_confidence": "conservative",
            }
        else:
            state.pop("preflop", None)
            state["preflop_tracker"] = {
                "status": "unconfirmed",
                "reason": self._sync_reason,
            }

        self._remember_observation(state)
        self._previous_street = street
        return state

    def _reset(self, hand_key: tuple[Any, ...]) -> None:
        self._hand_key = hand_key
        self._history = None
        self._previous_bets = {}
        self._previous_statuses = {}
        self._previous_pot_bb = None
        self._pending_contradictory_transition = False
        self._folded_seats = set()
        self._sequence = 0
        self._sync_reason = "new_hand_waiting_for_first_hero_decision"
        self._trusted_seat_layout = None

    def _stabilize_seat_layout(self, state: dict[str, Any]) -> None:
        """Keep one hand's seat-to-position mapping stable across a weak D match."""

        candidate = seat_layout(state)
        if candidate is None:
            return
        trusted = self._trusted_seat_layout
        if trusted is None:
            if dealer_layout_is_trusted(state):
                self._trusted_seat_layout = candidate
            return
        if candidate["dealer_seat"] == trusted["dealer_seat"]:
            if dealer_layout_is_trusted(state):
                self._trusted_seat_layout = candidate
            return
        if dealer_layout_is_trusted(state):
            self._trusted_seat_layout = candidate
            return

        apply_seat_layout(state, trusted)
        source = as_dict(state.get("source"))
        source["seat_layout_stabilized"] = True
        source["seat_layout_reason"] = "low_confidence_dealer_change"
        state["source"] = source

    def _remember_observation(self, state: dict[str, Any]) -> None:
        self._previous_bets = seat_bets(state)
        self._previous_statuses = seat_statuses(state)
        self._previous_pot_bb = number_or_none(as_dict(state.get("table")).get("pot_bb"))

    def _record_transitions(self, state: dict[str, Any]) -> None:
        if self._history is None or not self._previous_bets:
            return

        seats = ordered_seats(state)
        current_bets = seat_bets(state)
        current_statuses = seat_statuses(state)
        if self._record_all_in_raise_from_pot_change(state, seats):
            return
        raise_to = largest_raise_to(self._history, opening_floor(blind_sizes(state)))
        raise_count = raise_events(self._history)
        changed: list[tuple[int, dict[str, Any], float, float]] = []
        for seat in seats:
            seat_name = str(seat.get("seat") or "")
            if not seat_name:
                continue
            if seat_name in self._folded_seats:
                continue
            old = self._previous_bets.get(seat_name, blind_post(seat, blind_sizes(state)))
            new = current_bets.get(seat_name, blind_post(seat, blind_sizes(state)))
            if new > old + EPSILON_BB:
                changed.append((action_order(seat), seat, old, new))

        # Screen sampling can observe two updates in one frame. Action order is
        # the least surprising deterministic tie-breaker for that case.
        for _order, seat, _old, new in sorted(changed, key=lambda item: item[0]):
            action = "call"
            if new > raise_to + EPSILON_BB:
                action = raise_action_name(raise_count)
                raise_to = new
                raise_count += 1
            if self._repeated_action_without_intervening_raise(seat, action):
                return
            self._append(seat, action, new)

        for seat in seats:
            seat_name = str(seat.get("seat") or "")
            before = self._previous_statuses.get(seat_name, "")
            after = current_statuses.get(seat_name, "")
            if (
                seat_name not in self._folded_seats
                and before == "active_or_showdown"
                and after == "folded_or_empty"
            ):
                self._append(seat, "fold", None)
                self._folded_seats.add(seat_name)

    def _record_all_in_raise_from_pot_change(self, state: dict[str, Any], seats: list[dict[str, Any]]) -> bool:
        """用底池跳变和 Hero 跟注按钮确认一次全下加注。"""

        controls = as_dict(state.get("action_controls"))
        hero_turn = as_dict(state.get("hero_turn"))
        actions = {str(action).lower() for action in list(controls.get("actions") or [])}
        call_amount = number_or_none(controls.get("call_amount_bb"))
        hero = as_dict(state.get("hero"))
        hero_bet = number_or_none(hero.get("bet_bb")) or 0.0
        current_pot = number_or_none(as_dict(state.get("table")).get("pot_bb"))
        if (
            not hero_turn.get("is_turn")
            or not {"fold", "call"}.issubset(actions)
            or call_amount is None
            or call_amount <= EPSILON_BB
            or hero_bet <= EPSILON_BB
            or self._previous_pot_bb is None
            or current_pot is None
        ):
            return False

        prior_raise_to = largest_raise_to(self._history or [], opening_floor(blind_sizes(state)))
        raise_to = hero_bet + call_amount
        if raise_to <= prior_raise_to + EPSILON_BB:
            return False

        # 只接受仍持牌的、此前已参与加注的对手；这样不会把普通底池
        # 变化猜成全下。
        active_seats = {
            str(seat.get("seat") or "")
            for seat in seats
            if seat.get("has_cards") and str(seat.get("seat") or "") != str(hero.get("seat") or "")
        }
        prior_raiser = next(
            (
                event
                for event in reversed(self._history or [])
                if event.get("action") in raise_actions() and str(event.get("seat") or "") in active_seats
            ),
            None,
        )
        if prior_raiser is None:
            return False
        prior_seat = str(prior_raiser.get("seat") or "")
        prior_contribution = self._previous_bets.get(prior_seat, 0.0)
        added_amount = raise_to - prior_contribution
        pot_increase = current_pot - self._previous_pot_bb
        if added_amount <= EPSILON_BB or abs(pot_increase - added_amount) > 0.35:
            return False

        seat = next((item for item in seats if str(item.get("seat") or "") == prior_seat), None)
        if seat is None:
            return False
        self._append(seat, raise_action_name(raise_events(self._history or [])), raise_to)
        self._pending_contradictory_transition = False
        return True

    def _repeated_action_without_intervening_raise(self, seat: dict[str, Any], action: str) -> bool:
        """拒绝同一座位无中间加注的“跟注后又加注”矛盾序列。"""

        if self._history is None or action not in raise_actions():
            return False
        seat_name = str(seat.get("seat") or "")
        if not seat_name:
            return False
        prior_index = next(
            (
                index
                for index in range(len(self._history) - 1, -1, -1)
                if str(self._history[index].get("seat") or "") == seat_name
            ),
            None,
        )
        if prior_index is None or self._history[prior_index].get("action") != "call":
            return False
        intervening_raise = any(
            str(event.get("seat") or "") != seat_name and event.get("action") in raise_actions()
            for event in self._history[prior_index + 1 :]
        )
        if intervening_raise:
            return False
        self._history.pop(prior_index)
        self._sequence = len(self._history)
        self._sync_reason = f"contradictory_bet_transition:{seat_name}:call_to_{action}_without_intervening_raise"
        self._pending_contradictory_transition = True
        return True

    def _append(self, seat: dict[str, Any], action: str, amount_bb: float | None) -> None:
        if self._history is None:
            return
        self._sequence += 1
        event: dict[str, Any] = {
            "index": self._sequence,
            "position": seat.get("position"),
            "seat": seat.get("seat"),
            "action": action,
            "is_hero": str(seat.get("seat") or "") == "bottom_hero",
            "source": "cv_temporal_preflop_tracker",
        }
        if amount_bb is not None:
            event["amount_bb"] = round(amount_bb, 2)
        self._history.append(event)


def infer_visible_prehero_history(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Infer the first Hero decision only when all prior seats are accounted for."""

    history, _reason = infer_visible_prehero_history_with_reason(state)
    return history


def infer_visible_prehero_history_with_reason(state: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str]:
    """Return the conservative first-decision history and its refusal reason."""

    hero = as_dict(state.get("hero"))
    hero_order = int_or_none(hero.get("preflop_action_order"))
    if hero_order is None:
        return None, "hero_action_order_unavailable"
    sizes = blind_sizes(state)
    hero_bet = number_or_none(hero.get("bet_bb")) or 0.0
    hero_post = blind_post(hero, sizes)
    if str(hero.get("position") or "") == "THIRD_BLIND":
        return None, "three_blind_option_not_supported"
    if hero_bet > hero_post + EPSILON_BB:
        reconstructed = infer_utg_open_after_later_raises(state, sizes, hero_bet)
        if reconstructed is not None:
            return reconstructed, "utg_open_after_later_raises_reconstructed_from_pot"
        reconstructed = infer_cold_call_facing_later_raise(state, sizes, hero_bet)
        if reconstructed is not None:
            return reconstructed, "cold_call_facing_later_raise_reconstructed_from_pot"
        if visible_four_bet_or_more_depth_confirmed(state, sizes, hero_bet):
            return None, "four_bet_or_more_visible_levels"
        return None, "hero_already_invested_before_sync"
    if not blind_posts_confirmed(state, sizes):
        if not missing_big_blind_is_safely_implied(state, hero_order, sizes):
            return None, "blind_posts_unconfirmed"

    history: list[dict[str, Any]] = []
    raise_to = opening_floor(sizes)
    raise_count = 0
    pot_confirms_all_visible_contributions = visible_preflop_contributions_match_pot(
        state,
        include_missing_blinds=True,
    )
    for seat in ordered_seats(state):
        order = action_order(seat)
        if order <= 0 or order >= hero_order:
            continue
        status = str(seat.get("status") or "")
        amount = number_or_none(seat.get("bet_bb")) or 0.0
        post = blind_post(seat, sizes)
        position = seat.get("position")

        if status == "folded_or_empty" and amount <= post + EPSILON_BB:
            history.append(history_event(len(history) + 1, seat, "fold", None))
            continue
        if amount <= post + EPSILON_BB:
            # 某些白色头像会被旧的牌背检测误判为仍持牌。若底池金额已与
            # 所有可见下注精确相符，当前下注未超过强制盲注的座位不可能
            # 暗中跟注，因而可安全记为弃牌。
            if pot_confirms_all_visible_contributions:
                history.append(history_event(len(history) + 1, seat, "fold", None))
                continue
            # A seat that still looks active but has neither a blind nor a
            # contribution is not sufficient evidence that it folded.
            return None, f"prior_seat_unresolved:{position or seat.get('seat') or '?'}"

        if amount > raise_to + EPSILON_BB:
            action = raise_action_name(raise_count)
            raise_to = amount
            raise_count += 1
        else:
            action = "call"
        history.append(history_event(len(history) + 1, seat, action, amount))

    return history, "confirmed"


def visible_four_bet_or_more_depth_confirmed(
    state: dict[str, Any],
    sizes: dict[str, float],
    hero_bet: float,
) -> bool:
    """Detect unsupported re-raise depth without inventing the missing history."""

    controls = as_dict(state.get("action_controls"))
    actions = {str(action).lower() for action in list(controls.get("actions") or [])}
    if not controls.get("visible") or "call" not in actions:
        return False
    call_amount = trusted_call_amount(state)
    if call_amount is None or call_amount <= EPSILON_BB:
        return False

    floor = opening_floor(sizes)
    distinct_levels: list[float] = []
    for seat in ordered_seats(state):
        amount = number_or_none(seat.get("bet_bb")) or 0.0
        if amount <= floor + EPSILON_BB:
            continue
        if not any(abs(amount - existing) <= EPSILON_BB for existing in distinct_levels):
            distinct_levels.append(amount)
    if len(distinct_levels) < 3:
        return False
    if not any(abs(hero_bet - amount) <= EPSILON_BB for amount in distinct_levels):
        return False

    largest_visible = max(distinct_levels)
    expected_call = largest_visible - hero_bet
    tolerance = max(0.5, largest_visible * 0.01)
    return expected_call > EPSILON_BB and abs(call_amount - expected_call) <= tolerance


def infer_utg_open_after_later_raises(
    state: dict[str, Any],
    sizes: dict[str, float],
    hero_bet: float,
) -> list[dict[str, Any]] | None:
    """Reconstruct a raised UTG pot only when every visible amount agrees."""

    hero = as_dict(state.get("hero"))
    if int_or_none(hero.get("preflop_action_order")) != 1:
        return None
    if hero_bet <= opening_floor(sizes) + EPSILON_BB:
        return None
    if not blind_posts_present_in_raised_pot(state, sizes):
        return None
    if not visible_preflop_contributions_match_pot(state):
        return None

    controls = as_dict(state.get("action_controls"))
    hero_turn = as_dict(state.get("hero_turn"))
    is_hero_turn = bool(hero_turn.get("is_turn")) if hero_turn else bool(controls.get("visible"))
    actions = {str(action).lower() for action in list(controls.get("actions") or [])}
    if not is_hero_turn or "call" not in actions:
        return None

    seats = ordered_seats(state)
    hero_seat = str(hero.get("seat") or "")
    actions_by_seat: dict[str, tuple[str, float]] = {}
    raise_to = hero_bet
    raise_count = 1  # Hero 的枪口开局已经是第一轮加注。
    for seat in seats:
        if str(seat.get("seat") or "") == hero_seat or action_order(seat) <= 1:
            continue
        amount = number_or_none(seat.get("bet_bb")) or 0.0
        forced_blind = blind_post(seat, sizes)
        if amount <= forced_blind + EPSILON_BB:
            continue
        if amount > raise_to + EPSILON_BB:
            action = raise_action_name(raise_count)
            raise_to = amount
            raise_count += 1
        elif abs(amount - raise_to) <= EPSILON_BB:
            action = "call"
        else:
            # 这不是当前下注额的平跟，却又超过了该座位的强制盲注；
            # 单帧无法判断它是何时投入的，因此拒绝静态补写历史。
            return None
        actions_by_seat[str(seat.get("seat") or "")] = (action, amount)

    if raise_count < 2:
        return None

    call_amount = trusted_call_amount(state)
    if call_amount is None or abs(call_amount - (raise_to - hero_bet)) > EPSILON_BB:
        return None

    history: list[dict[str, Any]] = [history_event(1, hero, "raise", hero_bet)]
    for seat in seats:
        if action_order(seat) <= 1:
            continue
        action_and_amount = actions_by_seat.get(str(seat.get("seat") or ""))
        if action_and_amount is not None:
            action, amount = action_and_amount
            history.append(history_event(len(history) + 1, seat, action, amount))
        else:
            history.append(history_event(len(history) + 1, seat, "fold", None))
    return history


def infer_cold_call_facing_later_raise(
    state: dict[str, Any],
    sizes: dict[str, float],
    hero_bet: float,
) -> list[dict[str, Any]] | None:
    """Reconstruct a cold call followed by a later raise from one reconciled frame."""

    hero = as_dict(state.get("hero"))
    hero_order = int_or_none(hero.get("preflop_action_order"))
    if hero_order is None or hero_order <= 1:
        return None
    if not blind_posts_present_in_raised_pot(state, sizes):
        return None
    if not visible_preflop_contributions_match_pot(state):
        return None

    controls = as_dict(state.get("action_controls"))
    hero_turn = as_dict(state.get("hero_turn"))
    actions = {str(action).lower() for action in list(controls.get("actions") or [])}
    if not hero_turn.get("is_turn") or "call" not in actions:
        return None

    hero_seat = str(hero.get("seat") or "")
    history: list[dict[str, Any]] = []
    raise_to = opening_floor(sizes)
    raise_count = 0
    hero_called = False
    later_raise_seen = False
    for seat in ordered_seats(state):
        order = action_order(seat)
        if order <= 0:
            continue
        amount = number_or_none(seat.get("bet_bb")) or 0.0
        post = blind_post(seat, sizes)
        is_hero = str(seat.get("seat") or "") == hero_seat
        if is_hero:
            if raise_count != 1 or abs(hero_bet - raise_to) > EPSILON_BB:
                return None
            history.append(history_event(len(history) + 1, seat, "call", hero_bet))
            hero_called = True
            continue
        if amount <= post + EPSILON_BB:
            history.append(history_event(len(history) + 1, seat, "fold", None))
            continue
        if amount > raise_to + EPSILON_BB:
            action = raise_action_name(raise_count)
            raise_to = amount
            raise_count += 1
            if hero_called:
                later_raise_seen = True
        elif abs(amount - raise_to) <= EPSILON_BB:
            action = "call"
        else:
            return None
        history.append(history_event(len(history) + 1, seat, action, amount))

    call_amount = trusted_call_amount(state)
    if (
        not hero_called
        or not later_raise_seen
        or call_amount is None
        or abs(call_amount - (raise_to - hero_bet)) > EPSILON_BB
    ):
        return None
    return history


def blind_posts_present_in_raised_pot(state: dict[str, Any], sizes: dict[str, float]) -> bool:
    """Accept blind seats that have later called or raised, not only exact blind chips."""

    by_position = {str(seat.get("position") or ""): seat for seat in ordered_seats(state)}
    for position, minimum in forced_blind_posts(sizes).items():
        amount = number_or_none(as_dict(by_position.get(position)).get("bet_bb"))
        if amount is None or amount + EPSILON_BB < minimum:
            return False
    return True


def visible_preflop_contributions_match_pot(
    state: dict[str, Any],
    *,
    include_missing_blinds: bool = False,
) -> bool:
    """仅在底池金额和当前每个座位的可见下注相符时启用保守补全。"""

    table = as_dict(state.get("table"))
    pot_bb = number_or_none(table.get("pot_bb"))
    if pot_bb is None or pot_bb <= 0:
        return False
    confidence = number_or_none(as_dict(state.get("confidence")).get("pot_ocr"))
    if confidence is not None and confidence < 0.70:
        return False
    visible_total = sum(
        max(0.0, number_or_none(seat.get("bet_bb")) or 0.0)
        for seat in ordered_seats(state)
    )
    if include_missing_blinds:
        sizes = blind_sizes(state)
        forced_posts = forced_blind_posts(sizes)
        for seat in ordered_seats(state):
            position = str(seat.get("position") or "")
            if position in forced_posts and number_or_none(seat.get("bet_bb")) is None:
                visible_total += forced_posts[position]
    return abs(pot_bb - visible_total) <= EPSILON_BB


def observed_hand_key(state: dict[str, Any]) -> tuple[Any, ...] | None:
    hero = as_dict(state.get("hero"))
    cards = tuple(str(card) for card in list(hero.get("cards") or []) if card)
    if len(cards) != 2 or any("?" in card for card in cards):
        return None
    table = as_dict(state.get("table"))
    return (cards, table.get("dealer_seat"))


def dealer_layout_is_trusted(state: dict[str, Any]) -> bool:
    source = as_dict(state.get("source"))
    if source.get("dealer_button_cached") is True:
        return False
    confidence = number_or_none(as_dict(state.get("confidence")).get("dealer_button"))
    return confidence is None or confidence >= TRUSTED_DEALER_CONFIDENCE


def seat_layout(state: dict[str, Any]) -> dict[str, Any] | None:
    table = as_dict(state.get("table"))
    dealer_seat = str(table.get("dealer_seat") or "")
    if not dealer_seat:
        return None
    seat_values: dict[str, dict[str, Any]] = {}
    for item in list(state.get("seats") or []):
        seat = as_dict(item)
        name = str(seat.get("seat") or "")
        if not name:
            continue
        seat_values[name] = {
            key: seat.get(key)
            for key in (
                "position",
                "gto_position",
                "distance_from_dealer_clockwise",
                "preflop_action_order",
                "postflop_action_order",
            )
        }
    return {
        "dealer_seat": dealer_seat,
        "dealer_seat_index": table.get("dealer_seat_index"),
        "dealer_position": table.get("dealer_position"),
        "seats": seat_values,
    }


def apply_seat_layout(state: dict[str, Any], layout: dict[str, Any]) -> None:
    table = as_dict(state.get("table"))
    table["dealer_seat"] = layout.get("dealer_seat")
    table["dealer_seat_index"] = layout.get("dealer_seat_index")
    table["dealer_position"] = layout.get("dealer_position")
    state["table"] = table
    by_seat = as_dict(layout.get("seats"))
    hero = as_dict(state.get("hero"))
    hero_seat = str(hero.get("seat") or "")
    for item in list(state.get("seats") or []):
        seat = as_dict(item)
        values = as_dict(by_seat.get(str(seat.get("seat") or "")))
        if not values:
            continue
        seat.update(values)
        if str(seat.get("seat") or "") == hero_seat:
            hero.update(values)
    state["hero"] = hero


def ordered_seats(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [as_dict(seat) for seat in list(state.get("seats") or [])],
        key=action_order,
    )


def blind_sizes(state: dict[str, Any]) -> dict[str, float]:
    seats = [as_dict(seat) for seat in list(state.get("seats") or [])]
    sb = next((number_or_none(seat.get("bet_bb")) for seat in seats if seat.get("position") == "SB"), None)
    bb = next((number_or_none(seat.get("bet_bb")) for seat in seats if seat.get("position") == "BB"), None)
    bb_value = bb if bb is not None and 0.5 <= bb <= 1.5 else 1.0
    sb_value = sb if sb is not None and 0 < sb <= bb_value else 0.4
    sizes = {"sb": round(sb_value, 2), "bb": round(bb_value, 2)}
    structure = as_dict(as_dict(state.get("table")).get("blind_structure"))
    posts = as_dict(structure.get("posts_bb"))
    third = number_or_none(posts.get("THIRD_BLIND"))
    if structure.get("kind") == "three_blind" and third is not None and third > bb_value + EPSILON_BB:
        sizes["third_blind"] = round(third, 2)
    return sizes


def blind_post(seat: dict[str, Any], sizes: dict[str, float]) -> float:
    position = str(seat.get("position") or "")
    if position == "SB":
        return sizes["sb"]
    if position == "BB":
        return sizes["bb"]
    if position == "THIRD_BLIND":
        return sizes.get("third_blind", 0.0)
    return 0.0


def blind_posts_confirmed(state: dict[str, Any], sizes: dict[str, float]) -> bool:
    """Require the seat-to-bet association to agree with the blind layout.

    The money detector can occasionally attach a chip amount to its nearest
    visual seat rather than its owning seat.  Without both blind posts at their
    expected positions, a static preflop reconstruction is unsafe.
    """

    seats = [as_dict(seat) for seat in list(state.get("seats") or [])]
    by_position = {str(seat.get("position") or ""): seat for seat in seats}
    hero_position = str(as_dict(state.get("hero")).get("position") or "")
    forced_posts = forced_blind_posts(sizes)
    if blind_posts_present_in_raised_pot(state, sizes):
        return True
    if all(amount_matches(by_position.get(position), expected) for position, expected in forced_posts.items()):
        return True
    bb = by_position.get("BB")
    if amount_matches(bb, sizes["bb"]):
        if "third_blind" not in sizes and (
            hero_position == "BB" or amount_matches(by_position.get("SB"), sizes["sb"])
        ):
            return True
    if hero_position == "BB":
        return False

    # A forced blind can briefly disappear under an animation. Accept the
    # omission only when the trusted pot closes the exact missing-blind gap.
    for position, expected in forced_posts.items():
        observed = number_or_none(as_dict(by_position.get(position)).get("bet_bb"))
        if observed is not None and abs(observed - expected) > EPSILON_BB:
            return False
    return visible_preflop_contributions_match_pot(state, include_missing_blinds=True)


def missing_big_blind_is_safely_implied(state: dict[str, Any], hero_order: int, sizes: dict[str, float]) -> bool:
    """Allow a narrow static fallback when only the big-blind chip is unreadable.

    At a normal Hero decision, a known small blind plus an earlier non-blind
    investment equal to Hero's call price proves that the table has progressed
    beyond the forced blinds.  This avoids dropping an otherwise clear open
    merely because the 1 BB chip was not assigned to the BB seat in this frame.
    """

    if "third_blind" in sizes:
        return False
    hero = as_dict(state.get("hero"))
    if str(hero.get("position") or "") == "BB":
        return False
    seats = [as_dict(seat) for seat in list(state.get("seats") or [])]
    by_position = {str(seat.get("position") or ""): seat for seat in seats}
    big_blind = by_position.get("BB")
    if number_or_none(as_dict(big_blind).get("bet_bb")) is not None:
        return False
    if not amount_matches(by_position.get("SB"), sizes["sb"]):
        return False

    controls = as_dict(state.get("action_controls"))
    hero_turn = as_dict(state.get("hero_turn"))
    normal_actions = {"call", "raise"}
    is_hero_turn = bool(hero_turn.get("is_turn")) if hero_turn else bool(controls.get("visible"))
    if not is_hero_turn or not normal_actions.intersection(
        {str(action) for action in list(controls.get("actions") or [])}
    ):
        return False
    call_amount = trusted_call_amount(state)
    if call_amount is None or call_amount <= sizes["bb"] + EPSILON_BB:
        return False

    for seat in ordered_seats(state):
        position = str(seat.get("position") or "")
        if position in {"SB", "BB"} or action_order(seat) >= hero_order:
            continue
        amount = number_or_none(seat.get("bet_bb")) or 0.0
        if amount > sizes["bb"] + EPSILON_BB and abs(amount - call_amount) <= EPSILON_BB:
            return True
    return False


def trusted_call_amount(state: dict[str, Any]) -> float | None:
    """Prefer table contributions when button OCR accidentally reads Hero's blind."""

    controls = as_dict(state.get("action_controls"))
    control_call = number_or_none(controls.get("call_amount_bb"))
    table_call = number_or_none(as_dict(state.get("table")).get("to_call_bb"))
    if table_call is not None and control_call is not None:
        if abs(table_call - control_call) > EPSILON_BB:
            return table_call
    return control_call if control_call is not None else table_call


def amount_matches(seat: dict[str, Any] | None, expected: float) -> bool:
    if not seat:
        return False
    observed = number_or_none(seat.get("bet_bb"))
    return observed is not None and abs(observed - expected) <= EPSILON_BB


def seat_bets(state: dict[str, Any]) -> dict[str, float]:
    return {
        str(seat.get("seat")): number_or_none(seat.get("bet_bb")) or 0.0
        for seat in ordered_seats(state)
        if seat.get("seat")
    }


def seat_statuses(state: dict[str, Any]) -> dict[str, str]:
    return {
        str(seat.get("seat")): str(seat.get("status") or "")
        for seat in ordered_seats(state)
        if seat.get("seat")
    }


def forced_blind_posts(sizes: dict[str, float]) -> dict[str, float]:
    posts = {"SB": sizes["sb"], "BB": sizes["bb"]}
    if "third_blind" in sizes:
        posts["THIRD_BLIND"] = sizes["third_blind"]
    return posts


def opening_floor(sizes: dict[str, float]) -> float:
    return max(forced_blind_posts(sizes).values())


def largest_raise_to(history: list[dict[str, Any]], default: float = 1.0) -> float:
    return max(
        (number_or_none(item.get("amount_bb")) or 0.0 for item in history if item.get("action") in raise_actions()),
        default=default,
    )


def raise_events(history: list[dict[str, Any]]) -> int:
    return sum(1 for item in history if item.get("action") in raise_actions())


def raise_action_name(raise_count: int) -> str:
    return {0: "raise", 1: "3bet", 2: "4bet", 3: "5bet"}.get(raise_count, "all_in")


def raise_actions() -> set[str]:
    return {"raise", "open", "3bet", "4bet", "5bet", "all_in"}


def history_event(index: int, seat: dict[str, Any], action: str, amount_bb: float | None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "index": index,
        "position": seat.get("position"),
        "seat": seat.get("seat"),
        "action": action,
        "is_hero": str(seat.get("seat") or "") == "bottom_hero",
        "source": "cv_visible_prehero",
    }
    if amount_bb is not None:
        event["amount_bb"] = round(amount_bb, 2)
    return event


def action_order(seat: dict[str, Any]) -> int:
    return int_or_none(seat.get("preflop_action_order")) or 10_000


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
