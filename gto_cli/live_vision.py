from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .cv_advisor import attach_gto_advice
from .preflop_tracker import PreflopActionTracker
from .video_vision import (
    BOARD_CARD_ROIS,
    HERO_CARD_ROIS,
    analyze_video_frame,
    annotate_video_frame,
    choose_template,
    detect_action_controls,
    load_cv,
    load_ocr,
    sample_times,
    scale_roi,
)


def analyze_realtime_video(
    video_path: Path,
    output_dir: Path,
    template_path: Path | None = None,
    seat_count: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    every_sec: float = 1.0,
    middle: bool = False,
    max_frames: int | None = None,
    min_confidence: float = 0.45,
    trigger: str = "state-change",
    use_ocr: bool = True,
    visual_threshold: float = 2.4,
    min_event_gap_sec: float = 1.0,
    dealer_refresh_frames: int = 1,
    save_frames: bool = False,
    save_annotated: bool = False,
    with_advice: bool = False,
    advice_iterations: int = 600,
    effective_stack_bb: float = 100.0,
    villain_profile: str = "standard",
    ocr_scale: float = 1.0,
    ocr_action_only: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    cv2, _np = load_cv()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "event_frames"
    annotated_dir = output_dir / "event_annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 19.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps else 0.0
    if middle:
        start_sec = duration_sec * 0.25 if start_sec is None else start_sec
        end_sec = duration_sec * 0.75 if end_sec is None else end_sec
    else:
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = duration_sec if end_sec is None else end_sec

    template_path = choose_template(template_path)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"cannot read dealer template: {template_path}")

    ocr = load_ocr() if use_ocr else None
    sequential_read = trigger == "frame" and every_sec <= (1.5 / max(fps, 1.0))
    jsonl_path = output_dir / "events.jsonl"
    events_path = output_dir / "events.json"
    current_path = output_dir / "current_state.json"
    summary_path = output_dir / "realtime_summary.json"

    events: list[dict[str, Any]] = []
    previous_signature: str | None = None
    previous_visual: Any | None = None
    last_visual_event_sec = float("-inf")
    dealer_button_cache: dict[str, Any] | None = None
    last_dealer_refresh_sample = -10**9
    card_roi_cache_signature: str | None = None
    card_roi_cache: dict[str, Any] | None = None
    card_cache_hits = 0
    card_cache_misses = 0
    hero_card_cache: dict[str, Any] | None = None
    preflop_tracker = PreflopActionTracker()
    processed_frames = 0
    emitted_events = 0

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample_index, timestamp, frame_index, frame in iter_video_samples(
            cap,
            fps=fps,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            every_sec=every_sec,
            max_frames=max_frames,
            sequential_read=sequential_read,
        ):
            loop_started = time.perf_counter()
            processed_frames += 1
            visual_small = cv2.resize(frame, (160, 112), interpolation=cv2.INTER_AREA)
            visual_gray = cv2.cvtColor(visual_small, cv2.COLOR_BGR2GRAY)
            visual_diff = 0.0 if previous_visual is None else float(cv2.absdiff(visual_gray, previous_visual).mean())
            previous_visual = visual_gray

            if trigger == "visual-change":
                enough_gap = timestamp - last_visual_event_sec >= min_event_gap_sec
                visually_changed = previous_signature is None or (visual_diff >= visual_threshold and enough_gap)
                if not visually_changed:
                    continue
                last_visual_event_sec = timestamp

            try:
                refresh_dealer = (
                    dealer_button_cache is None
                    or dealer_refresh_frames <= 1
                    or sample_index - last_dealer_refresh_sample >= dealer_refresh_frames
                )
                cards_hint = None
                if trigger == "frame":
                    card_signature = roi_signature(frame, (*HERO_CARD_ROIS, *BOARD_CARD_ROIS))
                    if card_signature == card_roi_cache_signature:
                        cards_hint = card_roi_cache
                        card_cache_hits += 1
                    else:
                        card_roi_cache_signature = card_signature
                        card_cache_misses += 1
                frame_ocr = ocr
                ocr_mode = "full" if ocr is not None else "disabled"
                if ocr_action_only and ocr is not None:
                    quick_action_controls = detect_action_controls(frame, [])
                    if quick_action_controls.get("visible"):
                        frame_ocr = ocr
                        ocr_mode = "action_only_used"
                    else:
                        frame_ocr = None
                        ocr_mode = "action_only_skipped"
                frame_result = analyze_video_frame(
                    frame,
                    template,
                    seat_count=seat_count,
                    min_confidence=min_confidence,
                    ocr=frame_ocr,
                    dealer_button_hint=None if refresh_dealer else dealer_button_cache,
                    cards_hint=cards_hint,
                    ocr_scale=ocr_scale,
                )
                if refresh_dealer:
                    dealer_button_cache = frame_result.get("dealer_button")
                    last_dealer_refresh_sample = sample_index
                if trigger == "frame" and cards_hint is None:
                    card_roi_cache = frame_result.get("cards")
                frame_result["ok"] = True
                state = build_realtime_state(
                    frame_result,
                    video_path=video_path,
                    timestamp_sec=round(float(timestamp), 3),
                    frame_index=frame_index,
                    sample_index=sample_index,
                )
                hero_card_cache = stabilize_hero_cards(state, hero_card_cache)
                state["source"]["dealer_button_cached"] = not refresh_dealer and dealer_button_cache is not None
                preflop_tracker.update(state)
                if with_advice:
                    attach_gto_advice(
                        state,
                        iterations=advice_iterations,
                        effective_stack_bb=effective_stack_bb,
                        villain_profile=villain_profile,
                    )
                state["source"]["visual_diff"] = round(float(visual_diff), 4)
                state["source"]["ocr_mode"] = ocr_mode
                state["source"]["cv_timing_ms"] = frame_result.get("timing_ms") or {}
                state["source"]["ocr_item_count"] = frame_result.get("ocr_item_count")
                state["source"]["cards_hint_used"] = frame_result.get("cards_hint_used")
                state["source"]["card_cache_hit"] = cards_hint is not None
                if trigger == "frame":
                    signature = "frame"
                    should_emit = True
                    reason = "frame"
                else:
                    signature = state_signature(state)
                    should_emit = signature != previous_signature
                    if trigger == "visual-change":
                        reason = "initial" if previous_signature is None else "visual_changed"
                    else:
                        reason = "initial" if previous_signature is None else "state_changed"
                    previous_signature = signature
            except Exception as error:
                state = build_error_state(
                    error,
                    video_path=video_path,
                    timestamp_sec=round(float(timestamp), 3),
                    frame_index=frame_index,
                    sample_index=sample_index,
                )
                state["source"]["visual_diff"] = round(float(visual_diff), 4)
                if trigger == "frame":
                    signature = "frame-error"
                    should_emit = True
                    reason = "error"
                else:
                    signature = state_signature(state)
                    should_emit = signature != previous_signature
                    reason = "error" if previous_signature is None else ("visual_error" if trigger == "visual-change" else "error_changed")
                    previous_signature = signature

            if not should_emit:
                continue

            event_index = emitted_events
            event_frame_path = ""
            annotated_path = ""
            basename = f"event_{event_index:04d}_{int(timestamp):06d}s"
            if save_frames:
                event_frame_path = str(frames_dir / f"{basename}.png")
                cv2.imwrite(event_frame_path, frame)
            if save_annotated and state.get("ok"):
                annotated_path = str(annotated_dir / f"{basename}.png")
                annotate_video_frame(frame, frame_result, Path(annotated_path))

            state["source"]["frame_path"] = event_frame_path
            state["source"]["annotated_path"] = annotated_path
            state["source"]["analysis_ms"] = round((time.perf_counter() - loop_started) * 1000, 1)
            state["event"] = {
                "index": event_index,
                "trigger": trigger,
                "reason": reason,
                "signature": signature,
            }
            stream.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            events.append(state)
            current_json = (
                json.dumps(state, ensure_ascii=False, separators=(",", ":"))
                if trigger == "frame"
                else json.dumps(state, ensure_ascii=False, indent=2)
            )
            current_path.write_text(current_json, encoding="utf-8")
            emitted_events += 1

    cap.release()
    elapsed_sec = time.perf_counter() - started_at
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ok": True,
        "source": {
            "kind": "video",
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
        },
        "template": str(template_path),
        "output_dir": str(output_dir),
        "trigger": trigger,
        "ocr_enabled": bool(use_ocr and ocr is not None),
        "visual_threshold": visual_threshold if trigger == "visual-change" else None,
        "min_event_gap_sec": min_event_gap_sec if trigger == "visual-change" else None,
        "dealer_refresh_frames": dealer_refresh_frames,
        "advice_enabled": with_advice,
        "ocr_scale": ocr_scale if use_ocr and ocr is not None else None,
        "ocr_action_only": ocr_action_only if use_ocr and ocr is not None else None,
        "sequential_read": sequential_read,
        "sample": {
            "start_sec": round(float(start_sec), 3),
            "end_sec": round(float(end_sec), 3),
            "every_sec": every_sec,
            "processed_frames": processed_frames,
            "emitted_events": emitted_events,
            "wall_time_sec": round(elapsed_sec, 3),
            "avg_processed_frame_ms": round(elapsed_sec * 1000 / processed_frames, 2) if processed_frames else None,
            "effective_processing_fps": round(processed_frames / elapsed_sec, 3) if elapsed_sec > 0 else None,
            "card_cache_hits": card_cache_hits,
            "card_cache_misses": card_cache_misses,
        },
        "timing": event_source_timing_summary(events),
        "files": {
            "events_jsonl": str(jsonl_path),
            "events_json": str(events_path),
            "current_state": str(current_path),
            "summary": str(summary_path),
        },
        "events": events,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not current_path.exists():
        current_path.write_text(json.dumps({"ok": False, "error": "no events emitted"}, indent=2), encoding="utf-8")
    return summary


def build_realtime_state(
    frame_result: dict[str, Any],
    video_path: Path,
    timestamp_sec: float,
    frame_index: int,
    sample_index: int,
) -> dict[str, Any]:
    hero = frame_result.get("hero", {})
    dealer = frame_result.get("dealer", {})
    cards = frame_result.get("cards", {})
    board = cards.get("board", [])
    pot = frame_result.get("pot") or {}
    seats = [normalize_seat(seat) for seat in frame_result.get("seats", [])]
    bets = [normalize_bet(seat) for seat in seats if seat.get("bet_bb") is not None]
    hero_bet = normalized_amount(hero.get("bet_bb")) or 0.0
    max_bet = max((bet["amount_bb"] for bet in bets), default=0.0)
    to_call = max(0.0, round(max_bet - hero_bet, 2))
    card_confidence = build_card_confidence(cards)
    action_controls = frame_result.get("action_controls") or {}
    hero_turn = build_hero_turn(action_controls)
    # "过牌/下注"是翻后无人下注时的常规面板。若公共牌仍在发牌动画，
    # 不能把暂时空白的公共牌区域当成翻前并要求补齐翻前行动历史。
    board_pending = board_dealing_animation(action_controls, board, hero_turn)
    street = "flop" if board_pending else street_from_board(board)

    return {
        "ok": True,
        "source": {
            "kind": "video",
            "path": str(video_path),
            "timestamp_sec": timestamp_sec,
            "frame_index": frame_index,
            "sample_index": sample_index,
            "frame_path": "",
            "annotated_path": "",
        },
        "table": {
            "seat_count": len(seats),
            "street": street,
            "dealer_seat_index": dealer.get("seat_index"),
            "dealer_seat": dealer.get("seat"),
            "dealer_position": dealer.get("position"),
            "pot_bb": normalized_amount(pot.get("amount_bb")),
            "to_call_bb": to_call,
            "board": board,
            "board_pending": board_pending,
        },
        "hero": {
            "seat_index": hero.get("seat_index"),
            "seat": hero.get("seat"),
            "position": hero.get("position"),
            "gto_position": hero.get("gto_position"),
            "cards": hero.get("cards", []),
            "distance_from_dealer_clockwise": hero.get("distance_from_dealer_clockwise"),
            "preflop_action_order": hero.get("preflop_action_order"),
            "postflop_action_order": hero.get("postflop_action_order"),
            "status": hero.get("status"),
            "has_cards": bool(hero.get("has_cards")),
            "bet_bb": normalized_amount(hero.get("bet_bb")),
            "is_turn": hero_turn["is_turn"],
        },
        "seats": seats,
        "bets": bets,
        "action_controls": action_controls,
        "hero_turn": hero_turn,
        "confidence": {
            "dealer_button": normalized_amount(frame_result.get("dealer_button", {}).get("confidence"), 4),
            "pot_ocr": normalized_amount(pot.get("confidence"), 4),
            "cards": card_confidence,
        },
    }


def board_dealing_animation(
    action_controls: dict[str, Any],
    board: list[str],
    hero_turn: dict[str, Any],
) -> bool:
    """Return true only for the brief postflop board-dealing transition."""

    if board or not hero_turn.get("is_turn"):
        return False
    actions = {str(action).lower() for action in action_controls.get("actions") or []}
    # 翻前的正常选择是弃牌/跟注/加注；只有翻后无人下注时才会出现过牌/下注。
    return {"check", "bet"}.issubset(actions) and not {"call", "raise"}.intersection(actions)


def stabilize_hero_cards(
    state: dict[str, Any],
    cache: dict[str, Any] | None,
    *,
    max_age_sec: float = 75.0,
    confirmations_required: int = 2,
) -> dict[str, Any] | None:
    if not state.get("ok"):
        return cache
    source = state.setdefault("source", {})
    hero = state.get("hero") or {}
    cards = normalize_card_list(hero.get("cards") or [])
    timestamp = float(source.get("timestamp_sec") or 0.0)
    confirmations_required = max(1, int(confirmations_required))

    if cache and timestamp - float(cache.get("last_seen_sec") or cache.get("timestamp_sec") or 0.0) > max_age_sec:
        cache = None

    if cache and hero_hand_boundary_detected(state, cache):
        cache = None

    if not cards:
        source["hero_cards_stabilized"] = False
        if cache and (not hero.get("has_cards") or str(hero.get("status") or "") == "folded_or_empty"):
            cache = dict(cache)
            cache["boundary_seen"] = True
            cache["last_seen_sec"] = timestamp
        return cache

    if is_complete_cards(cards):
        if cache is None or hero_hand_context_changed(state, cache):
            source["hero_cards_stabilized"] = False
            source["hero_cards_confirmation"] = f"1/{confirmations_required}"
            return hero_card_cache_from_state(
                state,
                cards,
                timestamp,
                confirmations_required=confirmations_required,
            )

        cache = refresh_hero_card_cache_context(cache, state, timestamp)
        cached_cards = list(cache.get("cards") or [])
        if bool(cache.get("confirmed")):
            if cards != cached_cards:
                apply_stabilized_hero_cards(
                    state,
                    cached_cards,
                    raw_cards=cards,
                    cache=cache,
                    reason="locked_same_hand",
                )
            else:
                source["hero_cards_stabilized"] = False
                source["hero_cards_locked"] = True
            return cache

        counts = dict(cache.get("candidate_counts") or {})
        candidate_key = cards_key(cards)
        counts[candidate_key] = int(counts.get(candidate_key) or 0) + 1
        cache["candidate_counts"] = counts
        cache.setdefault("candidate_values", {})[candidate_key] = list(cards)
        previous_key = cards_key(list(cache.get("cards") or []))
        winner_key = previous_key if previous_key in counts else candidate_key
        winner_count = int(counts.get(winner_key) or 0)
        for key, count in counts.items():
            if int(count) > winner_count:
                winner_key = key
                winner_count = int(count)
        winner_cards = list((cache.get("candidate_values") or {}).get(winner_key) or cards)
        cache["cards"] = winner_cards
        cache["confirmation_count"] = int(winner_count)
        cache["confirmed"] = int(winner_count) >= confirmations_required
        source["hero_cards_confirmation"] = f"{int(winner_count)}/{confirmations_required}"
        if cache["confirmed"]:
            source["hero_cards_locked"] = True
            if cards != winner_cards:
                apply_stabilized_hero_cards(
                    state,
                    winner_cards,
                    raw_cards=cards,
                    cache=cache,
                    reason="temporal_consensus",
                )
            else:
                source["hero_cards_stabilized"] = False
        else:
            source["hero_cards_locked"] = False
            if cards != winner_cards:
                apply_stabilized_hero_cards(
                    state,
                    winner_cards,
                    raw_cards=cards,
                    cache=cache,
                    reason="candidate_hysteresis",
                )
            else:
                source["hero_cards_stabilized"] = False
        return cache

    if cache and should_fill_hero_cards_from_cache(state, cards, cache, timestamp, max_age_sec=max_age_sec):
        apply_stabilized_hero_cards(
            state,
            list(cache["cards"]),
            raw_cards=cards,
            cache=cache,
            reason="partial_read",
        )
        cache = refresh_hero_card_cache_context(cache, state, timestamp)
        return cache
    source["hero_cards_stabilized"] = False
    return cache


def normalize_card_list(cards: list[Any]) -> list[str]:
    return [str(card) for card in cards if str(card or "").strip()]


def is_complete_cards(cards: list[str]) -> bool:
    return len(cards) == 2 and all("?" not in card for card in cards)


def hero_card_cache_from_state(
    state: dict[str, Any],
    cards: list[str],
    timestamp: float,
    *,
    confirmations_required: int = 2,
) -> dict[str, Any]:
    hero = state.get("hero") or {}
    table = state.get("table") or {}
    count = 1
    return {
        "cards": list(cards),
        "timestamp_sec": timestamp,
        "last_seen_sec": timestamp,
        "dealer_position": table.get("dealer_position"),
        "dealer_seat": table.get("dealer_seat"),
        "hero_position": hero.get("gto_position") or hero.get("position"),
        "hero_seat": hero.get("seat"),
        "board": normalize_card_list(table.get("board") or []),
        "street": table.get("street"),
        "pot_bb": table.get("pot_bb"),
        "max_pot_bb": table.get("pot_bb"),
        "confirmed": count >= max(1, int(confirmations_required)),
        "confirmation_count": count,
        "candidate_counts": {cards_key(cards): count},
        "candidate_values": {cards_key(cards): list(cards)},
        "boundary_seen": False,
    }


def hero_hand_context_changed(state: dict[str, Any], cache: dict[str, Any]) -> bool:
    if hero_hand_boundary_detected(state, cache):
        return True
    hero = state.get("hero") or {}
    current_position = hero.get("gto_position") or hero.get("position")
    cached_position = cache.get("hero_position")
    if current_position and cached_position and current_position != cached_position:
        return True
    current_seat = hero.get("seat")
    cached_seat = cache.get("hero_seat")
    return bool(current_seat and cached_seat and current_seat != cached_seat)


def hero_hand_boundary_detected(state: dict[str, Any], cache: dict[str, Any]) -> bool:
    if cache.get("boundary_seen"):
        return True
    table = state.get("table") or {}
    current_board = normalize_card_list(table.get("board") or [])
    cached_board = normalize_card_list(cache.get("board") or [])
    current_street = str(table.get("street") or "").lower()
    cached_street = str(cache.get("street") or "").lower()
    if current_street == "preflop" and not current_board:
        if cached_board or cached_street in {"flop", "turn", "river"}:
            return True
        current_dealer = table.get("dealer_seat")
        cached_dealer = cache.get("dealer_seat")
        current_pot = table.get("pot_bb")
        try:
            opening_pot = current_pot is not None and float(current_pot) <= 3.6
        except (TypeError, ValueError):
            opening_pot = False
        if opening_pot and current_dealer and cached_dealer and current_dealer != cached_dealer:
            return True
    return False


def refresh_hero_card_cache_context(
    cache: dict[str, Any],
    state: dict[str, Any],
    timestamp: float,
) -> dict[str, Any]:
    refreshed = dict(cache)
    hero = state.get("hero") or {}
    table = state.get("table") or {}
    refreshed["last_seen_sec"] = timestamp
    refreshed["dealer_position"] = table.get("dealer_position")
    refreshed["dealer_seat"] = table.get("dealer_seat")
    refreshed["hero_position"] = hero.get("gto_position") or hero.get("position")
    refreshed["hero_seat"] = hero.get("seat")
    refreshed["street"] = table.get("street")
    current_pot = table.get("pot_bb")
    refreshed["pot_bb"] = current_pot
    try:
        if current_pot is not None:
            refreshed["max_pot_bb"] = max(
                float(current_pot),
                float(refreshed.get("max_pot_bb") or 0.0),
            )
    except (TypeError, ValueError):
        pass
    current_board = normalize_card_list(table.get("board") or [])
    if len(current_board) >= len(refreshed.get("board") or []):
        refreshed["board"] = current_board
    refreshed["boundary_seen"] = False
    return refreshed


def apply_stabilized_hero_cards(
    state: dict[str, Any],
    cards: list[str],
    *,
    raw_cards: list[str],
    cache: dict[str, Any],
    reason: str,
) -> None:
    hero = state.get("hero") or {}
    hero["cards"] = list(cards)
    hero["has_cards"] = True
    source = state.setdefault("source", {})
    source["hero_cards_stabilized"] = True
    source["hero_cards_stabilized_from_sec"] = cache.get("timestamp_sec")
    source["hero_cards_stabilization_reason"] = reason
    source["hero_cards_raw"] = list(raw_cards)
    source["hero_cards_locked"] = bool(cache.get("confirmed"))
    confidence = state.setdefault("confidence", {}).setdefault("cards", {})
    confidence["hero_stabilized_from"] = list(cards)


def cards_key(cards: list[str]) -> str:
    return "|".join(str(card) for card in cards)


def should_fill_hero_cards_from_cache(
    state: dict[str, Any],
    current_cards: list[str],
    cache: dict[str, Any],
    timestamp: float,
    *,
    max_age_sec: float,
    allow_partial_overlap: bool = False,
) -> bool:
    if not current_cards:
        return False
    if timestamp - float(cache.get("timestamp_sec") or 0.0) > max_age_sec:
        return False
    hero = state.get("hero") or {}
    table = state.get("table") or {}
    current_position = hero.get("gto_position") or hero.get("position")
    if cache.get("dealer_position") != table.get("dealer_position"):
        return False
    if cache.get("hero_position") != current_position:
        return False
    if cache.get("hero_seat") and hero.get("seat") and cache.get("hero_seat") != hero.get("seat"):
        return False
    if not board_is_compatible(cache.get("board") or [], normalize_card_list(table.get("board") or [])):
        return False
    cached_cards = list(cache.get("cards") or [])
    known_current = [card for card in current_cards if "?" not in card]
    if known_current and allow_partial_overlap:
        if not any(card in cached_cards for card in known_current):
            return False
    elif known_current and not all(card in cached_cards for card in known_current):
        return False
    current_ranks = {card[0] for card in current_cards if card and card[0] != "?"}
    cached_ranks = {card[0] for card in cached_cards if card}
    return bool(current_ranks & cached_ranks)


def board_is_compatible(previous_board: list[str], current_board: list[str]) -> bool:
    if previous_board and not current_board:
        return False
    if not previous_board:
        return True
    shorter, longer = (previous_board, current_board) if len(previous_board) <= len(current_board) else (current_board, previous_board)
    return longer[: len(shorter)] == shorter


def hero_cards_high_confidence(state: dict[str, Any]) -> bool:
    hero_confidence = ((state.get("confidence") or {}).get("cards") or {}).get("hero") or []
    if len(hero_confidence) < 2:
        return False
    for item in hero_confidence[:2]:
        rank_confidence = normalized_amount(item.get("rank_confidence")) or 0.0
        suit_confidence = normalized_amount(item.get("suit_confidence")) or 0.0
        if rank_confidence < 0.62 or suit_confidence < 0.55:
            return False
    return True


def iter_video_samples(
    cap: Any,
    fps: float,
    start_sec: float,
    end_sec: float,
    every_sec: float,
    max_frames: int | None,
    sequential_read: bool,
) -> Any:
    if sequential_read:
        start_frame = max(0, int(round(start_sec * fps)))
        end_frame = max(start_frame, int(round(end_sec * fps)))
        frame_step = max(1, int(round(every_sec * fps)))
        cap.set(1, start_frame)
        frame_index = start_frame
        sample_index = 0
        emitted = 0
        while frame_index <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_index - start_frame) % frame_step == 0:
                yield sample_index, frame_index / fps, frame_index, frame
                sample_index += 1
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            frame_index += 1
        return

    for sample_index, timestamp in enumerate(sample_times(start_sec, end_sec, every_sec, max_frames)):
        frame_index = int(round(timestamp * fps))
        cap.set(1, frame_index)
        ok, frame = cap.read()
        if ok:
            yield sample_index, timestamp, frame_index, frame


def roi_signature(frame: Any, rois: tuple[tuple[float, float, float, float], ...]) -> str:
    cv2, _np = load_cv()
    height, width = frame.shape[:2]
    digest = hashlib.sha1()
    for roi in rois:
        x1, y1, x2, y2 = scale_roi(roi, width, height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 12), interpolation=cv2.INTER_AREA)
        digest.update((small // 16).astype("uint8").tobytes())
    return digest.hexdigest()


def event_source_timing_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    sources = [event.get("source") or {} for event in events if event.get("ok")]
    analysis_values: list[float] = []
    cv_component_values: dict[str, list[float]] = {}
    screen_component_values: dict[str, list[float]] = {}
    ocr_modes: dict[str, int] = {}
    cards_hint_used = 0
    card_cache_hit = 0
    for source in sources:
        ocr_mode = str(source.get("ocr_mode") or "")
        if ocr_mode:
            ocr_modes[ocr_mode] = ocr_modes.get(ocr_mode, 0) + 1
        if source.get("cards_hint_used"):
            cards_hint_used += 1
        if source.get("card_cache_hit"):
            card_cache_hit += 1
        analysis_ms = source.get("analysis_ms")
        if analysis_ms is not None:
            analysis_values.append(float(analysis_ms))
        cv_timing = source.get("cv_timing_ms") or {}
        for key, value in cv_timing.items():
            if value is None:
                continue
            cv_component_values.setdefault(str(key), []).append(float(value))
        screen_timing = source.get("screen_timing_ms") or {}
        for key, value in screen_timing.items():
            if value is None:
                continue
            screen_component_values.setdefault(str(key), []).append(float(value))
    return {
        "events": len(sources),
        "analysis_ms": numeric_stats(analysis_values),
        "cv_timing_ms": {key: numeric_stats(values) for key, values in sorted(cv_component_values.items())},
        "screen_timing_ms": {key: numeric_stats(values) for key, values in sorted(screen_component_values.items())},
        "cards_hint_used": cards_hint_used,
        "card_cache_hit": card_cache_hit,
        "ocr_modes": ocr_modes,
    }


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "median": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)
    p90_index = min(len(ordered) - 1, max(0, int(round(len(ordered) * 0.9)) - 1))
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 2),
        "median": round(float(statistics.median(ordered)), 2),
        "p90": round(float(ordered[p90_index]), 2),
        "max": round(float(max(ordered)), 2),
    }


def build_error_state(
    error: Exception,
    video_path: Path,
    timestamp_sec: float,
    frame_index: int,
    sample_index: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "source": {
            "kind": "video",
            "path": str(video_path),
            "timestamp_sec": timestamp_sec,
            "frame_index": frame_index,
            "sample_index": sample_index,
            "frame_path": "",
            "annotated_path": "",
        },
    }


def normalize_seat(seat: dict[str, Any]) -> dict[str, Any]:
    return {
        "seat_index": seat.get("index"),
        "seat": seat.get("name"),
        "position": seat.get("position"),
        "gto_position": seat.get("gto_position"),
        "distance_from_dealer_clockwise": seat.get("distance_from_dealer_clockwise"),
        "preflop_action_order": seat.get("preflop_action_order"),
        "postflop_action_order": seat.get("postflop_action_order"),
        "status": seat.get("status"),
        "has_cards": bool(seat.get("has_cards")),
        "bet_bb": normalized_amount(seat.get("bet_bb")),
    }


def normalize_bet(seat: dict[str, Any]) -> dict[str, Any]:
    return {
        "seat_index": seat.get("seat_index"),
        "seat": seat.get("seat"),
        "amount_bb": normalized_amount(seat.get("bet_bb")) or 0.0,
    }


def build_card_confidence(cards: dict[str, Any]) -> dict[str, Any]:
    return {
        "hero": [card_confidence_item(item) for item in cards.get("hero_details", [])],
        "board": [card_confidence_item(item) for item in cards.get("board_details", [])],
    }


def card_confidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "card": item.get("card"),
        "slot": item.get("index"),
        "rank_confidence": normalized_amount(item.get("rank_confidence"), 4),
        "suit_confidence": normalized_amount(item.get("suit_confidence"), 4),
        "suit_margin": normalized_amount(item.get("suit_margin"), 4),
        "color": item.get("color"),
    }


def state_signature(state: dict[str, Any]) -> str:
    payload = {
        "ok": state.get("ok"),
        "error": state.get("error"),
        "table": state.get("table"),
        "hero": state.get("hero"),
        "seats": state.get("seats"),
        "bets": state.get("bets"),
        "action_controls": state.get("action_controls"),
        "hero_turn": state.get("hero_turn"),
        "gto_advice": state.get("gto_advice"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def street_from_board(board: list[str]) -> str:
    count = len(board)
    if count >= 5:
        return "river"
    if count == 4:
        return "turn"
    if count == 3:
        return "flop"
    return "preflop"


def normalized_amount(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_hero_turn(action_controls: dict[str, Any]) -> dict[str, Any]:
    actions = list(action_controls.get("actions") or [])
    red_buttons = list(action_controls.get("red_button_regions") or [])
    bottom_texts = list(action_controls.get("bottom_texts") or [])
    visible = bool(action_controls.get("visible"))
    if not visible:
        return {
            "is_turn": False,
            "confidence": 0.0,
            "reason": "action_controls_not_visible",
            "actions": [],
        }

    # The client can expose a pre-action quick-fold button before it is
    # actually Hero's turn.  A fold-only surface is therefore not enough to
    # request normal GTO advice.
    normal_actions = {"call", "raise", "check", "bet"}
    if not normal_actions.intersection(actions):
        return {
            "is_turn": False,
            "confidence": 0.72 if "fold" in actions or red_buttons else 0.0,
            "reason": "fast_fold_only",
            "actions": actions,
            "call_amount_bb": normalized_amount(action_controls.get("call_amount_bb")),
            "raise_amount_bb": normalized_amount(action_controls.get("raise_amount_bb")),
        }

    # 这套客户端在真正轮到 Hero 时会显示可点击的红色操作面板。
    # 单独出现“加注”等文字也可能只是上一手动作标签，不能据此输出建议。
    if not red_buttons:
        return {
            "is_turn": False,
            "confidence": 0.0,
            "reason": "action_buttons_not_visible",
            "actions": actions,
            "call_amount_bb": normalized_amount(action_controls.get("call_amount_bb")),
            "raise_amount_bb": normalized_amount(action_controls.get("raise_amount_bb")),
        }

    if red_buttons and actions:
        confidence = 0.96
        reason = "red_buttons_and_action_text"
    elif red_buttons:
        confidence = 0.90
        reason = "red_buttons_visible"

    return {
        "is_turn": True,
        "confidence": confidence,
        "reason": reason,
        "actions": actions,
        "call_amount_bb": normalized_amount(action_controls.get("call_amount_bb")),
        "raise_amount_bb": normalized_amount(action_controls.get("raise_amount_bb")),
    }


def format_realtime_summary(payload: dict[str, Any], limit: int = 12) -> str:
    sample = payload["sample"]
    lines = [
        f"Realtime CV source: {payload['source']['path']}",
        f"Output dir: {payload['output_dir']}",
        f"Trigger: {payload['trigger']}",
        f"OCR enabled: {payload.get('ocr_enabled')}",
        f"Sequential read: {payload.get('sequential_read')}; dealer refresh frames: {payload.get('dealer_refresh_frames')}",
        f"Processed frames: {sample['processed_frames']}; emitted events: {sample['emitted_events']}",
        f"Processing: {sample.get('avg_processed_frame_ms')} ms/frame; {sample.get('effective_processing_fps')} fps",
        f"Card cache: {sample.get('card_cache_hits')} hits; {sample.get('card_cache_misses')} misses",
    ]
    timing = payload.get("timing") or {}
    cv_timing = timing.get("cv_timing_ms") or {}
    if timing:
        analysis = timing.get("analysis_ms") or {}
        cards = cv_timing.get("cards_ms") or {}
        ocr = cv_timing.get("ocr_ms") or {}
        lines.append(
            "Timing: "
            f"analysis median={analysis.get('median')}ms p90={analysis.get('p90')}ms; "
            f"cards median={cards.get('median')}ms; ocr median={ocr.get('median')}ms"
        )
        lines.append(
            f"Cache hints used: {timing.get('cards_hint_used')} / {timing.get('events')} | OCR modes: "
            f"{json.dumps(timing.get('ocr_modes') or {}, ensure_ascii=False)}"
        )
    lines.extend(
        [
            f"JSONL stream: {payload['files']['events_jsonl']}",
            f"Current state: {payload['files']['current_state']}",
        ]
    )
    for event in payload.get("events", [])[:limit]:
        if not event.get("ok"):
            lines.append(f"{event['source']['timestamp_sec']:>7.1f}s error: {event.get('error')}")
            continue
        table = event["table"]
        hero = event["hero"]
        bets = ", ".join(f"{bet['seat']} {bet['amount_bb']:g}BB" for bet in event["bets"]) or "none"
        advice = event.get("gto_advice") or {}
        advice_text = ""
        if advice:
            advice_text = f" advice={advice.get('summary') or advice.get('reason')}"
        lines.append(
            f"{event['source']['timestamp_sec']:>7.1f}s "
            f"{event['event']['reason']} street={table['street']} D={table['dealer_seat']} "
            f"hero={hero['position']} cards={' '.join(hero['cards']) or '-'} "
            f"board={' '.join(table['board']) or '-'} pot={table['pot_bb']}BB bets={bets}{advice_text}"
        )
    remaining = len(payload.get("events", [])) - limit
    if remaining > 0:
        lines.append(f"... {remaining} more events in events.jsonl/events.json")
    return "\n".join(lines)
