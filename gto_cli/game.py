from __future__ import annotations

import random
import uuid
from typing import Any

from .cards import build_deck, evaluate_7, score_category_name
from .simulator import describe_cards, hand_category_label

STREETS = ("preflop", "flop", "turn", "river")
STARTING_STACK = 50.0
SMALL_BLIND = 0.5
BIG_BLIND = 1.0
STREET_LABELS = {
    "preflop": "翻前",
    "flop": "翻牌",
    "turn": "转牌",
    "river": "河牌",
}
STREET_TEXT = {
    "preflop": "翻前：还没发公共牌",
    "flop": "翻牌：桌上有 3 张公共牌",
    "turn": "转牌：桌上有 4 张公共牌",
    "river": "河牌：5 张公共牌都发完了",
}
PLAYER_NAMES = ("你", "左下电脑", "左上电脑", "对面电脑", "右上电脑", "右下电脑")
PLAYER_IDS = ("hero", "bot1", "bot2", "bot3", "bot4", "bot5")
MATCH_MODE_TEXT = {
    "small": "小对局",
    "big": "大对局",
    "ai": "AI 对战",
}


def new_game(
    level: str = "simple",
    seed: int | None = None,
    match_mode: str = "small",
    starting_stack: float = STARTING_STACK,
    stacks: dict[str, float] | None = None,
    hand_number: int = 1,
    match_log: list[str] | None = None,
    hand_records: list[dict[str, Any]] | None = None,
    first_turn_index: int | None = None,
    score: float = 0.0,
    score_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deck = build_deck()
    rng.shuffle(deck)
    match_mode = normalize_match_mode(match_mode)
    starting_stack = normalize_starting_stack(starting_stack)
    first_turn_index = normalize_player_index(
        rng.randrange(len(PLAYER_NAMES)) if first_turn_index is None else first_turn_index
    )
    dealer_index = dealer_index_for_first_turn(first_turn_index)
    small_blind_index = normalize_player_index(dealer_index + 1)
    big_blind_index = normalize_player_index(dealer_index + 2)
    players = []
    pot = 0.0
    for index, (player_id, name) in enumerate(zip(PLAYER_IDS, PLAYER_NAMES)):
        player_starting_stack = round((stacks or {}).get(player_id, starting_stack), 1)
        blind = blind_amount_for_index(index, small_blind_index, big_blind_index, player_starting_stack)
        pot = round(pot + blind, 1)
        players.append(
            {
                "id": player_id,
                "name": name,
                "cards": [deck.pop(), deck.pop()],
                "stack": round(player_starting_stack - blind, 1),
                "folded": False,
                "acted": False,
                "round_bet": blind,
                "blind_paid": blind,
                "blind_role": blind_role_for_index(index, small_blind_index, big_blind_index),
                "last_action": "等待",
                "is_hero": index == 0,
            }
        )

    mode_text = MATCH_MODE_TEXT[match_mode]
    first_actor = PLAYER_NAMES[first_turn_index]
    hero_order = hero_order_position(first_turn_index)
    small_blind_player = players[small_blind_index]
    big_blind_player = players[big_blind_index]
    intro = (
        f"{mode_text}第 {hand_number} 手，翻前。"
        f"{small_blind_player['name']} 是小盲，先出 {small_blind_player['round_bet']:g} pt；"
        f"{big_blind_player['name']} 是大盲，先出 {big_blind_player['round_bet']:g} pt。"
        f"底池现在 {pot:g} pt。"
        f"这手从{first_actor}开始做决定，你第 {hero_order} 个做决定。"
    )
    game = {
        "id": uuid.uuid4().hex,
        "level": level,
        "match_mode": match_mode,
        "starting_stack": starting_stack,
        "hand_number": hand_number,
        "match_done": False,
        "eliminated": [],
        "match_log": list(match_log or []),
        "hand_records": list(hand_records or []),
        "score": round(score, 1),
        "last_score_delta": 0.0,
        "score_history": list(score_history or []),
        "deck": deck,
        "players": players,
        "board": [],
        "street": "preflop",
        "pot": pot,
        "pot_awarded": False,
        "table_bet": players[big_blind_index]["round_bet"],
        "first_turn_index": first_turn_index,
        "turn_index": first_active_index(players, first_turn_index),
        "status": "playing",
        "winner": None,
        "message": intro,
        "log": [intro],
        "rng_seed": rng.randrange(1, 1_000_000),
    }
    return game


def public_game(game: dict[str, Any], auto_bots: bool = True) -> dict[str, Any]:
    if auto_bots:
        run_bots_until_hero_or_end(game)
    done = game["status"] != "playing"
    hero = hero_player(game)
    order_map = turn_order_map(game)
    dealer_index = dealer_index_for_game(game)
    small_blind_index = small_blind_index_for_game(game)
    big_blind_index = big_blind_index_for_game(game)
    dealer_player = game["players"][dealer_index]
    active_turn = current_player(game)
    return {
        "id": game["id"],
        "level": game["level"],
        "match_mode": game.get("match_mode", "small"),
        "match_mode_text": MATCH_MODE_TEXT.get(game.get("match_mode", "small"), "小对局"),
        "starting_stack": round(game.get("starting_stack", STARTING_STACK), 1),
        "hand_number": game.get("hand_number", 1),
        "match_done": game.get("match_done", False),
        "eliminated": game.get("eliminated", []),
        "match_log": game.get("match_log", [])[-20:],
        "hand_log": game.get("log", []),
        "hand_records": game.get("hand_records", [])[-20:],
        "score": round(game.get("score", 0.0), 1),
        "last_score_delta": round(game.get("last_score_delta", 0.0), 1),
        "score_history": game.get("score_history", [])[-80:],
        "first_actor": game["players"][game.get("first_turn_index", 0)]["name"],
        "dealer": dealer_player["name"],
        "dealer_id": dealer_player["id"],
        "dealer_index": dealer_index,
        "small_blind": blind_info(game, small_blind_index, "小盲"),
        "big_blind": blind_info(game, big_blind_index, "大盲"),
        "current_player": active_turn["name"] if active_turn else None,
        "current_player_id": active_turn["id"] if active_turn else None,
        "hero_order": order_map[hero["id"]],
        "turn_order": turn_order_names(game),
        "players": [
            public_player(
                player,
                done,
                game.get("eliminated", []),
                order_map[player["id"]],
                player["id"] == dealer_player["id"],
            )
            for player in game["players"]
        ],
        "hero_cards": hero["cards"],
        "villain_cards": visible_villain_cards(game) if done else [],
        "board": game["board"],
        "street": game["street"],
        "street_label": STREET_LABELS[game["street"]],
        "street_text": STREET_TEXT[game["street"]],
        "pot": round(game["pot"], 1),
        "hero_stack": round(hero["stack"], 1),
        "to_call": round(to_call_for_player(game, hero), 1),
        "status": game["status"],
        "winner": game["winner"],
        "message": game["message"],
        "log": game["log"][-12:],
        "actions": available_actions(game),
        "hint": game_hint(game),
    }


def public_player(
    player: dict[str, Any],
    reveal_cards: bool,
    eliminated: list[str],
    order_position: int,
    is_dealer: bool,
) -> dict[str, Any]:
    return {
        "id": player["id"],
        "name": player["name"],
        "cards": player["cards"] if (player["is_hero"] or reveal_cards) else [],
        "stack": round(player["stack"], 1),
        "folded": player["folded"],
        "round_bet": round(player["round_bet"], 1),
        "blind_paid": round(player.get("blind_paid", 0.0), 1),
        "last_action": player["last_action"],
        "is_hero": player["is_hero"],
        "is_dealer": is_dealer,
        "blind_role": player.get("blind_role"),
        "out": player["name"] in eliminated,
        "order_position": order_position,
    }


def visible_villain_cards(game: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    for player in game["players"]:
        if not player["is_hero"]:
            cards.extend(player["cards"])
    return cards


def available_actions(game: dict[str, Any]) -> list[str]:
    if game["status"] != "playing":
        if game.get("match_mode") in {"big", "ai"}:
            return ["new_match"] if game.get("match_done") else ["new_hand"]
        return ["new_hand"]
    player = current_player(game)
    if not player or not player["is_hero"]:
        return []
    if to_call_for_player(game, player) > 0:
        return ["fold", "call"]
    return ["check", "bet", "fold"]


def apply_action(game: dict[str, Any], action: str, auto_bots: bool = True) -> dict[str, Any]:
    if auto_bots:
        run_bots_until_hero_or_end(game)
    action = normalize_action(action)
    if game["status"] != "playing":
        if action == "new_hand" and game.get("match_mode") in {"big", "ai"} and not game.get("match_done"):
            start_next_hand(game)
        return game
    if action == "bot_step":
        if not auto_bots:
            step_bot_once(game)
        return game
    player = current_player(game)
    if not player or not player["is_hero"]:
        game["message"] = f"现在轮到 {player['name']}。" if player else "这一手正在等待。"
        return game
    if action not in available_actions(game):
        game["message"] = "这个动作现在不能选。"
        return game
    perform_action(game, hero_player(game), action)
    continue_after_action(game)
    if auto_bots:
        run_bots_until_hero_or_end(game)
    return game


def run_bots_until_hero_or_end(game: dict[str, Any]) -> None:
    guard = 0
    while game["status"] == "playing":
        if not step_bot_once(game):
            return
        guard += 1
        if guard > 80:
            game["message"] = "这局卡住了，建议新开一局。"
            return


def step_bot_once(game: dict[str, Any]) -> bool:
    player = current_player(game)
    if not player or player["is_hero"] or game["status"] != "playing":
        return False
    perform_action(game, player, choose_bot_action(game, player))
    continue_after_action(game)
    return True


def perform_action(game: dict[str, Any], player: dict[str, Any], action: str) -> None:
    call_amount = to_call_for_player(game, player)
    if action == "fold":
        player["folded"] = True
        player["acted"] = True
        player["last_action"] = "不玩了"
        game["log"].append(f"{player['name']} 不玩了。")
        return

    if action == "call":
        amount = pay(game, player, call_amount)
        player["acted"] = True
        if amount <= 0:
            player["last_action"] = "没钱可放，等结果"
            game["log"].append(f"{player['name']} 没钱可放，等结果。")
        else:
            player["last_action"] = f"跟上，放了 {amount:g} 份"
            game["log"].append(f"{player['name']} 跟上，放了 {amount:g} 份。")
        return

    if action == "check":
        player["acted"] = True
        player["last_action"] = "先不出"
        game["log"].append(f"{player['name']} 先不出。")
        return

    if action == "bet":
        amount = bet_size(game, player)
        pay(game, player, amount)
        game["table_bet"] = player["round_bet"]
        for other in active_players(game):
            if other["id"] != player["id"]:
                other["acted"] = False
        player["acted"] = True
        player["last_action"] = f"先出 {amount:g} 份"
        game["log"].append(f"{player['name']} 先出 {amount:g} 份。")


def continue_after_action(game: dict[str, Any]) -> None:
    if len(active_players(game)) == 1:
        finish_by_last_player(game)
        return
    if betting_round_complete(game):
        if game["street"] == "river":
            finish_by_showdown(game)
        else:
            next_street(game)
        return
    game["turn_index"] = next_active_index(game, game["turn_index"])
    player = current_player(game)
    game["message"] = f"现在轮到 {player['name']}。"


def betting_round_complete(game: dict[str, Any]) -> bool:
    players = active_players(game)
    if not players:
        return True
    return all(
        player["acted"] and (player["round_bet"] >= game["table_bet"] or player["stack"] <= 0)
        for player in players
    )


def next_street(game: dict[str, Any]) -> None:
    index = STREETS.index(game["street"])
    game["street"] = STREETS[index + 1]
    needed = {"flop": 3, "turn": 4, "river": 5}[game["street"]]
    while len(game["board"]) < needed:
        game["board"].append(game["deck"].pop())
    game["table_bet"] = 0.0
    for player in game["players"]:
        player["round_bet"] = 0.0
        player["acted"] = False
        if not player["folded"]:
            player["last_action"] = "等待"
    game["turn_index"] = first_active_index(game["players"], postflop_first_index_for_game(game))
    game["log"].append(f"进入{STREET_LABELS[game['street']]}，现在桌上有 {len(game['board'])} 张公共牌。")
    player = current_player(game)
    game["message"] = f"进入{STREET_LABELS[game['street']]}，底池 {game['pot']:g} pt。现在轮到 {player['name']}。"


def finish_by_last_player(game: dict[str, Any]) -> None:
    winner = active_players(game)[0]
    game["status"] = "finished"
    game["winner"] = winner["id"]
    award_pot(game, [winner])
    complete_hand(game, f"{winner['name']} 拿走桌上的钱。")


def finish_by_showdown(game: dict[str, Any]) -> None:
    players = active_players(game)
    ranked = []
    for player in players:
        score = evaluate_7([*player["cards"], *game["board"]])
        ranked.append((score, player, hand_category_label(score_category_name(score))))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score = ranked[0][0]
    winners = [item for item in ranked if item[0] == best_score]
    game["status"] = "finished"
    if len(winners) == 1:
        winner = winners[0][1]
        game["winner"] = winner["id"]
        award_pot(game, [winner])
        result = f"{winner['name']} 赢了，牌大概是：{winners[0][2]}。"
    else:
        names = "、".join(item[1]["name"] for item in winners)
        game["winner"] = "tie"
        award_pot(game, [item[1] for item in winners])
        result = f"{names} 平分，牌大概都是：{winners[0][2]}。"
    complete_hand(game, result)


def award_pot(game: dict[str, Any], winners: list[dict[str, Any]]) -> None:
    if game.get("pot_awarded") or not winners:
        return
    share = round(game["pot"] / len(winners), 1)
    remainder = round(game["pot"] - share * len(winners), 1)
    for index, winner in enumerate(winners):
        amount = round(share + (remainder if index == 0 else 0), 1)
        winner["stack"] = round(winner["stack"] + amount, 1)
    game["pot_awarded"] = True


def complete_hand(game: dict[str, Any], result: str) -> None:
    game["message"] = result
    game["log"].append(result)
    score_delta: float | None = None
    if game.get("match_mode") == "ai":
        score_delta = round(hero_player(game)["stack"] - game.get("starting_stack", STARTING_STACK), 1)
        game["score"] = round(game.get("score", 0.0) + score_delta, 1)
        game["last_score_delta"] = score_delta
        score_line = f"你的积分变化：{signed_amount(score_delta)}，总积分 {game['score']:g}。"
        game["message"] = f"{result} {score_line}"
        game["log"].append(score_line)
        game.setdefault("score_history", []).append(
            {
                "hand_number": game.get("hand_number", 1),
                "delta": score_delta,
                "score": game["score"],
            }
        )
    game.setdefault("hand_records", []).append(
        {
            "hand_number": game.get("hand_number", 1),
            "mode_text": MATCH_MODE_TEXT.get(game.get("match_mode", "small"), "小对局"),
            "result": result,
            "score_delta": score_delta,
            "score": round(game.get("score", 0.0), 1),
            "starting_stack": round(game.get("starting_stack", STARTING_STACK), 1),
            "first_actor": game["players"][game.get("first_turn_index", 0)]["name"],
            "dealer": game["players"][dealer_index_for_game(game)]["name"],
            "small_blind": blind_info(game, small_blind_index_for_game(game), "小盲"),
            "big_blind": blind_info(game, big_blind_index_for_game(game), "大盲"),
            "hero_order": hero_order_position(game.get("first_turn_index", 0)),
            "turn_order": turn_order_names(game),
            "board": list(game["board"]),
            "log": list(game["log"]),
            "stacks": stack_summary(game),
            "players": [
                {
                    "name": player["name"],
                    "stack": round(player["stack"], 1),
                    "cards": list(player["cards"]),
                    "folded": player["folded"],
                }
                for player in game["players"]
            ],
        }
    )
    if game.get("match_mode") == "ai":
        game.setdefault("match_log", []).append(
            f"第 {game.get('hand_number', 1)} 手：{result} 积分 {signed_amount(score_delta or 0)}，总积分 {game['score']:g}。"
        )
    else:
        game.setdefault("match_log", []).append(
            f"第 {game.get('hand_number', 1)} 手：{result} 现在的钱：{stack_summary(game)}。"
        )

    if game.get("match_mode") != "big":
        return

    eliminated = [player["name"] for player in game["players"] if player["stack"] <= 0]
    if eliminated:
        game["match_done"] = True
        game["eliminated"] = eliminated
        ending = f"{'、'.join(eliminated)} 没钱了，大对局结束。"
        game["message"] = f"{result} {ending}"
        game["log"].append(ending)
        game["match_log"].append(ending)
    else:
        game["message"] = f"{result} 这一手结束了，下一手会沿用现在的钱。"


def start_next_hand(game: dict[str, Any]) -> None:
    if game.get("status") == "playing" or game.get("match_done"):
        return
    if game.get("match_mode") == "ai":
        stacks = None
    else:
        stacks = {player["id"]: player["stack"] for player in game["players"]}
    next_game = new_game(
        level=game.get("level", "simple"),
        match_mode=game.get("match_mode", "big"),
        starting_stack=game.get("starting_stack", STARTING_STACK),
        stacks=stacks,
        hand_number=game.get("hand_number", 1) + 1,
        match_log=game.get("match_log", []),
        hand_records=game.get("hand_records", []),
        first_turn_index=game.get("first_turn_index", 0) + 1,
        score=game.get("score", 0.0),
        score_history=game.get("score_history", []),
    )
    next_game["id"] = game["id"]
    game.clear()
    game.update(next_game)


def choose_bot_action(game: dict[str, Any], player: dict[str, Any]) -> str:
    if player["stack"] <= 0:
        return "call" if to_call_for_player(game, player) > 0 else "check"
    chance = estimate_win_chance(player, game)
    to_call = to_call_for_player(game, player)
    if to_call > 0:
        pressure = (to_call / max(game["pot"] + to_call, 1)) * 100
        return "call" if chance >= pressure + 12 else "fold"
    if chance >= 64 and player["stack"] > 0:
        return "bet"
    return "check"


def estimate_win_chance(player: dict[str, Any], game: dict[str, Any], trials: int = 140) -> float:
    opponents = [item for item in active_players(game) if item["id"] != player["id"]]
    if not opponents:
        return 100.0
    if len(game["board"]) == 5:
        score = evaluate_7([*player["cards"], *game["board"]])
        better = 0
        ties = 0
        for opponent in opponents:
            other = evaluate_7([*opponent["cards"], *game["board"]])
            if other > score:
                better += 1
            elif other == score:
                ties += 1
        if better:
            return 0.0
        return 50.0 if ties else 100.0

    rng = random.Random(len(game["deck"]) * 37 + len(game["board"]) * 19 + len(player["cards"]))
    wins = 0
    ties = 0
    needed = 5 - len(game["board"])
    for _ in range(trials):
        sample = rng.sample(game["deck"], needed)
        board = [*game["board"], *sample]
        score = evaluate_7([*player["cards"], *board])
        opponent_scores = [evaluate_7([*opponent["cards"], *board]) for opponent in opponents]
        best_other = max(opponent_scores)
        if score > best_other:
            wins += 1
        elif score == best_other:
            ties += 1
    return ((wins + ties * 0.5) / trials) * 100


def bet_size(game: dict[str, Any], player: dict[str, Any]) -> float:
    if game["street"] == "preflop":
        amount = 2.0
    else:
        amount = max(2.0, round(game["pot"] * 0.45, 1))
    max_other = max((item["stack"] for item in active_players(game) if item["id"] != player["id"]), default=0)
    return round(min(amount, player["stack"], max_other), 1)


def pay(game: dict[str, Any], player: dict[str, Any], amount: float) -> float:
    paid = round(min(amount, player["stack"]), 1)
    player["stack"] = round(player["stack"] - paid, 1)
    player["round_bet"] = round(player["round_bet"] + paid, 1)
    game["pot"] = round(game["pot"] + paid, 1)
    return paid


def game_hint(game: dict[str, Any]) -> list[str]:
    hero = hero_player(game)
    if game.get("match_mode") == "ai":
        opening = (
            f"这是 AI 对战第 {game.get('hand_number', 1)} 手。每手结束后会计算你的积分变化，"
            f"下一手继续从 {game.get('starting_stack', STARTING_STACK):g} pt 开始。"
        )
    elif game.get("match_mode") == "big":
        opening = (
            f"这是大对局第 {game.get('hand_number', 1)} 手。每个人的钱会带到下一手，"
            "谁的钱变成 0 就出局。"
        )
    else:
        opening = f"这是小对局。每个人从 {game.get('starting_stack', STARTING_STACK):g} pt 开始，这一手结束就结束。"
    lines = [
        opening,
        f"现在是{STREET_LABELS[game['street']]}，底池 {game['pot']:g} pt。",
        f"{game['players'][small_blind_index_for_game(game)]['name']} 是小盲，已经出 {game['players'][small_blind_index_for_game(game)].get('blind_paid', 0):g} pt；"
        f"{game['players'][big_blind_index_for_game(game)]['name']} 是大盲，已经出 {game['players'][big_blind_index_for_game(game)].get('blind_paid', 0):g} pt。",
        f"翻前从{game['players'][game.get('first_turn_index', 0)]['name']}开始做决定，你第 {hero_order_position(game.get('first_turn_index', 0))} 个做决定。",
        "你只能看到自己的牌、公共牌、每个人出了多少钱，以及别人刚才做了什么。",
        "德州扑克不能换牌。你的两张手牌会一直跟着你。",
        f"你的手牌是 {describe_cards(hero['cards'])}。",
    ]
    if game["board"]:
        lines.append(f"桌上的公共牌是 {describe_cards(game['board'])}。")
    else:
        lines.append("现在还没有公共牌。")
    call_amount = to_call_for_player(game, hero)
    if call_amount > 0:
        lines.append(f"有人已经出钱了。你要再放 {call_amount:g} 份才能继续。")
    else:
        lines.append("现在没人逼你出钱。牌弱可以先不出，牌强可以先出一些。")
    lines.append("右侧记录会告诉你：谁不玩了、谁跟上了、谁先出钱了。")
    return lines


def active_players(game: dict[str, Any]) -> list[dict[str, Any]]:
    return [player for player in game["players"] if not player["folded"]]


def hero_player(game: dict[str, Any]) -> dict[str, Any]:
    return game["players"][0]


def current_player(game: dict[str, Any]) -> dict[str, Any] | None:
    if game["status"] != "playing":
        return None
    return game["players"][game["turn_index"]]


def first_active_index(players: list[dict[str, Any]], start_index: int) -> int:
    count = len(players)
    for offset in range(count):
        index = (start_index + offset) % count
        player = players[index]
        if not player["folded"]:
            return index
    return normalize_player_index(start_index)


def next_active_index(game: dict[str, Any], start_index: int) -> int:
    count = len(game["players"])
    for offset in range(1, count + 1):
        index = (start_index + offset) % count
        if not game["players"][index]["folded"]:
            return index
    return start_index


def to_call_for_player(game: dict[str, Any], player: dict[str, Any]) -> float:
    return round(max(0.0, game["table_bet"] - player["round_bet"]), 1)


def turn_order_indexes(game: dict[str, Any]) -> list[int]:
    start = normalize_player_index(game.get("first_turn_index", 0))
    return [(start + offset) % len(game["players"]) for offset in range(len(game["players"]))]


def turn_order_map(game: dict[str, Any]) -> dict[str, int]:
    return {game["players"][index]["id"]: position + 1 for position, index in enumerate(turn_order_indexes(game))}


def turn_order_names(game: dict[str, Any]) -> list[str]:
    return [game["players"][index]["name"] for index in turn_order_indexes(game)]


def blind_amount_for_index(index: int, small_blind_index: int, big_blind_index: int, stack: float) -> float:
    if index == small_blind_index:
        return round(min(SMALL_BLIND, stack), 1)
    if index == big_blind_index:
        return round(min(BIG_BLIND, stack), 1)
    return 0.0


def blind_role_for_index(index: int, small_blind_index: int, big_blind_index: int) -> str | None:
    if index == small_blind_index:
        return "small"
    if index == big_blind_index:
        return "big"
    return None


def blind_info(game: dict[str, Any], index: int, label: str) -> dict[str, Any]:
    player = game["players"][index]
    return {
        "id": player["id"],
        "name": player["name"],
        "label": label,
        "amount": round(player.get("blind_paid", 0.0), 1),
    }


def dealer_index_for_first_turn(first_turn_index: int) -> int:
    return normalize_player_index(first_turn_index - 3)


def dealer_index_for_game(game: dict[str, Any]) -> int:
    return dealer_index_for_first_turn(game.get("first_turn_index", 0))


def small_blind_index_for_game(game: dict[str, Any]) -> int:
    return normalize_player_index(dealer_index_for_game(game) + 1)


def big_blind_index_for_game(game: dict[str, Any]) -> int:
    return normalize_player_index(dealer_index_for_game(game) + 2)


def postflop_first_index_for_game(game: dict[str, Any]) -> int:
    return normalize_player_index(dealer_index_for_game(game) + 1)


def hero_order_position(first_turn_index: int) -> int:
    return ((0 - normalize_player_index(first_turn_index)) % len(PLAYER_NAMES)) + 1


def normalize_player_index(index: int) -> int:
    return int(index) % len(PLAYER_NAMES)


def stack_summary(game: dict[str, Any]) -> str:
    return "，".join(f"{player['name']} {player['stack']:g} 份" for player in game["players"])


def normalize_match_mode(match_mode: str) -> str:
    if str(match_mode) in {"ai", "score", "challenge", "AI 对战", "ai对战"}:
        return "ai"
    if str(match_mode) in {"big", "long", "large", "大对局"}:
        return "big"
    return "small"


def normalize_starting_stack(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = STARTING_STACK
    return round(min(max(number, 5.0), 5000.0), 1)


def signed_amount(value: float) -> str:
    return f"+{value:g}" if value > 0 else f"{value:g}"


def normalize_action(action: str) -> str:
    aliases = {
        "fold": "fold",
        "不玩了": "fold",
        "call": "call",
        "跟上": "call",
        "check": "check",
        "先不出": "check",
        "bet": "bet",
        "先出一些": "bet",
        "new_hand": "new_hand",
        "new_match": "new_match",
        "bot_step": "bot_step",
    }
    return aliases.get(action, action)
