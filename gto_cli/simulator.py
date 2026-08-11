from __future__ import annotations

import random
from typing import Any

from .advisor import advise_state
from .cards import build_deck
from .strategy import normalize_position, normalize_scenario

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
SCENARIOS = ("rfi", "vs_open", "vs_3bet")
STREETS = ("preflop", "flop", "turn", "river")
VILLAIN_PROFILES = ("tight", "standard", "wide")
CARD_RANK_LABEL = {
    "A": "A",
    "K": "K",
    "Q": "Q",
    "J": "J",
    "T": "10",
}
CARD_SUIT_LABEL = {
    "s": "黑桃",
    "h": "红桃",
    "d": "方块",
    "c": "梅花",
}
STREET_LABEL = {
    "preflop": "还没发公共牌",
    "flop": "桌上已经有 3 张公共牌",
    "turn": "桌上已经有 4 张公共牌",
    "river": "5 张公共牌都发完了",
}
SCENARIO_LABEL = {
    "rfi": "前面没人多出钱",
    "vs_open": "前面有人多出钱",
    "vs_3bet": "前面已经加过两轮钱",
}
POSITION_LABEL = {
    "UTG": "很早行动",
    "HJ": "较早行动",
    "CO": "较晚行动",
    "BTN": "最后行动",
    "SB": "先放一小份底钱的位置",
    "BB": "先放一大份底钱的位置",
}
ACTION_EXPLANATION = {
    "fold": "不玩了：放弃这手牌，不再往里放钱。",
    "call": "跟上：补到和别人一样多，继续看后面的牌。",
    "check": "先不出：现在没人逼你出钱，你先不出，把选择交给别人。",
    "bet": "先出一些：现在没人出钱，你主动往桌上放钱。",
    "raise": "多出一些：别人已经出钱了，你出更多。",
    "limp": "只补最少：还没发公共牌时，只补到最低的钱；新手先少用。",
    "3bet": "再多出一些：别人已经多出钱，你再加回去。",
    "4bet": "再再多出一些：别人又加回来，你继续加。",
}
ACTION_LABEL = {
    "fold": "不玩了",
    "call": "跟上",
    "check": "先不出",
    "bet": "先出一些",
    "raise": "多出一些",
    "limp": "只补最少",
    "3bet": "再多出一些",
    "4bet": "再再多出一些",
}
LEVEL_ALIASES = {
    "beginner": "simple",
    "easy": "simple",
    "simple": "simple",
    "newbie": "simple",
    "简单": "simple",
    "medium": "medium",
    "normal": "medium",
    "intermediate": "medium",
    "中等": "medium",
    "advanced": "advanced",
    "hard": "advanced",
    "高级": "advanced",
    "master": "master",
    "expert": "master",
    "大师": "master",
}
LEVEL_CONFIG = {
    "simple": {
        "streets": ("preflop",),
        "street_weights": (1,),
        "positions": ("CO", "BTN", "SB"),
        "scenarios": ("rfi", "vs_open"),
        "scenario_weights": (4, 1),
        "villains": ("standard",),
        "stacks": (100,),
    },
    "medium": {
        "streets": ("preflop", "flop"),
        "street_weights": (3, 2),
        "positions": ("HJ", "CO", "BTN", "SB", "BB"),
        "scenarios": ("rfi", "vs_open"),
        "scenario_weights": (3, 2),
        "villains": ("standard", "wide"),
        "stacks": (75, 100, 125),
    },
    "advanced": {
        "streets": ("preflop", "flop", "turn", "river"),
        "street_weights": (3, 3, 2, 1),
        "positions": POSITIONS,
        "scenarios": SCENARIOS,
        "scenario_weights": (3, 3, 2),
        "villains": VILLAIN_PROFILES,
        "stacks": (40, 60, 75, 100, 125, 150),
    },
    "master": {
        "streets": STREETS,
        "street_weights": (2, 3, 3, 2),
        "positions": POSITIONS,
        "scenarios": SCENARIOS,
        "scenario_weights": (2, 3, 3),
        "villains": VILLAIN_PROFILES,
        "stacks": (25, 40, 60, 75, 100, 125, 150, 200),
    },
}


def generate_spot(
    level: str = "medium",
    street: str = "random",
    position: str = "random",
    scenario: str = "random",
    villain_profile: str = "level",
    seed: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    normalized_level = normalize_level(level)
    config = LEVEL_CONFIG[normalized_level]
    chosen_street = choose_street(street, rng, config)
    chosen_position = choose_position(position, rng, config)
    chosen_scenario = choose_scenario(scenario, chosen_position, rng, config)
    stack_bb = rng.choice(config["stacks"])

    deck = build_deck()
    rng.shuffle(deck)
    hero_cards = [deck.pop(), deck.pop()]
    board_count = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[chosen_street]
    board = [deck.pop() for _ in range(board_count)]
    pot_bb, to_call_bb = random_pot(chosen_street, chosen_scenario, rng)

    return {
        "hero": {
            "cards": hero_cards,
            "position": chosen_position,
            "stack_bb": stack_bb,
        },
        "table": {
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb,
            "effective_stack_bb": stack_bb,
            "board": board,
        },
        "action": {
            "scenario": chosen_scenario,
            "street": chosen_street,
        },
        "villain": {
            "profile": choose_villain(villain_profile, rng, config),
        },
        "practice": {
            "level": normalized_level,
        },
        "seed": rng.randrange(1, 1_000_000),
    }


def build_practice_round(
    level: str = "medium",
    street: str = "random",
    position: str = "random",
    scenario: str = "random",
    villain_profile: str = "level",
    seed: int | None = None,
    iterations: int = 700,
) -> dict[str, Any]:
    state = generate_spot(
        level=level,
        street=street,
        position=position,
        scenario=scenario,
        villain_profile=villain_profile,
        seed=seed,
    )
    answer = advise_state(state, iterations=iterations)
    return {
        "state": state,
        "answer": answer,
        "actions": available_actions(answer),
        "lesson": build_lesson(state, answer),
    }


def available_actions(answer: dict[str, Any]) -> list[str]:
    mix = answer.get("decision", {}).get("mix", {})
    return list(mix.keys())


def build_lesson(state: dict[str, Any], answer: dict[str, Any]) -> dict[str, list[str]]:
    actions = available_actions(answer)
    table = state["table"]
    action = state["action"]
    street = action["street"]
    scenario = action["scenario"]
    pot = float(table["pot_bb"])
    to_call = float(table["to_call_bb"])
    before = [
        "你手里有 2 张牌。桌上最多会发 5 张大家都能用的牌。最后谁能凑出更大的 5 张牌，谁就赢。",
        f"现在是：{STREET_LABEL[street]}。{street_hint(street)}",
        scenario_lesson(street, scenario),
        f"你的位置是：{POSITION_LABEL[state['hero']['position']]}。",
        f"你的手牌是 {describe_cards(state['hero']['cards'])}。",
    ]
    if table["board"]:
        before.append(f"公牌是 {describe_cards(table['board'])}。")
    if to_call > 0:
        before.append(
            f"桌上已有 {pot:g} 份钱。你如果想继续，需要再放 {to_call:g} 份钱。"
        )
    else:
        before.append(f"桌上已有 {pot:g} 份钱。现在没人逼你出钱。")
    before.append("可选动作：" + "；".join(action_explanation(item, state) for item in actions))
    before.append(thinking_hint(state, answer))

    after = build_after_lesson(state, answer)
    return {
        "before": before,
        "after": after,
    }


def street_hint(street: str) -> str:
    if street == "preflop":
        return "这时只看你手里的 2 张牌、你行动早晚、前面有没有人多出钱。"
    if street == "flop":
        return "现在要看你有没有凑成对子，或者后面还有没有机会变强。"
    if street == "turn":
        return "还剩最后 1 张公共牌，你变强的机会少了一点。"
    return "不会再发新牌了，现在基本就是比谁的牌更大。"


def scenario_lesson(street: str, scenario: str) -> str:
    label = SCENARIO_LABEL[scenario]
    if street != "preflop":
        return f"前面的情况：{label}。现在先看你的牌强不强、桌上的钱多不多。"
    return f"前面的情况：{label}。{scenario_hint(scenario)}"


def scenario_hint(scenario: str) -> str:
    if scenario == "rfi":
        return "轮到你决定要不要主动玩这手牌。"
    if scenario == "vs_open":
        return "你要决定是不玩、跟上，还是出更多。"
    return "前面已经有人出了很多，通常大家手里的牌会更强，新手先保守一点。"


def action_explanation(action: str, state: dict[str, Any]) -> str:
    street = state["action"]["street"]
    to_call = float(state["table"]["to_call_bb"])
    if action == "raise" and street == "preflop" and to_call == 0:
        return "多出一些：还没发公共牌时主动多出钱，表示你想认真玩这手。"
    if action == "raise" and to_call == 0:
        return "多出一些：现在没人出钱时，你主动往桌上放钱。"
    return ACTION_EXPLANATION.get(action, action)


def thinking_hint(state: dict[str, Any], answer: dict[str, Any]) -> str:
    metrics = answer.get("metrics", {})
    decision = answer.get("decision", {})
    mix = decision.get("mix", {})
    top_frequency = max(mix.values()) if mix else 0
    if answer.get("mode") == "preflop":
        if top_frequency >= 80:
            return "思考提示：这题比较清楚，优先选电脑最常建议的动作。"
        return "思考提示：这题有点接近分界线。行动越晚，通常能多玩一点；别人已经多出钱时，要更小心。"
    if float(state["table"]["to_call_bb"]) > 0:
        return "思考提示：你要先想：为了继续要放的钱多不多？你的牌有没有足够机会赢回来？"
    return "思考提示：没人逼你出钱时，牌强就可以主动出钱；牌弱就先不出。"


def primary_action_hint(answer: dict[str, Any]) -> str:
    decision = answer.get("decision", {})
    primary = decision.get("primary_action")
    mix = decision.get("mix", {})
    if not primary or not mix:
        return "先看懂局面，再考虑动作。"
    frequency = mix.get(primary, 0)
    return f"最推荐的动作是「{plain_action_label(primary)}」。电脑大约 {frequency}% 的时候会这么做；数字越高，说明越确定。"


def build_after_lesson(state: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    decision = answer.get("decision", {})
    metrics = answer.get("metrics", {})
    primary = decision.get("primary_action")
    lines = [primary_action_hint(answer)]
    if decision.get("recommended_size_bb"):
        lines.append(f"如果选「{plain_action_label(primary)}」，大概放 {decision['recommended_size_bb']} 份钱。")
    if answer.get("mode") == "postflop":
        if "hand_category" in metrics:
            lines.append(f"你现在大概是：{hand_category_label(str(metrics['hand_category']))}。")
        if "equity_pct" in metrics:
            lines.append(f"电脑粗略估计，你最后赢的机会大约是 {metrics['equity_pct']}%。")
        if float(state["table"]["to_call_bb"]) > 0 and "required_equity_pct" in metrics:
            lines.append(f"如果要继续玩，至少要有大约 {metrics['required_equity_pct']}% 的赢面才比较划算。")
    else:
        lines.append("这题还没发公共牌，所以主要看两张手牌强不强、你行动早晚、前面有没有人多出钱。")
    return lines


def hand_category_label(category: str) -> str:
    labels = {
        "high_card": "还没凑成对子",
        "one_pair": "一对",
        "two_pair": "两对",
        "three_of_a_kind": "三张一样",
        "straight": "五张连续",
        "flush": "五张同花色",
        "full_house": "三张一样加一对",
        "four_of_a_kind": "四张一样",
        "straight_flush": "同花色的五张连续",
    }
    return labels.get(category, category)


def judge_action(raw_action: str, answer: dict[str, Any]) -> dict[str, Any]:
    mix = answer.get("decision", {}).get("mix", {})
    if not mix:
        return {"ok": False, "grade": "error", "message": "No action mix in answer."}

    action = normalize_user_action(raw_action, mix)
    primary = answer.get("decision", {}).get("primary_action")
    frequency = int(mix.get(action, 0))

    if action == primary:
        return {
            "ok": True,
            "grade": "correct",
            "action": action,
            "message": f"正确，最推荐的是「{plain_action_label(primary)}」。",
        }
    if frequency >= 30:
        return {
            "ok": True,
            "grade": "mixed",
            "action": action,
            "message": f"可以，这个动作也常会用到。电脑大约 {frequency}% 的时候会这么做。",
        }
    return {
        "ok": False,
        "grade": "miss",
        "action": action,
        "message": f"这次不太好。电脑很少这样做，最推荐的是「{plain_action_label(primary)}」。",
    }


def normalize_user_action(raw_action: str, mix: dict[str, int]) -> str:
    action = raw_action.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "f": "fold",
        "fold": "fold",
        "c": "call",
        "call": "call",
        "x": "check",
        "check": "check",
        "b": "bet",
        "bet": "bet",
        "r": "raise",
        "raise": "raise",
        "allin": "raise",
        "jam": "raise",
        "l": "limp",
        "limp": "limp",
        "3b": "3bet",
        "3bet": "3bet",
        "threebet": "3bet",
        "4b": "4bet",
        "4bet": "4bet",
        "fourbet": "4bet",
        "不玩了": "fold",
        "弃牌": "fold",
        "跟上": "call",
        "跟": "call",
        "先不出": "check",
        "过": "check",
        "先出一些": "bet",
        "下注": "bet",
        "多出一些": "raise",
        "加注": "raise",
        "只补最少": "limp",
        "补最少": "limp",
    }
    normalized = aliases.get(action, action)
    if normalized == "raise":
        if "raise" in mix:
            return "raise"
        if "3bet" in mix:
            return "3bet"
        if "4bet" in mix:
            return "4bet"
        if "bet" in mix:
            return "bet"
    if normalized not in mix:
        return normalized
    return normalized


def normalize_level(level: str) -> str:
    value = level.strip().lower()
    if value not in LEVEL_ALIASES:
        raise ValueError(f"unknown practice level: {level!r}")
    return LEVEL_ALIASES[value]


def choose_street(street: str, rng: random.Random, config: dict[str, Any]) -> str:
    value = street.strip().lower()
    if value == "random":
        return rng.choices(config["streets"], weights=config["street_weights"], k=1)[0]
    if value not in STREETS:
        raise ValueError(f"unknown street: {street!r}")
    return value


def choose_position(position: str, rng: random.Random, config: dict[str, Any]) -> str:
    if position.strip().lower() == "random":
        return rng.choice(config["positions"])
    return normalize_position(position)


def choose_scenario(
    scenario: str,
    position: str,
    rng: random.Random,
    config: dict[str, Any],
) -> str:
    if scenario.strip().lower() == "random":
        if position == "BB":
            available = [item for item in config["scenarios"] if item != "rfi"] or ("vs_open",)
            return rng.choice(available)
        return rng.choices(config["scenarios"], weights=config["scenario_weights"], k=1)[0]
    return normalize_scenario(scenario)


def choose_villain(villain_profile: str, rng: random.Random, config: dict[str, Any]) -> str:
    value = villain_profile.strip().lower()
    if value in ("level", "default"):
        return rng.choice(config["villains"])
    if value == "random":
        return rng.choice(VILLAIN_PROFILES)
    return villain_profile


def random_pot(street: str, scenario: str, rng: random.Random) -> tuple[float, float]:
    if street == "preflop":
        if scenario == "rfi":
            return 1.5, 0.0
        if scenario == "vs_open":
            open_size = rng.choice([2.0, 2.2, 2.5, 3.0])
            return round(1.5 + open_size, 1), open_size
        three_bet = rng.choice([7.0, 8.0, 9.0, 10.0])
        return round(three_bet + 3.5, 1), round(three_bet - rng.choice([2.0, 2.2, 2.5]), 1)

    pot = rng.choice([6.5, 8.0, 10.5, 12.0, 16.0, 22.0, 30.0])
    facing_bet = rng.random() < 0.65
    if not facing_bet:
        return pot, 0.0
    fraction = rng.choice([0.33, 0.5, 0.66, 0.75, 1.0])
    return pot, round(pot * fraction, 1)


def spot_title(state: dict[str, Any]) -> str:
    hero = state["hero"]
    table = state["table"]
    action = state["action"]
    board = describe_cards(table["board"]) if table["board"] else "-"
    return (
        f"{STREET_LABEL[action['street']]} | 位置：{POSITION_LABEL[hero['position']]} | "
        f"前面：{SCENARIO_LABEL[action['scenario']]} | 手牌：{describe_cards(hero['cards'])} | "
        f"公共牌：{board} | 桌上已有：{table['pot_bb']} 份 | 继续要放：{table['to_call_bb']} 份"
    )


def describe_cards(cards: list[str]) -> str:
    return " ".join(describe_card(card) for card in cards)


def describe_card(card: str) -> str:
    rank = CARD_RANK_LABEL.get(card[0], card[0])
    suit = CARD_SUIT_LABEL.get(card[1], card[1])
    return f"{rank}{suit}"


def plain_action_label(action: str) -> str:
    return ACTION_LABEL.get(action, action)
