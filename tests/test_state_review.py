from __future__ import annotations

import json
from pathlib import Path

from gto_cli.state_review import build_state_review


def make_event(*, index: int, street: str, board: list[str], hero_turn: bool, source_frame: str) -> dict:
    return {
        "ok": True,
        "source": {
            "timestamp_sec": float(index),
            "card_sample": {"frame": source_frame},
            "hero_cards_locked": True,
        },
        "event": {"index": index},
        "table": {
            "street": street,
            "dealer_seat": "top",
            "dealer_position": "BTN",
            "pot_bb": 7.5,
            "to_call_bb": 2.5,
            "board": board,
        },
        "hero": {
            "position": "UTG+1",
            "gto_position": "UTG",
            "preflop_action_order": 2,
            "postflop_action_order": 4,
            "cards": ["As", "Kd"],
        },
        "hero_turn": {"is_turn": hero_turn, "confidence": 0.96, "reason": "red_buttons"},
        "action_controls": {"visible": hero_turn, "actions": ["fold", "call", "raise"], "call_amount_bb": 2.5, "raise_amount_bb": 8},
        "bets": [{"seat": "HJ", "amount_bb": 2.5}],
        "seats": [{"seat": "HJ", "status": "active_or_showdown"}],
        "gto_advice": {"action": "fold", "scenario": "vs_open"},
    }


def test_state_review_exports_hero_turn_screenshots_and_ignores_old_advice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "preflop.png").write_bytes(b"image-a")
    (assets / "flop.png").write_bytes(b"image-b")
    events_path = tmp_path / "events.jsonl"
    rows = [
        make_event(index=1, street="preflop", board=[], hero_turn=True, source_frame="assets/preflop.png"),
        make_event(index=2, street="flop", board=["2c", "7d", "Th"], hero_turn=True, source_frame="assets/flop.png"),
        make_event(index=3, street="turn", board=["2c"], hero_turn=True, source_frame="assets/flop.png"),
    ]
    events_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    payload = build_state_review(events_path=events_path, output_dir=tmp_path / "review", limit=4)

    assert payload["ok"]
    assert payload["stats"]["eligible_candidates"] == 2
    assert len(payload["cases"]) == 2
    assert Path(payload["report_path"]).is_file()
    report = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert "案例 01 | PREFLOP" in report
    assert "旧版 `gto_advice` 已被忽略" in report
    assert "翻前场景：**未知**" in report
    assert "单帧翻前观察" in report
    assert all(Path(case["image"]).is_file() for case in payload["cases"])
