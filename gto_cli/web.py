from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .game import apply_action, new_game, public_game
from .simulator import build_practice_round


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
GAMES: dict[str, dict] = {}


class PracticeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/round":
            self.send_round(parsed.query)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/game/new":
            self.send_new_game()
            return
        if parsed.path == "/api/game/action":
            self.send_game_action()
            return
        self.send_json({"ok": False, "error": "unknown endpoint"}, status=404)

    def send_round(self, query: str) -> None:
        params = parse_qs(query)
        level = params.get("level", ["simple"])[0]
        try:
            payload = build_practice_round(level=level, iterations=500)
            self.send_json(payload)
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, status=400)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def send_new_game(self) -> None:
        body = self.read_json()
        game = new_game(
            level=str(body.get("level", "simple")),
            match_mode=str(body.get("match_mode", "small")),
            starting_stack=body.get("starting_stack", 50),
        )
        GAMES[game["id"]] = game
        self.send_json(public_game(game, auto_bots=not bool(body.get("slow_bots"))))

    def send_game_action(self) -> None:
        body = self.read_json()
        game_id = str(body.get("game_id", ""))
        action = str(body.get("action", ""))
        game = GAMES.get(game_id)
        if not game:
            self.send_json({"ok": False, "error": "game not found"}, status=404)
            return
        slow_bots = bool(body.get("slow_bots"))
        apply_action(game, action, auto_bots=not slow_bots)
        self.send_json(public_game(game, auto_bots=not slow_bots))

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def run_web(host: str = "127.0.0.1", port: int = 8765) -> None:
    if not WEB_ROOT.exists():
        raise RuntimeError(f"missing web directory: {WEB_ROOT}")
    server = ThreadingHTTPServer((host, port), PracticeHandler)
    print(f"网页练习盘已启动：http://{host}:{port}/")
    print("按 Ctrl+C 停止。")
    server.serve_forever()
