from __future__ import annotations

import csv
import html
import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .card_label_queue import LABEL_QUEUE_COLUMNS


RANKS = set("AKQJT98765432")
SUITS = set("shdc")


def serve_card_label_queue(
    *,
    queue_csv: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    if not queue_csv.exists():
        raise ValueError(f"queue csv does not exist: {queue_csv}")
    rows, fieldnames = load_queue_csv(queue_csv)
    if not rows:
        raise ValueError(f"queue csv has no rows: {queue_csv}")

    handler = make_handler(queue_csv)
    server = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{int(port)}/"
    print(f"Card label queue server: {url}", flush=True)
    print(f"Queue CSV: {queue_csv}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {
        "ok": True,
        "queue_csv": str(queue_csv),
        "url": url,
        "row_count": len(rows),
        "field_count": len(fieldnames),
    }


def make_handler(queue_csv: Path) -> type[BaseHTTPRequestHandler]:
    class CardLabelQueueHandler(BaseHTTPRequestHandler):
        server_version = "CardLabelQueue/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self.send_text(render_index_html(queue_csv), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/rows":
                rows, _fieldnames = load_queue_csv(queue_csv)
                self.send_json({"ok": True, "queue_csv": str(queue_csv), "rows": public_rows(rows), "progress": progress(rows)})
                return
            if parsed.path == "/api/progress":
                rows, _fieldnames = load_queue_csv(queue_csv)
                self.send_json({"ok": True, "progress": progress(rows)})
                return
            if parsed.path == "/file":
                query = urllib.parse.parse_qs(parsed.query)
                file_path = query.get("path", [""])[0]
                self.send_file(file_path)
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/update":
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                result = update_queue_csv(queue_csv, payload)
                self.send_json({"ok": True, **result})
            except ValueError as error:
                self.send_json({"ok": False, "error": str(error)}, status=400)
            except Exception as error:
                self.send_json({"ok": False, "error": str(error)}, status=500)

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path_text: str) -> None:
            path = Path(urllib.parse.unquote(path_text))
            if not path.exists() or not path.is_file():
                self.send_error(404, "file not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return CardLabelQueueHandler


def load_queue_csv(queue_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(queue_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for column in LABEL_QUEUE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for row in rows:
        for column in fieldnames:
            row.setdefault(column, "")
    return rows, fieldnames


def write_queue_csv(queue_csv: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with Path(queue_csv).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_queue_csv(queue_csv: Path, payload: dict[str, Any]) -> dict[str, Any]:
    label_id = str(payload.get("label_id") or "").strip()
    if not label_id:
        raise ValueError("label_id is required")
    final_card0 = normalize_card_input(payload.get("final_card0"))
    final_card1 = normalize_card_input(payload.get("final_card1"))
    notes = str(payload.get("notes") or "").strip()
    rows, fieldnames = load_queue_csv(queue_csv)
    target = None
    for row in rows:
        if str(row.get("label_id") or "") == label_id:
            target = row
            break
    if target is None:
        raise ValueError(f"label_id not found: {label_id}")
    target["final_card0"] = final_card0
    target["final_card1"] = final_card1
    target["notes"] = notes
    write_queue_csv(queue_csv, rows, fieldnames)
    return {"label_id": label_id, "progress": progress(rows)}


def normalize_card_input(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("10", "T")
    if len(text) != 2:
        raise ValueError(f"card must be like As, Td, 7h, or empty: {text}")
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"card must be like As, Td, 7h, or empty: {text}")
    return f"{rank}{suit}"


def progress(rows: list[dict[str, Any]]) -> dict[str, int]:
    labeled_rows = 0
    labeled_slots = 0
    total_slots = 0
    for row in rows:
        row_labeled = False
        for slot in (0, 1):
            if row.get(f"card{slot}") or row.get(f"card{slot}_card_path"):
                total_slots += 1
                if row.get(f"final_card{slot}"):
                    labeled_slots += 1
                    row_labeled = True
        if row_labeled:
            labeled_rows += 1
    return {"rows": len(rows), "labeled_rows": labeled_rows, "total_slots": total_slots, "labeled_slots": labeled_slots}


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "label_id",
        "priority",
        "reason",
        "video",
        "timestamp_sec",
        "frame_index",
        "class",
        "street",
        "dealer",
        "hero_position",
        "hero_turn",
        "raw_hero_cards",
        "stabilized_hero_cards",
        "card0",
        "card0_consensus",
        "card1",
        "card1_consensus",
        "final_card0",
        "final_card1",
        "notes",
        "asset_table",
        "asset_card0",
        "asset_rank0",
        "asset_suit0",
        "asset_card1",
        "asset_rank1",
        "asset_suit1",
        "table_frame_path",
        "card0_card_path",
        "card0_rank_path",
        "card0_suit_path",
        "card1_card_path",
        "card1_rank_path",
        "card1_suit_path",
    ]
    return [{key: str(row.get(key) or "") for key in keys} for row in rows]


def render_index_html(queue_csv: Path) -> str:
    title = "Card Label Queue"
    escaped_queue = html.escape(str(queue_csv))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{margin:0;background:#141414;color:#eee;font-family:Arial,'Microsoft YaHei',sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#202020;border-bottom:1px solid #444;padding:10px 16px;display:flex;gap:18px;align-items:center}}
button,input{{font:inherit}}
button{{background:#333;color:#eee;border:1px solid #666;border-radius:4px;padding:5px 10px;cursor:pointer}}
button.primary{{background:#9b2226;border-color:#c43;color:white}}
input{{background:#111;color:#fff;border:1px solid #555;border-radius:4px;padding:5px;width:58px;text-transform:uppercase}}
textarea{{background:#111;color:#fff;border:1px solid #555;border-radius:4px;width:210px;height:42px}}
.wrap{{padding:14px 16px}}
.row{{display:grid;grid-template-columns:130px minmax(280px,410px) minmax(360px,1fr) 300px;gap:12px;border:1px solid #3c3c3c;background:#1c1c1c;margin-bottom:14px;padding:10px}}
.meta{{font-size:13px;color:#bbb;line-height:1.55}}
.reason{{color:#ffdf62;word-break:break-word}}
.table-img{{max-width:400px;max-height:240px;border:1px solid #333}}
.cards{{display:flex;gap:12px;align-items:flex-start}}
.cardbox{{min-width:150px}}
.card-img{{max-width:120px;max-height:150px;border:1px solid #333;background:#222}}
.glyph{{max-width:48px;max-height:48px;image-rendering:pixelated;background:#222;border:1px solid #333}}
.current{{font-weight:bold;margin-bottom:5px}}
.save-status{{font-size:12px;color:#8de48d;min-height:18px}}
.muted{{color:#999}}
.hidden{{display:none}}
</style>
</head>
<body>
<header>
  <strong>Card Label Queue</strong>
  <span class="muted">{escaped_queue}</span>
  <span id="progress">loading...</span>
  <button onclick="loadRows()">Refresh</button>
  <label><input id="hideDone" type="checkbox" onchange="render()"> hide labeled</label>
</header>
<div class="wrap" id="rows"></div>
<script>
let rows = [];
function esc(s) {{
  return String(s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function fileUrl(path) {{
  if (!path) return '';
  return '/file?path=' + encodeURIComponent(path);
}}
function img(path, cls) {{
  return path ? `<img class="${{cls}}" src="${{fileUrl(path)}}">` : '<span class="muted">missing</span>';
}}
function isLabeled(r) {{
  return Boolean(r.final_card0 || r.final_card1);
}}
async function loadRows() {{
  const res = await fetch('/api/rows');
  const data = await res.json();
  rows = data.rows || [];
  updateProgress(data.progress);
  render();
}}
function updateProgress(p) {{
  document.getElementById('progress').textContent = `rows ${{p.labeled_rows}}/${{p.rows}} | slots ${{p.labeled_slots}}/${{p.total_slots}}`;
}}
function render() {{
  const hideDone = document.getElementById('hideDone').checked;
  const root = document.getElementById('rows');
  root.innerHTML = '';
  for (const r of rows) {{
    if (hideDone && isLabeled(r)) continue;
    const row = document.createElement('div');
    row.className = 'row';
    row.id = 'row-' + r.label_id;
    row.innerHTML = `
      <div class="meta">
        <div><b>${{esc(r.label_id)}}</b> | priority ${{esc(r.priority)}}</div>
        <div class="reason">${{esc(r.reason || '-')}}</div>
        <div>${{esc((r.video || '').split(/[\\\\/]/).pop())}}</div>
        <div>t=${{esc(r.timestamp_sec)}} frame=${{esc(r.frame_index)}}</div>
        <div>${{esc(r.street || '-')}} dealer=${{esc(r.dealer || '-')}} hero=${{esc(r.hero_position || '-')}} turn=${{esc(r.hero_turn || '-')}}</div>
        <div>raw=${{esc(r.raw_hero_cards || '-')}}</div>
        <div>stable=${{esc(r.stabilized_hero_cards || '-')}}</div>
      </div>
      <div>${{img(r.asset_table || r.table_frame_path, 'table-img')}}</div>
      <div class="cards">
        ${{cardBox(r, 0)}}
        ${{cardBox(r, 1)}}
      </div>
      <div>
        <div>final_card0 <input id="${{r.label_id}}-f0" value="${{esc(r.final_card0)}}"></div>
        <div style="margin-top:8px">final_card1 <input id="${{r.label_id}}-f1" value="${{esc(r.final_card1)}}"></div>
        <div style="margin-top:8px"><textarea id="${{r.label_id}}-notes" placeholder="notes">${{esc(r.notes)}}</textarea></div>
        <button class="primary" style="margin-top:8px" onclick="saveRow('${{esc(r.label_id)}}')">Save</button>
        <button style="margin-top:8px" onclick="copyCurrent('${{esc(r.label_id)}}')">Use current</button>
        <div class="save-status" id="${{r.label_id}}-status"></div>
      </div>`;
    root.appendChild(row);
  }}
}}
function cardBox(r, slot) {{
  const card = r['card' + slot] || '';
  const consensus = r['card' + slot + '_consensus'] || '';
  return `<div class="cardbox">
    <div class="current">card${{slot}} cur=${{esc(card || '-')}} cons=${{esc(consensus || '-')}}</div>
    ${{img(r['asset_card' + slot] || r['card' + slot + '_card_path'], 'card-img')}}
    <div>${{img(r['asset_rank' + slot] || r['card' + slot + '_rank_path'], 'glyph')}} ${{img(r['asset_suit' + slot] || r['card' + slot + '_suit_path'], 'glyph')}}</div>
  </div>`;
}}
function copyCurrent(labelId) {{
  const r = rows.find(item => item.label_id === labelId);
  if (!r) return;
  document.getElementById(labelId + '-f0').value = r.card0 || '';
  document.getElementById(labelId + '-f1').value = r.card1 || '';
}}
async function saveRow(labelId) {{
  const status = document.getElementById(labelId + '-status');
  status.textContent = 'saving...';
  const payload = {{
    label_id: labelId,
    final_card0: document.getElementById(labelId + '-f0').value,
    final_card1: document.getElementById(labelId + '-f1').value,
    notes: document.getElementById(labelId + '-notes').value
  }};
  const res = await fetch('/api/update', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
  const data = await res.json();
  if (!data.ok) {{
    status.style.color = '#ff8b8b';
    status.textContent = data.error || 'save failed';
    return;
  }}
  status.style.color = '#8de48d';
  status.textContent = 'saved';
  await loadRows();
}}
loadRows();
</script>
</body>
</html>"""


def format_card_label_server_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Queue CSV: {payload.get('queue_csv')}",
            f"URL: {payload.get('url')}",
            f"Rows: {payload.get('row_count')}",
        ]
    )
