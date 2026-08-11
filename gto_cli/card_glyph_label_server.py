from __future__ import annotations

import csv
import html
import json
import mimetypes
import re
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .card_glyph_label_queue import GLYPH_LABEL_QUEUE_COLUMNS, normalize_glyph_label

REVIEW_RANK_ORDER = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
REVIEW_SUIT_ORDER = ("s", "h", "d", "c")


def serve_card_glyph_label_queue(
    *,
    queue_csv: Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = False,
) -> dict[str, Any]:
    queue_csv = Path(queue_csv)
    if not queue_csv.exists():
        raise ValueError(f"queue csv does not exist: {queue_csv}")
    rows, fieldnames = load_glyph_queue_csv(queue_csv)
    if not rows:
        raise ValueError(f"queue csv has no rows: {queue_csv}")

    server = ThreadingHTTPServer((host, int(port)), make_glyph_handler(queue_csv))
    url = f"http://{host}:{int(port)}/"
    print(f"Card glyph audit: {url}", flush=True)
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


def make_glyph_handler(queue_csv: Path) -> type[BaseHTTPRequestHandler]:
    class CardGlyphLabelHandler(BaseHTTPRequestHandler):
        server_version = "CardGlyphAudit/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self.send_text(render_glyph_audit_html(queue_csv), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/rows":
                rows, _fieldnames = load_glyph_queue_csv(queue_csv)
                self.send_json(
                    {
                        "ok": True,
                        "queue_csv": str(queue_csv),
                        "rows": public_glyph_rows(rows),
                        "progress": glyph_progress(rows),
                    }
                )
                return
            if parsed.path == "/file":
                query = urllib.parse.parse_qs(parsed.query)
                self.send_file(query.get("path", [""])[0])
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
                result = update_glyph_queue_csv(queue_csv, payload)
                self.send_json({"ok": True, **result})
            except ValueError as error:
                self.send_json({"ok": False, "error": str(error)}, status=400)
            except Exception as error:
                self.send_json({"ok": False, "error": str(error)}, status=500)

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path_text: str) -> None:
            path = Path(urllib.parse.unquote(path_text))
            if not path.exists() or not path.is_file():
                self.send_error(404, "file not found")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return CardGlyphLabelHandler


def load_glyph_queue_csv(queue_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(queue_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for column in GLYPH_LABEL_QUEUE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for row in rows:
        for column in fieldnames:
            row.setdefault(column, "")
    return rows, fieldnames


def write_glyph_queue_csv(
    queue_csv: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with Path(queue_csv).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_glyph_queue_csv(queue_csv: Path, payload: dict[str, Any]) -> dict[str, Any]:
    label_id = str(payload.get("label_id") or "").strip()
    if not label_id:
        raise ValueError("label_id is required")
    rows, fieldnames = load_glyph_queue_csv(queue_csv)
    target = next((row for row in rows if str(row.get("label_id") or "") == label_id), None)
    if target is None:
        raise ValueError(f"label_id not found: {label_id}")
    if payload.get("ignore_card"):
        target_context = infer_glyph_context_paths(target)
        ignored_label_ids = []
        for row in rows:
            context = infer_glyph_context_paths(row)
            if (
                context.get("sample_id") == target_context.get("sample_id")
                and context.get("group") == target_context.get("group")
                and context.get("slot") == target_context.get("slot")
            ):
                row["ignored"] = "1"
                row["final_label"] = ""
                notes = str(row.get("notes") or "").strip()
                if "ignored_noncard" not in notes:
                    row["notes"] = f"{notes} | ignored_noncard".strip(" |")
                ignored_label_ids.append(str(row.get("label_id") or ""))
        write_glyph_queue_csv(queue_csv, rows, fieldnames)
        return {
            "label_id": label_id,
            "ignored": True,
            "ignored_label_ids": ignored_label_ids,
            "progress": glyph_progress(rows),
        }
    kind = str(target.get("kind") or "").strip().lower()
    raw_label = str(payload.get("final_label") or "").strip()
    final_label = normalize_glyph_label(raw_label, kind)
    if raw_label and not final_label:
        expected = "A K Q J T 9..2" if kind == "rank" else "s h d c"
        raise ValueError(f"invalid {kind} label: {raw_label}; expected {expected}")
    target["final_label"] = final_label
    target["ignored"] = ""
    if "notes" in payload:
        target["notes"] = str(payload.get("notes") or "").strip()
    write_glyph_queue_csv(queue_csv, rows, fieldnames)
    return {"label_id": label_id, "final_label": final_label, "progress": glyph_progress(rows)}


def glyph_progress(rows: list[dict[str, Any]]) -> dict[str, int]:
    active_rows = [row for row in rows if not glyph_row_is_ignored(row)]
    rank_total = sum(1 for row in active_rows if row.get("kind") == "rank")
    suit_total = sum(1 for row in active_rows if row.get("kind") == "suit")
    rank_labeled = sum(1 for row in active_rows if row.get("kind") == "rank" and row.get("final_label"))
    suit_labeled = sum(1 for row in active_rows if row.get("kind") == "suit" and row.get("final_label"))
    return {
        "rows": len(active_rows),
        "ignored": len(rows) - len(active_rows),
        "labeled": rank_labeled + suit_labeled,
        "rank_total": rank_total,
        "rank_labeled": rank_labeled,
        "suit_total": suit_total,
        "suit_labeled": suit_labeled,
    }


def glyph_row_is_ignored(row: dict[str, Any]) -> bool:
    return str(row.get("ignored") or "").strip().lower() in {"1", "true", "yes", "ignored"}


def public_glyph_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(rows, key=glyph_review_sort_key):
        context = infer_glyph_context_paths(row)
        output.append(
            {
                "label_id": str(row.get("label_id") or ""),
                "kind": str(row.get("kind") or ""),
                "current_label": str(row.get("current_label") or ""),
                "current_confidence": str(row.get("current_confidence") or ""),
                "current_margin": str(row.get("current_margin") or ""),
                "final_label": str(row.get("final_label") or ""),
                "ignored": str(row.get("ignored") or ""),
                "reason": str(row.get("reason") or ""),
                "notes": str(row.get("notes") or ""),
                "glyph_path": str(row.get("asset_path") or row.get("input_path") or ""),
                **context,
            }
        )
    return output


def glyph_review_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Keep the audit UI in one low-switching classification mode at a time."""
    kind = str(row.get("kind") or "").strip().lower()
    current = str(row.get("current_label") or "").strip()
    if kind == "suit":
        label = current.lower()
        label_order = REVIEW_SUIT_ORDER.index(label) if label in REVIEW_SUIT_ORDER else len(REVIEW_SUIT_ORDER)
        kind_order = 0
    elif kind == "rank":
        label = current.upper()
        label_order = REVIEW_RANK_ORDER.index(label) if label in REVIEW_RANK_ORDER else len(REVIEW_RANK_ORDER)
        kind_order = 1
    else:
        label = current
        label_order = 999
        kind_order = 2
    return (kind_order, label_order, label, str(row.get("label_id") or ""))


def infer_glyph_context_paths(row: dict[str, Any]) -> dict[str, str]:
    input_text = str(row.get("input_path") or "").strip()
    if not input_text:
        return {
            "card_path": "",
            "overlay_path": "",
            "frame_path": "",
            "sample_id": "",
            "group": "",
            "slot": "",
            "observed_card": "",
        }
    input_path = Path(input_text)
    stem = input_path.stem
    match = re.match(r"^(hero|board)_slot(\d+)_([^_]+)_(rank|suit)$", stem)
    for suffix in ("_rank", "_suit"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    card_path = input_path.with_name(f"{stem}_card{input_path.suffix}")
    parent = input_path.parent
    return {
        # Keep API assembly independent of disk latency. /file validates the
        # selected asset only when the browser requests it.
        "card_path": str(card_path),
        "overlay_path": str(parent / "diagnostic_overlay.png"),
        "frame_path": str(parent / "frame.png"),
        "sample_id": parent.name,
        "group": match.group(1) if match else "",
        "slot": match.group(2) if match else "",
        "observed_card": match.group(3) if match else "",
    }


def render_glyph_audit_html(queue_csv: Path) -> str:
    escaped_queue = html.escape(str(queue_csv))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>牌面字形人工校对</title>
<style>
body{{margin:0;background:#111;color:#eee;font-family:Arial,'Microsoft YaHei',sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#202020;border-bottom:1px solid #444;padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
button,select,input{{font:inherit}}
button{{background:#333;color:#eee;border:1px solid #666;border-radius:4px;padding:8px 12px;cursor:pointer}}
button:hover{{background:#494949}} button.primary{{background:#a3232a;border-color:#d34}} button.ignore{{background:#4a3131;border-color:#a65;color:#ffd6d6}}
.wrap{{padding:16px}} .grid{{display:grid;grid-template-columns:minmax(420px,1.35fr) minmax(320px,.65fr);gap:18px}}
.panel{{background:#1c1c1c;border:1px solid #3d3d3d;padding:12px}}
.overlay{{max-width:100%;max-height:540px;border:1px solid #555}}
.images{{display:flex;gap:18px;align-items:flex-start;justify-content:center}}
.card{{max-width:210px;max-height:280px;background:#fff;border:1px solid #666}}
.glyph{{width:180px;height:230px;object-fit:contain;image-rendering:pixelated;background:#000;border:1px solid #666}}
.meta{{color:#bbb;line-height:1.65;word-break:break-all}} .current{{font-size:34px;font-weight:bold;color:#ffd166}}
.buttons{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px}}
.buttons button{{font-size:22px;min-height:52px}} textarea{{width:100%;height:58px;background:#111;color:#fff;border:1px solid #555;margin-top:12px}}
.status{{min-height:24px;color:#8de48d;margin-top:8px}} .danger{{color:#ff8f8f}} .muted{{color:#999}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <strong>牌面字形人工校对</strong>
  <span id="progress">加载中...</span>
  <select id="kind" onchange="refreshFilter()"><option value="all">全部</option><option value="rank">只看数字/字母</option><option value="suit">只看花色</option></select>
  <select id="group" onchange="refreshFilter()"><option value="all">手牌+公共牌</option><option value="hero">只看手牌</option><option value="board">只看公共牌</option></select>
  <label><input id="hideDone" type="checkbox" checked onchange="refreshFilter()"> 隐藏已校对</label>
  <button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button>
  <span class="muted">{escaped_queue}</span>
</header>
<div class="wrap"><div id="empty" class="panel" style="display:none">当前筛选已全部校对完成。</div>
<div id="content" class="grid">
  <div class="panel"><img id="overlay" class="overlay"></div>
  <div class="panel">
    <div id="meta" class="meta"></div>
    <div class="images"><img id="card" class="card"><img id="glyph" class="glyph"></div>
    <div>当前预测：<span id="current" class="current"></span></div>
    <button class="primary" style="margin-top:10px" onclick="acceptCurrent()">预测正确，直接接受</button>
    <button class="ignore" style="margin-top:10px" onclick="ignoreCard()">非牌面，整张忽略</button>
    <div id="buttons" class="buttons"></div>
    <textarea id="notes" placeholder="可选备注，例如：黑桃被识别成草花"></textarea>
    <div id="status" class="status"></div>
  </div>
</div></div>
<script>
let rows=[], filtered=[], index=0;
const rankLabels=['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
const suitOrder=['s','h','d','c'];
function reviewSortKey(r){{
  const label=(r.current_label||'?').toUpperCase();
  if(r.kind==='suit'){{const i=suitOrder.indexOf(label.toLowerCase());return [0,i<0?99:i,label,r.label_id];}}
  if(r.kind==='rank'){{const i=rankLabels.indexOf(label);return [1,i<0?99:i,label,r.label_id];}}
  return [2,99,label,r.label_id];
}}
function compareReviewRows(a,b){{
  const ak=reviewSortKey(a),bk=reviewSortKey(b);for(let i=0;i<ak.length;i++){{if(ak[i]<bk[i])return -1;if(ak[i]>bk[i])return 1;}}return 0;
}}
function reviewBucketKey(r){{return `${{r.kind}}:${{(r.current_label||'?').toUpperCase()}}`;}}
const suitLabels=[['s','♠ 黑桃'],['h','♥ 红桃'],['d','♦ 方片'],['c','♣ 草花']];
const fileUrl=p=>p?'/file?path='+encodeURIComponent(p):'';
async function loadRows(){{
  const data=await (await fetch('/api/rows')).json(); rows=data.rows||[]; updateProgress(data.progress); refreshFilter();
}}
function updateProgress(p){{document.getElementById('progress').textContent=`已校对 ${{p.labeled}}/${{p.rows}} | 已忽略 ${{p.ignored||0}} | rank ${{p.rank_labeled}}/${{p.rank_total}} | suit ${{p.suit_labeled}}/${{p.suit_total}}`;}}
function refreshFilter(){{
  const kind=document.getElementById('kind').value, group=document.getElementById('group').value, hide=document.getElementById('hideDone').checked;
  const current=filtered[index]?.label_id; filtered=rows.filter(r=>!r.ignored&&(kind==='all'||r.kind===kind)&&(group==='all'||r.group===group)&&(!hide||!r.final_label)).sort(compareReviewRows);
  index=Math.max(0,filtered.findIndex(r=>r.label_id===current)); if(index<0)index=0; render();
}}
function move(delta){{if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render();}}
function render(){{
  const empty=!filtered.length; document.getElementById('empty').style.display=empty?'block':'none';document.getElementById('content').style.display=empty?'none':'grid';if(empty)return;
  const r=filtered[index]; document.getElementById('overlay').src=fileUrl(r.overlay_path||r.frame_path);
  document.getElementById('card').src=fileUrl(r.card_path);document.getElementById('glyph').src=fileUrl(r.glyph_path);
  document.getElementById('current').textContent=r.current_label||'?';document.getElementById('notes').value=r.notes||'';
  const groupName=r.group==='hero'?'手牌':'公共牌';
  const kindName=r.kind==='rank'?'数字/字母':'花色';
  const confidence=r.current_confidence||'-',margin=r.current_margin||'-';
  document.getElementById('meta').innerHTML=`<b>${{groupName}} 第${{Number(r.slot)+1}}张 | ${{kindName}}</b><br>当时整张牌预测：${{r.observed_card||'?'}}<br>置信度：${{confidence}} | 差值：${{margin}}<br>${{r.label_id}} | ${{index+1}}/${{filtered.length}}<br>${{r.sample_id}}<br>${{r.reason}}`;
  const bucket=reviewBucketKey(r),bucketRows=filtered.filter(item=>reviewBucketKey(item)===bucket),bucketIndex=bucketRows.findIndex(item=>item.label_id===r.label_id)+1;
  document.getElementById('meta').insertAdjacentHTML('afterbegin',`<b>分组：${{r.kind==='suit'?'花色':'数字/字母'}} ${{r.current_label||'?'}}（${{bucketIndex}}/${{bucketRows.length}}）</b><br><span class="muted">顺序：花色 → 数字/字母；同预测连续</span><br>`);
  const root=document.getElementById('buttons');root.innerHTML='';
  const values=r.kind==='rank'?rankLabels.map(x=>[x,x==='T'?'10 (T)':x]):suitLabels;
  for(const [value,label] of values){{const b=document.createElement('button');b.textContent=label;b.onclick=()=>save(value);root.appendChild(b);}}
  document.getElementById('status').textContent='';
}}
function acceptCurrent(){{if(filtered.length)save(filtered[index].current_label);}}
async function ignoreCard(){{
  if(!filtered.length)return;const r=filtered[index],status=document.getElementById('status');status.textContent='正在忽略整张非牌面...';
  const data=await (await fetch('/api/update',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{label_id:r.label_id,ignore_card:true,notes:document.getElementById('notes').value}})}})).json();
  if(!data.ok&&String(data.error||'').includes('label_id not found')){{await loadRows();status.textContent='Queue changed; reloaded.';return;}}
  if(!data.ok){{status.textContent=data.error||'忽略失败';status.className='status danger';return;}}
  const ids=new Set(data.ignored_label_ids||[]);for(const row of rows){{if(ids.has(row.label_id))row.ignored='1';}}updateProgress(data.progress);refreshFilter();
}}
async function save(label){{
  if(!filtered.length)return;const r=filtered[index],status=document.getElementById('status');status.textContent='保存中...';
  const data=await (await fetch('/api/update',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{label_id:r.label_id,final_label:label,notes:document.getElementById('notes').value}})}})).json();
  if(!data.ok&&String(data.error||'').includes('label_id not found')){{await loadRows();status.textContent='Queue changed; reloaded.';return;}}
  if(!data.ok){{status.textContent=data.error||'保存失败';status.className='status danger';return;}}
  r.final_label=data.final_label;const original=rows.find(x=>x.label_id===r.label_id);if(original)Object.assign(original,r);updateProgress(data.progress);status.textContent='已保存';
  if(document.getElementById('hideDone').checked)refreshFilter();else move(1);
}}
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='TEXTAREA')return;
  if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key==='Enter')acceptCurrent();
  else if(filtered.length){{const r=filtered[index];let key=e.key.toUpperCase();if(key==='0')key='T';const valid=r.kind==='rank'?rankLabels:['S','H','D','C'];if(valid.includes(key))save(r.kind==='rank'?key:key.toLowerCase());}}
}});
loadRows();
</script>
</body></html>"""


def format_card_glyph_label_server_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Glyph audit UI: {payload.get('url')}",
            f"Queue CSV: {payload.get('queue_csv')}",
            f"Rows: {payload.get('row_count')}",
        ]
    )
