const state = {
  level: "simple",
  startingStack: 50,
  game: null,
  busy: false,
  handKey: "",
  lastSeenLogCount: 0,
  recentNames: new Set(),
  recentTimer: null,
  botTimer: null,
  playbackMessage: "",
};

const seatTargets = {
  hero: "seatHero",
  bot1: "seatLeftBottom",
  bot2: "seatLeftTop",
  bot3: "seatTop",
  bot4: "seatRightTop",
  bot5: "seatRightBottom",
};

const seatLabels = {
  hero: "下方位置",
  bot1: "左下位置",
  bot2: "左上位置",
  bot3: "上方位置",
  bot4: "右上位置",
  bot5: "右下位置",
};

const actionLabels = {
  fold: "不玩了",
  call: "跟上",
  check: "先不出",
  bet: "出一些",
  new_hand: "下一手",
  new_match: "新对局",
};

const actionHints = {
  fold: "放弃这手",
  call: "补到一样多",
  check: "这轮先过",
  bet: "主动放 pt",
  new_hand: "继续打",
  new_match: "重新开始",
};

const rankLabels = {
  A: "A",
  K: "K",
  Q: "Q",
  J: "J",
  T: "10",
};

const suitLabels = {
  s: { text: "黑桃", icon: "♠", red: false },
  h: { text: "红桃", icon: "♥", red: true },
  d: { text: "方块", icon: "♦", red: true },
  c: { text: "梅花", icon: "♣", red: false },
};

const el = {
  noticeText: document.querySelector("#noticeText"),
  scoreText: document.querySelector("#scoreText"),
  scoreDelta: document.querySelector("#scoreDelta"),
  scoreNow: document.querySelector("#scoreNow"),
  scoreBars: document.querySelector("#scoreBars"),
  customPoints: document.querySelector("#customPoints"),
  ptOptions: document.querySelectorAll(".pt-option"),
  levelOptions: document.querySelectorAll(".level-option"),
  startButton: document.querySelector("#startButton"),
  handText: document.querySelector("#handText"),
  dealerText: document.querySelector("#dealerText"),
  potText: document.querySelector("#potText"),
  smallBlindText: document.querySelector("#smallBlindText"),
  bigBlindText: document.querySelector("#bigBlindText"),
  boardCards: document.querySelector("#boardCards"),
  tableMessage: document.querySelector("#tableMessage"),
  resultBlock: document.querySelector("#resultBlock"),
  resultText: document.querySelector("#resultText"),
  handLog: document.querySelector("#handLog"),
  matchLog: document.querySelector("#matchLog"),
  turnText: document.querySelector("#turnText"),
  needText: document.querySelector("#needText"),
  actionButtons: document.querySelector("#actionButtons"),
};

async function startMatch() {
  state.busy = true;
  state.handKey = "";
  state.lastSeenLogCount = 0;
  state.recentNames = new Set();
  state.playbackMessage = "";
  clearBotTimer();
  showNotice(`正在开新桌：每人 ${formatNumber(state.startingStack)} pt。`);
  renderActions([]);
  try {
    const response = await fetch("/api/game/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        level: state.level,
        match_mode: "ai",
        starting_stack: state.startingStack,
        slow_bots: true,
      }),
    });
    if (!response.ok) throw new Error("开桌失败");
    state.game = await response.json();
    showNotice(`已开 AI 连续桌第 ${state.game.hand_number} 手。`);
  } catch (error) {
    showNotice(`${error.message}，刷新页面再试。`);
  } finally {
    state.busy = false;
    render();
    scheduleBotStep();
  }
}

async function chooseAction(action) {
  if (state.busy || !state.game) return;
  clearBotTimer();
  if (action === "new_match") {
    await startMatch();
    return;
  }
  state.busy = true;
  renderActions([]);
  try {
    const response = await fetch("/api/game/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: state.game.id, action, slow_bots: true }),
  });
    if (!response.ok) throw new Error("动作失败");
    state.game = await response.json();
  } catch (error) {
    showNotice(`${error.message}，可以开新桌继续。`);
  } finally {
    state.busy = false;
    render();
    scheduleBotStep();
  }
}

async function stepBot() {
  if (state.busy || !state.game || !shouldStepBot(state.game)) return;
  state.busy = true;
  try {
    const response = await fetch("/api/game/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: state.game.id, action: "bot_step", slow_bots: true }),
    });
    if (!response.ok) throw new Error("电脑行动失败");
    state.game = await response.json();
  } catch (error) {
    showNotice(`${error.message}，可以刷新页面再试。`);
  } finally {
    state.busy = false;
    render();
    scheduleBotStep();
  }
}

function render() {
  const game = state.game;
  if (!game) {
    el.tableMessage.textContent = "正在开桌。";
    return;
  }

  syncRecentActors(game);
  renderTopLine(game);
  renderCards(el.boardCards, game.board, 5);
  renderSeats(game);
  renderLogs(game);
  renderScore(game);
  renderResult(game);
  renderTurn(game);
  renderActions(game.actions || []);
}

function renderTopLine(game) {
  el.handText.textContent = `第 ${game.hand_number} 手 · ${game.street_label || game.street_text || ""}`;
  el.dealerText.textContent = `Dealer：${game.dealer || "-"}`;
  el.potText.textContent = `底池 ${formatNumber(game.pot)} pt`;
  el.smallBlindText.textContent = blindLine(game.small_blind);
  el.bigBlindText.textContent = blindLine(game.big_blind);
  el.tableMessage.textContent = cleanText(game.message);
  showNotice(
    state.playbackMessage ||
      `${game.street_label || "当前"}。底池 ${formatNumber(game.pot)} pt。${blindLine(game.small_blind)}，${blindLine(game.big_blind)}。`,
  );
}

function renderSeats(game) {
  for (const player of game.players) {
    const target = document.querySelector(`#${seatTargets[player.id]}`);
    if (!target) continue;

    target.className = "seat";
    if (player.id === "hero") target.classList.add("seat-hero");
    if (player.id === "bot1") target.classList.add("seat-left-bottom");
    if (player.id === "bot2") target.classList.add("seat-left-top");
    if (player.id === "bot3") target.classList.add("seat-top");
    if (player.id === "bot4") target.classList.add("seat-right-top");
    if (player.id === "bot5") target.classList.add("seat-right-bottom");
    if (player.folded) target.classList.add("folded");
    if (player.id === game.current_player_id) target.classList.add("is-turn");
    if (state.recentNames.has(player.name)) target.classList.add("recent");
    if (game.status !== "playing" && game.winner === player.id) target.classList.add("winner");

    const cardsHtml = buildCardsHtml(player.cards, 2, !player.cards.length);
    const turnLabel = player.id === game.current_player_id ? (player.is_hero ? "轮到你" : "轮到他") : seatStateText(player);
    const dealer = player.is_dealer ? '<span class="dealer-chip">Dealer</span>' : "";
    const blind = blindChip(player);

    target.innerHTML = `
      <div class="seat-head">
        <div class="seat-title">
          <strong>${escapeHtml(player.name)}</strong>
          <span>${escapeHtml(seatLabels[player.id] || "座位")} · 行动第 ${player.order_position} 个 · ${turnLabel}</span>
        </div>
        ${dealer}
        ${blind}
      </div>
      <div class="seat-cards">${cardsHtml}</div>
      <div class="seat-money">
        <span>${seatMoneyText(player)}</span>
      </div>
      <div class="last-action">刚才：${escapeHtml(cleanText(player.last_action))}</div>
    `;
  }
}

function seatStateText(player) {
  if (player.out) return "出局";
  if (player.folded) return "已不玩";
  return "等待";
}

function blindLine(blind) {
  if (!blind) return "";
  return `${blind.label}：${blind.name} 出 ${formatNumber(blind.amount)} pt`;
}

function blindChip(player) {
  if (player.blind_role === "small") return '<span class="blind-chip">小盲</span>';
  if (player.blind_role === "big") return '<span class="blind-chip">大盲</span>';
  return "";
}

function seatMoneyText(player) {
  if (player.blind_role === "small") return `小盲出 ${formatNumber(player.blind_paid)} pt`;
  if (player.blind_role === "big") return `大盲出 ${formatNumber(player.blind_paid)} pt`;
  return `本轮出 ${formatNumber(player.round_bet)} pt`;
}

function renderTurn(game) {
  if (game.status !== "playing") {
    el.turnText.textContent = "本手结束";
    el.needText.textContent = `${game.street_label || "结束"} · 底池 ${formatNumber(game.pot)} pt · 本手积分 ${signedNumber(game.last_score_delta)} pt`;
    return;
  }

  if (game.current_player_id === "hero") {
    el.turnText.textContent = `${game.street_label || "当前"}，轮到你做决定`;
  } else {
    el.turnText.textContent = `${game.street_label || "当前"}，轮到 ${game.current_player || "电脑"}`;
  }
  el.needText.textContent =
    game.to_call > 0
      ? `底池 ${formatNumber(game.pot)} pt，你需要补 ${formatNumber(game.to_call)} pt`
      : `底池 ${formatNumber(game.pot)} pt，现在不需要补 pt`;
}

function renderActions(actions) {
  el.actionButtons.innerHTML = "";
  if (!actions.length) {
    const empty = document.createElement("div");
    empty.className = "no-action";
    empty.textContent = state.busy ? "正在处理..." : "等轮到你";
    el.actionButtons.append(empty);
    return;
  }

  for (const action of actions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `play-action ${action}`;
    button.disabled = state.busy;
    button.innerHTML = `
      <span>${escapeHtml(actionLabels[action] || action)}</span>
      <small>${escapeHtml(actionHints[action] || "")}</small>
    `;
    button.addEventListener("click", () => chooseAction(action));
    el.actionButtons.append(button);
  }
}

function renderLogs(game) {
  renderLogList(el.handLog, (game.hand_log || game.log || []).slice(-8).reverse(), "这手还没有记录。");
  renderLogList(el.matchLog, (game.match_log || []).slice(-10).reverse(), "打完一手后会记录到这里。");
}

function renderLogList(container, items, emptyText) {
  container.innerHTML = "";
  const shown = items.length ? items : [emptyText];
  for (const item of shown) {
    const li = document.createElement("li");
    li.textContent = cleanText(item);
    container.append(li);
  }
}

function renderScore(game) {
  el.scoreText.textContent = formatNumber(game.score);
  el.scoreNow.textContent = formatNumber(game.score);
  el.scoreDelta.textContent = `上一手 ${signedNumber(game.last_score_delta)} pt`;
  el.scoreBars.innerHTML = "";

  const history = (game.score_history || []).slice(-28);
  if (!history.length) {
    const empty = document.createElement("span");
    empty.className = "empty-score";
    empty.textContent = "打完一手后显示";
    el.scoreBars.append(empty);
    return;
  }

  const maxDelta = Math.max(...history.map((item) => Math.abs(item.delta)), 1);
  for (const item of history) {
    const bar = document.createElement("div");
    bar.className = "score-bar";
    if (item.delta > 0) bar.classList.add("up");
    if (item.delta < 0) bar.classList.add("down");
    bar.style.height = `${Math.max(8, Math.round((Math.abs(item.delta) / maxDelta) * 62))}px`;
    bar.title = `第 ${item.hand_number} 手：${signedNumber(item.delta)}，总积分 ${formatNumber(item.score)}`;
    el.scoreBars.append(bar);
  }
}

function renderResult(game) {
  if (game.status === "playing") {
    el.resultBlock.hidden = true;
    return;
  }
  el.resultBlock.hidden = false;
  const winner = game.winner === "tie" ? "平分" : game.players.find((player) => player.id === game.winner)?.name;
  el.resultText.textContent = `${winner || "本手"}结束，本手 ${signedNumber(game.last_score_delta)} pt，总积分 ${formatNumber(game.score)}。`;
}

function syncRecentActors(game) {
  const handKey = `${game.id}:${game.hand_number}`;
  if (state.handKey !== handKey) {
    state.handKey = handKey;
    state.lastSeenLogCount = 0;
    state.recentNames = new Set();
  }

  const allLog = game.hand_log || game.log || [];
  const fresh = allLog.slice(state.lastSeenLogCount);
  state.lastSeenLogCount = allLog.length;
  const names = fresh.map((line) => actorName(line, game.players)).filter(Boolean);
  if (fresh.length) state.playbackMessage = cleanText(fresh[fresh.length - 1]);
  if (!names.length) return;

  state.recentNames = new Set(names);
  window.clearTimeout(state.recentTimer);
  state.recentTimer = window.setTimeout(() => {
    state.recentNames = new Set();
    render();
  }, 1800);
}

function shouldStepBot(game) {
  return game.status === "playing" && game.current_player_id && game.current_player_id !== "hero";
}

function scheduleBotStep() {
  clearBotTimer();
  if (!state.game || !shouldStepBot(state.game) || state.busy) return;
  state.botTimer = window.setTimeout(stepBot, 950);
}

function clearBotTimer() {
  window.clearTimeout(state.botTimer);
  state.botTimer = null;
}

function actorName(line, players) {
  return players.map((player) => player.name).find((name) => line.startsWith(name)) || "";
}

function renderCards(container, cards, total) {
  container.innerHTML = buildCardsHtml(cards, total, false);
}

function buildCardsHtml(cards, total, useBacks) {
  const parts = [];
  for (let index = 0; index < total; index += 1) {
    const card = cards[index];
    if (!card && useBacks) {
      parts.push('<span class="table-card back">?</span>');
    } else if (!card) {
      parts.push('<span class="table-card empty">?</span>');
    } else {
      const suit = suitLabels[card[1]];
      const rank = rankLabels[card[0]] || card[0];
      const redClass = suit.red ? " red" : "";
      parts.push(
        `<span class="table-card${redClass}" title="${escapeHtml(rank + suit.text)}">${escapeHtml(rank + suit.icon)}</span>`,
      );
    }
  }
  return parts.join("");
}

function setStartingStack(value) {
  const number = Number(value);
  state.startingStack = Math.min(Math.max(Number.isFinite(number) ? number : 50, 5), 5000);
  el.customPoints.value = formatNumber(state.startingStack);
}

function showNotice(text) {
  el.noticeText.textContent = text;
}

function cleanText(value) {
  return String(value || "")
    .replace(/(\d+(?:\.\d+)?) 份/g, "$1 pt")
    .replaceAll("份钱", "pt")
    .replaceAll("pt钱", "pt")
    .replaceAll("份", "pt");
}

function formatNumber(value) {
  const number = Number(value) || 0;
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function signedNumber(value) {
  const number = Number(value) || 0;
  return number > 0 ? `+${formatNumber(number)}` : formatNumber(number);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

el.ptOptions.forEach((button) => {
  button.addEventListener("click", () => {
    setStartingStack(button.dataset.points);
    el.ptOptions.forEach((item) => item.classList.toggle("active", item === button));
  });
});

el.customPoints.addEventListener("change", () => {
  setStartingStack(el.customPoints.value);
  el.ptOptions.forEach((item) => item.classList.remove("active"));
});

el.customPoints.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    setStartingStack(el.customPoints.value);
    el.ptOptions.forEach((item) => item.classList.remove("active"));
    startMatch();
  }
});

el.levelOptions.forEach((button) => {
  button.addEventListener("click", () => {
    state.level = button.dataset.level;
    el.levelOptions.forEach((item) => item.classList.toggle("active", item === button));
  });
});

el.startButton.addEventListener("click", startMatch);

startMatch();
