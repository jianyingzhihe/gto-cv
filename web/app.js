const state = {
  level: "simple",
  mode: "quiz",
  gameKind: "small",
  startingStack: 50,
  round: null,
  game: null,
  answered: false,
  score: 0,
  total: 0,
  handKey: "",
  lastSeenLogCount: 0,
  recentEvents: [],
  recentActorNames: new Set(),
  globalRecords: [],
  seenGlobalRecords: new Set(),
};

const actionLabels = {
  fold: "不玩了",
  call: "跟上",
  check: "先不出",
  bet: "先出一些",
  raise: "多出一些",
  limp: "只补最少",
  "3bet": "再多出一些",
  "4bet": "再再多出一些",
  new_hand: "再开小对局",
  new_match: "重新开始大对局",
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

const stageLabels = {
  preflop: "还没发公共牌",
  flop: "桌上已经有 3 张公共牌",
  turn: "桌上已经有 4 张公共牌",
  river: "5 张公共牌都发完了",
};

const positionLabels = {
  UTG: "很早行动",
  HJ: "较早行动",
  CO: "较晚行动",
  BTN: "最后行动",
  SB: "先放一小份底钱的位置",
  BB: "先放一大份底钱的位置",
};

const scenarioLabels = {
  rfi: "前面没人多出钱",
  vs_open: "前面有人多出钱",
  vs_3bet: "前面已经加过两轮钱",
};

const el = {
  modes: document.querySelectorAll(".mode"),
  levels: document.querySelectorAll(".level"),
  gameKindControls: document.querySelector("#gameKindControls"),
  gameKinds: document.querySelectorAll(".game-kind"),
  pointControls: document.querySelector("#pointControls"),
  pointChoices: document.querySelectorAll(".point-choice"),
  customPoints: document.querySelector("#customPoints"),
  applyCustomPoints: document.querySelector("#applyCustomPoints"),
  scoreText: document.querySelector("#scoreText"),
  stageText: document.querySelector("#stageText"),
  moneyText: document.querySelector("#moneyText"),
  boardCards: document.querySelector("#boardCards"),
  heroCards: document.querySelector("#heroCards"),
  spotTitle: document.querySelector("#spotTitle"),
  hintList: document.querySelector("#hintList"),
  actionButtons: document.querySelector("#actionButtons"),
  resultPanel: document.querySelector("#resultPanel"),
  resultText: document.querySelector("#resultText"),
  afterList: document.querySelector("#afterList"),
  noticeBox: document.querySelector("#noticeBox"),
  globalPanel: document.querySelector("#globalPanel"),
  globalSummary: document.querySelector("#globalSummary"),
  globalList: document.querySelector("#globalList"),
  scorePanel: document.querySelector("#scorePanel"),
  scoreNow: document.querySelector("#scoreNow"),
  scoreChange: document.querySelector("#scoreChange"),
  scoreChart: document.querySelector("#scoreChart"),
  recentBox: document.querySelector("#recentBox"),
  recentList: document.querySelector("#recentList"),
  logBox: document.querySelector("#logBox"),
  logList: document.querySelector("#logList"),
  playersBox: document.querySelector("#playersBox"),
  playerList: document.querySelector("#playerList"),
  nextButton: document.querySelector("#nextButton"),
  rulesButton: document.querySelector("#rulesButton"),
  rulesDialog: document.querySelector("#rulesDialog"),
  closeRules: document.querySelector("#closeRules"),
};

async function loadRound() {
  if (state.mode === "game") {
    await startGame();
    return;
  }
  state.answered = false;
  el.resultPanel.hidden = true;
  el.actionButtons.innerHTML = "";
  el.spotTitle.textContent = "正在发题...";
  const response = await fetch(`/api/round?level=${encodeURIComponent(state.level)}`);
  if (!response.ok) {
    el.spotTitle.textContent = "出题失败，刷新页面再试。";
    return;
  }
  state.round = await response.json();
  renderRound();
}

async function startGame() {
  state.answered = false;
  el.resultPanel.hidden = true;
  el.logBox.hidden = false;
  el.actionButtons.innerHTML = "";
  el.spotTitle.textContent = "正在开始一局...";
  const response = await fetch("/api/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      level: state.level,
      match_mode: state.gameKind,
      starting_stack: state.startingStack,
    }),
  });
  state.game = await response.json();
  renderGame();
}

function renderRound() {
  el.logBox.hidden = true;
  el.playersBox.hidden = true;
  el.recentBox.hidden = true;
  el.globalPanel.hidden = true;
  el.scorePanel.hidden = true;
  el.noticeBox.hidden = true;
  const round = state.round;
  const spot = round.state;
  el.stageText.textContent = stageLabels[spot.action.street];
  el.moneyText.textContent = `桌上已有 ${spot.table.pot_bb} 份`;
  el.spotTitle.textContent = [
    `你在${positionLabels[spot.hero.position]}`,
    scenarioLabels[spot.action.scenario],
    spot.table.to_call_bb > 0 ? `继续要放 ${spot.table.to_call_bb} 份` : "现在没人逼你出钱",
  ].join("，");

  renderCards(el.heroCards, spot.hero.cards, 2);
  renderCards(el.boardCards, spot.table.board, 5);
  renderList(el.hintList, round.lesson.before);
  renderActions(round.actions);
}

function renderGame() {
  const game = state.game;
  const hero = game.players.find((player) => player.is_hero);
  const stillPlaying = game.players.filter((player) => !player.folded).length;
  syncGameRecords(game);
  el.resultPanel.hidden = game.status === "playing";
  el.scoreText.textContent = game.match_mode === "ai" ? `积分 ${formatScore(game.score)}` : "对局中";
  el.stageText.textContent = game.street_text;
  el.moneyText.textContent = `${game.match_mode_text} 第 ${game.hand_number} 手，桌上已有 ${game.pot} 份`;
  el.spotTitle.textContent = [
    game.message,
    `每人起始 ${formatScore(game.starting_stack)} pt`,
    `这手从${game.first_actor}开始，你第 ${game.hero_order} 个做决定`,
    `还有 ${stillPlaying} 个人没有退出这局`,
  ].join("。");
  el.playersBox.hidden = false;
  el.logBox.hidden = false;
  renderCards(el.heroCards, game.hero_cards, 2);
  renderCards(el.boardCards, game.board, 5);
  renderPlayers(game.players, game.status !== "playing");
  renderList(el.hintList, game.hint);
  renderRecentEvents();
  renderList(el.logList, game.log);
  renderGlobalRecords(game);
  renderScorePanel(game);
  renderGameActions(game.actions, game);

  if (game.status !== "playing") {
    el.resultPanel.hidden = false;
    el.resultText.textContent = resultText(game);
    renderList(el.afterList, revealPlayers(game));
  }
}

function renderPlayers(players, done) {
  el.playerList.innerHTML = "";
  for (const player of players) {
    const node = document.createElement("div");
    node.className = "player";
    if (player.is_hero) node.classList.add("hero");
    if (player.folded) node.classList.add("folded");
    if (done && player.cards.length) node.classList.add("revealed");
    if (!player.is_hero && state.recentActorNames.has(player.name)) node.classList.add("recent");

    const cards = player.cards.length ? describeCards(player.cards) : "暂时看不到";
    const status = player.out ? "没钱了" : player.folded ? "已经退出这局" : "还在这局里";
    node.innerHTML = `
      <div class="player-top">
        <strong>${player.name}</strong>
        <span>${status}</span>
      </div>
      <div class="player-meta">
        <span>第 ${player.order_position} 个做决定</span>
        <span>还剩 ${player.stack} 份</span>
        <span>这轮已放 ${player.round_bet} 份</span>
      </div>
      <div class="player-action">${player.last_action}</div>
      <div class="player-cards">${cards}</div>
    `;
    el.playerList.append(node);
  }
}

function syncGameRecords(game) {
  const handKey = `${game.id}:${game.hand_number}`;
  if (state.handKey !== handKey) {
    state.handKey = handKey;
    state.lastSeenLogCount = 0;
    state.recentEvents = [];
    state.recentActorNames = new Set();
  }

  const startAdded = addGlobalRecord({
    key: `start:${handKey}`,
    title: `已开${game.match_mode_text}第 ${game.hand_number} 手`,
    detail:
      game.match_mode === "big"
        ? "这是大对局，钱会一直带到下一手，直到有人没钱。"
        : game.match_mode === "ai"
          ? `这是 AI 对战，每手从 ${formatScore(game.starting_stack)} pt 开始，按你的输赢累积积分。`
        : `这是小对局，每个人从 ${formatScore(game.starting_stack)} pt 开始，这一手结束就结束。`,
    lines: [
      `行动顺序：${game.turn_order.join(" → ")}。`,
      `每人起始 ${formatScore(game.starting_stack)} pt。`,
      game.hand_log?.[0] || game.message,
    ],
  });
  if (startAdded) showNotice(`已开${game.match_mode_text}第 ${game.hand_number} 手`);

  const allLog = game.hand_log || game.log || [];
  const fresh = allLog.slice(state.lastSeenLogCount);
  state.lastSeenLogCount = allLog.length;
  state.recentEvents = fresh.filter(isOpponentEvent);
  state.recentActorNames = new Set(state.recentEvents.map(actorName).filter(Boolean));

  for (const record of game.hand_records || []) {
    addGlobalRecord({
      key: `finish:${game.id}:${record.hand_number}`,
      title: `${record.mode_text}第 ${record.hand_number} 手结果`,
      detail: `${record.result} 结束后：${record.stacks}。`,
      lines: buildReviewLines(record),
    });
  }
}

function addGlobalRecord(record) {
  if (state.seenGlobalRecords.has(record.key)) return false;
  state.seenGlobalRecords.add(record.key);
  state.globalRecords.unshift(record);
  if (state.globalRecords.length > 40) state.globalRecords.length = 40;
  return true;
}

function showNotice(text) {
  el.noticeBox.textContent = text;
  el.noticeBox.hidden = false;
}

function isOpponentEvent(line) {
  return ["左下电脑", "左上电脑", "对面电脑", "右上电脑", "右下电脑"].some((name) =>
    line.startsWith(name),
  );
}

function actorName(line) {
  return (
    ["左下电脑", "左上电脑", "对面电脑", "右上电脑", "右下电脑"].find((name) =>
      line.startsWith(name),
    ) || ""
  );
}

function renderRecentEvents() {
  if (!state.recentEvents.length) {
    el.recentBox.hidden = true;
    return;
  }
  el.recentBox.hidden = false;
  renderList(el.recentList, state.recentEvents);
}

function renderGlobalRecords(game) {
  el.globalPanel.hidden = state.mode !== "game";
  el.globalSummary.textContent =
    game.match_mode === "ai"
      ? `已记录 ${state.globalRecords.length} 条，当前积分 ${formatScore(game.score)}`
      : `已记录 ${state.globalRecords.length} 条，当前是${game.match_mode_text}第 ${game.hand_number} 手`;
  el.globalList.innerHTML = "";

  for (const [index, record] of state.globalRecords.entries()) {
    const item = document.createElement("details");
    item.className = "review-item";
    item.open = index < 2;

    const summary = document.createElement("summary");
    summary.textContent = record.title;
    item.append(summary);

    const detail = document.createElement("p");
    detail.textContent = record.detail;
    item.append(detail);

    const list = document.createElement("ol");
    for (const line of record.lines) {
      const li = document.createElement("li");
      li.textContent = line;
      list.append(li);
    }
    item.append(list);
    el.globalList.append(item);
  }
}

function renderScorePanel(game) {
  if (game.match_mode !== "ai") {
    el.scorePanel.hidden = true;
    return;
  }
  el.scorePanel.hidden = false;
  el.scoreNow.textContent = formatScore(game.score);
  el.scoreChange.textContent = `上一手 ${signedScore(game.last_score_delta)}`;
  el.scoreChart.innerHTML = "";

  const history = game.score_history.slice(-24);
  if (!history.length) {
    const empty = document.createElement("span");
    empty.className = "score-empty";
    empty.textContent = "打完一手后显示走势";
    el.scoreChart.append(empty);
    return;
  }

  const maxDelta = Math.max(...history.map((item) => Math.abs(item.delta)), 1);
  for (const item of history) {
    const bar = document.createElement("div");
    bar.className = "score-bar";
    if (item.delta > 0) bar.classList.add("up");
    if (item.delta < 0) bar.classList.add("down");
    bar.style.height = `${Math.max(8, Math.round((Math.abs(item.delta) / maxDelta) * 64))}px`;
    bar.title = `第 ${item.hand_number} 手：${signedScore(item.delta)}，总分 ${formatScore(item.score)}`;
    el.scoreChart.append(bar);
  }
}

function buildReviewLines(record) {
  const orderLine = record.turn_order
    ? `行动顺序：${record.turn_order.join(" → ")}。你第 ${record.hero_order} 个做决定。`
    : "行动顺序：旧记录没有保存。";
  const lines = [orderLine, ...record.log];
  if (record.score_delta !== null && record.score_delta !== undefined) {
    lines.push(`本手积分：${signedScore(record.score_delta)}，总积分 ${formatScore(record.score)}。`);
  }
  if (record.starting_stack) lines.push(`本手每人起始 ${formatScore(record.starting_stack)} pt。`);
  if (record.board.length) lines.push(`最后桌上的公共牌：${describeCards(record.board)}。`);
  for (const player of record.players) {
    lines.push(`${player.name}：${describeCards(player.cards)}，最后还剩 ${player.stack} 份。`);
  }
  return lines;
}

function resultText(game) {
  if (game.match_mode === "ai") {
    return `本手 ${signedScore(game.last_score_delta)}，总积分 ${formatScore(game.score)}`;
  }
  if (game.match_done && game.eliminated.length) return `${game.eliminated.join("、")} 没钱了，大对局结束`;
  if (game.winner === "hero") return "这局你赢了";
  if (game.winner === "tie") return "这局平分";
  const winner = game.players.find((player) => player.id === game.winner);
  return winner ? `这局 ${winner.name} 赢了` : "这局结束了";
}

function revealPlayers(game) {
  const shown = game.players
    .filter((player) => !player.is_hero && player.cards.length)
    .map((player) => `${player.name} 的手牌是 ${describeCards(player.cards)}。`);
  const history = game.match_mode === "big" || game.match_mode === "ai" ? game.match_log.slice(-3) : [];
  return [...shown, ...history, ...game.log.slice(-3)];
}

function renderCards(container, cards, total) {
  container.innerHTML = "";
  for (let index = 0; index < total; index += 1) {
    const card = cards[index];
    const node = document.createElement("div");
    node.className = "card";
    if (!card) {
      node.classList.add("empty");
      node.textContent = "?";
    } else {
      const suit = suitLabels[card[1]];
      const rank = rankLabels[card[0]] || card[0];
      if (suit.red) node.classList.add("red");
      node.textContent = `${rank}${suit.icon}`;
      node.title = `${rank}${suit.text}`;
    }
    container.append(node);
  }
}

function renderList(container, items) {
  container.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    container.append(li);
  }
}

function renderActions(actions) {
  el.actionButtons.innerHTML = "";
  for (const action of actions) {
    const button = document.createElement("button");
    button.className = "action";
    button.type = "button";
    button.dataset.action = action;
    button.textContent = actionLabels[action] || action;
    button.addEventListener("click", () => chooseAction(action));
    el.actionButtons.append(button);
  }
}

function renderGameActions(actions, game) {
  el.actionButtons.innerHTML = "";
  for (const action of actions) {
    const button = document.createElement("button");
    button.className = "action";
    button.type = "button";
    button.dataset.action = action;
    button.textContent = gameActionLabel(action, game);
    button.addEventListener("click", () => chooseGameAction(action));
    el.actionButtons.append(button);
  }
}

function gameActionLabel(action, game) {
  if (action === "new_hand") {
    return game.match_mode === "big" || game.match_mode === "ai" ? "下一手" : "再开小对局";
  }
  return actionLabels[action] || action;
}

function chooseAction(action) {
  if (state.answered || !state.round) return;
  state.answered = true;
  state.total += 1;

  const mix = state.round.answer.decision.mix;
  const primary = state.round.answer.decision.primary_action;
  const chosenValue = mix[action] || 0;
  let points = 0;
  let message = "";

  if (action === primary) {
    points = 1;
    message = `正确，最推荐的是「${actionLabels[primary]}」。`;
  } else if (chosenValue >= 30) {
    points = 0.5;
    message = `可以，这个选择有时也会用。最推荐的是「${actionLabels[primary]}」。`;
  } else {
    message = `这次不太好。最推荐的是「${actionLabels[primary]}」。`;
  }

  state.score += points;
  el.scoreText.textContent = `${formatScore(state.score)} / ${state.total}`;
  el.resultText.textContent = message;
  renderList(el.afterList, simplifyAfter(state.round.lesson.after));
  el.resultPanel.hidden = false;

  for (const button of el.actionButtons.querySelectorAll("button")) {
    const current = button.dataset.action;
    if (current === primary) button.classList.add("correct");
    if (current === action && action !== primary) button.classList.add("miss");
    button.disabled = true;
  }
}

async function chooseGameAction(action) {
  if (!state.game) return;
  if (
    action === "new_match" ||
    (action === "new_hand" && !["big", "ai"].includes(state.game.match_mode))
  ) {
    await startGame();
    return;
  }
  const response = await fetch("/api/game/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: state.game.id, action }),
  });
  state.game = await response.json();
  renderGame();
}

function simplifyAfter(items) {
  return items.map((item) =>
    item
      .replaceAll("翻前手牌", "你的两张牌")
      .replaceAll("当前范围继续频率约", "这类牌通常会继续玩的比例大约")
      .replaceAll("建议尺度约", "如果选择多出一些，建议放")
      .replaceAll("主线动作", "最推荐的动作")
      .replaceAll("权益", "赢面")
      .replaceAll("底池", "桌上的钱"),
  );
}

function formatScore(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function signedScore(value) {
  const number = Number(value) || 0;
  return number > 0 ? `+${formatScore(number)}` : formatScore(number);
}

function describeCards(cards) {
  return cards
    .map((card) => {
      const suit = suitLabels[card[1]];
      const rank = rankLabels[card[0]] || card[0];
      return `${rank}${suit.text}`;
    })
    .join(" ");
}

el.modes.forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    updateModeView(button);
    loadRound();
  });
});

function updateModeView(activeButton) {
  el.modes.forEach((item) => item.classList.toggle("active", item === activeButton));
  el.gameKindControls.hidden = state.mode !== "game";
  el.pointControls.hidden = state.mode !== "game";
  el.scoreText.textContent = state.mode === "quiz" ? `${formatScore(state.score)} / ${state.total}` : "对局中";
  el.nextButton.textContent = state.mode === "quiz" ? "下一题" : "重新开始";
}

el.levels.forEach((button) => {
  button.addEventListener("click", () => {
    state.level = button.dataset.level;
    el.levels.forEach((item) => item.classList.toggle("active", item === button));
    loadRound();
  });
});

el.gameKinds.forEach((button) => {
  button.addEventListener("click", () => {
    state.gameKind = button.dataset.gameKind;
    el.gameKinds.forEach((item) => item.classList.toggle("active", item === button));
    if (state.mode === "game") loadRound();
  });
});

el.pointChoices.forEach((button) => {
  button.addEventListener("click", () => {
    setStartingStack(button.dataset.points);
    el.pointChoices.forEach((item) => item.classList.toggle("active", item === button));
    el.customPoints.value = formatScore(state.startingStack);
    if (state.mode === "game") loadRound();
  });
});

el.applyCustomPoints.addEventListener("click", () => {
  setStartingStack(el.customPoints.value);
  el.customPoints.value = formatScore(state.startingStack);
  el.pointChoices.forEach((item) => item.classList.remove("active"));
  if (state.mode === "game") loadRound();
});

el.customPoints.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    el.applyCustomPoints.click();
  }
});

function setStartingStack(value) {
  const number = Number(value);
  state.startingStack = Math.min(Math.max(Number.isFinite(number) ? number : 50, 5), 5000);
}

el.nextButton.addEventListener("click", loadRound);
el.rulesButton.addEventListener("click", () => el.rulesDialog.showModal());
el.closeRules.addEventListener("click", () => el.rulesDialog.close());

loadRound();
