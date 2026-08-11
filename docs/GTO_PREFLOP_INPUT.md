# GTO 翻前输入与输出

GTO 与 CV 已拆开。CV 只提供牌桌事实；GTO 只读取明确的牌局状态 JSON，绝不把 `to_call_bb > 0` 自动当成前位有人加注，因为盲注本身也会产生这个数。

## 先看当前牌桌

实时运行时，想看 CV 已识别的完整事实而不是只看建议，用原来的 `screen-cv` 命令加：

```powershell
--console-mode full
```

该模式会显示庄家、原始位置、策略位置、行动顺序、手牌、公牌、底池、可见下注和是否轮到你。可见下注只是观察结果，不等于完整的翻前行动历史。

## 翻前决策需要什么

GTO 输入必须包含：

- `hero.cards`: 两张手牌
- `hero.position`: 原始位置，例如 `UTG+1`、`LJ`、`CO`
- `hero.gto_position`: 策略桶；8 人桌中 `UTG+1 -> UTG`，`LJ -> HJ`
- `hero.preflop_action_order`: 你翻前第几个行动
- `table.to_call_bb`: 现在要补多少 BB
- `preflop.action_history`: 到你当前回合前，按顺序发生的动作

历史以 `hero_to_act` 结束。盲注记为 `post_sb` / `post_bb` 时会被忽略；普通 `raise`、`3bet` 会被计数。

```json
"preflop": {
  "action_history": [
    {"position": "UTG", "action": "fold"},
    {"position": "HJ", "action": "raise", "amount_bb": 2.5},
    {"action": "hero_to_act"}
  ]
}
```

当前策略层支持三种翻前场景：

- 没有 raise: `RFI`，输出 `OPEN RAISE` 或 `FOLD`
- 一个 raise: `vs_open`，输出 `3BET`、`CALL` 或 `FOLD`
- 两个 raise: `vs_3bet`，输出 `4BET`、`CALL` 或 `FOLD`

前面有 limp、面对 4bet 以上、没有行动历史，都会输出 `WAIT`，并写出缺少什么；不会伪造一个 `FOLD`。

## 三个可运行样例

```powershell
python gto.py advise --state samples\preflop_utg1_unopened_aks.json --format text
python gto.py advise --state samples\preflop_co_facing_hj_open_ato.json --format text
python gto.py advise --state samples\preflop_co_facing_btn_3bet_aks.json --format text
```

输出会明确写成：

```text
翻前状态：前位有人 open/raise
位置：CO -> 策略桶 CO
行动顺序：第5个
已知前位行动：UTG fold -> HJ raise 2.5BB
建议：CALL 2.5 BB
频率：3BET 10% / CALL 45% / FOLD 45%
```

CV 适配层目前没有可靠的完整行动历史时，会返回 `preflop_context_incomplete`，不会再通过底池、盲注或单帧可见下注猜测 `vs_open`。

## 历史实盘截图复盘

实时录制结束后，直接运行：

```powershell
python gto.py review-states --limit 10
```

它读取默认的 `video_frames\screen_live\events.jsonl`，只挑有原始截图的 Hero 回合候选，生成：

```text
video_frames\screen_live\state_review\state_review.md
```

报告会复制原图，并逐张写明：街道、D、Hero 原始位置/策略桶、行动顺序、手牌、公牌、底池、可见下注、按钮 OCR、Hero 回合证据评级和单帧翻前假设。`疑似面对 open` 只是可见下注与位置产生的线索，不能直接进入 GTO；按钮区域太小或 CALL 金额与底池冲突时会明确标为不可信。
