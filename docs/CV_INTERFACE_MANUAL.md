# 扑克 CV 接口手册

本文只覆盖 **CV 识别**：截屏、牌桌定位、手牌和公共牌识别、庄家/座位/下注/底池识别、覆盖层、状态文件、问题帧保存、离线回放和人工校对。

不包含 GTO 建议、下注决策或任何自动操作。本文所有实时命令都不使用 `--with-advice`。

## 1. 最常用：实盘实时识别

在 PowerShell 中执行：

```powershell
cd E:\dezhou
python gto.py screen-cv --bbox-file "video_frames\screen_calibrate\bbox.json" --lock-layout --hero-name "鱼寻欢" --trigger frame --every 1 --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --console-mode full --output-dir "video_frames\screen_live" --show-overlay --format text
```

- 终端会持续打印完整牌桌状态。
- 屏幕会显示透明覆盖层，框出手牌、公共牌、庄家和识别结果。
- 停止时按 `Ctrl+C`。
- `--trigger frame --every 1` 表示每秒分析一帧；它是当前实时信息流模式。

启动后最重要的输出文件是：

```text
video_frames\screen_live\current_state.json
video_frames\screen_live\events.jsonl
video_frames\screen_live\latest_overlay.png
```

## 2. 第一次使用或窗口比例变化：三步定位

### 第 1 步：圈住整个牌桌窗口

先把 WPT 牌桌窗口完整显示在屏幕上，再执行：

```powershell
cd E:\dezhou
python gto.py screen-cv --pick-bbox --hero-name "鱼寻欢" --output-dir "video_frames\screen_calibrate"
```

鼠标拖一个尽量包含整个 WPT 窗口的外框，按 `Enter` 或 `Space` 确认。它会保存初始外框和一组后续命令。

### 第 2 步：检查并修正自动定位

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_review_auto_bbox_command.txt")
```

弹出的图会显示程序根据外框推导出的牌桌区域、公共牌区和英雄手牌区。若位置正确，按提示确认；若不正确，重新拖动修正。确认后会写入：

```text
video_frames\screen_calibrate\analysis_bbox.json
```

这是之后稳定识别的布局基准。`--lock-layout` 会固定使用它，避免每帧为了“纠偏”而把卡牌区漂移到头像或动画上。

### 第 3 步：必要时单独修正两张手牌

当覆盖层里 H1/H2 没有准确框住两张底部手牌时，执行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_pick_hero_cards_command.txt")
```

依次圈 H1 和 H2：

- H1 要包含左牌可读的点数和花色，不要包含头像、昵称或桌边。
- H2 要从右牌自己的点数角开始圈，只包含右牌可读部分，不能把左牌与右牌重叠的区域一起圈进去。

确认后会保存手牌固定相对位置，并生成 `run_live_overlay_command.txt`。此后应运行这个新生成的命令，而不是继续使用未带手牌框的旧命令：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_live_overlay_command.txt")
```

如果脚本已生成 `run_live_overlay_command.txt`，也可以运行它进行覆盖层检查；注意旧版生成文件可能带 `--with-advice`，那一项与 CV 无关。日常纯 CV 请优先使用第 1 节命令。

## 3. `screen-cv` 命令接口

### 画面来源与布局

| 参数 | 含义 | 日常建议 |
| --- | --- | --- |
| `--pick-bbox` | 手工选择第一层外框 | 仅首次/缩放变化时用 |
| `--bbox "x,y,w,h"` | 直接传入外框 | 临时调试用，不建议把 `x,y,w,h` 字样原样复制 |
| `--bbox-file path` | 从文件读校准外框 | 日常使用手工外框 `bbox.json`；程序会自动读取同目录的内部牌桌框 `analysis_bbox.json` |
| `--latest-bbox` | 读取最近一次保存的外框 | 临时恢复用 |
| `--monitor 1` | 选择显示器编号 | 多屏时指定 |
| `--hero-name "鱼寻欢"` | 用固定昵称辅助定位英雄座位 | 日常保留 |
| `--lock-layout` | 锁定校准后的相对牌面位置 | 日常保留 |
| `--review-auto-bbox` | 打开第二层布局检查/修正 | 校准时使用 |
| `--pick-hero-cards` | 手工校准 H1/H2 两张手牌 | H1/H2 框错时使用 |
| `--hero-cards-file path` | 读取已保存的 H1/H2 配置 | 一般由校准命令自动处理 |
| `--auto-bbox` | 在外框内自动寻找牌桌 | 仅作备用，不替代确认后的固定布局 |
| `--auto-bbox-refresh 10` | 每 10 秒重新尝试自动定位 | 仅窗口会移动或被遮挡时使用 |

### 采样与实时输出

| 参数 | 含义 | 日常建议 |
| --- | --- | --- |
| `--trigger frame` | 固定频率逐帧采样 | 实盘推荐 |
| `--trigger state-change` | 仅状态变化时写事件 | 适合复盘，输出较少 |
| `--trigger visual-change` | 画面差异超过阈值才分析 | 适合降低 CPU 占用 |
| `--every 1` | 每秒分析次数 | 实盘从 `1` 开始 |
| `--visual-threshold 2.4` | `visual-change` 模式的变化阈值 | 一般保持默认 |
| `--min-event-gap 1` | 相邻事件最短间隔（秒） | 一般保持默认 |
| `--console-mode full` | 打印完整 CV 状态 | 实盘排查推荐 |
| `--console-heartbeat 10` | 无变化时的心跳输出间隔 | 可选 |
| `--format text/json` | 终端输出格式 | 人看用 `text`，程序接入用 `json` |
| `--compact` | 压缩 JSON 输出 | 程序接入时可选 |
| `--duration 300` | 自动运行 300 秒后停止 | 录制排查时可选 |

### 识别与保存

| 参数 | 含义 | 日常建议 |
| --- | --- | --- |
| `--min-confidence 0.35` | 目标检测最低置信度 | 当前校准版建议保留 |
| `--ocr-scale 0.65` | 金额/文本 OCR 的缩放 | 当前校准版建议保留 |
| `--no-ocr` | 关闭 OCR | 仅性能排查时使用 |
| `--ocr-action-only` | 只 OCR 操作区 | 仅性能排查时使用 |
| `--dealer-refresh-frames 4` | 每 4 个采样帧刷新一次庄家 D 检测 | 当前建议保留 |
| `--seats 8` | 牌桌座位数 | 六人桌请改为 `6` |
| `--show-overlay` | 显示透明 CV 覆盖层 | 调试/实盘检查推荐 |
| `--save-frames` | 保存普通截屏帧 | 离线复盘时用，磁盘占用较大 |
| `--save-annotated` | 保存带标注的截屏帧 | 离线复盘时用 |
| `--no-problem-frames` | 不保存异常帧 | 不建议；默认会留证据 |
| `--problem-frame-limit 240` | 最多保留的异常帧数量 | 可按磁盘空间调整 |
| `--no-card-samples` | 不保存卡牌样本 | 不建议；默认样本可用于后续校对 |
| `--card-sample-interval 30` | 卡牌样本采样间隔 | 默认即可 |
| `--card-sample-limit 1000` | 卡牌样本上限 | 默认即可 |

## 4. 换电脑或新显示比例：先预检，再开实时流

新电脑、不同显示缩放比例、不同腾讯会议窗口大小，都不能直接复用旧电脑的手牌相对位置。请完整执行第 2 节的第 1、2 步，然后先运行一帧预检：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_preflight_command.txt")
```

预检正常的最低条件是：

- 覆盖层的 H1/H2 分别落在底部两张真实手牌上。
- `hero.cards` 有两张牌，而不是 `[]`、一张牌或带 `?` 的牌。
- `ocr_mode` 不是 `disabled`。

若看到 `fallback_fixed_roi`、`hero_cards_incomplete` 或 `hero.cards: []`，这不是屏幕抓取失败，而是程序没有在当前窗口中找到手牌，已经退回到不适用的固定裁剪位置。立刻按 `Ctrl+C` 停止实时程序，执行第 2 节第 3 步的手牌框校准，再执行新生成的 `run_live_overlay_command.txt`。

`hero_turn_not_confirmed` 出现在上述情况之后，是安全保护结果：手牌不完整时，程序不会给出可操作的建议。应先修复 H1/H2，不要先调整行动判断。

若预检或终端显示 `ocr_mode: disabled`，说明当前 Python 环境缺少文字识别组件；在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python -c "from rapidocr_onnxruntime import RapidOCR; print('OCR OK')"
```

再重新运行预检。文字识别恢复前，底池、跟注额和按钮文字可能为空，不能把相应数值当作可靠输入。

## 5. CV 状态数据接口

`current_state.json` 始终保存最新一次结果。外部程序只需要轮询这个文件即可，不需要读取终端文本。

结构要点如下：

```json
{
  "ok": true,
  "source": {
    "kind": "screen",
    "timestamp_sec": 12.34,
    "frame_index": 12,
    "analysis_ms": 860.3,
    "screen_region": {"left": 100, "top": 200, "width": 1200, "height": 820},
    "overlay_path": "video_frames\\screen_live\\latest_overlay.png"
  },
  "event": {"index": 12, "trigger": "frame", "reason": "frame"},
  "table": {
    "seat_count": 8,
    "street": "preflop",
    "dealer_seat": "bottom_left",
    "dealer_position": "BTN",
    "pot_bb": 3.4,
    "to_call_bb": 2.0,
    "board": []
  },
  "hero": {
    "seat": "bottom_hero",
    "position": "BTN",
    "cards": ["As", "Kd"],
    "status": "active_or_showdown",
    "is_turn": false
  },
  "seats": [],
  "bets": [],
  "hero_turn": {"is_turn": false},
  "confidence": {}
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `ok` | 本帧是否成功形成状态；`false` 时不要把该帧当牌局事实使用 |
| `source.analysis_ms` | 这一帧 CV 推理耗时（毫秒） |
| `table.street` | `preflop`、`flop`、`turn`、`river` |
| `table.pot_bb` | 识别到的底池，单位 BB |
| `table.to_call_bb` | Hero 当前需补齐的金额，单位 BB |
| `table.board` | 公共牌数组，如 `["9s", "Kd", "Tc"]` |
| `hero.cards` | Hero 两张手牌；不完整或低置信度时可能包含 `?` |
| `table.dealer_seat` | D 按钮所在座位 |
| `table.dealer_position` | D 按钮对应的位置，如 `BTN` |
| `seats` | 每座位的状态、位置、下注、弃牌等详情 |
| `bets` | 当前可见下注列表，金额单位 BB |
| `confidence` | 庄家、底池、手牌、公共牌等置信信息 |

`events.jsonl` 是历史事件流：每一行是一份完整 JSON 状态。适合后续做视频回放、统计或状态机，不要把它当普通 CSV 读取。

## 6. 覆盖层和异常保存接口

| 路径 | 内容 | 何时看 |
| --- | --- | --- |
| `video_frames\screen_live\latest_overlay.png` | 最近一帧覆盖层截图 | 先看框是否落在正确的牌/桌面位置 |
| `video_frames\screen_live\current_state.json` | 当前机器可读状态 | 外部程序接入/终端结果异常时看 |
| `video_frames\screen_live\events.jsonl` | 全部历史状态事件 | 复盘某一时刻、追踪状态变化 |
| `video_frames\screen_live\problem_frames\` | 桌面被遮挡、布局错误或识别异常的现场截图 | “为什么这帧没识别到”时看 |
| `video_frames\screen_live\card_debug\` | 卡牌识别调试包 | 数字/花色识别错误时看 |
| `video_frames\screen_live\card_samples\` | 采集的手牌/公共牌样本 | 离线重新识别和人工校对 |
| `video_frames\screen_live\card_sample_predictions.csv` | 卡牌样本逐项预测及来源 | 生成数字/花色人工校对队列时使用 |
| `video_frames\screen_calibrate\bbox.json` | 手工确认的完整窗口外框 | 日常 `--bbox-file` 使用，也是底部操作区的坐标来源 |
| `video_frames\screen_calibrate\analysis_bbox.json` | 已确认的内部牌桌框 | 由程序随 `bbox.json` 自动读取，用于牌面、庄家和座位识别 |

一个 `card_debug` 样本通常包含：

```text
frame.png                 原始识别帧
screen_context.png        屏幕上下文
diagnostic_overlay.png    框和结果叠加图
*_card.png                整张牌裁剪
*_rank.png                数字/字母裁剪
*_suit.png                花色裁剪
metadata.json             裁剪位置、结果、置信度和原因
```

排错顺序固定为：先看 `diagnostic_overlay.png` 的框是否正确，再看 `*_card.png`，再看 `*_rank.png` / `*_suit.png`。不要先假设是分类器错误，裁剪位置错也会导致任何模型输出错误。

## 7. 异常状态的含义

| 现象/文本 | CV 含义 | 应该做什么 |
| --- | --- | --- |
| `poker table occluded` | 牌桌区域被其他窗口、动画或画面切换遮挡 | 看 `problem_frames`，等稳定画面后恢复 |
| `hero_cards_incomplete` | H1/H2 未识别完整，或只有一张牌可见 | 看覆盖层；若出现 `fallback_fixed_roi` 或 `hero.cards: []`，先重新校准 H1/H2；持续发生则完整重做第 2 节 |
| `hero_turn_not_confirmed` | 尚无足够证据确认轮到 Hero | 若同时有 `hero_cards_incomplete`，先修复 H1/H2；否则再检查底部操作区是否完整可见 |
| `ocr_mode: disabled` | 当前 Python 缺少文字识别组件，或启动时显式关闭了 OCR | 执行 `python -m pip install -r requirements.txt`，然后重新启动 |
| `hero_action_controls_not_visible` | Hero 的底部操作区当前不可见 | 仅表示无法从按钮确认是否轮到 Hero；牌桌信息仍可继续读取 |
| 手牌/公共牌出现 `?` | 当前裁剪或分类置信度不足 | 查看 `card_debug`，不要把 `?` 强行当成真实牌 |
| 框跑到头像/桌面 logo | 当前布局不匹配或尚未锁定正确 H1/H2 | 重新执行三步定位，然后用 `--lock-layout` |

## 8. 离线视频与卡牌回放接口

### 视频 CV

对录制视频生成逐帧/事件流结果：

```powershell
python gto.py live-cv "C:\path\to\table.mp4" --trigger state-change --every 1 --output-dir "video_frames\cv_video_review" --format text
```

- 想看每个采样帧：使用 `--trigger frame`。
- 想减少重复帧：使用 `--trigger state-change`。
- 输出目录同样会包含状态、事件、标注和问题帧；具体保存项取决于命令参数。

### 使用固定布局重新识别历史卡牌样本

当已更新卡牌裁剪/识别逻辑，需要用固定布局重新跑历史样本时：

```powershell
python gto.py replay-fixed-card-samples --samples-dir "video_frames\screen_live\card_samples" --layout-profile "video_frames\screen_calibrate\analysis_bbox.json" --output-dir "video_frames\screen_live\card_sample_replay" --format text
```

可选增加 `--old-queue-csv <旧校对队列>`，把完全相同的历史人工标签迁移过去；迁移后仍应在校对界面抽查，不把旧标签当成新模型真值。

## 9. 人工校对接口

### 生成待校对队列

```powershell
python gto.py prepare-card-glyph-label-queue --predictions-csv "video_frames\screen_live\card_sample_replay\glyph_predictions.csv" --output-dir "video_frames\screen_live\glyph_label_queue" --max-rows 10000 --format text
```

如有多个预测 CSV，可重复传入 `--predictions-csv`。队列将把数字/字母与花色拆开，保存原图、二值图、当前预测、人工最终标签和忽略状态。

### 启动网页校对

```powershell
python gto.py serve-card-glyph-label-queue --queue-csv "video_frames\screen_live\glyph_label_queue\glyph_label_queue.csv" --port 8778 --format text
```

浏览器打开：

```text
http://127.0.0.1:8778/
```

校对时：

- 数字/字母项：点击正确的 `A K Q J T 9 ... 2`。
- 花色项：点击正确的黑桃、红桃、方片或草花。
- 头像、动画、牌背或不是牌面：点击“非牌面，整张忽略”。
- “预测正确，直接接受”只在你确认当前预测确实正确时点击。

队列 CSV 是人工真值的唯一保存位置：

```text
video_frames\screen_live\glyph_label_queue\glyph_label_queue.csv
```

其中 `final_label` 是最终人工标签，`ignored` 表示该整张候选不是可训练/可评估的牌面。校对服务器启动失败且提示 `label_id not found` 时，说明浏览器页面仍指向旧队列；关闭旧页面、用当前 CSV 重新启动服务器并刷新即可。

## 10. Python 级接口

供其他本地脚本调用的核心函数在 `gto_cli` 包中：

| 函数 | 文件 | 输入/输出 |
| --- | --- | --- |
| `analyze_video_frame(...)` | `gto_cli/video_vision.py` | 输入一帧 OpenCV 图像和布局/模板，输出单帧视觉识别结果 |
| `build_realtime_state(...)` | `gto_cli/live_vision.py` | 把单帧识别结果规范化为 `current_state.json` 同构状态 |
| `replay_fixed_card_samples(...)` | `gto_cli/card_fixed_replay.py` | 用固定布局批量回放 `card_samples`，输出新的卡牌预测与调试产物 |
| `prepare_card_glyph_label_queue(...)` | `gto_cli/card_glyph_label_queue.py` | 把预测 CSV 转为人工校对队列 |
| `serve_card_glyph_label_queue(...)` | `gto_cli/card_glyph_label_server.py` | 本地启动人工校对网页 |

外部代码推荐优先消费 `current_state.json` 或 `events.jsonl`，不要耦合终端文本或覆盖层像素。

## 11. CV 工作流速查

```text
第一次/窗口比例变化
  画整个窗口外框
  -> 复核自动布局
  -> 必要时圈 H1/H2
  -> 固定 analysis_bbox.json

实盘
  -> screen-cv + --bbox-file + --lock-layout + --show-overlay
  -> current_state.json / events.jsonl
  -> 识别异常时查看 problem_frames / card_debug

离线纠错
  -> replay-fixed-card-samples
  -> prepare-card-glyph-label-queue
  -> serve-card-glyph-label-queue
  -> 人工标注 final_label / ignored
```
