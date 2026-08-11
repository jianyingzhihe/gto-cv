# 实时牌桌 CV + GTO 操作手册

更新时间：2026-08-11

这份手册对应当前 `E:\dezhou` 最新版本。默认终端使用精简 advice
模式；完整牌桌数据继续写入覆盖层、`current_state.json` 和
`events.jsonl`。

## 0. 小白版：第一次只执行这三步

先让师兄把腾讯会议共享画面中的完整牌桌显示出来，不要让聊天窗口、会议工具栏
或 PowerShell 挡住牌桌。

打开 PowerShell：

```powershell
cd E:\dezhou
```

### 第一步：手动拖一个包含完整牌桌的大框

```powershell
python gto.py screen-cv --pick-bbox --hero-name "鱼寻欢" --output-dir "video_frames\screen_calibrate"
```

操作方法：

1. 鼠标从牌桌窗口左上角拖到右下角。
2. 框可以稍微大一点，但必须包含完整牌桌、手牌和底部操作按钮。
3. 拖完后按 Enter 或 Space。
4. PowerShell 出现 `Saved bbox` 就表示第一步完成。

`--hero-name` 后面的名字要和牌桌底部自己的名字完全一致。如果实际显示的是
“于寻欢”而不是“鱼寻欢”，就把命令中的名字改成实际文字。

### 第二步：检查程序自动找到的精确牌桌框

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_review_auto_bbox_command.txt")
```

程序会在第一步的大框里面寻找牌桌，并显示一个青色框：

- 青色框正确包住完整牌桌：直接按 Enter 或 Space。
- 青色框不准：按 `R`，重新拖动精确牌桌框，再按 Enter 或 Space。
- 想取消：按 `C` 或 Esc。

接受后会保存 `analysis_bbox.json`。这个文件只描述内部牌桌；第一步手工拖出的
`bbox.json` 仍是完整窗口和底部操作区的唯一坐标来源。内部牌桌框会在本次布局中固定，不会继续自动缩小或漂移。

### 第三步：先做一帧预检，再启动全部覆盖层和实时建议

先运行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_preflight_command.txt")
```

确认覆盖层中的 H1/H2 落在两张真实手牌上，且输出中的 `hero.cards` 有两张完整牌、`ocr_mode` 不是 `disabled`。若不满足，先按第 3 节重新校准手牌；不要直接启动实时建议。

预检通过后运行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_overlay_command.txt")
```

这一个命令同时完成：

- 连续截取腾讯会议中的牌桌。
- 显示当前系统的完整覆盖层。
- 显示牌桌框、`H1/H2` 手牌框、`B1..B5` 公共牌框和庄家 `D`。
- 实时识别手牌、公共牌、底池、下注和轮到谁。
- 轮到自己时在 PowerShell 打印 `ADVICE`。

看到覆盖层后不需要再运行另一个 live 命令。按 `Ctrl+C` 停止。

如果只有 `H1/H2` 手牌框不准，再执行本手册第 3 节的手牌框校准。其他情况下，第一次使用到这里就结束了。

## 1. 每次开始

打开 PowerShell：

```powershell
cd E:\dezhou
```

如果已经完成二次定位，且窗口位置、大小和比例没有变化，优先运行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_live_command.txt")
```

需要同时查看识别框时运行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_overlay_command.txt")
```

如果尚未完成二次定位，才使用 `run_live_command.txt` 或
`run_overlay_diagnostic_command.txt`。如果已经手动框过 H1/H2，会额外生成
`run_live_overlay_command.txt`，它的优先级最高。

按 `Ctrl+C` 停止。

## 2. 新窗口或位置变化时重新校准

先框一个包含完整牌桌窗口的大框：

```powershell
python gto.py screen-cv --pick-bbox --hero-name "鱼寻欢" --output-dir "video_frames\screen_calibrate"
```

鼠标拖框后按 Enter 或 Space。然后运行自动二次定位：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_review_auto_bbox_command.txt")
```

在青色框检查窗口中：

- Enter 或 Space：接受当前牌桌框。
- `R`：当前框不准，重新拖动。
- `C` 或 Esc：取消。

接受后会生成固定的 `analysis_bbox.json` 和 `run_reviewed_*` 运行命令。本次布局不会再被
自动 bbox 缩小或漂移。

## 3. 手牌框仍不准时

先运行固定牌桌框的覆盖层：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_overlay_command.txt")
```

覆盖层中的 `H1/H2` 应分别落在两张手牌自己的可读牌面上。H1 包含左牌的点数和花色；H2 从右牌自己的点数角开始，不能包含与左牌重叠的区域、头像或昵称。若只有这里不准，运行：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_pick_hero_cards_command.txt")
```

依次拖动左牌 `H1`、右牌 `H2`，每次按 Enter 或 Space。完成后使用新生成的：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_live_overlay_command.txt")
```

手牌框按牌桌比例保存。同一比例只发生整体缩放时通常不需要重画；换电脑、牌桌 UI 比例或布局明显变化时必须重新执行本节。

## 3.1 换电脑时的必做检查

换电脑后，屏幕抓取、庄家、桌面和座位都正常，并不代表手牌框仍然可用。不同的显示缩放、腾讯会议窗口大小或窗口边距会让旧手牌相对位置失效。不要跳过第二步后直接开实时流。

完整顺序是：第一步框完整窗口，第二步接受或重画内部牌桌框，执行第 4 节的一帧预检；如果 H1/H2 没有落在两张真实手牌上，再执行本节的手牌框校准。

出现以下任一项时，表示程序未找到真实手牌而回退到了不适用的固定位置：`fallback_fixed_roi`、`hero.cards: []`、`hero_cards_incomplete`。这不是屏幕抓取失败。按 `Ctrl+C` 停止，重新画 H1/H2，然后执行新生成的 `run_live_overlay_command.txt`。

随后出现的 `hero_turn_not_confirmed` 是安全保护：两张手牌不完整时，程序拒绝给出建议。先修复手牌框，不要先处理行动判断。

若状态里出现 `ocr_mode: disabled`，则当前电脑的 Python 少了文字识别组件。执行：

```powershell
python -m pip install -r requirements.txt
python -c "from rapidocr_onnxruntime import RapidOCR; print('OCR OK')"
```

重新启动预检。未恢复前，底池、跟注额和按钮文字不应被当作可靠结果。

## 4. 正式运行前预检

完成二次定位后，先只分析一帧：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_preflight_command.txt")
```

重点检查：

- `hero.cards` 是两张完整手牌。
- `ocr_mode` 不是 `disabled`。
- `table.dealer_seat` 正确。
- `table.pot_bb` 和可见下注合理。
- `hero_turn.is_turn` 只有出现操作按钮时为 `true`。
- 覆盖图的 `H1/H2`、`B1..B5` 没有 `CLIPPED`。

预检文件默认在 `video_frames\screen_preflight`。

## 5. 正式实时识别

推荐运行固定二次定位后的命令：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_reviewed_live_command.txt")
```

如果执行过手动 H1/H2 定位，则改用：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_live_overlay_command.txt")
```

默认终端是精简 advice 模式：

```text
[0012.345s #7] ADVICE | 3BET 8.5 BB (3bet 80% / call 20%) | hero=CO AhKd | preflop pot=3.4BB call=2BB board=- | 418.2ms
```

没有轮到自己时只显示等待原因和必要上下文：

```text
[0022.100s #15] WATCH | wait=hero_action_controls_not_visible | hero=CO AhKd | preflop pot=9BB call=0BB board=- | 310.4ms
```

状态不变时不会逐帧重复刷屏，默认每 10 秒打印一次心跳。建议、手牌、公共牌、
街道或轮到自己发生变化时会立即打印。

临时恢复旧版完整逐帧输出：

```powershell
python gto.py screen-cv --bbox-file "video_frames\screen_calibrate\analysis_bbox.json" --lock-layout --trigger frame --every 1 --with-advice --console-mode full --output-dir "video_frames\screen_live" --format text
```

修改精简模式心跳间隔：

```powershell
--console-heartbeat 20
```

设为 `0` 表示状态不变时完全不打印重复行：

```powershell
--console-heartbeat 0
```

## 6. 如何看 advice

`ADVICE` 表示当前识别条件完整，并且确认轮到自己。

例如：

```text
bet 33% pot 3.3BB 21.6% / bet 66% pot 6.6BB 36% / check 28%
```

含义是混合策略：

- 21.6% 的频率下注 3.3BB。
- 36% 的频率下注 6.6BB。
- 28% 的频率过牌。
- 其余频率可能在其他动作中，具体以完整 advice 文本为准。

这不是“固定下注一个数”。若只选最高频动作，这个例子是下注 6.6BB；严格执行
混合策略时应按概率随机选择。

常见等待原因：

- `hero_action_controls_not_visible`：还没轮到自己。
- `hero_cards_incomplete`：手牌不完整。
- `board_cards_incomplete`：公共牌不完整。
- `hero_cards_confirmation_pending`：新手牌仍在多帧确认。

等待状态下不要根据残缺识别结果行动。

## 7. 完整数据在哪里

终端精简不影响数据保存：

- `video_frames\screen_live\current_state.json`：最新完整状态。
- `video_frames\screen_live\events.jsonl`：逐事件完整记录。
- `video_frames\screen_live\latest_overlay.png`：最新覆盖图。
- `video_frames\screen_live\problem_frames`：自动保存的问题帧。
- `video_frames\screen_live\card_debug`：问题帧的手牌、rank、suit 裁剪。
- `video_frames\screen_live\card_samples`：自动采样的训练数据。

## 8. 数字或花色识别错了

实时识别会在牌面预测变化时立即保存一次。预测不变时默认每 30 秒补存一次，
所以不会因为逐帧写图片拖慢实时识别。保存内容包括整帧、覆盖图、整张牌、
数字/字母裁剪、花色裁剪、模型结果和置信度。

停止实时程序后生成全量标注队列：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_prepare_card_sample_labels_command.txt")
```

打开浏览器校对页：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_serve_card_sample_labels_command.txt")
```

页面每次显示一条，左边是当时的完整覆盖图，右边是整张牌和放大的字形：

- 预测正确：点“预测正确，直接接受”，或按 `Enter`。
- 预测错误：直接点正确的数字、字母或花色。
- 上一条/下一条：点击按钮，或按左右方向键。
- 数字/字母快捷键：`A K Q J T 9 ... 2`，其中 `0` 也表示 `T`。
- 花色快捷键：`S` 黑桃、`H` 红桃、`D` 方片、`C` 草花。
- 每次点击后立即写入 `glyph_label_queue.csv`，不需要另存。

全部校对完成后，在运行页面的 PowerShell 窗口按 `Ctrl+C` 停止，再应用标注：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_apply_card_sample_labels_command.txt")
```

本次 2026-07-29 四张截图对应的 66 条优先样本可以先单独校对：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_serve_card_sample_audit_screenshots_20260729_command.txt")
```

完成后应用这批标注：

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_apply_card_sample_audit_screenshots_20260729_command.txt")
```

不要直接把自动预测全部当成真值训练。未人工点击确认的行保持空白，不会进入
已标注数据。

## 9. 快速排错

终端一直 `hero_cards_incomplete`：

1. 打开 `latest_overlay.png`。
2. 检查 `H1/H2` 是否完整覆盖牌面。
3. 若框偏了，或看到 `fallback_fixed_roi` / `hero.cards: []`，重新执行第 3 节。
4. 框正确但字符错，检查最新 `card_debug\...\*_rank.png` 和
   `*_suit.png`，这是分类器问题。

终端一直 `hero_action_controls_not_visible`：

1. 确认底部红色操作按钮确实出现。
2. 检查牌桌底部是否被腾讯会议工具栏或其他窗口遮挡。
3. 查看覆盖层是否完整包含操作按钮区域。

终端出现 `ocr_mode: disabled`：

1. 在项目目录执行 `python -m pip install -r requirements.txt`。
2. 执行 `python -c "from rapidocr_onnxruntime import RapidOCR; print('OCR OK')"`。
3. 重启 `screen-cv`，再跑一次第 4 节预检。

模型更新后仍显示旧结果：

1. 按 `Ctrl+C` 停止旧进程。
2. 重新启动 `screen-cv`。

模型在进程内有缓存，不重启不会加载新模型。
