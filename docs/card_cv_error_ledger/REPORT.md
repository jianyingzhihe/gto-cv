# 扑克 CV 识别错误账本

- 生成时间：`2026-07-31T23:55:14+08:00`
- 数据集：`screen_live sample_20260729_231042`
- 版本闸门：`阻塞`

## 零错误验收闸门

- [ ] 人工校对完成：152/1012 个字形
- [ ] 已校对样本中的错误数：9
- [x] 全帧重放：187/187 个源帧
- [x] 自动化测试：174 passed

> 只有全部字形完成校对、已校对结果零错误、最后一次修复后重新跑完全部帧，
> 并且自动化测试通过，才允许把这一版标记为通过。

## 当前校对状态

| 组件 | 已校对 | 正确 | 错误 | 总数 |
|---|---:|---:|---:|---:|
| 手牌数字/字母 | 52 | 52 | 0 | 332 |
| 手牌花色 | 66 | 65 | 1 | 332 |
| 公共牌数字/字母 | 26 | 22 | 4 | 174 |
| 公共牌花色 | 8 | 4 | 4 | 174 |

## 已校对但仍错误的样本

| ID | 区域 | 当前输出 | 人工真值 | 置信度 / margin | 样本 | 原因 |
|---|---|---:|---:|---|---|---|
| G0004 | board rank | `?` | `9` | `0.000000 / 0.000000` | [sample_20260729_231042_0005_0054p267s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0005_0054p267s/board_slot0_item_rank.png) | `live_rank_low_score_or_margin` |
| G0005 | board suit | `?` | `s` | `0.000000 / 0.000000` | [sample_20260729_231042_0005_0054p267s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0005_0054p267s/board_slot0_item_suit.png) | `live_suit_low_score_or_margin` |
| G0007 | board rank | `?` | `9` | `0.000000 / 0.000000` | [sample_20260729_231042_0007_0057p508s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0007_0057p508s/board_slot0_item_rank.png) | `live_rank_low_score_or_margin` |
| G0008 | board suit | `?` | `s` | `0.000000 / 0.000000` | [sample_20260729_231042_0007_0057p508s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0007_0057p508s/board_slot0_item_suit.png) | `live_suit_low_score_or_margin` |
| G0013 | board rank | `?` | `6` | `0.000000 / 0.000000` | [sample_20260729_231042_0009_0060p664s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0009_0060p664s/board_slot4_item_rank.png) | `live_rank_low_score_or_margin` |
| G0014 | board suit | `?` | `c` | `0.000000 / 0.000000` | [sample_20260729_231042_0009_0060p664s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0009_0060p664s/board_slot4_item_suit.png) | `live_suit_low_score_or_margin` |
| G0017 | board rank | `?` | `6` | `0.000000 / 0.000000` | [sample_20260729_231042_0010_0062p064s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0010_0062p064s/board_slot4_item_rank.png) | `live_rank_low_score_or_margin` |
| G0018 | board suit | `?` | `c` | `0.000000 / 0.000000` | [sample_20260729_231042_0010_0062p064s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0010_0062p064s/board_slot4_item_suit.png) | `live_suit_low_score_or_margin` |
| G0142 | hero suit | `?` | `s` | `0.698900 / 0.064600` | [sample_20260729_231042_0160_1145p720s](../../video_frames/screen_live/card_sample_fixed_replay_20260730/samples/sample_20260729_231042_0160_1145p720s/hero_slot1_6_suit.png) | `live_suit_low_score_or_margin` |

## 错误案例

### CV-001 - 公共牌 7 被识别成 J [已修复]

- 区域：`公共牌数字/字母`
- 样本：`sample_20260729_231042_0038_0315p831s 第 4 张公共牌`
- 人工真值：`7`
- 修复前输出：`J, score 0.5805, margin 0.0783`
- 根因：正确的左上窗口给出 7，但向下偏移的窗口截掉字形顶部后得到了更高的 J 模板分；旧 KNN 也没有覆盖这种公共牌字体。
- 修复：增加固定字体公共牌 rank 模型，只有高置信度的 clean-corner 结果才能覆盖旧多窗口结果。
- 回归证据：目标数字变为 7，score 1.0000、margin 0.1200；当轮已校对公共牌数字从 5/15 提升到 11/15。

**修复前**

![CV-001 修复前](assets/CV-001_before_board_7_as_J.png)

**修复后**

![CV-001 修复后](assets/CV-001_after_board_7.png)

### CV-002 - 第二张手牌 7 被规则抬成 K [已修复]

- 区域：`手牌数字与 ROI 仲裁`
- 样本：`sample_20260729_231042_0050_0411p415s 第 2 张手牌`
- 人工真值：`7`
- 修复前输出：`K, score 0.6815, margin 0.3631`
- 根因：偏移 H2 裁剪削弱了 7 的左上结构并触发 black-K hint，人为增加了 K 分数；两路都缺花色时，旧选择器又无条件偏向偏移框。
- 修复：raw 与 shifted 冲突时，无论牌是否完整，都按真实 rank 置信度选择；只有偏移框证据更强时才保留它。
- 回归证据：raw H2 为 7（0.8055），shifted 为 K（0.6815）；已校对手牌数字 23/23。

**修复前**

![CV-002 修复前](assets/CV-002_before_hero_7_as_K.png)

**修复后**

![CV-002 修复后](assets/CV-002_after_hero_7c.png)

### CV-003 - 手牌草花二值图混入牌边和相邻黑块 [已修复]

- 区域：`手牌花色裁剪与分类`
- 样本：`sample_20260729_231042_0050_0411p415s 第 2 张手牌`
- 人工真值：`c`
- 修复前输出：`unknown, score 0.8222, margin 0.0171`
- 根因：花色 ROI 同时包含牌边、数字尾部和相邻黑色圆块；归一化把组合前景拉伸成类似黑桃的轮廓。
- 修复：使用固定小手牌花色窗口，过滤细边，只保留中心花色连通域；独立手牌花色模型只有超过严格阈值才接管。
- 回归证据：目标变为 7c，花色 score 0.9890、margin 0.1080；已校对手牌花色 33/33。

**修复前**

![CV-003 修复前](assets/CV-003_before_club_binary.png)

**修复后**

![CV-003 修复后](assets/CV-003_after_hero_7c.png)

**修复后字形**

![CV-003 修复后字形](assets/CV-003_after_club_glyph.png)

### CV-004 - 第二张手牌 5 被识别成 J [已修复]

- 区域：`手牌 H2 裁剪仲裁`
- 样本：`sample_20260729_231042_0017_0151p481s 第 2 张手牌`
- 人工真值：`5`
- 修复前输出：`J`
- 根因：偏移 H2 框给出完整但错误的 J，而 raw 固定框保留了真实的 5。
- 修复：当 raw clean-corner 结果明确、shifted 结果不干净时优先 raw。
- 回归证据：固定重放在目标帧识别为 Ah 5h。

**修复前**

![CV-004 修复前](assets/CV-004_before_hero_5_as_J.png)

**修复后**

![CV-004 修复后](assets/CV-004_after_hero_5.png)

### CV-005 - 清晰的手牌 4 被拒绝成未知 [已修复]

- 区域：`手牌 H2 裁剪回退`
- 样本：`sample_20260729_231042_0011_0066p329s 第 2 张手牌`
- 人工真值：`4`
- 修复前输出：`unknown`
- 根因：shifted 结果不完整时，流程没有稳定回退到完整的 raw 固定框。
- 修复：加入 raw-box 回退，并为单字符 H2 rank 增加回归测试。
- 回归证据：固定重放在目标帧识别为 5h 4d。

**修复前**

![CV-005 修复前](assets/CV-005_before_hero_4_unknown.png)

**修复后**

![CV-005 修复后](assets/CV-005_after_hero_4.png)

### CV-006 - 数字 10 与牌边及相邻内容粘连 [已修复]

- 区域：`rank 二值化`
- 样本：`sample_20260729_231042_0035_0284p567s 手牌`
- 人工真值：`T`
- 修复前输出：`K or 8`
- 根因：宽 rank 裁剪保留了牌边和相邻牌边；形态学 closing 又把 1 和 0 合并。
- 修复：增加不使用 closing 的 clean rank 连通域归一化，并允许明确的 clean-corner T 覆盖旧窗口。
- 回归证据：目标双十帧识别为 T，干净字形保持两个连通域。

**修复前**

![CV-006 修复前](assets/CV-006_before_ten_merged.png)

**修复后**

![CV-006 修复后](assets/CV-006_after_ten.png)

### CV-007 - 公共牌在 rank/suit 分类前被丢弃 [未解决]

- 区域：`公共牌可见性门控`
- 样本：`样本 0005、0007、0009、0010`
- 人工真值：`相邻帧中的 9s 和 6c`
- 修复前输出：`unknown with score 0 and margin 0`
- 根因：牌在字形分类前已经被拒绝，因此 rank 和 suit 模型根本没有收到可见牌面裁剪。
- 修复：待修复。先诊断公共牌可见性和槽位接受门控，不能继续盲调 rank/suit 分类器。
- 回归证据：当前队列有 4 条公共牌 rank 和 4 条公共牌 suit 已校对错误，来自这些物理牌。

### CV-008 - 清晰的公共牌 6 被多窗口仲裁识别成 4 [已修复]

- 区域：`公共牌 rank 多窗口裁剪与仲裁`
- 样本：`sample_20260729_231042_0054_0435p227s 第 3 张公共牌（重放后 G0462）`
- 人工真值：`6`
- 修复前输出：`4, score 0.6679, margin 0.0973`
- 根因：整张牌 ROI 正确，原图是清晰的 6s。对齐窗口的模板第一名是 6（0.5234），但 clean 公共牌模型误投 8 且 margin 仅 0.0250，未达到接管阈值；随后向右下偏移的窗口削掉 6 的顶部和左侧，通用 KNN 将残缺轮廓判成 4（0.6679，margin 0.0973），多窗口选择器把这个错误窗口选为最终结果。审查页面展示的是完整字形，不是最终获胜的偏移候选，因此视觉上会显得完全不合理。
- 修复：公共牌 clean-corner 同时评估原始左上角和左侧内缩 6 像素的去边框候选，按 score、margin 选择证据更强的一路；只有高分和足够 margin 的 clean 结果才覆盖多窗口结果。
- 回归证据：第二次全量重放后，目标 6s 的 rank score 为 0.9756、margin 0.1969，原本正确的 2c 保持 score 1.0000、margin 0.4851；187/187 帧、530 张牌和 1060 个字形已重放，人工标签迁移 83 条，完整测试 166 passed。

**修复前**

![CV-008 修复前](assets/CV-008_before_board_6_as_4.png)

**修复后**

![CV-008 修复后](assets/CV-008_after_board_6s.png)

### CV-009 - 无手牌时比熊头像和牌背进入手牌字形队列 [已修复]

- 区域：`手牌调试样本导出门控`
- 样本：`G0021、G0033、G0037、G0040、G0050、G0071、G0084、G0093、G0106 对应的 9 组样本`
- 人工真值：`非牌面，不应进入 rank/suit 校对队列`
- 修复前输出：`hero card ??，rank/suit 均以 item 和 0 分导出，共污染 18 条队列记录`
- 根因：固定 hero ROI 在无手牌或只显示牌背时覆盖到了圆形比熊头像。牌面识别已经返回 ?? 且 rank/suit 置信度均为 0，但调试导出层仍无条件保存 rank 和 suit 字形，导致头像碎片进入人工校对队列。
- 修复：导出字形前过滤 group=hero、card 为未知且 rank/suit 置信度同时为 0 的条目；只过滤 hero 非牌面，公共牌未知样本继续保留用于诊断可见性门控。
- 回归证据：187 帧全量重放后字形队列从 1060 降到 1042，hero unknown item 从 18 条降到 0，仍保留 10 条 board unknown item；105 条人工标签全部迁移，完整测试 167 passed。

**修复前**

![CV-009 修复前](assets/CV-009_before_avatar_as_hero_glyph.png)

### CV-010 - 比熊头像被非零置信度硬猜成 7s [已修复]

- 区域：`手牌牌面几何门控与人工忽略`
- 样本：`sample_20260729_231042_0081_0596p327s 第 2 张手牌（原 G0031）`
- 人工真值：`非牌面，不应进入 rank/suit 校对队列`
- 修复前输出：`7s，rank score 0.5049、margin 0.1394，suit score 0.6870、margin 0.1467`
- 根因：第一版只过滤未知牌且 rank/suit 同为 0 的条目；该头像被字符模型硬猜出了 7s，因此绕过过滤。其牌面矩形度实际异常：face_fill 0.7171、face_cover 0.5675、face_aspect 1.0460。
- 修复：增加低矩形度 hero 非牌面过滤，同时为校对网页增加“非牌面，整张忽略”按钮；点击一次会同时忽略相同 sample/group/slot 的 rank 和 suit，并从校对总数、错误报告和训练输入中排除，忽略状态会在后续全量重放时迁移。
- 回归证据：当前 187 帧中只有该头像命中低矩形度规则；全量重放后队列从 1042 降到 1040，该样本两条记录均消失，105 条人工标签保留。校对服务已重启，页面和 API 均确认支持整张忽略，完整测试 169 passed。

**修复前**

![CV-010 修复前](assets/CV-010_before_avatar_as_7s.png)

### CV-011 - 清晰的手牌 Q 被偏移窗口识别成 8 [已修复]

- 区域：`手牌 rank 多窗口候选过滤`
- 样本：`sample_20260729_231042_0143_1044p558s 第 1 张手牌（回放后 G0066）`
- 人工真值：`Q`
- 修复前输出：`8s，rank score 0.7115、margin 0.0667`
- 根因：手牌 ROI、原始裁剪和二值字形均正确。对齐的 64 像素宽窗口已将完整 Q 分类为 Q，score 0.9213、margin 0.0533；但旧的宽窗口弱候选规则要求 margin 至少 0.055，只差 0.0017 就把这个高分 Q 丢弃。流程随后只能采用右下偏移窗口裁出的残缺轮廓，并误判为 8。
- 修复：宽手牌窗口改为分层门控：rank score 至少 0.90 且 margin 至少 0.04 时保留完整高分候选，不再被统一的 0.055 margin 门限误杀；低分或极低 margin 的宽窗口仍按原规则拒绝。
- 回归证据：目标裁剪直接识别为 Qs，rank score 0.9213、margin 0.0533；187 帧全量重放完成，530 张牌、1040 条字形记录，127 条人工标签与 4 条非牌面忽略全部保留。

**修复前**

![CV-011 修复前](assets/CV-011_before_hero_Q_as_8.png)

**修复后**

![CV-011 修复后](assets/CV-011_after_hero_Qs.png)

### CV-012 - 清晰的红色手牌 3 被纠偏规则改成 8 [已修复]

- 区域：`手牌 rank clean-corner 与红色 3/8 仲裁`
- 样本：`sample_20260729_231042_0180_1338p033s 第 1 张手牌（G0091）`
- 人工真值：`3`
- 修复前输出：`8d，rank score 0.7152、margin 0.1000`
- 根因：手牌 ROI 和二值字形正确，clean-corner 模型已经输出 3，score 0.8733、margin 0.0753；旧 clean 接管门限要求 margin 至少 0.10，因此拒绝了正确的 3。后续红色 3/8 纠偏又采纳右下偏移窗口中的残缺 8 票，将结果改成 8。
- 修复：为固定 WPT 字体的 3 增加专用 clean-corner 接管条件：score 至少 0.86 且 margin 至少 0.07 时优先采用对齐的完整 3，其他牌仍使用原有严格门限。对当前 332 条有效手牌 rank 做反事实检查，只有该样本发生 8→3，45 条已校对手牌 rank 均无回归。
- 回归证据：目标裁剪和 187 帧全量重放均输出 3d，rank score 0.8733、margin 0.0753；530 张牌、1040 条字形记录重放完成，139 条人工标签与 28 条非牌面忽略全部保留。

**修复前**

![CV-012 修复前](assets/CV-012_before_hero_3_as_8.png)

**修复后**

![CV-012 修复后](assets/CV-012_after_hero_3d.png)

### CV-013 - 公共牌草花被截断并识别成黑桃 [已修复]

- 区域：`公共牌花色固定 ROI 与连通域分割`
- 样本：`sample_20260729_231042_0004_0046p301s 第 2 张公共牌（回放后 G0002）`
- 人工真值：`c`
- 修复前输出：`Js，suit score 0.8730、margin 0.2765`
- 根因：旧公共牌花色框直接截取固定矩形 y=56:84、x=6:34。该区域同时包含 J 的下钩，又切掉草花的右叶和下方；归一化把 rank 残片与截断草花组合成近似黑桃的轮廓。完整连通域候选实际输出 c，score 0.8982、margin 0.0123，但旧候选门限未采纳。
- 修复：不增加滑动窗口；在已知公共牌左上角固定区域内进行连通域分割，排除 rank 尾部、牌边和相邻牌的大花色，只将完整的本牌花色连通域交给黑桃/草花分类器。高分完整黑色花色组件优先于截断矩形候选。
- 回归证据：目标公共牌输出 Jc，score 0.8982、margin 0.0123；同帧三张公共牌为 9s Jc 2c。187 帧全量重放后，可见黑色公共牌分为 40 个黑桃与 67 个草花，4 个先于花色分类被可见性门控拒绝的样本仍保持未知；151 条人工标签与 28 条忽略全部保留。

**修复前**

![CV-013 修复前](assets/CV-013_before_board_club_as_spade.png)

**修复后**

![CV-013 修复后](assets/CV-013_after_board_Jc.png)

### CV-014 - 公共牌花色校对页展示了错误候选二值图 [已修复]

- 区域：`公共牌花色调试证据导出`
- 样本：`sample_20260729_231042_0004_0046p301s 第 3 张公共牌 2c`
- 人工真值：`校对页应展示实际分类输入`
- 修复前输出：`二值图包含 rank 残片和截断花色，虽然当前整牌结果为 2c`
- 根因：花色识别已优先使用完整目标连通域，但 safe_suit_debug_image 仍导出 normalized_suit_candidates 的第一个旧固定矩形候选，导致网页展示与实际分类输入不一致。
- 修复：公共牌花色调试图改为直接导出 normalized_suit_component_by_label，与当前 clean board suit 分类输入一致；旧批与今晚批均重新导出并重建合并队列。大花色可稳定裁出，但现有模型仅针对小花色训练，本样本对大草花给出的 s/c 差值仅 0.00050，因此暂不允许大花色覆盖，仅保留为后续独立模型输入。
- 回归证据：两批共 360 帧、2032 条字形重新导出，152 条人工标签和 28 条忽略迁移完成；新增测试要求公共牌花色调试图无 rank 顶部残片，完整测试 174 passed。

**修复前**

![CV-014 修复前](assets/CV-014_before_board_suit_debug_mismatch.png)

**修复后**

![CV-014 修复后](assets/CV-014_after_board_suit_component.png)

## 每轮迭代流程

1. 在校对页面填写真实数字、字母或花色。
2. 在 `cases.json` 新增或更新一个案例，记录例图、错误输出和人工真值。
3. 判断故障阶段：牌桌/槽位定位、裁剪、二值化、分类器、时序稳定或状态门控。
4. 修复时同时加入针对该例的回归测试。
5. 先运行聚焦测试，确认该例修复且相邻案例没有退化。
6. 重新识别全部源帧，并按物理样本和槽位保留已有人工标签。
7. 根据新队列和重放摘要重新生成本报告。
8. 继续人工校对；发现错误后重复以上流程，直到零错误闸门通过。

## 命令

全帧重新识别：

```powershell
python gto.py replay-fixed-card-samples --samples-dir "video_frames\screen_live\card_samples" --sample-prefix "sample_20260729_231042_" --layout-profile "video_frames\screen_live\layout_profile.json" --old-queue-csv "video_frames\screen_live\card_sample_audit_20260729_today\glyph_label_queue.csv" --output-dir "video_frames\screen_live\card_sample_fixed_replay_20260730" --format text
```

更新本报告：

```powershell
python -m gto_cli.card_error_report --ledger "docs\card_cv_error_ledger\cases.json" --queue-csv "video_frames\screen_live\card_sample_fixed_replay_20260730\glyph_label_queue\glyph_label_queue.csv" --replay-summary "video_frames\screen_live\card_sample_fixed_replay_20260730\fixed_replay_summary.json" --output "docs\card_cv_error_ledger\REPORT.md"
```

人工校对页面：http://127.0.0.1:8767/
