# 2026-08-17 第二批跨机器牌面样本

这些字形来自用户提供的第二批真实错误帧，标签已按原始完整牌面人工确认：

- 公共牌：`Q`、`3`
- Hero 手牌：`6`

训练时以当前已发布模型作为种子，仅追加这里的已确认字形，并保留原模型原型。对应完整牌面裁剪位于
`tests/fixtures/card_regressions/20260817_batch2/`，用于防止以后再次把 `Qh` 读成 `9h`、把 `3h`
读成 `Jh`，或把 `6d` 读成 `Jd`。

重新生成模型的命令：

```powershell
python gto.py train-card-classifier --seed-model "pict/card_models/card_glyph_board_knn.npz" --glyph-dir "pict/card_training_regressions/20260817_batch2/board_rank" --model "pict/card_models/card_glyph_board_knn.npz" --no-templates --augment 2 --format text
python gto.py train-card-classifier --seed-model "pict/card_models/card_glyph_hero_rank_knn.npz" --glyph-dir "pict/card_training_regressions/20260817_batch2/hero_rank" --model "pict/card_models/card_glyph_hero_rank_knn.npz" --no-templates --augment 2 --format text
```
