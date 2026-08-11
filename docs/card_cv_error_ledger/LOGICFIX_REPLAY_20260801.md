# 2026-08-01 Logic-Fix Replay Comparison

## Scope

- Baseline queue: `video_frames/screen_live/card_sample_combined_replay_20260731/glyph_label_queue/glyph_label_queue.csv`
- Replay queue: `video_frames/screen_live/card_sample_combined_replay_20260801_logicfix/glyph_label_queue/glyph_label_queue.csv`
- Shared manually labeled glyphs (ignored rows excluded): **245**
- Evaluation uses labels entered before this replay; no new predictions were accepted as training labels.

## Glyph Accuracy

| Kind | Baseline | Logic-fix replay | Delta |
|---|---:|---:|---:|
| rank | 78/87 (89.7%) | 83/87 (95.4%) | +5 |
| suit | 134/158 (84.8%) | 148/158 (93.7%) | +14 |

## Complete-Card Accuracy

Only cards with both rank and suit manually labeled are counted: **37 cards**.

| Baseline | Logic-fix replay | Delta |
|---:|---:|---:|
| 33/37 (89.2%) | 33/37 (89.2%) | +0 |

## Changed Labeled Glyphs

- Changed predictions: **19**
- Improved: **19**
- Regressed: **0**
- Other changed: **0**

| Sample | Position | Kind | Label | Before | After | Result |
|---|---|---|---|---|---|---|
| sample_20260731_223845_0003_0038p492s | board slot 0 | rank | 7 | 9 | 7 | improved |
| sample_20260731_223845_0003_0038p492s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0004_0050p513s | board slot 0 | rank | 7 | 9 | 7 | improved |
| sample_20260731_223845_0004_0050p513s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0005_0064p543s | board slot 0 | rank | 7 | 9 | 7 | improved |
| sample_20260731_223845_0005_0064p543s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0006_0066p227s | board slot 0 | rank | 7 | 9 | 7 | improved |
| sample_20260731_223845_0006_0066p227s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0007_0081p005s | board slot 0 | rank | 7 | 9 | 7 | improved |
| sample_20260731_223845_0007_0081p005s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0019_0170p105s | board slot 0 | suit | c | s | c | improved |
| sample_20260731_223845_0019_0170p105s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0020_0183p233s | board slot 0 | suit | c | s | c | improved |
| sample_20260731_223845_0020_0183p233s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0021_0195p739s | board slot 0 | suit | c | s | c | improved |
| sample_20260731_223845_0021_0195p739s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0021_0195p739s | board slot 4 | suit | c | s | c | improved |
| sample_20260731_223845_0022_0204p154s | board slot 2 | suit | c | s | c | improved |
| sample_20260731_223845_0022_0204p154s | board slot 4 | suit | c | s | c | improved |

## Notes

- The replay contains the rank clean-corner selection fix and the public-card black-suit clean-component fix.
- Hero-card suit thresholds were intentionally not relaxed: on the currently labeled set, that change would create regressions. Those samples remain in the queue for continued labeling and later model retraining.
