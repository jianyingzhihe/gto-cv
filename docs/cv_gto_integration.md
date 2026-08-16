# CV to GTO Advice Integration

This project can now attach a GTO-style recommendation to live CV states.

## Runtime Command

### Tencent Meeting / Live Screen

Run the promoted-config health check before a live session:

```powershell
python gto.py cv-health `
  --bbox "136,123,1534,1058" `
  --hero-name "于寻欢" `
  --output-dir "video_frames\cv_health_promoted" `
  --fail-on-not-ready `
  --format text
```

If you have already run the draggable selector, prefer the saved bbox file so
you do not have to copy coordinates by hand:

```powershell
python gto.py cv-health `
  --bbox-file "video_frames\screen_calibrate\bbox.json" `
  --output-dir "video_frames\cv_health_promoted" `
  --fail-on-not-ready `
  --format text
```

Equivalent shortcut:

```powershell
python gto.py cv-health --latest-bbox --output-dir "video_frames\cv_health_promoted" --fail-on-not-ready --format text
```

When `--bbox-file` is used, the generated `run_preflight_command.txt`,
`run_live_command.txt`, and `run_fast_live_command.txt` keep the same
`--bbox-file` argument. This is the recommended low-friction live path because
the command survives process restarts without recopying coordinates.

The command verifies the promoted KNN model, latest `validate-cv --all`
summary, and model promotion gate summary. Deep fallback files are checked only
when `--deep-card-model-dir` is passed explicitly. It writes:

- `video_frames\cv_health_promoted\cv_health_summary.json`
- `video_frames\cv_health_promoted\cv_health_report.md`
- `video_frames\cv_health_promoted\run_preflight_command.txt`
- `video_frames\cv_health_promoted\run_live_command.txt`
- `video_frames\cv_health_promoted\run_fast_live_command.txt`

Only use the live command when `cv-health` prints `Decision: READY`.
The current promoted default should show `pict\card_models\card_glyph_knn.npz`,
`real_problem=0`, `board_bad=0`, median about `29.9 ms`, p90 about
`48.0 ms`, and a promoted gate summary.
The generated `run_preflight_command.txt`, `run_live_command.txt`, and
`run_fast_live_command.txt` include the current recommended OCR,
dealer-refresh, villain-profile, model, and layout-locking options. The default
live commands intentionally omit `--deep-card-model-dir` for lower latency.
Use `run_live_command.txt` for the fullest pot/bet information stream. Use
`run_fast_live_command.txt` when you mainly need low-delay advice when action
buttons appear; it adds `--ocr-action-only`.
Replace `136,123,1534,1058` with the current table rectangle from
`screen-cv --pick-bbox`. Placeholder values such as `x,y,w,h` fail the health
check unless `--allow-placeholder-bbox` is set for template generation.

No extra recorder is required for Tencent Meeting. Keep the poker table visible
on screen. If the meeting view contains a full shared desktop, let the tool find
the poker window first:

```powershell
python gto.py screen-cv `
  --auto-bbox `
  --auto-bbox-refresh 10 `
  --lock-layout `
  --hero-name "于寻欢" `
  --output-dir "video_frames\screen_live" `
  --trigger frame `
  --every 1 `
  --with-advice `
  --effective-stack 100 `
  --min-confidence 0.35 `
  --ocr-scale 0.65 `
  --dealer-refresh-frames 4 `
  --format text
```

If automatic detection cannot find the table, open the draggable calibration
selector:

```powershell
python gto.py screen-cv --pick-bbox --hero-name "于寻欢" --output-dir "video_frames\screen_calibrate"
```

### Two-Step Full-Window Calibration

For a new Tencent Meeting layout, select one generous full poker window. It must
contain the hero cards and bottom action buttons. The selector writes the live
overlay command directly:

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_live_overlay_command.txt")
```

There is no mandatory second review window. `bbox.json` is the only manual
full-window coordinate source. The live command automatically projects the
existing stable inner-table/card layout into the new full window, or locates the
inner table within the selected region when no compatible layout exists.

底部操作按钮有意使用另一张输入图：每个实时画面先截取完整扑克窗口，
再裁出已复核的内部牌桌用于识别手牌、公共牌和牌桌信息。
`analysis_bbox.json` 同时保存两种区域：顶层坐标是内部牌桌，
`outer_region` 是包含底部按钮的完整扑克窗口。不能把内部牌桌裁图直接
作为按钮识别输入；否则会切掉底部一行按钮，即使屏幕上显示了弃牌、过牌、
跟注或加注，程序也会错误报告“未看到我方操作按钮”。

If the whole-table bbox is correct but only `H1/H2` are still misplaced, run
the optional hero-card picker:

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_calibrate\run_pick_hero_cards_command.txt")
```

Drag a generous outer region containing the complete poker window, then press
Enter or Space. The command prints
and saves the selected `x,y,width,height` box plus two follow-up commands:

- `video_frames\screen_calibrate\run_health_command.txt`
- `video_frames\screen_calibrate\run_preflight_command.txt`
- `video_frames\screen_calibrate\run_live_command.txt`
- `video_frames\screen_calibrate\run_fast_live_command.txt`
- `video_frames\screen_calibrate\run_live_overlay_command.txt`
- `video_frames\screen_calibrate\run_pick_hero_cards_command.txt`

The health and preflight commands remain optional diagnostics. Normal use runs
`run_live_overlay_command.txt` immediately after selecting the full window.

### Live ROI Overlay And Manual Hero Cards

When live recognition disagrees with what is visibly on screen, start the same
capture with `--show-overlay`. It creates a transparent, topmost, click-through
overlay on the selected screen region and continuously writes an inspectable
copy to `video_frames\screen_live\latest_overlay.png`:

```powershell
python gto.py screen-cv `
  --bbox-file "video_frames\screen_calibrate\bbox.json" `
  --show-overlay `
  --lock-layout `
  --output-dir "video_frames\screen_live" `
  --trigger frame `
  --every 1 `
  --with-advice `
  --effective-stack 100 `
  --min-confidence 0.35 `
  --ocr-scale 0.65 `
  --format text
```

The cyan rectangle is the analyzed table. `H1/H2` and `B1..B5` are the exact
card crops used by recognition. A red label with `CLIPPED` means that crop hits
an analyzed-frame border, so the table bbox is truncating the card. A clean
crop whose visible glyph disagrees with the prediction points to the classifier
instead of localization.

To override automatic hero-card localization, keep the cards visible and run:

```powershell
python gto.py screen-cv `
  --bbox-file "video_frames\screen_calibrate\bbox.json" `
  --pick-hero-cards `
  --output-dir "video_frames\screen_calibrate"
```

Select the full visible white face of the left card (`H1`) and then the right
card (`H2`). Press Enter or Space after each selection. The command saves:

- `video_frames\screen_calibrate\hero_card_rois.json`
- `video_frames\screen_calibrate\hero_card_rois_preview.png`
- `video_frames\screen_calibrate\run_live_overlay_command.txt`

Run `run_live_overlay_command.txt` for the calibrated session. Manual card
boxes are normalized, so the same table aspect ratio may be scaled without
reselecting them. If the table layout or aspect ratio changes materially,
repeat `--pick-hero-cards`.

```powershell
python gto.py screen-cv `
  --bbox "x,y,width,height" `
  --auto-bbox `
  --auto-bbox-refresh 10 `
  --lock-layout `
  --hero-name "于寻欢" `
  --output-dir "video_frames\screen_live" `
  --trigger frame `
  --every 1 `
  --with-advice `
  --effective-stack 100 `
  --min-confidence 0.35 `
  --ocr-scale 0.65 `
  --dealer-refresh-frames 4 `
  --villain standard `
  --format text
```

You can also use the saved bbox file directly for live screen capture:

```powershell
python gto.py screen-cv `
  --bbox-file "video_frames\screen_calibrate\bbox.json" `
  --auto-bbox `
  --auto-bbox-refresh 10 `
  --lock-layout `
  --output-dir "video_frames\screen_live" `
  --trigger frame `
  --every 1 `
  --with-advice `
  --effective-stack 100 `
  --min-confidence 0.35 `
  --ocr-scale 0.65 `
  --dealer-refresh-frames 4 `
  --format text
```

The saved fast live command has the same shape plus `--ocr-action-only` and
uses `video_frames\screen_live_fast` as its output directory.

Use `--monitor 2` if the meeting window is on another display. Stop the command
with `Ctrl+C`. The latest state and recommendation are continuously written to
`current_state.json`, while the event stream is appended to `events.jsonl`.

### Automatic Card Samples For Manual Correction

Live `screen-cv` now records deduplicated hero and board card observations by
default. A changed prediction is saved immediately; an unchanged prediction is
sampled again every 30 seconds. Each package contains the table frame, overlay,
whole-card crop, rank crop, suit crop, prediction, confidence, and margin:

- `video_frames\screen_live\card_samples\sample_*`
- `video_frames\screen_live\card_samples\glyph_predictions.csv`

After stopping the live process, build a correction queue. Current predictions
are prefilled, so only incorrect rank or suit labels need to be changed:

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_prepare_card_sample_labels_command.txt")
```

Review `card_sample_label_queue\glyph_label_queue_sheet.jpg`, then edit
`card_sample_label_queue\glyph_label_queue.csv`. Put the corrected rank
(`A K Q J T 9 ... 2`) or suit (`s h d c`) in `final_label`. Apply the completed
labels with:

```powershell
Invoke-Expression (Get-Content -Raw "video_frames\screen_live\run_apply_card_sample_labels_command.txt")
```

Use `--no-card-samples` to disable recording, or
`--card-sample-interval 10` to keep more repeated examples.

When a live frame is saved because of a card-recognition problem, the screen
runner also writes a focused card debug package:

- `problem_frames\*.png`: the full table frame.
- `card_debug\<event>\frame.png`: the same frame for the debug package.
- `card_debug\<event>\screen_context.png`: the wider selected capture with the
  inner analysis bbox drawn on top (saved when auto-bbox crops inside it).
- `card_debug\<event>\diagnostic_overlay.png`: the exact H1/H2/board boxes,
  predictions, confidence, and any `CLIPPED` warning for that failed frame.
- `card_debug\<event>\metadata.json`: hero/board cards, rank/suit confidence,
  ROI boxes, and links to every exported crop.
- `card_debug\<event>\*_card.png`: the card crop used by the recognizer.
- `card_debug\<event>\*_rank.png`: normalized rank glyph, ready for labeling
  or teacher/probe inspection.
- `card_debug\<event>\*_suit.png`: normalized suit glyph, ready for labeling
  or teacher/probe inspection.

This is most useful when the console prints
`advice=wait(hero_cards_incomplete)`: open the latest
`video_frames\screen_live\card_debug\...\metadata.json` and inspect whether the
failure is the card crop, the rank glyph, or the suit glyph.

To turn all saved live card-debug samples into a normal review package:

```powershell
python gto.py collect-card-debug-review --input-dir "video_frames\screen_live\card_debug" --output-dir "video_frames\screen_live_card_debug_review" --prepare-label-queue --prepare-glyph-label-queue --format text
```

This writes both queue types:

- `label_queue\label_queue.csv`: whole-card review, fill `final_card0` /
  `final_card1`.
- `glyph_label_queue\glyph_label_queue.csv`: split rank/suit review, fill one
  `final_label` per crop.

If you skipped `--prepare-label-queue`, build the browser labeling queue from
that review CSV:

```powershell
python gto.py prepare-card-label-queue --review-csv "video_frames\screen_live_card_debug_review\review.csv" --output-dir "video_frames\screen_live_card_debug_review\label_queue" --max-rows 80 --format text
python gto.py serve-card-label-queue --queue-csv "video_frames\screen_live_card_debug_review\label_queue\label_queue.csv" --open-browser
```

If the issue is only a bad digit/rank or a bad suit, use the split glyph queue:

```powershell
python gto.py prepare-card-glyph-label-queue --review-csv "video_frames\screen_live_card_debug_review\review.csv" --output-dir "video_frames\screen_live_card_debug_review\glyph_label_queue" --max-rows 160 --format text
python gto.py apply-card-glyph-label-queue --queue-csv "video_frames\screen_live_card_debug_review\glyph_label_queue\glyph_label_queue.csv" --output-dir "video_frames\screen_live_card_debug_review\glyph_label_applied" --format text
```

The queue command also writes `label_queue_sheet.jpg`, a compact image with
the table frame, full card crop, rank crop, and suit crop per row. Use it first
when checking whether an error is caused by card cropping, rank recognition, or
suit recognition.

After filling `final_card0/final_card1`, use the command written in
`video_frames\screen_live_card_debug_review\runbook.md` to audit the queue and
run:

```powershell
python gto.py retrain-card-label-queue --queue-csv "video_frames\screen_live_card_debug_review\label_queue\label_queue.csv" --output-dir "video_frames\screen_live_card_debug_review\label_retrain" --format text
```

That command applies the labels, trains a candidate KNN model, exports a
candidate review set, validates all recorded videos, and runs the promotion
gate. Do not promote unless the gate says `promote`.

If the selector window cannot open, save a plain screenshot instead:

```powershell
python gto.py screen-cv --snapshot-only --output-dir "video_frames\screen_calibrate"
```

Then open `video_frames\screen_calibrate\event_frames\screen_snapshot.png` and
choose the same rectangle manually.

### Auto-BBox Diagnostics

To test whether automatic table localization is robust on recorded validation
videos, run the light diagnostic. It stress-tests native, loose-border,
shifted-border, and tight-crop capture situations:

```powershell
python gto.py diagnose-auto-bbox --all --video-dir "video_frames" --output-dir "video_frames\auto_bbox_diagnostics_20260709_full" --every 300 --max-frames 2 --min-confidence 0.35 --format text
```

Result:

```text
Videos: 8
Rows: 45
Failures: 0
IoU median/p90/min: 1.0 / 1.0 / 1.0
Timing median/p90/max: 431.6 ms / 721.1 ms / 948.7 ms
Methods: visual-titlebar-row=22, action-buttons=14, dealer-button-anchor=7, current-region-table=2
```

Report: `video_frames\auto_bbox_diagnostics_20260709_full\auto_bbox_diagnostics_report.md`.
It writes:

- `auto_bbox_diagnostics_summary.json`
- `auto_bbox_diagnostics_rows.csv`
- `auto_bbox_diagnostics_report.md`
- `problem_frames\*.png` for hard failures

### Recorded Video

```powershell
python gto.py live-cv "C:\path\to\table.mp4" `
  --output-dir "video_frames\live_advice" `
  --trigger frame `
  --every 1 `
  --dealer-refresh-frames 30 `
  --with-advice `
  --effective-stack 100 `
  --villain standard `
  --format text
```

The same output is written to:

- `events.jsonl`
- `events.json`
- `current_state.json`

## Advice Gate

Advice is emitted only when all conditions are true:

- Hero action controls are visible at the bottom of the table.
- Hero has exactly two complete cards, with no unknown rank/suit marker.
- Board cards are complete if the hand is postflop.
- The CV state is otherwise valid.

If any condition fails, `gto_advice.ready` is `false` and `reason` explains why.

## Output Fields

When ready:

```json
{
  "gto_advice": {
    "ready": true,
    "should_act": true,
    "action": "3bet",
    "amount_bb": 8.5,
    "target_bet_bb": 8.5,
    "scenario": "vs_open",
    "summary": "3BET 8.5 BB  (3bet 80% / call 20% / fold 0%)"
  }
}
```

When not ready:

```json
{
  "gto_advice": {
    "ready": false,
    "should_act": false,
    "reason": "hero_cards_incomplete"
  }
}
```

## Current Limitations

- The strategy engine is the existing local heuristic advisor, not a commercial solver.
- Effective stack defaults to `100BB` unless passed with `--effective-stack`.
- If the mouse or animation blocks a card, advice is intentionally withheld instead of guessing.
