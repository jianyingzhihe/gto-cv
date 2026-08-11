# Card CV Model Pipeline

This project uses a two-stage card recognition pipeline:

1. Lock the poker table layout once per live session.
2. Recognize the already-cropped card corner glyphs as separate rank and suit tasks.

The live path stays conservative. Template/kNN recognition remains the default because it is fast and already stable on WPT UI. Deep models are optional low-confidence fallbacks.

## Preflight Health

Before a live Tencent Meeting session, check the promoted runtime configuration:

```powershell
python gto.py cv-health --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\cv_health_promoted" --fail-on-not-ready --format text
```

Current promoted evidence should report:

```text
Decision: READY
KNN: pict\card_models\card_glyph_knn.npz
Deep: disabled
Validation: real_problem=0, board_bad=0, median=29.9 ms, p90=48.0 ms
Card health: hero_complete=123 hero_incomplete_or_missed=0 hero_turn_blocked=0 board_bad=0
Card issues: {}
```

The command writes `run_preflight_command.txt` and `run_live_command.txt` in
the output directory. Run the preflight command first after replacing the bbox
with the current table rectangle. It captures exactly one frame, writes
`current_state.json`, and saves raw/annotated frames for inspection. If the
single-frame state looks sane, run the live command. Both commands include the
recommended OCR scale, dealer refresh, villain profile, model, and
layout-locking options.
Placeholder bbox values such as `x,y,w,h` are rejected unless
`--allow-placeholder-bbox` is used for template generation.

`validate-cv` and `cv-health` now expose a dedicated `card_health` block in the
JSON summaries. It splits card failures into hero visible-card misses,
incomplete hero cards, unknown rank/suit, blocked hero-turn frames, board
unknowns, and duplicate cards. A clean live-ready run should have
`hero_incomplete_or_missed=0`, `hero_turn_blocked=0`, `board_bad=0`, and an
empty `Card issues` map.

Current card-health smoke:

```powershell
python gto.py validate-cv --all --video-dir "video_frames" --output-dir "video_frames\cv_validation_card_health_smoke_20260709" --every 300 --max-frames 2 --min-confidence 0.35 --dealer-refresh-frames 4 --format text
python gto.py cv-health --bbox-file "video_frames\screen_calibrate\bbox.json" --validation-summary-json "video_frames\cv_validation_card_health_smoke_20260709\cv_validation_all_summary.json" --output-dir "video_frames\cv_health_card_health_20260709" --format text
```

```text
Hero card health: visible=9 complete=9 incomplete_or_missed=0 turn_blocked=0
Board card health: frames=1 bad=0 duplicates=0
Card issues: {}
```

Latest current-goal health output:

```text
video_frames\cv_health_current_goal\run_preflight_command.txt
video_frames\cv_health_current_goal\run_live_command.txt
video_frames\cv_health_current_goal\run_fast_live_command.txt
```

Use `run_live_command.txt` when you want the right-side information stream to
keep pot and visible bets as complete as possible. Use
`run_fast_live_command.txt` when the priority is lower delay around hero action;
it adds `--ocr-action-only`, so OCR is skipped until bottom action buttons are
visible.

## Live Layout

Pick the Tencent Meeting / poker window region once:

```powershell
python gto.py screen-cv --pick-bbox --hero-name "于寻欢" --output-dir "video_frames\screen_calibrate"
```

The picker saves:

```text
video_frames\screen_calibrate\bbox.json
video_frames\screen_calibrate\run_health_command.txt
video_frames\screen_calibrate\run_preflight_command.txt
video_frames\screen_calibrate\run_live_command.txt
```

Run `run_health_command.txt` first and only continue when it prints
`Decision: READY`. Then run `run_preflight_command.txt` once and inspect the
saved state/images before starting `run_live_command.txt`.

Run live recognition with one-time layout locking. Use this low-latency command
for normal play:

```powershell
python gto.py screen-cv --bbox "x,y,w,h" --auto-bbox --lock-layout --hero-name "于寻欢" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

The older deep fallback command below is kept as a historical optional-fallback
example, not the default live command:

```powershell
python gto.py screen-cv --bbox "x,y,w,h" --auto-bbox --lock-layout --hero-name "于寻欢" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

The locked layout profile is saved as:

```text
video_frames\screen_live\layout_profile.json
```

The promoted card recognizer is the default
`pict\card_models\card_glyph_knn.npz`, so live commands do not need
`--card-knn-model`. Normal live commands also omit `--deep-card-model-dir`;
deep models are reserved for offline review/teacher runs unless a later gate
shows a latency-safe live benefit.

Current automatic bbox diagnostics across all 8 validation videos and 5 bbox
variants per video:

```text
rows=40
failures=0
methods=visual-titlebar-row, dealer-button-anchor, action-buttons, current-region-table
timing median=1318.0 ms
```

This means auto bbox is suitable for startup and periodic correction. It should
not be run every frame; keep `--lock-layout` enabled after the first strong
profile is found.

OCR mode can be smoke-tested on recorded video:

```powershell
python gto.py validate-cv "video_frames\屏幕录制 2026-06-11 230741.mp4" --output-dir "video_frames\cv_validation_ocr_action_only_smoke" --every 120 --max-frames 3 --min-confidence 0.35 --with-ocr --ocr-action-only --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format text
```

Current smoke comparison on the same 3 sampled frames:

```text
no OCR:       median 198.6ms, max 4871.2ms
full OCR:     median 1995.5ms, max 3345.5ms
action-only:  median 1461.7ms, max 2195.8ms, skipped 1/3 OCR frames
```

The first no-OCR sample can still be expensive because auto bbox/layout locking
runs at startup; after locking, visual-only frames are much cheaper.

## Glyph Dataset Export

Export cropped rank/suit/card images from recorded videos:

```powershell
python gto.py export-card-glyphs --all --output-dir "video_frames\card_glyph_export_v2" --every 2 --max-frames 90 --min-rank-confidence 0.75 --min-suit-confidence 0.35
```

Output structure:

```text
rank\<label>\*.png
suit\<label>\*.png
card\<rank+suit>\*.png
manifest.jsonl
summary.json
```

`manifest.jsonl` includes `source`, `card_index`, `roi_mode`, `frame_index`, and the crop paths, so later audits can group the same hero/board card across nearby frames.

## External Dataset Ingest

External full-card datasets can be converted into the same rank/suit glyph layout when labels appear in file or folder names, for example `As`, `10h`, `queen_spades`, or `cards-[C0]-001.jpg`.
The parser also accepts suit-first compact labels such as `SA`, `h10`, `[D5]`,
`[5D]`, and Unicode suit symbols such as `A♠`.

The bundled HuggingFace dataset can be downloaded or inspected with:

```powershell
python gto.py download-card-dataset
```

By default this inspects the existing local directory and does not use the network when the directory is already present:

```text
pict\card_datasets\hf_f1nn21_playing_cards
```

Use `--refresh` to force a HuggingFace snapshot refresh.

Current default local result:

```text
repo: F1NN21/playing-cards
images: 52
label dirs: 52
likely root: pict\card_datasets\hf_f1nn21_playing_cards\organized_playing_cards
```

Convert it into the live rank/suit glyph layout:

```powershell
python gto.py ingest-card-images --dataset-dir "pict\card_datasets\hf_f1nn21_playing_cards\organized_playing_cards" --output-dir "video_frames\external_ingest_f1nn21_playing_cards" --format text
```

Current ingest result:

```text
ingested cards: 52
rank images: 52
suit images: 52
skipped: 0
```

```powershell
python gto.py ingest-card-images --dataset-dir "video_frames\external_datasets\lordloh-playing-cards\img" --output-dir "video_frames\card_glyph_export_external_lordloh"
```

The lordloh dataset was cloned from:

```text
https://github.com/lordloh/playing-cards
```

It produced 216 external cards with no skipped images in the current run. Directly mixing it into the live model lowered validation accuracy because its card style differs from the WPT UI, so it should be used as auxiliary or teacher data, not as the main live training set.

## Synthetic Glyph Balancing

Generate normalized black-background rank/suit glyphs for class balancing:

```powershell
python gto.py generate-card-synthetic --output-dir "video_frames\card_glyph_synthetic_v1" --per-class 40
```

This produces training-ready folders:

```text
rank\<label>\*.png
suit\<label>\*.png
manifest.jsonl
summary.json
```

The synthetic set is useful as an auxiliary `--extra-glyph-dir`, especially for rare ranks and suits. It should not replace real WPT crops.

## Deep Model Training

Offline MobileNet teacher model:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v1" --model-dir "pict\card_models\deep_v1" --kind both --arch mobilenet_v3_small --pretrained --epochs 3 --batch-size 32 --image-size 96
```

Current result:

```text
rank val acc: 0.782
suit val acc: 0.621
```

Realtime lightweight fallback model:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v1" --model-dir "pict\card_models\deep_realtime_v1" --kind both --arch simple_cnn --epochs 10 --batch-size 32 --image-size 64
```

Current result:

```text
rank val acc: 0.509
suit val acc: 0.819
```

## Larger Teacher Models

The cropped regions are already split into two independent recognition tasks:

- `rank`: 13 classes, `A K Q J T 9 8 7 6 5 4 3 2`
- `suit`: 4 classes, `s h d c`

This means a larger public/pretrained vision model does not need to understand poker UI. It only sees the small normalized glyph crop and predicts one label. The project now supports these torchvision backbones:

```text
simple_cnn
mobilenet_v3_small
resnet18
resnet50
efficientnet_b0
efficientnet_b2
convnext_tiny
swin_t
vit_b_16
```

Recommended offline teacher command:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --model-dir "pict\card_models\deep_teacher_convnext_v1" --kind both --arch convnext_tiny --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 6 --batch-size 16 --image-size 160
```

Current ConvNeXt teacher result:

```text
rank val acc: 0.682
suit val acc: 0.707
```

The rank and suit tasks can also use different larger backbones while still writing
`deep_rank.pt` and `deep_suit.pt` into one model directory:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --model-dir "pict\card_models\deep_teacher_split_v1" --kind both --rank-arch convnext_tiny --rank-image-size 160 --suit-arch resnet50 --suit-image-size 128 --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 6 --batch-size 16
```

If CPU training is too slow, train the two parts into separate directories:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --model-dir "pict\card_models\deep_teacher_rank_v1" --kind rank --arch convnext_tiny --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 6 --batch-size 16 --image-size 160

python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --model-dir "pict\card_models\deep_teacher_suit_v1" --kind suit --arch resnet50 --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 5 --batch-size 16 --image-size 128
```

The large teacher is for dataset auditing, not for live Tencent Meeting inference on this CPU box. Use it like this:

```powershell
python gto.py audit-card-glyphs --manifest "video_frames\card_glyph_export_v2\manifest.jsonl" --output-dir "video_frames\card_glyph_audit_v5_convnext_temporal" --teacher-model-dir "pict\card_models\deep_teacher_convnext_v1" --realtime-model-dir "pict\card_models\deep_realtime_v1" --max-review 160 --temporal-window-frames 120 --temporal-min-support 2
```

If the rank and suit teachers live in separate directories, use the split teacher
arguments:

```powershell
python gto.py audit-card-glyphs --manifest "video_frames\card_glyph_export_v2\manifest.jsonl" --output-dir "video_frames\card_glyph_audit_split_teacher_v1" --teacher-rank-model-dir "pict\card_models\deep_teacher_rank_v1" --teacher-suit-model-dir "pict\card_models\deep_teacher_suit_v1" --realtime-model-dir "pict\card_models\deep_realtime_v2_temporal" --max-review 160 --temporal-window-frames 120 --temporal-min-support 2
```

Current ConvNeXt + temporal audit result:

```text
audited: 348
needs_review: 55
accepted_copied: 293
review_sheet: video_frames\card_glyph_audit_v5_convnext_temporal\review_sheet.jpg
```

Then train a small realtime model from WPT-specific samples plus reviewed/accepted samples:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --model-dir "pict\card_models\deep_realtime_v2_temporal" --kind both --arch simple_cnn --class-balanced-loss --weighted-sampler --epochs 14 --batch-size 32 --image-size 64
```

Current realtime v2 result:

```text
rank val acc: 0.732
suit val acc: 0.888
```

Live inference also applies two conservative runtime guards:

- Hero card stabilization: if the same hand briefly loses one card or emits a low-confidence complete pair, reuse the previous complete hand only when dealer position, hero position, and board progression are compatible.
- Second-card K heuristic: for black second hero cards where the corner `K` template is tied with noisy alternatives, prefer `K` only in that narrow high-risk layout. This fixes the WPT overlap case where `Ks` was read as `Ts`.

Note: `vit_b_16` must be trained with `--image-size 224`.

Use the realtime model only as an optional fallback:

```powershell
python gto.py screen-cv --bbox "x,y,w,h" --auto-bbox --lock-layout --hero-name "于寻欢" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

Runtime can also load rank and suit from different model directories. This is
the preferred interface when a larger offline teacher uses different backbones
for the two tasks:

```powershell
python gto.py screen-cv --bbox "x,y,w,h" --auto-bbox --lock-layout --hero-name "于寻欢" --deep-rank-card-model-dir "pict\card_models\deep_teacher_rank_v1" --deep-suit-card-model-dir "pict\card_models\deep_teacher_suit_v1" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

For normal live play, use the promoted KNN/default fusion path and reserve deep
models for offline review/distillation. Validation without human labels checks
completeness and runtime regressions, not true card accuracy; complete but
wrong cards still need teacher or manual audit.

Split inference smoke test:

```powershell
python gto.py validate-cv "video_frames\屏幕录制 2026-06-11 230741.mp4" --output-dir "video_frames\cv_validation_split_smoke" --every 10 --max-frames 2 --min-confidence 0.35 --deep-rank-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --deep-suit-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

## Validation

Baseline after the deep-model work:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_baseline_after_deep" --every 30 --max-frames 20 --min-confidence 0.35
```

Result:

```text
complete: 35
empty_or_no_hand: 7
incomplete: 2
real_problem_count: 2
board_bad_count: 0
median latency: 143.2 ms
```

Realtime deep fallback validation:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_deep_realtime_v1" --every 30 --max-frames 20 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v1"
```

Result:

```text
complete: 35
empty_or_no_hand: 7
incomplete: 2
real_problem_count: 2
board_bad_count: 0
median latency: 167.4 ms
```

After model prewarming, the same realtime fallback path validated as:

```text
complete: 35
empty_or_no_hand: 7
incomplete: 2
real_problem_count: 2
board_bad_count: 0
median latency: 145.2 ms
```

Realtime v2 fallback with locked layout, temporal stabilization, and the narrow second-card K heuristic:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_deep_realtime_v2_temporal_heuristic_dense" --every 10 --max-frames 80 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal"
```

Result over 138 sampled frames:

```text
complete: 114
empty_or_no_hand: 20
incomplete: 2
obstructed_animation: 2
real_problem_count: 2
board_bad_count: 0
median latency: 149.6 ms
```

Matched dense baseline without deep fallback:

```text
complete: 105
empty_or_no_hand: 17
incomplete: 11
obstructed_animation: 5
real_problem_count: 11
board_bad_count: 0
median latency: 57.8 ms
```

The remaining 2 dense problems are the same 2026-06-09 desktop/window recording duplicated under two filenames; the 2026-06-10 and 2026-06-11 validation videos have `real_problem_count: 0`.

MobileNet fallback is useful offline but too heavy for live CPU use:

```text
median latency with MobileNet gated fallback: 349.8 ms
median latency with MobileNet ungated fallback: 469.1 ms
```

## Recommendation

For live Tencent Meeting use:

1. Use `--lock-layout --hero-name "于寻欢"` every time.
2. Do not enable MobileNet in live mode on this CPU machine.
3. Use `pict\card_models\deep_realtime_v2_temporal` as the current recommended low-confidence card fallback.
4. Keep exporting problem frames and periodically retrain on WPT-specific samples. More WPT samples are more valuable than generic public playing-card images.

## Active Learning

Generate a focused review package from exported WPT glyphs:

```powershell
python gto.py audit-card-glyphs --manifest "video_frames\card_glyph_export_v2\manifest.jsonl" --output-dir "video_frames\card_glyph_audit_v5_convnext_temporal" --teacher-model-dir "pict\card_models\deep_teacher_convnext_v1" --realtime-model-dir "pict\card_models\deep_realtime_v1" --max-review 160 --temporal-window-frames 120 --temporal-min-support 2
```

Current result:

```text
audited: 348
needs_review: 55
accepted_copied: 293
review_csv: video_frames\card_glyph_audit_v5_convnext_temporal\review.csv
review_md: video_frames\card_glyph_audit_v5_convnext_temporal\review.md
review_sheet: video_frames\card_glyph_audit_v5_convnext_temporal\review_sheet.jpg
accepted_dir: video_frames\card_glyph_audit_v5_convnext_temporal\accepted
```

Manual correction workflow:

1. Open `video_frames\card_glyph_audit_v5_convnext_temporal\review_sheet.jpg` for visual scanning.
2. Fill `final_rank` and/or `final_suit` in `video_frames\card_glyph_audit_v5_convnext_temporal\review.csv`.
3. Apply the reviewed labels:

```powershell
python gto.py apply-card-review --review-csv "video_frames\card_glyph_audit_v5_convnext_temporal\review.csv" --output-dir "video_frames\card_glyph_review_applied_v2"
```

Train an active-learning candidate model from high-consensus accepted samples:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --model-dir "pict\card_models\deep_realtime_v2_temporal" --kind both --arch simple_cnn --class-balanced-loss --weighted-sampler --epochs 14 --batch-size 32 --image-size 64
```

Current promoted realtime result:

```text
rank val acc: 0.732
suit val acc: 0.888
validate dense real_problem_count: 2 vs baseline 11
validate board_bad_count: 0
validate dense median latency: 149.6 ms
```

## Latest 2026-07-08 Regression

After adding reviewed 2026-06-10 club/heart/diamond samples into `pict\card_templates`, the formerly failing `4?` / `A?` club cases now validate as `4c 3h` and `Ac 6h`.

Focused validation:

```powershell
python gto.py validate-cv "video_frames\屏幕录制 2026-06-10 200605.mp4" --output-dir "video_frames\cv_validation_0610_200605_after_club_templates" --every 10 --max-frames 8 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal"
```

Result:

```text
frames: 7
complete: 7
real_problem_count: 0
board_bad_count: 0
median latency: 403.4 ms
```

All-video validation:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_all_after_club_templates" --every 10 --max-frames 80 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal"
```

Result over 138 sampled frames:

```text
complete: 125
empty_or_no_hand: 13
real_problem_count: 0
board_bad_count: 0
median latency: 237.4 ms
p90 latency: 804.0 ms
```

Synthetic-data candidate:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --extra-glyph-dir "video_frames\card_review_0611_seed_applied" --extra-glyph-dir "video_frames\card_review_0610_200605_seed_applied" --extra-glyph-dir "video_frames\card_glyph_synthetic_v1" --model-dir "pict\card_models\deep_realtime_v6_synthetic" --kind both --arch simple_cnn --class-balanced-loss --weighted-sampler --epochs 8 --batch-size 32 --image-size 64
```

Candidate result:

```text
rank best val acc: 0.630
suit best val acc: 0.874
validate all real_problem_count: 0
card-output diff vs deep_realtime_v2_temporal on the 138-frame regression: 0
```

Because rank validation is weaker, `deep_realtime_v2_temporal` remains the recommended live fallback for now. The v6 synthetic model is retained as an experiment, not promoted. The large teacher remains an offline auditing tool, not a live CPU model.

After the hand-level truth audit and hero-card face filter, the safer all-video
sample is:

```text
sampled frames: 138
complete: 123
empty_or_no_hand: 14
obstructed_animation: 1
real_problem_count: 0
board_bad_count: 0
median latency: 208.7 ms
p90 latency: 725.5 ms
```

The follow-up hand audit found no strong complete-hand conflicts:

```text
audited hands: 124
strong consensus/open-suit disagreements: 0
needs_review: 80
ok: 44
```

The next offline step is to use external HuggingFace/CLIP teachers on the
already-cropped rank/suit regions as an independent signal:

```powershell
python gto.py label-card-crops-hf --input-dir "video_frames\card_review_all_after_truth_audit_fixes" --output-dir "video_frames\card_hf_clip_labeled_v1" --kind both --rank-score-threshold 0.58 --rank-margin-threshold 0.08 --suit-score-threshold 0.58 --suit-margin-threshold 0.08
```

Only high-confidence accepted samples from that folder should be mixed into a
new realtime model candidate; review rows should not be treated as truth.

External teacher smoke result on the current crops:

```text
openai/clip-vit-base-patch32, 40 samples:
rank correct: 5 / 20
suit correct: 11 / 20
```

Generic CLIP and TrOCR are therefore audit references only. They are not strong
enough to auto-promote into the realtime model without manual review.

## Latest 2026-07-08 Red-Five Fix

The hand-level audit found one real hidden card error:

```text
video: validation_20260610_crop.mp4
time: 40.0s
old read: Kd 8d
visual truth: Kd 5d
```

A broad manual rank-5 template fixed this frame but corrupted black 9 crops, so
that template was removed. The live code now uses a narrow red-card-only rank-5
override inside `recognize_card_rank`: it only triggers when the card is red,
the current prediction is a low-margin near-neighbor, and several width probes
independently vote for `5`.

Safe all-video validation after that fix:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_all_after_red5_override" --every 10 --max-frames 80 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

Result over 138 sampled frames:

```text
complete: 123
empty_or_no_hand: 14
obstructed_animation: 1
real_problem_count: 0
board_bad_count: 0
median latency: 196.4 ms
p90 latency: 695.9 ms
```

Review export after the same fix:

```powershell
python gto.py export-card-review --all --video-dir "video_frames" --output-dir "video_frames\card_review_all_after_red5_override" --every 10 --max-frames 80 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

Result:

```text
rows: 138
ok: 100
rank_low: 21
suit_low: 2
empty_or_no_hand: 14
obstructed_animation: 1
```

Hand audit after the red-5 fix:

```powershell
python gto.py audit-card-review --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --output-dir "video_frames\card_hand_audit_all_after_red5_override" --teacher-model-dir "pict\card_models\deep_teacher_convnext_v1" --realtime-model-dir "pict\card_models\deep_realtime_v2_temporal" --max-review 120 --format json --compact
```

Result:

```text
audited hands: 124
skipped_no_hand: 14
strong consensus/open-suit disagreements: 0
needs_review: 82
ok: 42
```

Only three frame-level outputs changed from the previous truth-audited review,
and all three are the same duplicated recording/frame:

```text
Kd 8d -> Kd 5d
```

Two retraining candidates were tested after organizing review crops:

```text
pict\card_models\deep_realtime_v7_organized
pict\card_models\deep_realtime_v8_ok_organized
```

Neither is promoted. `v7` used polluted labels from the full review folder, and
`v8` used only `review_reason=ok` rows but did not improve rank accuracy.
`pict\card_models\deep_realtime_v2_temporal` remains the recommended live
fallback.

## Latest 2026-07-08 Auto-BBox Hardening

Automatic table localization now has a `current-region-table` fallback. If the
capture region itself already looks like a poker table and has strong table
anchors, auto-bbox accepts the current capture instead of failing while looking
for a smaller child window. This covers the practical case where the user has
already dragged a tight table box or the Tencent Meeting stream is already a
cropped table.

Diagnostic command:

```powershell
python gto.py diagnose-auto-bbox --all --output-dir "video_frames\auto_bbox_diagnostics_all_light_after_current_region" --every 300 --max-frames 1 --min-confidence 0.35 --format json --compact
```

Result over 8 root validation videos, one frame per video, and five bbox stress
variants per frame:

```text
rows: 40
ok: 38
ok_inner_table: 2
failure_count: 0
methods:
  visual-titlebar-row: 19
  dealer-button-anchor: 8
  action-buttons: 12
  current-region-table: 1
```

Standard CV regression after the auto-bbox change:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_all_after_current_region_bbox" --every 10 --max-frames 80 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

Result over 138 sampled frames:

```text
complete: 123
empty_or_no_hand: 14
obstructed_animation: 1
real_problem_count: 0
board_bad_count: 0
median latency: 189.6 ms
p90 latency: 665.0 ms
```

## Card Recognition Benchmark

`validate-cv` checks whether the pipeline produces complete cards, but it is not
a full truth-label accuracy test. Use `benchmark-card-review` to evaluate card,
rank, and suit predictions against manual `final_card0/final_card1` labels and,
optionally, high-confidence `review_reason=ok` pseudo-truth.

Manual-only smoke test for the `Kd 5d` correction:

```powershell
python gto.py benchmark-card-review --review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --output-dir "video_frames\card_benchmark_manual_kd5d_smoke" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

Result:

```text
samples: 2 manual
current_csv card acc: 0.500
runtime card acc: 1.000
deep_realtime_v2_temporal card acc: 0.000
```

Mixed benchmark using 200 `ok` pseudo-truth slots plus the 2 manual `Kd 5d`
slots:

```powershell
python gto.py benchmark-card-review --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --output-dir "video_frames\card_benchmark_red5_runtime_v2" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --include-ok-pseudo --format text
```

Result:

```text
samples: 202
truth_sources: pseudo:ok 200, manual 2
current_csv card/rank/suit: 0.995 / 0.995 / 1.000
runtime card/rank/suit: 1.000 / 1.000 / 1.000
deep_realtime_v2_temporal card/rank/suit: 0.173 / 0.257 / 0.604
```

Standalone deep-model comparisons on the same benchmark:

```text
deep_teacher_convnext_v1: card 0.267, rank 0.505, suit 0.559
deep_realtime_v7_organized: card 0.540, rank 0.688, suit 0.767
deep_realtime_v8_ok_organized: card 0.480, rank 0.515, suit 0.802
deep_realtime_v9_red5_ok: card 0.208, rank 0.515, suit 0.401
```

Conclusion: the full runtime recognizer remains much stronger than any
standalone deep model on the current benchmark. Deep models should stay gated
fallbacks or offline audit signals until this benchmark improves materially.
Do not promote `v7`, `v8`, or `v9` as the live model.

The benchmark also reports a standalone KNN/template evaluator. On the same
202-slot mixed benchmark:

```text
default card_glyph_knn: card 0.713, rank 0.847, suit 0.866
deep_realtime_v2_temporal: card 0.173, rank 0.257, suit 0.604
runtime recognizer: card 1.000, rank 1.000, suit 1.000
```

This confirms that the live strength mostly comes from KNN/templates plus
runtime color/ROI/heuristic fusion; the deep model is a weak fallback.

KNN candidate training now supports WPT glyph directories directly:

```powershell
python gto.py train-card-classifier --glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --model "pict\card_models\card_glyph_knn_candidate_red5_ok.npz" --augment 8 --glyph-augment 6 --format text
```

Candidate KNN result:

```text
rank source: 810, suit source: 849
glyph rank/suit sources: 493 / 493
benchmark KNN card/rank/suit: 1.000 / 1.000 / 1.000
validate-cv --all: real_problem_count 0, board_bad_count 0
median latency: 224.4 ms
```

However, this candidate is not promoted. The benchmark overlaps with candidate
training data, and a full review diff found 12 card-output changes, including
3 high-risk changes such as:

```text
9d 6h -> 8d 6h
Jd 6s -> 6h 4s
7d 7c -> 7d 7s
```

The diff gate is now a first-class command:

```powershell
python gto.py diff-card-review --baseline-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --candidate-review-csv "video_frames\card_review_all_candidate_knn_red5_ok\review.csv" --output-dir "video_frames\card_review_diff_candidate_knn_red5_ok" --fail-on-risk --format text
```

The combined promotion gate wraps benchmark, diff, and validate-cv latency into
one decision:

```powershell
python gto.py gate-card-model --benchmark-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --benchmark-review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --baseline-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --candidate-review-csv "video_frames\card_review_all_candidate_external_f1nn21\review.csv" --baseline-validation-summary-json "video_frames\cv_validation_all_after_current_region_bbox\cv_validation_all_summary.json" --candidate-validation-summary-json "video_frames\cv_validation_all_candidate_external_f1nn21\cv_validation_all_summary.json" --output-dir "video_frames\card_model_gate_external_f1nn21_with_validation" --candidate-name "external_f1nn21_knn" --candidate-evaluator knn --knn-model "pict\card_models\card_glyph_knn_candidate_external_f1nn21.npz" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --include-ok-pseudo --require-validation --max-median-regression-ms 80 --max-p90-regression-ms 150 --fail-on-reject --format text
```

The external-data KNN candidate result:

```text
decision: REJECT
benchmark knn card/rank/suit: 1.000 / 1.000 / 1.000
review diff changed: 12
review diff risk: 3
validate-cv real_problem_count: 0
validate-cv board_bad_count: 0
median latency: 331.6 ms, allowed 269.6 ms
p90 latency: 913.1 ms, allowed 815.0 ms
exit code with --fail-on-reject: 2
```

Current result:

```text
matched rows: 138
slots: 247
changed: 12
risk: 3
risky examples:
- validation_20260610_crop.mp4 t=10.000 slot=0: 9d -> 8d
- 屏幕录制 2026-06-11 230741.mp4 t=790.000 slot=0: Jd -> 6h
- 屏幕录制 2026-06-11 230741.mp4 t=790.000 slot=1: 6s -> 4s
```

`gto.py` now propagates command return codes, so `--fail-on-risk` returns
exit code 2 when a candidate has more risky changes than `--max-risk`.

Use candidate KNN files only behind `--card-knn-model` until a non-overlapping
manual benchmark proves improvement:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_candidate_knn" --card-knn-model "pict\card_models\card_glyph_knn_candidate_red5_ok.npz" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal"
```

An external-data candidate was also trained after ingesting
`F1NN21/playing-cards`:

```powershell
python gto.py train-card-classifier --glyph-dir "video_frames\external_ingest_f1nn21_playing_cards" --glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --model "pict\card_models\card_glyph_knn_candidate_external_f1nn21.npz" --augment 8 --glyph-augment 6 --format text
```

Result:

```text
rank source: 862, suit source: 901
benchmark KNN card/rank/suit: 1.000 / 1.000 / 1.000
validate-cv --all: real_problem_count 0, board_bad_count 0
median latency: 331.6 ms
diff-card-review: changed 12, risk 3, exit code 2 with --fail-on-risk
```

This external-data candidate is also rejected for live use. More generic card
images did not fix the WPT-domain risky changes and made latency worse than the
current promoted default.

## Manual Label Queue

To grow a real hand-labeled dataset, merge suspicious review/audit rows into a
single prioritized queue:

```powershell
python gto.py prepare-card-label-queue --review-csv "video_frames\card_hand_audit_all_after_red5_override\review.csv" --output-dir "video_frames\card_label_queue_red5_top80" --max-rows 80 --format text
```

Current queue result:

```text
candidates: 82
deduped: 82
selected: 80
assets copied: 557 image files
queue_csv: video_frames\card_label_queue_red5_top80\label_queue.csv
queue_html: video_frames\card_label_queue_red5_top80\label_queue.html
```

Open the HTML file for visual review, then fill `final_card0` and
`final_card1`. Empty final cells are ignored, so it is safe to label only the
uncertain slot in a hand.

Preferred local labeling UI:

```powershell
python gto.py serve-card-label-queue --queue-csv "video_frames\card_label_queue_red5_top80\label_queue.csv" --open-browser
```

This starts a local `127.0.0.1` web UI that displays the table frame, card
crops, rank glyphs, and suit glyphs. The `Save` button writes directly back to
`label_queue.csv`. Smoke test result:

```text
loaded rows: 80
total slots: 159
POST /api/update wrote final_card0 back to CSV successfully
```

If the local UI is not convenient, edit `label_queue.csv` directly.

Apply completed labels into a trainable rank/suit/card dataset:

```powershell
python gto.py apply-card-review --review-csv "video_frames\card_label_queue_red5_top80\label_queue.csv" --output-dir "video_frames\card_label_queue_red5_top80_applied"
```

Then run the benchmark against that same queue:

```powershell
python gto.py benchmark-card-review --review-csv "video_frames\card_label_queue_red5_top80\label_queue.csv" --output-dir "video_frames\card_benchmark_label_queue_red5_top80" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format text
```

Finally, use the applied labels as an extra training source for a candidate
model, and only promote the candidate if it beats the current runtime benchmark
and the all-video `validate-cv` regression:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --extra-glyph-dir "video_frames\card_label_queue_red5_top80_applied" --model-dir "pict\card_models\deep_realtime_candidate_manual" --kind both --arch simple_cnn --class-balanced-loss --weighted-sampler --epochs 14 --batch-size 32 --image-size 64
```
