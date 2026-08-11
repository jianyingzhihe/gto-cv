# Card Big-Model Labeling Pipeline

This pipeline uses larger pretrained models as offline teachers for cropped card regions.
The live CV path should still use templates, cached layout, and the small realtime model.

## Current Recommended Flow

Use the big model offline only. The production path is:

1. Crop card regions with the existing CV layout code.
2. Run a frozen HuggingFace vision model over the cropped regions.
3. Train two small probes, one for rank and one for suit.
4. Auto-accept only high-confidence crops, preferably only when the probe agrees with the current trusted label.
5. Distill accepted crops back into the fast runtime KNN/deep model.
6. Promote only after the normal benchmark, diff, validation, and gate checks pass.
   Runtime candidates also have to pass the default latency gate:
   median <= 300 ms and p90 <= 900 ms.

The crop teacher is already split by task:

- `rank`: 13 labels, using the cropped printed rank glyph.
- `suit`: 4 labels, using the cropped suit/color glyph.

This split is intentional. Rank mistakes are usually shape/blur mistakes, while
suit mistakes are usually color/context mistakes. Keep the two classifiers and
thresholds separate, then combine the accepted rank+suit labels only after both
passes are confident.

## Current Split-Teacher Smoke

Latest local smoke command, verified 2026-07-09:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_split_smoke_20260709" --kind both --max-images 8 --batch-size 8 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

Result:

```text
model: rank=facebook/dinov2-base, suit=facebook/dinov2-base
processed: 16
accepted: 8
review: 8
predictions: video_frames\card_big_teacher_split_smoke_20260709\labeled\predictions.csv
review: video_frames\card_big_teacher_split_smoke_20260709\labeled\review.csv
runbook: video_frames\card_big_teacher_split_smoke_20260709\runbook.md
```

The accepted crops can be used as additional training data; rejected rows should
go through the label queue/manual review path.

Latest smoke on the current runtime crop directory, verified 2026-07-09:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_rank_large_suit_base_next" --output-dir "video_frames\big_model_split_smoke_20260709" --kind both --max-images 40 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --batch-size 8 --format text
```

Result:

```text
rank model: facebook/dinov2-large
suit model: facebook/dinov2-base
processed: 80
accepted: 46
review: 34
predictions: video_frames\big_model_split_smoke_20260709\labeled\predictions.csv
review: video_frames\big_model_split_smoke_20260709\labeled\review.csv
runbook: video_frames\big_model_split_smoke_20260709\runbook.md
```

This is a healthy offline-teacher result, not a promotion result. Continue to
use the promoted KNN model in live play unless a distilled candidate passes the
benchmark/diff/validation gate with zero new risk.

Full split-teacher run on the same current runtime crop directory, verified
2026-07-09:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_rank_large_suit_base_next" --output-dir "video_frames\big_model_split_full_20260709" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --batch-size 8 --format text
```

Result:

```text
processed: 494
accepted: 302
review: 192
rank review: 146
suit review: 46
predictions: video_frames\big_model_split_full_20260709\labeled\predictions.csv
review: video_frames\big_model_split_full_20260709\labeled\review.csv
glyph queue: video_frames\big_model_split_full_20260709\glyph_label_queue\glyph_label_queue.html
contact sheet: video_frames\big_model_split_full_20260709\glyph_label_queue\glyph_label_queue_sheet.jpg
```

The threshold sweep found no threshold set with complete rank/suit coverage:

```text
best automatic accepted rows: 317
rank accepted: 116/247
suit accepted: 201/247
missing rank labels: Q, 9, 7, 6, 5
report: video_frames\big_model_split_full_20260709\threshold_sweep\threshold_sweep.md
```

Treat this run as an active-learning batch. The rejected/uncertain 192 glyphs
should be manually reviewed before any distillation attempt; otherwise the
teacher can reintroduce known WPT-domain rank confusions such as `6 -> 4`,
`8 -> 9`, and `J -> 9`.

The ensemble voter now uses real majority voting for `--min-teachers`: with
three teacher CSVs, any label that receives two votes can win. This is useful
for reducing the manual queue without trusting a single model:

```powershell
python gto.py ensemble-card-hf-predictions --predictions-csv "video_frames\big_model_split_full_20260709\labeled\predictions.csv" --predictions-csv "video_frames\card_big_teacher_split_large_next_20260709\labeled\predictions.csv" --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --output-dir "video_frames\big_model_split_full_20260709\ensemble_vote2of3" --kind both --min-teachers 2 --rank-score-threshold 0.45 --rank-margin-threshold 0.04 --suit-score-threshold 0.60 --suit-margin-threshold 0.04 --format text
python gto.py sweep-card-hf-thresholds --predictions-csv "video_frames\big_model_split_full_20260709\ensemble_vote2of3\predictions.csv" --output-dir "video_frames\big_model_split_full_20260709\ensemble_vote2of3_threshold_sweep" --rank-score-threshold 0.25 --rank-score-threshold 0.35 --rank-score-threshold 0.45 --rank-score-threshold 0.55 --rank-margin-threshold 0.00 --rank-margin-threshold 0.02 --rank-margin-threshold 0.04 --rank-margin-threshold 0.08 --suit-score-threshold 0.45 --suit-score-threshold 0.55 --suit-score-threshold 0.60 --suit-score-threshold 0.65 --suit-margin-threshold 0.00 --suit-margin-threshold 0.02 --suit-margin-threshold 0.04 --suit-margin-threshold 0.06 --format text
python gto.py filter-card-hf-predictions --predictions-csv "video_frames\big_model_split_full_20260709\ensemble_vote2of3\predictions.csv" --output-dir "video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended" --kind both --rank-score-threshold 0.25 --rank-margin-threshold 0.04 --suit-score-threshold 0.60 --suit-margin-threshold 0.06 --require-current-agreement --format text
python gto.py prepare-card-glyph-label-queue --predictions-csv "video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended\predictions.csv" --output-dir "video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended\glyph_label_queue" --max-rows 160 --format text
```

Current result:

```text
2-of-3 ensemble rows: 494
recommended thresholds: rank_score=0.25 rank_margin=0.04 suit_score=0.60 suit_margin=0.06
accepted: 393
manual review: 101
manual queue: video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended\glyph_label_queue\glyph_label_queue.html
contact sheet: video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended\glyph_label_queue\glyph_label_queue_sheet.jpg
```

The 2-of-3 accepted set was also distilled into a runtime KNN candidate, but
the promotion gate rejected it:

```text
candidate: ensemble_vote2of3_recommended
benchmark card/rank/suit: 0.830 / 0.835 / 0.995
review_diff_risk: 71
validation real_problem=0 board_bad=0 median/p90=29.9 ms / 51.5 ms
gate report: video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended_distill\runtime_candidate\gate\card_model_gate_report.md
```

Use the ensemble result to shrink the manual labeling workload, not as an
automatic live-model promotion path.

## Pipeline-Generated Split-Teacher Commands

`card-cv-pipeline` now writes the full split-teacher follow-up path into
`<output-dir>\commands\`. This is the preferred way to avoid hand-copying long
commands:

```powershell
python gto.py card-cv-pipeline --crop-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_rank_large_suit_base_next" --output-dir "video_frames\card_cv_pipeline_split_teacher_commands_20260709" --no-candidate-summary --format text
```

Verified output:

```text
Crops: rank=247 suit=247
Probe ready: True
Dataset images: 52
Dataset deck: parsed=52/52 complete=True missing=0 duplicate_labels=0 unparsed=0
Crop quality: missing_rank=[] missing_suit=[] unreadable=0
```

`Dataset deck` is the quick safety check before running a bigger public model:
it should be `parsed=52/52 complete=True` for a clean one-deck external
dataset. If it reports missing, duplicate, or unparsed cards, fix that before
training a new rank/suit probe.

The generated command files now include:

```text
label_with_probe.txt              run the split rank/suit teacher
threshold_sweep.txt               tune thresholds from cached predictions.csv
filter_predictions_distill.txt    reuse predictions.csv and distill/gate a runtime candidate
prepare_glyph_label_queue.txt     build a manual rank/suit glyph queue
apply_glyph_label_queue.txt       apply completed glyph labels into a training folder
```

The important detail is that `threshold_sweep.txt` and
`filter_predictions_distill.txt` reuse
`big_teacher_label\labeled\predictions.csv`, so DINOv2/CLIP embeddings do not
need to be recomputed while tuning thresholds.

`train_probe.txt` now trains on both the current live/replay crop directory and
the ingested public-card glyph directory. The per-class cap is applied with
source-balanced round-robin sampling, so a large current-crop class no longer
pushes all external examples out of the training set. `label_with_probe.txt`
still labels only the current crop directory, so the external dataset is used
as training support and is not mixed into the live-review prediction set.

Current generated training input check:

```text
Probe train inputs: ['video_frames\\current_runtime_final_review', 'video_frames\\external_card_glyphs']
train_probe.txt:
python gto.py train-card-hf-probe --input-dir "video_frames\current_runtime_final_review" --input-dir "video_frames\external_card_glyphs" ...
```

The same glyph queue format can now be built from live/debug review files:

```powershell
python gto.py prepare-card-glyph-label-queue --review-csv "video_frames\screen_live_card_debug_review\review.csv" --output-dir "video_frames\screen_live_card_debug_review\glyph_label_queue" --max-rows 160 --format text
```

Use this when live `card_debug` shows a single bad rank or suit crop. It keeps
the correction at the rank/suit level instead of forcing a whole-hand label.

## Next Split Large-Model Track

Yes: the next useful step is to keep the cropper fixed and let larger public
vision models handle the already-cropped regions. Treat the two subtasks as
separate classifiers:

- `rank`: card rank glyphs, 13 classes: `A K Q J T 9 8 7 6 5 4 3 2`.
- `suit`: suit/color glyphs, 4 classes: `s h d c`.

Do not run these big models in the live Tencent Meeting loop. Use them as
offline teachers, then distill accepted labels back into the fast runtime card
model.

Current verified smoke on the latest crop directory:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_split_demo_latest" --kind both --max-images 8 --batch-size 8 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

Result:

```text
model: rank=facebook/dinov2-base, suit=facebook/dinov2-base
processed: 16
accepted: 5
review: 11
predictions: video_frames\card_big_teacher_split_demo_latest\labeled\predictions.csv
review: video_frames\card_big_teacher_split_demo_latest\labeled\review.csv
runbook: video_frames\card_big_teacher_split_demo_latest\runbook.md
```

For a heavier public-model experiment, train a new split probe. Start with DINO
for rank shape and CLIP for suit/color:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\current_runtime_final_review" --output-dir "pict\card_models\hf_probe_split_large_next" --kind both --rank-model "facebook/dinov2-large" --suit-model "openai/clip-vit-large-patch14" --max-images-per-class 32 --batch-size 8 --format text
```

Then label the same crop set with that probe:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_split_large_next" --output-dir "video_frames\card_big_teacher_split_large_next" --kind both --batch-size 8 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --format text
```

After the predictions are generated, sweep thresholds without recomputing
embeddings:

```powershell
python gto.py sweep-card-hf-thresholds --predictions-csv "video_frames\card_big_teacher_split_large_next\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_split_large_next\threshold_sweep" --rank-score-threshold 0.45 --rank-score-threshold 0.50 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --rank-margin-threshold 0.08 --rank-margin-threshold 0.12 --suit-score-threshold 0.60 --suit-score-threshold 0.65 --suit-score-threshold 0.70 --suit-margin-threshold 0.04 --suit-margin-threshold 0.06 --suit-margin-threshold 0.10 --format text
```

Only after thresholding and manual review should the accepted crops be distilled
back to the runtime model:

```powershell
python gto.py filter-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_split_large_next\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_split_large_next\filtered_runtime" --kind both --rank-score-threshold 0.50 --rank-margin-threshold 0.12 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text
```

## 2026-07-09 Split-Large Experiments

The current crop set is strong enough for large-model experiments:

```text
crop dir: video_frames\current_runtime_final_review
rank crops: 247
suit crops: 247
missing rank labels: []
missing suit labels: []
unreadable: 0
```

`card-cv-pipeline` now supports configurable split probe models and can train
the probe directly:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --crop-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_rank_large_suit_base_next" --probe-rank-model "facebook/dinov2-large" --probe-suit-model "facebook/dinov2-base" --probe-max-images-per-class 32 --probe-batch-size 8 --run-train-probe --output-dir "video_frames\card_cv_pipeline_rank_large_suit_base_train_20260709" --no-candidate-summary --format text
```

Local experiment summary:

| Probe | Rank Val | Suit Val | Accepted | Rank Accepted | Suit Accepted | Gate |
|---|---:|---:|---:|---:|---:|---|
| DINOv2-base / DINOv2-base | 0.8077 | 1.0 | 252/494 | 57 | 195 | old candidate, not promoted |
| DINOv2-large / CLIP ViT-L/14 | 0.8 | 0.7917 | 297/494 | 101 | 196 | reject |
| DINOv2-large / DINOv2-base | 0.8 | 0.8333 | 302/494 | 101 | 201 | reject |
| Union: stable base + old DINOv2 accepted + rank-large/suit-base accepted | - | - | 302 new + bases | - | - | reject |

The best split-large runtime candidate so far is:

```text
candidate: teacher_union_base_ranklarge_suitbase_std
gate: reject
benchmark card/rank/suit: 0.995 / 1.0 / 0.995
validation real_problem: 0
validation board_bad: 0
median/p90: 31.4 ms / 58.1 ms
risk: 34
report: video_frames\card_big_teacher_rank_large_suit_base_next_20260709\filtered_union_with_base_distill\runtime_candidate\gate\card_model_gate_report.md
```

Conclusion: the large teachers are useful as offline data sources, but the
current accepted set should not replace the promoted live card model. The rank
teacher is complementary rather than strictly better: DINOv2-base accepts some
`A/5/6`, while DINOv2-large accepts more `J/8/K/2`. Keep both teacher outputs
for manual review/active learning, then retrain only after the risk queue is
resolved or more real crops are labeled.

### Ensemble Agreement Pass

`ensemble-card-hf-predictions` accepts crops only when multiple teacher
prediction CSVs agree on the same label. This is meant to create a high-precision
offline dataset before distilling to the realtime model.

```powershell
python gto.py ensemble-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --predictions-csv "video_frames\card_big_teacher_rank_large_suit_base_next_20260709\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_ensemble_base_ranklarge_suitbase_20260709" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --distill-runtime --runtime-candidate-name "ensemble_base_ranklarge_suitbase_agree" --runtime-base-glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --runtime-risk-queue-max-rows 80 --format text
```

Result:

```text
processed: 494
accepted: 209
review: 285
teacher_disagrees: 106
gate: reject
benchmark card/rank/suit: 0.995 / 1.0 / 0.995
validation real_problem: 0
validation board_bad: 0
median/p90: 31.2 ms / 58.1 ms
risk: 34
report: video_frames\card_big_teacher_ensemble_base_ranklarge_suitbase_20260709\runtime_candidate\gate\card_model_gate_report.md
```

Conclusion: teacher agreement gives a cleaner offline crop set, but it still
does not clear the live promotion gate. Use the ensemble output as an
active-learning source, not as a promoted replacement.

### Glyph Disagreement Label Queue

Teacher disagreement is now converted into a single-glyph labeling queue instead
of the full-hand `label_queue.csv` format. This is the right format for rank and
suit crops because each row has one image and one `final_label`.

Generate the queue from ensemble predictions:

```powershell
python gto.py prepare-card-glyph-label-queue --predictions-csv "video_frames\card_big_teacher_ensemble_base_ranklarge_suitbase_detail_20260709\predictions.csv" --output-dir "video_frames\card_glyph_label_queue_ensemble_disagrees_detail_20260709" --allowed-reason teacher_disagrees --max-rows 160 --format text
```

Current output:

```text
candidates: 106
selected: 106
rank: 60
suit: 46
csv: video_frames\card_glyph_label_queue_ensemble_disagrees_detail_20260709\glyph_label_queue.csv
html: video_frames\card_glyph_label_queue_ensemble_disagrees_detail_20260709\glyph_label_queue.html
sheet: video_frames\card_glyph_label_queue_ensemble_disagrees_detail_20260709\glyph_label_queue_sheet.jpg
```

Each disagreement row preserves the teacher votes in `teacher_model`, for
example:

```text
ensemble_disagree[facebook/dinov2-base=3;facebook/dinov2-large=8]
```

After filling `final_label`, apply the queue into a rank/suit dataset:

```powershell
python gto.py apply-card-glyph-label-queue --queue-csv "video_frames\card_glyph_label_queue_ensemble_disagrees_detail_20260709\glyph_label_queue.csv" --output-dir "video_frames\card_glyph_label_applied_ensemble_disagrees"
```

Then add that applied directory as another `--glyph-dir` or `--base-glyph-dir`
in the next retraining/gate pass.

### Current-Prefill Pseudo-Truth Experiment

For a controlled pseudo-label experiment, `prepare-card-glyph-label-queue` can
prefill `final_label` from `current_label`. This is off by default and should be
treated as a candidate experiment, not manual truth.

```powershell
python gto.py prepare-card-glyph-label-queue --predictions-csv "video_frames\card_big_teacher_ensemble_base_ranklarge_suitbase_detail_20260709\predictions.csv" --output-dir "video_frames\card_glyph_label_queue_ensemble_disagrees_current_prefill_20260709" --allowed-reason teacher_disagrees --prefill-final-label current --max-rows 160 --format text

python gto.py apply-card-glyph-label-queue --queue-csv "video_frames\card_glyph_label_queue_ensemble_disagrees_current_prefill_20260709\glyph_label_queue.csv" --output-dir "video_frames\card_glyph_label_applied_ensemble_current_prefill_20260709" --format text
```

Applied labels:

```text
rows: 106
copied: 106
rank: 2=7, 3=10, 6=9, 7=5, 8=9, 9=8, J=12
suit: c=9, d=14, h=2, s=21
```

Training a seeded KNN candidate with this applied dataset improved the benchmark
to 1.0/1.0/1.0 but still failed promotion:

| Candidate | Card/Rank/Suit | Risk | Real Problems | Board Bad | Median | P90 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| non-compact current-prefill | 1.0 / 1.0 / 1.0 | 34 | 0 | 0 | 126.0 ms | 976.0 ms | reject |
| compact top4 min16 max96 | 1.0 / 1.0 / 1.0 | 40 | 0 | 0 | 98.0 ms | 793.1 ms | reject |
| compact top8 min24 max160 | 1.0 / 1.0 / 1.0 | 40 | 0 | 0 | 93.8 ms | 760.5 ms | reject |
| compact top16 min32 max320 | 1.0 / 1.0 / 1.0 | 38 | 0 | 0 | 104.8 ms | 860.6 ms | reject |

Best current-prefill speed/accuracy tradeoff so far:

```text
model: video_frames\glyph_current_prefill_candidate_20260709\glyph_current_prefill_candidate_compact_wide.npz
gate: video_frames\glyph_current_prefill_candidate_20260709\gate_compact_wide\card_model_gate_report.md
risk queue: video_frames\glyph_current_prefill_candidate_20260709\gate_compact_wide_risk_label_queue\label_queue.html
```

Conclusion: current-prefill is useful because it fixes benchmark accuracy and
keeps validation clean after compaction, but it still needs the 29-row risk queue
resolved before promotion.

## Recommended Full Split-Teacher Pass

The preferred large-model path is a frozen HuggingFace encoder plus two probes:
one prototype classifier for rank and one prototype classifier for suit. The
current local default uses `facebook/dinov2-base` for both probes.

If the probe already exists, label the current crop set directly:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_dinov2_base_full" --kind both --batch-size 16 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

If the crop distribution has changed a lot, retrain the probes first:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --output-dir "pict\card_models\hf_probe_dinov2_base_next" --kind both --model "facebook/dinov2-base" --max-images-per-class 24 --batch-size 16 --local-files-only --format text

python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_next" --output-dir "video_frames\card_big_teacher_dinov2_base_next" --kind both --batch-size 16 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

To use different public encoders for rank and suit, train a split probe:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --output-dir "pict\card_models\hf_probe_split_next" --kind both --rank-model "facebook/dinov2-base" --suit-model "openai/clip-vit-base-patch32" --max-images-per-class 24 --batch-size 16 --local-files-only --format text
```

Keep these large-model passes offline. Do not put DINO/CLIP in the normal live
Tencent Meeting command; use the accepted crops to train/gate a fast runtime
candidate instead.

## 2026-07-09 Current-Crop Full Teacher Pass

The current stable crop set is:

```text
video_frames\current_runtime_final_review
rank crops: 247
suit crops: 247
rank labels: complete
suit labels: complete
unreadable: 0
```

Full DINOv2-base teacher run with the existing review probe:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709" --kind both --batch-size 16 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

Result:

```text
processed: 494
accepted: 252
review: 242
rank accepted: 57 / 247
suit accepted: 195 / 247
```

Training a new probe directly on `current_runtime_final_review` did not improve
the rank teacher:

```text
probe: pict\card_models\hf_probe_dinov2_current_runtime_v1
rank val_acc: 0.807692
suit val_acc: 1.0
full teacher accepted: 228 / 494
```

Keep `pict\card_models\hf_probe_dinov2_base_v1_review` as the current teacher
probe unless a later probe beats it on both validation and full-run acceptance.

Scan teacher thresholds before distillation:

```powershell
python gto.py sweep-card-hf-thresholds --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\threshold_sweep" --format text
```

Current sweep recommendation:

```text
rank_score: 0.4
rank_margin: 0.12
suit_score: 0.65
suit_margin: 0.06
accepted: 348
rank accepted: 153 / 247
suit accepted: 195 / 247
report: video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\threshold_sweep\threshold_sweep.md
```

The rank `0.40` distillation candidate was gated with deep fallback disabled,
matching the current live-latency policy:

```powershell
python gto.py filter-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank040_distill" --kind both --rank-score-threshold 0.40 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --distill-runtime --runtime-candidate-name "dinov2_rank040_no_deep" --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text
```

Gate result:

```text
decision: reject
promote: false
benchmark card/rank/suit: 0.995 / 1.0 / 0.995
review diff risk: 29
missing rows: 3
validation real_problem: 0
validation board_bad: 0
validation median/p90: 27.6 ms / 55.4 ms
gate: video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank040_distill\runtime_candidate\gate\card_model_gate_report.md
risk queue: video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank040_distill\runtime_candidate\risk_label_queue\label_queue.html
```

Do not promote `dinov2_rank040_no_deep`. The risk queue should be manually
resolved first, or the accepted teacher set should be constrained further.

Additional threshold candidates were tested with the same no-deep runtime gate:

```powershell
python gto.py filter-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank045m012_distill" --kind both --rank-score-threshold 0.45 --rank-margin-threshold 0.12 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --distill-runtime --runtime-candidate-name "dinov2_rank045m012_no_deep" --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text

python gto.py filter-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank050m012_distill" --kind both --rank-score-threshold 0.50 --rank-margin-threshold 0.12 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --distill-runtime --runtime-candidate-name "dinov2_rank050m012_no_deep" --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text
```

Candidate comparison:

```text
Candidates: 3
Promotable: 0
best: dinov2_rank050m012_no_deep
card/rank/suit: 0.995 / 1.0 / 0.995
risk: 27
changed: 30
missing rows: 3
median: 28.1 ms
report: video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\candidate_comparison_after_thresholds\card_candidate_summary.md
```

Threshold tightening from `0.40` to `0.50` only reduced risk from `29` to
`27`. Stop blind threshold tuning here. The remaining risk is stable across
candidates and should be handled through the generated risk queue:

```text
video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank050m012_distill\runtime_candidate\risk_label_queue\label_queue.html
```

Important: use `review_with_truth.csv` for gate/diff truth only. Do not use a
candidate's `review_with_truth.csv` as training input when the candidate may
have slot drift or crop drift. In that case the truth label can be attached to
the wrong crop and poison the next model.

## Latest Overlap-ROI And Riskfix Iteration

Verified 2026-07-08:

- Code now treats overlap hand-card ROIs as backup only. They are not used to
  lock the initial layout profile.
- Overlap backup supports partial visible cards and merges best candidates by
  card index, so one card can come from overlap and the other from component
  search.
- The dealing-frame classifier now detects red card-back animation in the hero
  region, so frames like `problem_000710_incomplete.png` are classified as
  `obstructed_animation`, not a real recognition failure.

Targeted problem-frame checks with the clean/riskfix candidate:

```text
problem_000420_incomplete.png -> 3c 9c
problem_000460_incomplete.png -> 8s 9h
problem_000710_incomplete.png -> obstructed_animation / K?
```

The default promoted model plus these code-path changes was validated over the
current 8 root videos:

```text
frames: 138
real_problem: 0
board_bad: 0
counts: {"complete": 123, "empty_or_no_hand": 13, "obstructed_animation": 2}
summary: video_frames\promoted_default_after_overlap_backup_animation_validation\cv_validation_all_summary.json
```

After the red-rank selector and live-speed check, the recommended live path is
to omit `--deep-card-model-dir`. The deep fallback did not improve the current
video validation but increased p90 latency substantially. Latest no-deep
validation:

```text
frames: 138
real_problem: 0
board_bad: 0
median: 95.8 ms
p90: 706.3 ms
summary: video_frames\promoted_default_after_red8_no_deep_validation\cv_validation_all_summary.json
```

Use deep models for offline audit/teacher runs, not for the normal Tencent
Meeting live loop unless a later gate proves a latency-safe benefit.

Do not promote the current `manual10_overlap_v4/v5/v6` KNN candidates yet. They
fix the 420/460 overlap frames but still fail the promotion gate. The most
recent strict gate for `manual10_overlap_v4_clean_runtime` rejected with:

```text
benchmark card/rank/suit: 0.995 / 0.995 / 1.000
review diff risk: 15
validation real_problem: 1 before red-card-back animation classification
p90: 1386 ms
```

The reduced risk queue is:

```text
video_frames\card_big_teacher_dinov2_base_filter_s3_manual10_overlap_v4_clean_runtime\risk_label_queue\label_queue.html
```

Current risk groups:

```text
Ad->4d x5: regression, true card is Ad
4s->9s x2: regression, true card is 4s
8d->9d x1: old baseline appears wrong from table context, true card is 9d
3h->8h x2: old baseline appears wrong from table context, true card is 8h
```

The next model iteration should solve these as rank-window selection issues,
not by blindly adding more generic data. In particular, do not let tied
high-score wide windows override narrower windows with a real positive margin.

## Current Promoted Runtime Model

As of 2026-07-08, the default realtime KNN model has been promoted to the
compact teacher/review candidate:

- Active model: `pict\card_models\card_glyph_knn.npz`
- Previous-model backup:
  `pict\card_models\card_glyph_knn_before_20260708_weighted_compact_suitfix.npz`
- Candidate source:
  `video_frames\weighted_runtime_after_red_four_tied3_gate\current_risk_weighted_compact_256_512.npz`
- Full validation:
  `video_frames\promoted_default_suitfix_validation\cv_validation_all_summary.json`
- Promotion gate:
  `video_frames\promoted_default_suitfix_gate\card_model_gate_report.md`

Validation covered 8 root videos / 138 sampled frames with
`real_problem=0`, `board_bad=0`, median `164.3 ms`, and p90 `724.8 ms`.
The promotion gate passed with manual-truth benchmark accuracy `1.0`,
review-diff risk `0`, and the same latency gates.

The live command can now use the default model directly; no
`--card-knn-model` override is needed:

```powershell
python gto.py screen-cv --bbox "x,y,w,h" --auto-bbox --auto-bbox-refresh 10 --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --min-confidence 0.35 --dealer-refresh-frames 4 --format text
```

## Current Auto-Bbox Evidence

As of 2026-07-08, auto-bbox uses a fast path for live calibration:

- reuse dealer-button anchors already found during candidate generation;
- use component detection before expensive full-template matching;
- score large candidate crops on a capped-width copy while keeping output
  coordinates in the original frame;
- only allow titlebar candidates to override when they also have dealer
  confidence.

Latest recorded-video stress diagnostic:

```text
Videos: 8
Rows: 40
Variants: native, loose_8, loose_shift, tight_3, tight_6
Failures: 0
Median: 430.9 ms
P90: 795.1 ms
Max: 931.1 ms
```

Integrated pipeline check:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_autobbox_fast" --run-auto-bbox-diagnostics --auto-bbox-every 300 --auto-bbox-max-frames 1 --no-auto-bbox-problem-frames --format text
```

Report:
`video_frames\card_cv_pipeline_autobbox_fast\card_cv_pipeline_runbook.md`.

## One-Command Pipeline Check

Use `card-cv-pipeline` before a live session or before retraining. It checks the
current bbox, root videos, cropped rank/suit data, HuggingFace probe, promoted
KNN model, optional deep fallback, validation summary, and promotion gate. It
also writes BOM-encoded PowerShell command files so Chinese hero names stay
readable in Windows PowerShell. The generated default live command omits
`--deep-card-model-dir` for lower latency:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_current" --run-smoke --smoke-max-images 20 --format text
```

Current smoke output:

```text
Next stage: ready_for_live_and_training_iteration
Ready for live: True
Ready for training: True
Videos: 8
Crops: rank=247 suit=247
Probe ready: True
Crop quality: missing_rank=[] missing_suit=[] unreadable=0
Smoke: processed=40 accepted=19 review=21
```

The important outputs are:

- `video_frames\card_cv_pipeline_current\card_cv_pipeline_summary.json`
- `video_frames\card_cv_pipeline_current\card_cv_pipeline_runbook.md`
- `video_frames\card_cv_pipeline_current\commands\live.txt`
- `video_frames\card_cv_pipeline_current\commands\label_with_probe.txt`
- `video_frames\card_cv_pipeline_current\commands\distill_and_gate.txt`
- `video_frames\card_cv_pipeline_current\candidate_summary\card_candidate_summary.md`

To refresh the public full-card dataset and convert it into rank/suit glyphs:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_current_ingest" --download-dataset --ingest-dataset --max-external-ingest 200 --format text
```

The current local public dataset ingest produced 52 card images, 52 rank glyphs,
52 suit glyphs, and `skipped_count=0` in
`video_frames\external_card_glyphs\external_summary.json`.

A small real training smoke with balanced current+external+template inputs
completed:

```text
probe: pict\card_models\hf_probe_external_balanced_smoke_20260709
rank sources: current=26, external=13, templates=13
suit sources: current=8, external=4, templates=4
teacher smoke: processed=12 accepted=11 review=1
```

The suit accuracy above is from a deliberately tiny `--max-images-per-class 4`
smoke; it verifies the pipeline, not the final model quality. Use the generated
`train_probe.txt` with the larger per-class cap for the real probe.

Extra local full-card datasets can be merged into the same ingest command with
repeated `--extra-dataset-dir`. The pipeline skips the ingest output directory
if it is accidentally passed as an input, so the glyph export folder is not fed
back into itself:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_candidates" --extra-dataset-dir "D:\your_manual_card_images" --format text
```

The same run also scans existing promotion gates and writes a candidate ranking.
Current candidate scan:

```text
Candidates: 37
Promotable: 3
Best: weighted_compact_256_512_suitfix_candidate_csv
```

The ranking report is:
`video_frames\card_cv_pipeline_candidates\candidate_summary\card_candidate_summary.md`.

The pipeline also audits crop training quality. The current crop set has full
rank/suit coverage and no unreadable images:

```text
rank_count=247
suit_count=247
unreadable=0
missing_rank=[]
missing_suit=[]
rank_label_counts={'A': 9, 'K': 41, 'Q': 13, 'J': 36, 'T': 13, '9': 38, '8': 13, '7': 10, '6': 21, '5': 10, '4': 11, '3': 17, '2': 15}
suit_label_counts={'s': 60, 'h': 82, 'd': 85, 'c': 20}
```

Use `--min-rank-per-label` and `--min-suit-per-label` to raise the threshold
when preparing a larger training run. Use `--no-crop-image-audit` only when a
quick command-generation pass is needed.

To actually run the offline DINOv2 rank/suit teacher from the same pipeline
entry point:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_teacher_smoke" --run-teacher --teacher-max-images 20 --teacher-output-dir "video_frames\card_cv_pipeline_teacher_smoke\teacher" --format text
```

Current teacher smoke output:

```text
Teacher: processed=40 accepted=19 review=21
```

To prove the full teacher -> runtime candidate -> validation -> gate path works
without spending time on a full promotion run:

```powershell
python gto.py card-cv-pipeline --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\card_cv_pipeline_teacher_distill_smoke" --run-teacher --teacher-max-images 20 --teacher-output-dir "video_frames\card_cv_pipeline_teacher_distill_smoke\teacher" --teacher-distill-runtime --teacher-runtime-video "video_frames\validation_new_crop.mp4" --teacher-runtime-every 20 --teacher-runtime-max-frames 5 --teacher-runtime-max-benchmark-samples 80 --teacher-runtime-max-diff-rows 80 --format text
```

That smoke currently reaches the promotion gate and rejects only because the
tiny sample does not cover all baseline review rows:

```text
Teacher: processed=40 accepted=19 review=21
Teacher distill: decision=reject promote=False
Gate: benchmark_card_acc=1.0, review_diff_risk=0, review_diff_missing_rows=136
```

For a real promotion attempt, remove the tight smoke limits and use the full
`distill_and_gate.txt` command generated by `card-cv-pipeline`.

The one-command entry point is `card-big-teacher`. It can reuse an existing probe:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_split" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

To turn accepted teacher crops into a fast runtime candidate and gate it in the
same run, add `--distill-runtime`:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_distill_full" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text
```

The runtime distill and label-queue retrain commands default to
`--runtime-max-median-ms 300 --runtime-max-p90-ms 900` or
`--max-median-ms 300 --max-p90-ms 900`. Raise these only for offline
experiments, not for a model intended for live play.

For a quick mechanical smoke test, restrict the runtime validation to one video
and a few frames:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_distill_smoke" --kind both --max-images 20 --batch-size 16 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --distill-runtime --runtime-video "video_frames\validation_new_crop.mp4" --runtime-every 20 --runtime-max-frames 5 --runtime-max-benchmark-samples 80 --runtime-max-diff-rows 80 --format text
```

That smoke only proves the pipeline runs. It may be rejected by the gate because
the small candidate review does not cover every baseline review row. For a
promotion decision, run the full command above.

Or train a fresh split probe from trusted crops:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --trusted-dir "video_frames\card_review_all_after_red5_override" --output-dir "video_frames\card_big_teacher_dinov2_base" --kind both --model "facebook/dinov2-base" --max-images-per-class 12 --batch-size 16 --format text
```

Rank and suit can use different backbones when needed:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --trusted-dir "video_frames\card_review_all_after_red5_override" --output-dir "video_frames\card_big_teacher_split_models" --kind both --rank-model "facebook/dinov2-base" --suit-model "openai/clip-vit-base-patch32" --max-images-per-class 12 --batch-size 16 --format text
```

This writes:

- `summary.json`: machine-readable run result.
- `runbook.md`: next commands for this exact run.
- `labeled/predictions.csv`: all rank/suit predictions with confidence and margin.
- `labeled/review.csv`: uncertain or rejected crops to check by hand.
- `labeled/rank/<label>/*.png` and `labeled/suit/<label>/*.png`: accepted crops when copying is enabled.
- `runtime_candidate/runtime_distill_summary.json`: candidate training,
  validation, and gate result when `--distill-runtime` is enabled.
- `runtime_candidate/gate/card_model_gate_report.md`: promotion gate report.
- `runtime_candidate/risk_label_queue/label_queue.html`: risky changed crops
  to review when the gate rejects with review-diff risk.
- `runtime_candidate/risk_label_queue/label_queue_sheet.jpg`: one-page visual
  contact sheet with table, card, rank, and suit crops for fast manual triage.

## Why Split Rank And Suit

Rank and suit are different tasks:

- `rank`: 13 classes, `A K Q J T 9 8 7 6 5 4 3 2`. This is mostly printed-character recognition.
- `suit`: 4 classes, `s h d c`. This is mostly color and shape recognition.

Train and run them separately so a strong rank model does not contaminate suit predictions, and vice versa.

## External Big-Model Plan For Cropped Regions

Use external pretrained models only after the table/card layout code has already
cropped stable regions. The model inputs are small images, not the full Tencent
Meeting frame:

- `rank/*.png`: upper-left glyph window for the rank classifier.
- `suit/*.png`: lower glyph/window for the suit classifier.
- `cards/*.png`: optional context image for visual review, not the primary
  runtime classifier input.

The recommended structure is:

1. Export crops from real videos with the current CV pipeline.
2. Run a large frozen visual backbone over `rank` and `suit` crops separately.
3. Train or reuse two lightweight probes:
   - rank probe: 13-way classifier.
   - suit probe: 4-way classifier.
4. Accept only high-confidence, high-margin predictions.
5. Put disagreements into a label queue for manual review.
6. Distill accepted samples into the small realtime KNN/deep runtime model.
7. Promote only if benchmark, review-diff, CV validation, and latency gates all pass.

Do not run the large model in the live Tencent Meeting loop unless explicitly
testing latency. It is too slow and too brittle for the realtime decision path;
its job is to produce better labels for the fast model.

Recommended one-command offline pass:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_distill_full" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --format text
```

Latest local smoke check for the same split-teacher path:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_dinov2_base_smoke_latest" --kind both --max-images 20 --batch-size 8 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --format text
```

Result:

```text
model: rank=facebook/dinov2-base, suit=facebook/dinov2-base
processed: 40
accepted: 22
review: 18
predictions: video_frames\card_big_teacher_dinov2_base_smoke_latest\labeled\predictions.csv
review: video_frames\card_big_teacher_dinov2_base_smoke_latest\labeled\review.csv
```

Full DINOv2-base teacher/distill check:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_base_v1_review" --output-dir "video_frames\card_big_teacher_dinov2_base_distill_full" --kind both --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --local-files-only --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --runtime-risk-queue-max-rows 80 --format text
```

Result:

```text
processed: 494
accepted: 242
review: 252
runtime decision: REJECT
review diff risk: 15
missing rows: 3
validation real_problem_count: 2
median: 27.1 ms
p90: 39.7 ms
```

This confirms that the large model is useful as an offline teacher/review
signal, but its accepted crops are not safe to promote directly into the
runtime KNN model.

To tune thresholds without rerunning DINOv2 embeddings, reuse the generated
`predictions.csv`:

```powershell
python gto.py filter-card-hf-predictions --predictions-csv "video_frames\card_big_teacher_dinov2_base_distill_full\labeled\predictions.csv" --output-dir "video_frames\card_big_teacher_dinov2_base_filter_s3_distill" --rank-score-threshold 0.85 --rank-margin-threshold 0.12 --suit-score-threshold 0.88 --suit-margin-threshold 0.12 --require-current-agreement --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --runtime-risk-queue-max-rows 80 --format text
```

Strict-filter result:

```text
processed: 494
accepted: 153
review: 341
runtime decision: REJECT
review diff risk: 14
missing rows: 3
validation real_problem_count: 2
median: 26.8 ms
p90: 38.2 ms
```

A lower teacher weight with `--runtime-glyph-augment 2` did not improve the
gate: risk stayed at `15`. Therefore the current DINOv2-base crop teacher
should feed manual review queues and training experiments, not replace the
promoted realtime recognizer.

To reduce the rejected candidate into a small manual task, summarize only the
actual card-output changes:

```powershell
python gto.py summarize-card-diff-risks --diff-csv "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\gate\review_diff\card_review_diff_rows.csv" --output-dir "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_summary_card_changes" --no-include-same --format text
```

Current focused summary:

```text
selected rows: 10
risk rows: 14
changed rows: 24
groups: 6
top groups: Ad->4d x5, 9d->8d x1, Jd->6d x1, Js->9h x1, 9h->- x1, 9s->- x1
```

Then build the label queue from those risky changed rows:

```powershell
python gto.py prepare-card-diff-label-queue --diff-csv "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\gate\review_diff\card_review_diff_rows.csv" --output-dir "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_label_queue_card_changes" --format text
```

The default diff queue prefers candidate assets. This is the right mode for
diagnosing bad ROI/crop selection because the contact sheet shows the crop that
the candidate actually used. If the candidate crop is visibly partial or
damaged and the goal is to retrain from cleaner images, rebuild the queue with
baseline assets first:

```powershell
python gto.py prepare-card-diff-label-queue --diff-csv "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\gate\review_diff\card_review_diff_rows.csv" --output-dir "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_label_queue_training_baseline_assets" --prefer-baseline-assets --format text
```

Current queue:

```text
candidates: 10
deduped: 10
selected: 10
queue: video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_label_queue_card_changes\label_queue.html
contact sheet: video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_label_queue_card_changes\label_queue_sheet.jpg
```

Manual labels were filled for this queue on 2026-07-08:

```text
D0001 9d
D0002 Ad
D0003 Ad
D0004 Ad
D0005 Ad
D0006 8s
D0007 6d
D0008 Ad
D0009 9c
D0010 9h
```

The queue audit after labeling:

```text
rows: 10
labeled slots: 10 / 10
invalid labels: 0
missing assets: 0
ready_to_apply: True
ready_to_retrain: True
```

Two training attempts were tested from these labels and rejected by the video
gate:

```text
card_big_teacher_dinov2_base_filter_s3_manual10_runtime:
  benchmark with manual truth: card/rank/suit = 1.0/1.0/1.0
  validation: real_problem=2, median=168.8 ms, p90=742.0 ms
  gate: REJECT, review_diff_risk=13, missing_rows=3

card_big_teacher_dinov2_base_filter_s3_manual10_v2_runtime:
  benchmark with manual truth: card/rank/suit = 1.0/1.0/1.0
  validation: real_problem=3, median=162.8 ms, p90=674.1 ms
  gate: REJECT, review_diff_risk=22, missing_rows=3
```

Interpretation: the DINO/manual branch is useful for finding and correcting
truth labels, but static crop benchmark accuracy is not enough. A candidate can
score perfectly on the old crop set and still fail after the video pipeline
re-crops frames. Promotion still requires the full video-level gate.

The benchmark loader now deduplicates samples by `video/timestamp/frame/slot`
and lets manual `final_card` truth override pseudo `review_reason=ok` truth.
This is required because the manual queue deliberately fixes old pseudo-truth
mistakes such as `Js -> 8s`, `Jd -> 6d`, and `9s -> 9c`.

If rank and suit need different teachers:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --trusted-dir "video_frames\card_review_all_after_red5_override" --output-dir "video_frames\card_big_teacher_split_models" --kind both --rank-model "facebook/dinov2-base" --suit-model "openai/clip-vit-base-patch32" --max-images-per-class 12 --batch-size 16 --format text
```

When the gate rejects because of risky changes, review the generated queue:

```powershell
python gto.py serve-card-label-queue --queue-csv "video_frames\card_big_teacher_dinov2_base_filter_s3_distill\runtime_candidate\risk_label_queue_card_changes\label_queue.csv" --open-browser
```

## 1. Export Crops

Use review export for hero-card crops:

```powershell
python gto.py export-card-review --all --video-dir "video_frames" --output-dir "video_frames\card_review_all" --every 10 --only-suspicious --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal"
```

This creates:

- `video_frames\card_review_all\rank\*.png`
- `video_frames\card_review_all\suit\*.png`
- `video_frames\card_review_all\review.csv`
- `video_frames\card_review_all\review_sheet.jpg`

## 2. Train A Larger Teacher

Use pretrained torchvision backbones for offline teacher models. This can be slow, but it is not used live.

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_review_0611_seed_applied" --model-dir "pict\card_models\deep_teacher_split_v2" --kind both --rank-arch convnext_tiny --suit-arch efficientnet_b0 --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 12 --batch-size 32 --image-size 128
```

For a heavier rank teacher, use ViT only for rank:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_review_0611_seed_applied" --model-dir "pict\card_models\deep_teacher_rank_vit_v1" --kind rank --arch vit_b_16 --pretrained --freeze-backbone --class-balanced-loss --weighted-sampler --epochs 8 --batch-size 16 --image-size 224
```

## 3. Label Cropped Regions With The Teacher

Run the teacher on already-cropped rank/suit regions:

```powershell
python gto.py label-card-crops --input-dir "video_frames\card_review_all" --output-dir "video_frames\card_teacher_labeled_v1" --teacher-model-dir "pict\card_models\deep_teacher_split_v2" --rank-score-threshold 0.90 --rank-margin-threshold 0.20 --suit-score-threshold 0.88 --suit-margin-threshold 0.18
```

Or use separate rank/suit teachers:

```powershell
python gto.py label-card-crops --input-dir "video_frames\card_review_all" --output-dir "video_frames\card_teacher_labeled_v1" --teacher-rank-model-dir "pict\card_models\deep_teacher_rank_vit_v1" --teacher-suit-model-dir "pict\card_models\deep_teacher_split_v2"
```

Outputs:

- `rank/<label>/*.png` and `suit/<label>/*.png`: auto-accepted teacher labels.
- `predictions.csv`: every prediction with score and margin.
- `review.csv`: low-confidence or disagreed samples for manual labeling.
- `review/`: copied uncertain images.

## 4. Optional External HuggingFace Teacher

There is also a zero-shot HuggingFace/CLIP teacher for models that were not
trained inside this repo. It still uses the same cropped regions and still keeps
rank and suit separate:

```powershell
python gto.py label-card-crops-hf --input-dir "video_frames\card_review_all_after_truth_audit_fixes" --output-dir "video_frames\card_hf_clip_labeled_v1" --kind both --rank-model "openai/clip-vit-base-patch32" --suit-model "openai/clip-vit-base-patch32" --rank-score-threshold 0.58 --rank-margin-threshold 0.08 --suit-score-threshold 0.58 --suit-margin-threshold 0.08
```

For a heavier offline pass, use the larger CLIP model:

```powershell
python gto.py label-card-crops-hf --input-dir "video_frames\card_review_all_after_truth_audit_fixes" --output-dir "video_frames\card_hf_clip_large_labeled_v1" --kind both --rank-model "openai/clip-vit-large-patch14" --suit-model "openai/clip-vit-large-patch14" --rank-score-threshold 0.58 --rank-margin-threshold 0.08 --suit-score-threshold 0.58 --suit-margin-threshold 0.08
```

This command writes:

- `rank/<label>/*.png` and `suit/<label>/*.png`: high-confidence external labels.
- `predictions.csv`: every external-model prediction.
- `review.csv`: low-confidence or disagreed samples.
- `review/`: uncertain images copied for manual checking.

Use this as an independent teacher/audit signal. Generic CLIP is weaker than the
repo-trained teacher on the current tiny single-character crops:

```text
openai/clip-vit-base-patch32, 40-sample smoke test:
rank correct: 5 / 20
suit correct: 11 / 20
```

TrOCR small printed was also tested on rank crops and was not reliable enough
for promotion. Do not treat generic CLIP/TrOCR predictions as truth unless the
row is manually reviewed or independently confirmed by the WPT-trained teacher.

## 4b. HuggingFace Embedding Probe

The better external-model path is not zero-shot prompting. Use a public vision
model as a frozen feature extractor, then train two tiny prototype classifiers
on our cropped regions:

- `rank`: 13-way classifier over the cropped rank glyph.
- `suit`: 4-way classifier over the cropped suit glyph. Prefer raw
  `card_review_*` folders instead of already-organized folders, because raw
  folders still contain `cards/*_card.png`; the suit preprocessor can use that
  card crop to keep color/context.

Train a CLIP-base probe from trusted review crops:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --output-dir "pict\card_models\hf_probe_clip_base_v2_review" --kind both --max-images-per-class 12 --batch-size 16 --local-files-only --format text
```

Current smoke result:

```text
rank: source=156 train=130 val=26 val_acc=0.846
suit: source=48 train=40 val=8 val_acc=0.750
```

Use the trained probe to process cropped regions:

```powershell
python gto.py label-card-crops-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_clip_base_v2_review" --output-dir "video_frames\card_hf_probe_clip_base_v2_review_labeled_loose" --kind both --max-images 40 --batch-size 16 --local-files-only --rank-score-threshold 0.45 --rank-margin-threshold 0.03 --suit-score-threshold 0.70 --suit-margin-threshold 0.08 --require-current-agreement --format text
```

Current 80-crop smoke output:

```text
Processed: 80
Accepted: 28
Review: 52
Accepted disagreements with current trusted labels: 0
```

The files are:

- `pict\card_models\hf_probe_clip_base_v2_review\hf_rank_probe.npz`
- `pict\card_models\hf_probe_clip_base_v2_review\hf_suit_probe.npz`
- `video_frames\card_hf_probe_clip_base_v2_review_labeled_loose\predictions.csv`
- `video_frames\card_hf_probe_clip_base_v2_review_labeled_loose\review.csv`

For a heavier model, download/cache it first or omit `--local-files-only`:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --output-dir "pict\card_models\hf_probe_clip_large_v1" --kind both --model "openai/clip-vit-large-patch14" --max-images-per-class 12 --batch-size 8 --format text
```

A non-text vision feature model can also be used. DINOv2 small downloaded and
ran successfully in the current environment:

```powershell
python gto.py train-card-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --output-dir "pict\card_models\hf_probe_dinov2_small_v1_review" --kind both --model "facebook/dinov2-small" --max-images-per-class 12 --batch-size 16 --format text
```

The probe is still an offline teacher. Do not point live `screen-cv` at it until
its accepted labels are merged into a small realtime model and the candidate
passes `benchmark-card-review`, `diff-card-review`, `validate-cv --all`, and
`gate-card-model`.

The probe can now be evaluated as a first-class benchmark/gate evaluator:

```powershell
python gto.py benchmark-card-review --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --output-dir "video_frames\card_benchmark_hf_probe_clip_base_v2" --hf-probe-dir "pict\card_models\hf_probe_clip_base_v2_review" --hf-probe-local-files-only --include-ok-pseudo --format text
```

To run review-diff/gate, first rewrite a candidate `review.csv` from the same
source frames:

```powershell
python gto.py apply-card-hf-probe-review --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --output-dir "video_frames\card_review_all_candidate_hf_probe_clip_base_v2" --probe-dir "pict\card_models\hf_probe_clip_base_v2_review" --batch-size 32 --local-files-only --format text
```

Then gate it:

```powershell
python gto.py gate-card-model --benchmark-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --benchmark-review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --baseline-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --candidate-review-csv "video_frames\card_review_all_candidate_hf_probe_clip_base_v2\review.csv" --output-dir "video_frames\card_model_gate_hf_probe_clip_base_v2" --candidate-name "hf_probe_clip_base_v2" --candidate-evaluator hf_probe --hf-probe-dir "pict\card_models\hf_probe_clip_base_v2_review" --hf-probe-local-files-only --include-ok-pseudo --format text
```

Current gate results:

```text
hf_probe_clip_base_v2: REJECT
benchmark: card=0.737624 rank=0.861386 suit=0.856436
diff: changed=72 risk=53

hf_probe_clip_base_v3_more_wpt: REJECT
benchmark: card=0.737624 rank=0.856436 suit=0.866337
diff: changed=73 risk=53

hf_probe_dinov2_small_v1: REJECT
benchmark: card=0.742574 rank=0.831683 suit=0.876238
diff: changed=65 risk=52
```

Interpretation: public embedding probes are useful as auxiliary review signals
and can auto-label only conservative, agreement-checked crops. DINOv2 small is
the best current public-feature probe on card accuracy, but it is still not
safe as a live recognizer or direct replacement for the promoted runtime model.

The recommended split is:

- Rank recognition: one 13-class probe over `A,K,Q,J,T,9,8,7,6,5,4,3,2`.
- Suit recognition: one 4-class probe over `s,h,c,d`.

Keep them separate even when the source image is the same card corner. Rank
errors and suit errors have different failure modes: ranks are mostly shape
confusion and blur, while suits depend more on color/context and can use the
full card-corner crop. A single 52-class card classifier is useful for final
reporting, but it hides which half of the recognizer failed.

The DINOv2 probe can safely create a conservative teacher set only when it
agrees with the current trusted label:

```powershell
python gto.py label-card-crops-hf-probe --input-dir "video_frames\card_review_all_after_red5_override" --probe-dir "pict\card_models\hf_probe_dinov2_small_v1_review" --output-dir "video_frames\card_hf_probe_dinov2_small_v1_agree_labeled" --kind both --batch-size 32 --rank-score-threshold 0.55 --rank-margin-threshold 0.04 --suit-score-threshold 0.65 --suit-margin-threshold 0.06 --require-current-agreement --format text
```

Current DINO agreement output:

```text
processed=494 accepted=246 review=248 unreadable=0
rank accepted=51
suit accepted=195
accepted disagreements=0
```

Those accepted crops can be distilled back into the fast kNN runtime candidate:

```powershell
python gto.py train-card-classifier --glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --glyph-dir "video_frames\card_hf_probe_dinov2_small_v1_agree_labeled" --model "pict\card_models\card_glyph_knn_candidate_dino_agree_v1.npz" --augment 8 --glyph-augment 6 --format text
```

Do not promote this candidate yet. The current gate rejected it because the
benchmark was good but the review diff still had risky changed labels and a
small latency regression:

```text
candidate: card_glyph_knn_candidate_dino_agree_v1.npz
gate: REJECT
review diff risk=12
validation median=271.8 ms
```

To inspect exactly what the new teacher/candidate changed, build a compact
manual queue from the risky diff rows:

```powershell
python gto.py prepare-card-diff-label-queue --diff-csv "video_frames\card_review_diff_knn_dino_agree_v1\card_review_diff_rows.csv" --output-dir "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40" --max-rows 40 --format text
python gto.py audit-card-label-queue --queue-csv "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40\label_queue.csv" --output-dir "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40" --applied-output-dir "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40_applied" --format text
python gto.py serve-card-label-queue --queue-csv "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40\label_queue.csv" --open-browser
```

Current queue:

```text
selected=12
missing assets=0
labeled slots=0 / 12
ready_to_retrain=False
```

After filling all `final_card0` values in the browser UI, run the audit command
again. It should show `ready_to_retrain=True`; then apply the labels:

```powershell
python gto.py apply-card-review --review-csv "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40\label_queue.csv" --output-dir "video_frames\card_diff_label_queue_knn_dino_agree_v1_top40_applied"
```

Candidate gates can be summarized into one ranked table:

```powershell
python gto.py summarize-card-candidates --output-dir "video_frames\card_candidate_summary_current_goal" --format text
```

Current candidate ranking:

```text
1 self_runtime                    promote card=1.000 risk=0  median=189.6ms
2 knn_compact_guided_v2_runtime   reject  card=1.000 risk=4  median=193.2ms
3 knn_compact_guided_v1_runtime   reject  card=1.000 risk=24 median=173.7ms
4 knn_dino_agree_v1               reject  card=1.000 risk=12 median=271.8ms
5 external_f1nn21_knn             reject  card=1.000 risk=3  median=331.6ms
6 knn_compact_aug6_current        reject  card=0.713 risk=2  median=194.6ms
7 external_lowweight_current      reject  card=0.228 risk=55 median=212.8ms
```

The closest non-promoted candidate is `external_f1nn21_knn`. It has only three
risky changed labels, so it has its own smaller manual queue:

```powershell
python gto.py serve-card-label-queue --queue-csv "video_frames\card_diff_label_queue_external_f1nn21_top20\label_queue.csv" --open-browser
```

Current audit:

```text
rows=3
missing assets=0
labeled slots=0 / 3
ready_to_retrain=False
```

The KNN model can also be compacted using benchmark-guided prototype pruning:

```powershell
python gto.py compact-card-classifier --model "pict\card_models\card_glyph_knn.npz" --output-model "pict\card_models\card_glyph_knn_compact_guided_v2.npz" --benchmark-rows-csv "video_frames\card_model_gate_self_runtime_with_validation\benchmark\card_benchmark_rows.csv" --top-per-sample 8 --min-per-label 256 --max-per-label 512 --format text
```

Current compaction results:

```text
guided_v1: rank 4438->1664, suit 4984->512, median 173.7ms, risk 24, reject
guided_v2: rank 4438->3144, suit 4984->1024, median 193.2ms, risk 4, reject
```

So compaction is useful as an experiment tool, but neither compact candidate
should replace the promoted model yet. The v2 risk queue is:

```powershell
python gto.py serve-card-label-queue --queue-csv "video_frames\card_diff_label_queue_knn_compact_guided_v2_top20\label_queue.csv" --open-browser
```

## 4c. External Full-Card Image Datasets

Full-card image datasets are useful for pretraining or auxiliary data, but they
must first be converted into this repo's cropped rank/suit glyph format:

```powershell
python gto.py download-card-dataset --format text
python gto.py ingest-card-images --dataset-dir "pict\card_datasets\hf_f1nn21_playing_cards\organized_playing_cards" --output-dir "video_frames\external_ingest_f1nn21_playing_cards" --format text
```

Current local default dataset:

```text
repo: F1NN21/playing-cards
images: 52
ingested: 52
skipped: 0
```

It can be tested as a KNN/template candidate without touching the live default:

```powershell
python gto.py train-card-classifier --glyph-dir "video_frames\external_ingest_f1nn21_playing_cards" --glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --model "pict\card_models\card_glyph_knn_candidate_external_f1nn21.npz" --augment 8 --glyph-augment 6 --format text
```

Current result: benchmark is perfect on the overlapping 202-slot benchmark, but
the promotion gate rejects it:

```text
validate-cv --all: real_problem_count 0, board_bad_count 0, median 331.6 ms
diff-card-review: changed 12, risk 3, --fail-on-risk exit code 2
```

Do not promote external full-card data just because it improves or preserves an
overlapping benchmark. The external style is still out-of-domain relative to
WPT crops, so every external-data candidate must pass `diff-card-review`.

## 5. Organize Only Trusted Review Crops

Review exports are flat folders. Before retraining, convert only trusted rows
into training folders. If `--review-csv` is provided, the default allowed reason
is `ok`, so suspicious rows are skipped:

```powershell
python gto.py organize-card-crops --input-dir "video_frames\card_review_all_after_truth_audit_fixes" --review-csv "video_frames\card_review_all_after_truth_audit_fixes\review.csv" --output-dir "video_frames\card_review_all_after_truth_ok_organized" --kind both --format json --compact
```

Current safe organize result:

```text
processed: 494
copied: 388
skipped: 106
```

Do not train from an unfiltered full review export. A full organized export was
tested and rejected because it included polluted labels.

## 6. Retrain A Candidate Realtime Model

After teacher labeling and manual review, retrain a candidate small realtime
model:

```powershell
python gto.py train-deep-card-classifier --glyph-dir "video_frames\card_glyph_export_v2" --extra-glyph-dir "video_frames\card_teacher_labeled_v1" --extra-glyph-dir "video_frames\card_review_all_after_truth_ok_organized" --model-dir "pict\card_models\deep_realtime_candidate" --kind both --arch simple_cnn --class-balanced-loss --weighted-sampler --epochs 12 --batch-size 32 --image-size 64
```

Then validate:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_candidate" --every 10 --max-frames 80 --min-confidence 0.35 --deep-card-model-dir "pict\card_models\deep_realtime_candidate" --format text
```

Two recent candidates were tested and not promoted:

```text
pict\card_models\deep_realtime_v7_organized
pict\card_models\deep_realtime_v8_ok_organized
pict\card_models\deep_realtime_v9_red5_ok
```

Training validation accuracy is not enough to promote a model. Always run the
review benchmark:

```powershell
python gto.py benchmark-card-review --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --output-dir "video_frames\card_benchmark_candidate" --deep-card-model-dir "pict\card_models\deep_realtime_candidate" --include-ok-pseudo --format text
```

Current benchmark on 202 slots (`200` high-confidence pseudo-truth, `2` manual):

```text
runtime recognizer: card 1.000, rank 1.000, suit 1.000
deep_realtime_v2_temporal: card 0.173, rank 0.257, suit 0.604
deep_teacher_convnext_v1: card 0.267, rank 0.505, suit 0.559
deep_realtime_v7_organized: card 0.540, rank 0.688, suit 0.767
deep_realtime_v8_ok_organized: card 0.480, rank 0.515, suit 0.802
deep_realtime_v9_red5_ok: card 0.208, rank 0.515, suit 0.401
```

The `v9_red5_ok` candidate was trained from the red-5 post-fix `ok` organized
set:

```powershell
python gto.py organize-card-crops --input-dir "video_frames\card_review_all_after_red5_override" --review-csv "video_frames\card_review_all_after_red5_override\review.csv" --output-dir "video_frames\card_review_all_after_red5_ok_organized" --kind both --format json --compact
```

It produced `400` copied safe crops but still failed the independent benchmark,
so it is retained only as an experiment.

For future training, build manual labels first:

```powershell
python gto.py prepare-card-label-queue --review-csv "video_frames\card_hand_audit_all_after_red5_override\review.csv" --output-dir "video_frames\card_label_queue_red5_top80" --max-rows 80 --format text
```

After filling `final_card0/final_card1`, apply the labels:

```powershell
python gto.py serve-card-label-queue --queue-csv "video_frames\card_label_queue_red5_top80\label_queue.csv" --open-browser
```

The local UI writes labels directly back to `label_queue.csv`.

Apply the completed labels:

```powershell
python gto.py apply-card-review --review-csv "video_frames\card_label_queue_red5_top80\label_queue.csv" --output-dir "video_frames\card_label_queue_red5_top80_applied"
```

Treat a new deep or KNN/template model as promotable only after it passes all
gates:

```text
1. benchmark-card-review improves on the manual/pseudo review benchmark.
2. validate-cv --all remains real_problem_count=0 and board_bad_count=0.
   With new validation summaries, card_health must also stay clean:
   hero_incomplete_or_missed=0, hero_turn_blocked=0, board_health_bad_frames=0,
   and card_issue_count=0.
3. diff-card-review --fail-on-risk finds no risky card-output changes versus
   the current promoted baseline review.
```

Use the combined promotion command for benchmark, review-diff, and validation
latency gates:

```powershell
python gto.py gate-card-model --benchmark-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --benchmark-review-csv "video_frames\card_review_manual_20260708_kd5d\review.csv" --baseline-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --candidate-review-csv "video_frames\card_review_all_candidate_knn_red5_ok\review.csv" --baseline-validation-summary-json "video_frames\cv_validation_all_after_current_region_bbox\cv_validation_all_summary.json" --candidate-validation-summary-json "video_frames\cv_validation_all_candidate_knn_red5_ok\cv_validation_all_summary.json" --output-dir "video_frames\card_model_gate_candidate_knn_red5_ok" --candidate-name "candidate_knn_red5_ok" --candidate-evaluator knn --knn-model "pict\card_models\card_glyph_knn_candidate_red5_ok.npz" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --include-ok-pseudo --require-validation --max-median-regression-ms 80 --max-p90-regression-ms 150 --fail-on-reject --format text
```

`gate-card-model` writes a single promotion report plus the underlying
benchmark and review-diff artifacts. If validation summaries are provided, it
also checks `real_problem_count`, `board_bad_count`, the detailed
`card_health` counters, and optional latency regression thresholds. It returns
exit code 2 when `--fail-on-reject` is set and the candidate fails any gate.

Current card-health gate smoke:

```text
output: video_frames\card_model_gate_card_health_smoke_20260709
validation_hero_incomplete_or_missed: PASS 0 / 0
validation_hero_turn_blocked: PASS 0 / 0
validation_board_health_bad_frames: PASS 0 / 0
validation_card_issue_count: PASS 0 / 0
```

After a candidate is promoted, run the live preflight health check before using
it in a session:

```powershell
python gto.py cv-health --bbox "136,123,1534,1058" --hero-name "于寻欢" --output-dir "video_frames\cv_health_promoted" --fail-on-not-ready --format text
```

`cv-health` verifies the currently promoted KNN file, latest validation summary,
and promotion gate summary, then writes a `run_live_command.txt` file for the
live screen stream with the recommended OCR, dealer-refresh, villain-profile,
model, and layout-locking options. Deep fallback files are checked only when
`--deep-card-model-dir` is passed explicitly.
Placeholder bbox values such as `x,y,w,h` are rejected unless
`--allow-placeholder-bbox` is used for template generation.

The same rule applies to KNN/template models. `train-card-classifier` can now
consume WPT glyph folders directly:

```powershell
python gto.py train-card-classifier --glyph-dir "video_frames\card_glyph_audit_v5_convnext_temporal\accepted" --glyph-dir "video_frames\card_review_all_after_red5_ok_organized" --model "pict\card_models\card_glyph_knn_candidate_red5_ok.npz" --augment 8 --glyph-augment 6 --format text
```

Test a candidate without replacing the live default:

```powershell
python gto.py validate-cv --all --output-dir "video_frames\cv_validation_all_candidate_knn_red5_ok" --every 10 --max-frames 80 --min-confidence 0.35 --card-knn-model "pict\card_models\card_glyph_knn_candidate_red5_ok.npz" --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format json --compact
```

The current `card_glyph_knn_candidate_red5_ok.npz` is rejected despite passing
the overlapping benchmark, because the promotion gate sees 12 changed card
outputs and 3 risky changes such as `9d -> 8d` and `Jd 6s -> 6h 4s`.
The lower-level review-diff command is:

```powershell
python gto.py diff-card-review --baseline-review-csv "video_frames\card_review_all_after_red5_override\review.csv" --candidate-review-csv "video_frames\card_review_all_candidate_knn_red5_ok\review.csv" --output-dir "video_frames\card_review_diff_candidate_knn_red5_ok" --fail-on-risk --format text
```

## 2026-07-09 Split Rank/Suit Big-Model Route

The most useful "bigger model" path is not to run a heavy model live. Use the
heavy model offline on already-cropped glyphs, split into two independent
tasks:

```text
rank: 13 classes, A K Q J T 9 8 7 6 5 4 3 2
suit: 4 classes, s h d c
```

The split matters because rank and suit fail differently. Rank is mostly small
shape confusion, blur, and partial corner crops. Suit is mostly color/filled
symbol context. The current CLI already supports separate public vision
encoders:

```powershell
python gto.py card-big-teacher --input-dir "video_frames\card_review_all_after_red5_override" --trusted-dir "video_frames\card_review_all_after_red5_ok_organized" --output-dir "video_frames\card_big_teacher_split_rank_suit" --kind both --rank-model "facebook/dinov2-base" --suit-model "openai/clip-vit-large-patch14" --rank-score-threshold 0.50 --rank-margin-threshold 0.12 --suit-score-threshold 0.62 --suit-margin-threshold 0.08 --require-current-agreement --distill-runtime --runtime-every 10 --runtime-max-frames 80 --runtime-max-benchmark-samples 300 --runtime-max-diff-rows 300 --runtime-risk-queue-max-rows 80 --format text
```

For local/offline machines, use `--model auto --local-files-only` first. The
auto resolver checks local HuggingFace cache in this order:

```text
facebook/dinov2-base
facebook/dinov2-small
openai/clip-vit-base-patch32
openai/clip-vit-large-patch14
```

This creates four layers of artifacts:

```text
crops\              exported rank/suit crops, if videos were provided
probe\              hf_rank_probe.npz and hf_suit_probe.npz
labeled\            accepted rank/suit teacher labels plus review.csv
runtime_candidate\  distilled fast KNN model plus validation/gate report
```

Promotion rule: only the distilled runtime candidate can be promoted. The
heavy teacher is for labeling and diagnosis. A candidate must pass
`benchmark-card-review`, `validate-cv --all`, and `diff-card-review` through
`gate-card-model` before it becomes the live default.

Current manual-label finding:

```text
best candidate:
video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank050m012_distill\manual37_patch_seed_only_dedupe_no_deep

benchmark card/rank/suit: 1.000 / 1.000 / 1.000
validation: real_problem=0, board_bad=0, median=26.5 ms, p90=47.1 ms
gate: rejected, review_diff_risk=21, missing_rows=3
```

That result means the classifier part is already strong on clean cropped
glyphs. The remaining errors are mostly crop-quality/ROI-selection issues.
Do not train directly on diagnostic candidate crops when they are visibly
partial or damaged. The candidate-path round worsened review risk because
damaged partial crops polluted the prototype table. Use those crops to debug
ROI selection, not as trusted labels.

A baseline-assets queue was generated for the same remaining 9 risky rows:

```text
video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank050m012_distill\manual37_patch_seed_only_dedupe_no_deep\risk_label_queue_round4_baseline_assets
```

Its contact sheet contains complete Jd/Ad/8s crops, but direct retraining with
the default duplicate policy was rejected and got worse:

```text
candidate:
video_frames\card_big_teacher_dinov2_current_runtime_full_20260709\filtered_rank050m012_distill\manual46_baseline_assets_patch_seed_manual37_no_deep

benchmark card/rank/suit: 0.910 / 0.915 / 0.995
validation: real_problem=0, board_bad=0, median=26.7 ms, p90=49.1 ms
gate: rejected, review_diff_risk=41, missing_rows=3
```

The reason is important: those 9 crops were exact duplicate features of the
manual37 seed. With the default `manual-override` policy they overwrote existing
seed labels without adding new prototypes. A train-only check with
`--seed-conflict-policy keep-seed` produced a model whose `rank_features`,
`rank_labels`, `suit_features`, and `suit_labels` are byte-identical to the
manual37 seed. That is now the safe mode for baseline-assets diagnostic queues.

The manual queue workflow now has a contact sheet, so the fastest review loop is:

```powershell
python gto.py prepare-card-diff-label-queue --diff-csv "<candidate_diff.csv>" --output-dir "<label_queue_dir>" --max-rows 80 --format text
python gto.py serve-card-label-queue --queue-csv "<label_queue_dir>\label_queue.csv" --open-browser
python gto.py retrain-card-label-queue --queue-csv "<label_queue_dir>\label_queue.csv" --output-dir "<candidate_retrain_dir>" --base-glyph-dir "video_frames\empty_glyph_seed_only" --no-templates --format text
```

Use `--prefer-baseline-assets` in the first command when the candidate contact
sheet is diagnosing broken crop selection rather than supplying trainable
glyphs. If the queue is based on baseline assets and is being applied on top of
an already validated seed model, add `--seed-conflict-policy keep-seed` to the
retrain command unless you intentionally want duplicate glyph features in the
queue to overwrite seed prototypes.

`retrain-card-label-queue` now seeds from the promoted KNN by default and
deduplicates exact feature rows so manual labels override older pseudo labels.
Use `--no-seed-model` only for isolation experiments.

## 2026-07-09 Earlier Promoted Runtime Model

The earlier live card model was promoted after combining the big-teacher
active-learning labels with manually verified risk queues.

```text
candidate: glyph_current_prefill_balanced86
model: video_frames\glyph_current_prefill_candidate_20260709\risk_queue_balanced86_retrain\glyph_current_prefill_balanced86.npz
live default: pict\card_models\card_glyph_knn.npz
previous backup: pict\card_models\card_glyph_knn.before_balanced86_20260709.npz
truth queue: video_frames\glyph_current_prefill_candidate_20260709\combined_manual_truth_queue_balanced86\label_queue.csv
```

Promotion evidence:

```text
benchmark card/rank/suit: 1.000 / 1.000 / 1.000
review_diff_risk: 0
review_diff_missing_rows: 0
validation real_problem: 0
validation board_bad: 0
validation median/p90: 31.1 ms / 49.1 ms
gate report: video_frames\glyph_current_prefill_candidate_20260709\risk_queue_balanced86_retrain\gate\card_model_gate_report.md
```

Important ROI fixes from this promotion:

- Hero rank windows now include lower-left candidates for narrow tilted crops,
  which fixes cases like `Ac` where the printed rank sits lower than usual.
- Suit selection now lets a slightly lower raw score with a much stronger
  margin beat a low-margin template match, fixing narrow club crops.
- Black `4` has a classifier-backed override for cases where template matching
  confuses it with `J/K/9`.

## 2026-07-09 Compact Model Check

A conservative compaction attempt was tested against the then-promoted KNN
using the latest benchmark rows:

```powershell
python gto.py compact-card-classifier --model "pict\card_models\card_glyph_knn.npz" --output-model "video_frames\compact_promoted_safe_v3_20260709\card_glyph_knn_compact_safe_v3.npz" --benchmark-rows-csv "video_frames\promoted_default_suitfix_gate\benchmark\card_benchmark_rows.csv" --benchmark-rows-csv "video_frames\glyph_current_prefill_candidate_20260709\risk_queue_balanced86_retrain\gate\benchmark\card_benchmark_rows.csv" --benchmark-rows-csv "video_frames\card_model_gate_knn_compact_guided_v2_runtime\benchmark\card_benchmark_rows.csv" --top-per-sample 24 --min-per-label 384 --max-per-label 768 --format text
```

Compaction result:

```text
rank features: 5121 -> 4349
suit features: 4844 -> 1549
validation: real_problem=0, board_bad=0, hero_incomplete_or_missed=0
validation median/p90: 151.8 ms / 831.7 ms
gate: rejected
review_diff_risk: 72
gate report: video_frames\compact_promoted_safe_v3_20260709\gate\card_model_gate_report.md
```

Do not promote compact KNN candidates yet. Even conservative compaction keeps
the benchmark accuracy at 1.0 but changes fragile validation reads such as
`6h -> 4h`, `3h -> 8h`, and `9c -> 9s`.

## 2026-07-09 Split Teacher Seedguard Promotion

The current live card model is now a distilled runtime KNN trained from a
rank/suit split big-teacher ensemble. The heavy HuggingFace models are used
offline only: they label the cropped rank and suit glyphs, a 2-of-3 teacher vote
filters high-confidence crops, and the accepted crops are distilled into the
fast KNN recognizer used by `screen-cv`.

```text
teacher input: video_frames\big_model_split_full_20260709\ensemble_vote2of3\predictions.csv
accepted teacher crops: 393 / 494
candidate: ensemble_vote2of3_seedguard_defaulttruth_coldgate
model: video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended_seedguard_defaulttruth_coldgate_distill\runtime_candidate\ensemble_vote2of3_seedguard_defaulttruth_coldgate.npz
live default: pict\card_models\card_glyph_knn.npz
previous backup: pict\card_models\card_glyph_knn.before_bigteacher_seedguard_20260709.npz
truth queue: video_frames\glyph_current_prefill_candidate_20260709\combined_manual_truth_queue_balanced86\label_queue.csv
```

Promotion evidence:

```text
benchmark card/rank/suit: 1.000 / 1.000 / 1.000
truth sources: manual=81, pseudo-ok=119
review_diff_risk: 0
review_diff_missing_rows: 0
validation real_problem: 0
validation board_bad: 0
validation median/p90: 29.9 ms / 48.0 ms
gate report: video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended_seedguard_defaulttruth_coldgate_distill\runtime_candidate\gate\card_model_gate_report.md
health report: video_frames\cv_health_promoted_bigteacher_seedguard_20260709\cv_health_report.md
```

Runtime defaults were updated so future big-teacher distillation gates use the
manual truth queue when present, plus the pseudo-ok fallback review CSV. This
avoids treating the old baseline prediction as truth during model promotion.
The runtime gate also clears the glyph-classification cache between review
export and validation, so timing evidence is not inflated by reusing the same
sampled frame crops.

## Live Command

Use the current promoted realtime model, not the heavy teacher or unpromoted
candidate:

```powershell
python gto.py screen-cv --bbox-file "video_frames\screen_calibrate\bbox.json" --auto-bbox --auto-bbox-refresh 10 --lock-layout --hero-name "于寻欢" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --villain "standard" --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

The older deep fallback command below is kept only as an explicit experimental
fallback example:

```powershell
python gto.py screen-cv --bbox-file "video_frames\screen_calibrate\bbox.json" --auto-bbox --auto-bbox-refresh 10 --lock-layout --hero-name "于寻欢" --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --villain "standard" --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --deep-card-model-dir "pict\card_models\deep_realtime_v2_temporal" --format text
```
