# CV Completion Audit 2026-07-09

This audit records the current evidence for the poker CV workstream: table
localization/screenshots, rank/suit recognition, training data, offline teacher
models, runtime promotion, and live commands.

## Current Verdict

```text
Ready for live: true
Ready for training iteration: true
Current default model: pict\card_models\card_glyph_knn.npz
Best candidate: ensemble_vote2of3_seedguard_defaulttruth_coldgate
```

Primary audit output:

```text
video_frames\card_cv_pipeline_final_audit_20260709\card_cv_pipeline_summary.json
video_frames\card_cv_pipeline_final_audit_20260709\card_cv_pipeline_runbook.md
```

## Evidence Matrix

| Requirement | Evidence | Status |
|---|---|---|
| Table/window localization works on recorded validation videos | `video_frames\card_cv_pipeline_final_audit_20260709\auto_bbox\auto_bbox_diagnostics_summary.json`: 8 videos, 45 rows, 0 failures | PASS |
| Manual bbox is saved and reusable | `video_frames\screen_calibrate\bbox.json`: `141,382,1152,807` | PASS |
| Rank/suit crops exist for the current runtime review set | `video_frames\card_cv_pipeline_final_audit_20260709\card_cv_pipeline_summary.json`: rank=247, suit=247, missing labels empty | PASS |
| External public card dataset is available locally | `pict\card_datasets\hf_f1nn21_playing_cards\dataset_download_summary.json`: 52/52 parsed, complete deck, 0 missing, 0 duplicates | PASS |
| Offline larger teacher/probe path exists | `pict\card_models\hf_probe_rank_large_suit_base_next`; teacher predictions under `video_frames\big_model_split_full_20260709` | PASS |
| Runtime model is promoted by gate | `video_frames\promoted_default_bigteacher_seedguard_20260709_gate\card_model_gate_report.md`: decision promote | PASS |
| Runtime hand-card recognition has no validation problems | `video_frames\promoted_default_bigteacher_seedguard_20260709_validation\cv_validation_all_summary.json`: real_problem=0, board_bad=0, hero_incomplete_or_missed=0 | PASS |
| Current promoted health check is ready | `video_frames\cv_health_promoted_bigteacher_seedguard_20260709\cv_health_report.md`: Decision READY | PASS |
| Candidate scan ranks promoted model best | `video_frames\candidate_summary_after_bigteacher_seedguard_coldgate_20260709\card_candidate_summary.md`: best candidate is `ensemble_vote2of3_seedguard_defaulttruth_coldgate` | PASS |
| Regression tests pass | `rtk pytest -q`: 109 passed | PASS |

## Promoted Model

```text
candidate: ensemble_vote2of3_seedguard_defaulttruth_coldgate
source model:
video_frames\big_model_split_full_20260709\ensemble_vote2of3_filtered_recommended_seedguard_defaulttruth_coldgate_distill\runtime_candidate\ensemble_vote2of3_seedguard_defaulttruth_coldgate.npz

live default:
pict\card_models\card_glyph_knn.npz

previous backup:
pict\card_models\card_glyph_knn.before_bigteacher_seedguard_20260709.npz
```

Promotion metrics:

```text
benchmark card/rank/suit: 1.000 / 1.000 / 1.000
truth sources: manual=81, pseudo-ok=119
review_diff_risk: 0
review_diff_missing_rows: 0
validation real_problem: 0
validation board_bad: 0
validation median/p90: 29.9 ms / 48.0 ms
```

## Live Commands

Run the health check first:

```powershell
python gto.py cv-health --bbox-file "video_frames\screen_calibrate\bbox.json" --output-dir "video_frames\cv_health_promoted_bigteacher_seedguard_20260709" --fail-on-not-ready --format text
```

Then use the generated command:

```text
video_frames\cv_health_promoted_bigteacher_seedguard_20260709\run_live_command.txt
```

Current live command:

```powershell
python gto.py screen-cv --bbox-file "video_frames\screen_calibrate\bbox.json" --auto-bbox --auto-bbox-refresh 10 --lock-layout --output-dir "video_frames\screen_live" --trigger frame --every 1 --with-advice --effective-stack 100 --villain "standard" --min-confidence 0.35 --ocr-scale 0.65 --dealer-refresh-frames 4 --format text
```

## Repro Commands

Dataset summary:

```powershell
python gto.py download-card-dataset --local-files-only --format text
```

Final pipeline audit:

```powershell
python gto.py card-cv-pipeline --output-dir "video_frames\card_cv_pipeline_final_audit_20260709" --bbox-file "video_frames\screen_calibrate\bbox.json" --video-dir "video_frames" --crop-dir "video_frames\current_runtime_final_review" --probe-dir "pict\card_models\hf_probe_rank_large_suit_base_next" --local-files-only --candidate-output-dir "video_frames\card_cv_pipeline_final_audit_20260709\candidate_summary" --run-auto-bbox-diagnostics --auto-bbox-output-dir "video_frames\card_cv_pipeline_final_audit_20260709\auto_bbox" --auto-bbox-every 300 --auto-bbox-max-frames 2 --format text
```

Regression:

```powershell
rtk pytest -q
```
