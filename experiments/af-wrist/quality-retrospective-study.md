# Retrospective wrist quality replay

On the previously evaluated 25-person wrist cohort, adding causal PPG quality to the template ECG model increased AF episode detection and wrist false alerts. Both quality variants stayed within 10 percentage points of the historical model's sensitivity. False starts rose in the retrospective replay after falling from five to three on the development calibration group.

## Methods

1. **Freeze the two feature variants.** Both used the seven pulse-template interval features. The first added the existing PPG quality value. The second also added two products, quality times normalized RMSSD and quality times pNN50. The quality value came from the same trailing 30-second native PPG as the timing features. The code checks the cached quality value against the waveform on every usable row.
2. **Fit and calibrate on the original development groups.** A standard scaler and class-balanced logistic regression at fixed `C=1` were fit on the same 43 stable windows from five participants (21 AF, 22 non-AF). The original other five participants supplied every cutoff candidate and calibration outcome. The original rule minimizes false starts among cutoffs that reach 80% episode sensitivity. When every cutoff falls below the target, it maximizes sensitivity first and minimizes false starts second. Both quality models reached 12/17 (70.6%) with three false starts and 26.2 seconds of median delay. The historical calibration reached the same 12/17 with five false starts and 10.6 seconds of median delay.
3. **Verify the sealed archives and reconcile the historical model.** After freezing both variants, the replay loaded the original 25 participants, verified all five sealed archive scopes and the original model and coverage receipts, and reproduced the historical result. All three models used the original shared feature gate and 10-second immediate alert clock with native predecessor state. The replay used the cutoffs from development.
4. **Pair outcomes on the same reference clock.** Episode sensitivity uses the same 260 complete ECG episodes. The 82 partial episodes remain separate. Definite false starts use the same 7,500 seconds of ECG-observed wrist monitoring, of which 6,872 seconds were non-AF. The study pairs false starts by participant and decision timestamp. A start that shifts to another timestamp counts once on each side of the comparison.

## Retrospective results

| Model | Complete AF episodes detected | Sensitivity | Wrist false starts | False starts per observed wrist hour | Median delay among detected episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical template ECG | 155/260 | 59.6% | 45 | 21.60 | 22.3 s |
| Template ECG + quality | 165/260 | 63.5% | 53 | 25.44 | 26.6 s |
| Template ECG + quality and interactions | 168/260 | 64.6% | 48 | 23.04 | 27.7 s |

| Variant versus historical template ECG | AF episodes gained / lost | Median delay change on jointly detected episodes | False starts on both / historical only / variant only |
| --- | ---: | ---: | ---: |
| Quality | 28 / 18 | 0 s across 137 episodes | 29 / 16 / 24 |
| Quality and interactions | 32 / 19 | 0 s across 136 episodes | 28 / 17 / 20 |

For quality alone, 33 jointly detected episodes were detected earlier, 26 later and 78 at the same time. The interaction variant was earlier on 32, later on 31 and unchanged on 73. Each variant's median over all its detected episodes also includes episodes unique to that model.

All three models fell below the 80% sensitivity target. The interaction variant gained a net 13 detections and added three wrist false starts, with a 5.4-second higher median delay across all detected episodes. This comparison reuses a cohort already analyzed in the project. One participant contributed 157 of the 260 complete episodes, so the pooled counts give that participant substantial weight.

## Key code and reproduction

- Verify the trailing PPG window and align quality with the original feature rows. [`build_quality_matrices`](../../ppg_af/af_wrist_quality_study.py)
- Separate development fitting from retrospective inference. [`freeze_variants` and `replay_variants`](../../ppg_af/af_wrist_quality_retrospective.py)
- Pair first alerts within episodes and wrist false starts at their decision timestamps. [`compare_alert_runs`](../../ppg_af/af_wrist_vote_compare.py)
- Check common support, paired changes and receipt hashes. [Aggregate results](quality-retrospective-results.json)

Run the commands from the repository root with the original verified local archive location in `RAW`. Set `PRIVATE_LOCK` and `REPORT` to fresh local paths, and record the SHA-256 printed by `freeze` in `LOCK_SHA` before running `replay`.

```bash
python -m ppg_af.af_wrist_quality_retrospective freeze \
  --protocol-lock "$RAW/holdout-window-seeded-lock.json" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --development-parent-raw "$RAW/cohort-seeded-development" \
  --development-parent-features "$RAW/cohort-seeded-development-features" \
  --development-preroll "$RAW/cohort-seeded-development-preroll" \
  --development-stride-raw "$RAW/cohort-seeded-development-stride10" \
  --development-stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --output "$PRIVATE_LOCK"

python -m ppg_af.af_wrist_quality_retrospective replay \
  --protocol-lock "$RAW/holdout-window-seeded-lock.json" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --variant-lock "$PRIVATE_LOCK" --variant-lock-sha256 "$LOCK_SHA" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --historical-results experiments/af-wrist/template-independent-results.json \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --sealed-inputs "$RAW/template-independent-prepared" --output "$REPORT"
```

Keep the development lock and archives private. The aggregate JSON contains participant counts and event summaries.
