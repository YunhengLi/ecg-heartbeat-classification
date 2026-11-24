# PPG context length for pulse-template AF alerts

The [measured-ECG context comparison](measured-ecg-context.md) detected more episodes with a 10-second window than a 30-second window. This experiment checks whether the same change helps when the rhythm features come from an ECG template reconstructed from wrist PPG.

## Fixed comparison

1. Read the locked AFPPGECG development archives for the ten participants. Rebuild all 881 shared 30-second template feature rows from the original 100 Hz wrist PPG. Require exact agreement with the cached features and verify the raw and feature clocks.
2. Rebuild a second template from only the last 10 seconds of each existing input. Both lengths use the same calibration from 22 external PTT participants and two seconds of left reflection within the chosen input. Reflection supplies the template detector's left edge. Both variants use the same seven beat-timing features, and every sample comes from at or before the decision time.
3. Fit the same logistic model at C=1 using the original stable ECG labels. Preserve the five outer participant folds and inner partitions, fold-local threshold selection, original 10-second decision clock, and causal predecessor replay. Compare both models on the shared input rows. All 513 evaluation, 175 predecessor, and 114 stable fitting rows stayed eligible.

| PPG context | Detected AF episodes | Wrist false starts | Pre-onset starts | Stable-window ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| 30 seconds | 12/17 | 6 | 3 | 0.819 |
| 10 seconds | 12/17 | 9 | 4 | 0.822 |

The 30-second model reproduced the published full-support result on the same 46.5 minutes of wrist monitoring. Shortening context gained three episodes and lost three. Alert delay fell a median of 10 seconds among the nine episodes detected by both models. The median among each model's own detections rose from 20.5 to 33.8 seconds. The shorter rhythm window preserved pooled sensitivity and increased wrist and pre-onset false starts. The pre-onset comparison uses the ten episodes with a fully observed ECG-confirmed non-AF lead-in.

## Exploratory quality follow-up

After observing the plain-template comparison, repeat the same 30-second versus 10-second contrast with the [quality-interaction model](quality-study.md). Both fits receive the identical quality scalar computed from the trailing 30 seconds of native PPG. Multiply this scalar by the same two irregularity features from the corresponding template. This changes the context length used for rhythm features while keeping the quality input fixed at 30 seconds. Use only the original seed 2026 and the same participants and support.

| Rhythm context with 30-second quality | Detected AF episodes | Wrist false starts | Pre-onset starts | Stable-window ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| 30 seconds | 14/17 | 4 | 3 | 0.685 |
| 10 seconds | 8/17 | 5 | 1 | 0.711 |

The 30-second quality-interaction model reproduced its published result exactly. The 10-second variant gained one detection and lost seven. Alerting was a median of 10 seconds later among the seven events detected by both variants. Misses with all scores below cutoff grew from one to seven. The decrease in pre-onset starts and improvement in stable-window AUC came with fewer detections and more wrist false starts. The 30-second timing features remain the development reference for both template variants.

## Reproduce

Prepare the locked development and stride archives from the [decision cadence study](clock-study.md#reproduce) and the same local [PTT calibration inputs](../rddm/README.md#2-run-the-external-ptt-benchmark). Run from the repository root.

```powershell
python -m ppg_af.af_wrist_template_context `
  --ptt-data data/ptt-raw --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock `
  --clock-results experiments/af-wrist/clock-study-results.json `
  --quality-results experiments/af-wrist/quality-study-results.json `
  --output runs/template-context-results.json
```

The [experiment driver](../../ppg_af/af_wrist_template_context.py) verifies the input windows and reproduces both published 30-second models before comparing context lengths. The [aggregate results](template-context-results.json) record sample support, paired episode and false-start changes, alert delays, and calibration status.
