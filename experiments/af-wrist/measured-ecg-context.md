# Trailing measured ECG duration and AF alerts

The [measured ECG control](measured-ecg-control.md) showed strong stable-window discrimination with a 30-second input and few new alerts during complete AF episodes. This exploratory comparison asks whether a shorter input captures a new rhythm sooner on the same decision clock.

## Method

1. Read either the final 10 seconds or the final 30 seconds of native 500 Hz ECG ending at each existing decision time. Both routes use the same seven interval features derived from measured voltage. The original 30-second reference labels remain the training labels for both lengths.
2. Keep only rows supported by both ECG lengths and all three original PPG routes. The common footprint retained 114 stable fitting rows, 512 of the 513 original evaluation rows, and 173 of the 175 history rows. The 10-second route lacked supported intervals in one evaluation row. The 30-second route lacked locally cached ECG for two history rows. These three rows remain on the clock as unavailable decisions for both models.
3. Refit both classifiers at C=1 on the same fitting rows and exact original outer and inner participant assignments. Use seed 2026, five outer folds, at most three inner folds, and the existing cutoff objective. Keep the 10-second decision cadence, immediate alert policy and causal history replay. All 17 complete AF episodes and 46.5 ECG-observed wrist minutes remain in the denominators.

## Results

| Held-out outcome | 30-second ECG | 10-second ECG |
| --- | ---: | ---: |
| Stable-window ROC AUC | 0.9860 | 0.9849 |
| Complete AF episodes detected | 3/17 (17.6%) | 9/17 (52.9%) |
| Wrist false starts | 0 | 0 |
| Positive evaluation decisions | 93 | 105 |
| Pre-onset starts across 10 ECG-verified eligible episodes | 2 | 3 |
| Median delay among each model's detections | 13.530 s | 20.640 s |
| Inner calibration folds meeting the 80% target | 0/5 | 1/5 |

Stable-window ROC AUC used the same 34 AF and 80 non-AF rows. The refitted 30-second model reproduced the original event summary on the slightly smaller common footprint. The 10-second fit gained six episode detections and lost none. The number of misses with in-episode scores below cutoff fell from 13 to seven. One episode had no score in either route. The median paired delay moved 10 seconds earlier among the three detections shared by both models.

Shorter measured ECG context improved episode detection on this development split, with almost the same stable-window ROC AUC. The 10-second route still fell below the project's 80% episode target and added one pre-onset start. The comparison changes the feature input duration and refits and recalibrates both models. A separate comparison on PPG inputs would establish whether the effect transfers to the wearable route.

## Reproduce

Use the same [local measured ECG sources and development archives](measured-ecg-control.md#reproduce) as the 30-second control.

```powershell
python -m ppg_af.af_wrist_measured_ecg_context `
  --source-inventory data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --measured-control-results experiments/af-wrist/measured-ecg-control-results.json `
  --output runs/measured-ecg-context-results.json
```

The [context comparison](../../ppg_af/af_wrist_measured_ecg_context.py) checks matched support and original fold assignments. The [shared reader](../../ppg_af/af_wrist_measured_ecg.py) reads both signal durations from the pinned local source. The [aggregate results](measured-ecg-context-results.json) contain the coverage and paired outcomes.
