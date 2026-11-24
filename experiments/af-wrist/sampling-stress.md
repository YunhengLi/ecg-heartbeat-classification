# Synthetic 50 Hz sampling stress on fixed wrist classifiers

This development experiment measures how the fixed template and quality-interaction classifiers respond when their held-out wrist PPG inputs pass through synthetic 50 Hz sampling. It uses the original seed-2026 participant folds and cutoff policy from the [quality comparison](quality-study.md).

## Input and model controls

Recover each originally scoreable 30-second, 100 Hz PPG core from the original raw and finer-grid archives. Verify the original seven template features and quality interactions by re-extracting them on the native signals. Apply polyphase anti-aliasing for the 100 to 50 Hz step, then interpolate back to 100 Hz with a second polyphase filter. Each input is processed separately, with linear padding computed from its own endpoints. The 3,000 → 1,500 → 3,000 sample sequence keeps the same grid origin, trailing input period, and decision timestamp. No measured PPG samples after the decision enter the calculation.

For each outer fold, reproduce the original model using its native 100 Hz fitting rows. Check that held-out native scores match the original run exactly. Re-extract features from the transformed inputs with the same fixed external PTT template, score them with the original model, and apply its original inner-calibrated threshold. Preserve the original shared scoreability, predecessor replay, 17 complete episode references, and 2,790 seconds of wrist ECG exposure. No independent predictions are generated.

## Results

| Classifier | Complete ECG-reference episodes | Native 100 Hz | Synthetic 50 Hz round trip |
| --- | ---: | --- | --- |
| Plain template | 17 | 12 detections, 6 false starts | 12 detections, 7 false starts |
| Template with quality interactions | 17 | 14 detections, 4 false starts | 14 detections, 4 false starts |

All 881 originally scoreable rows kept finite scores, including 513 evaluation decisions and 175 predecessor decisions. Both classifiers retained the exact detected episodes and their alert timing. The quality-interaction model kept the exact four wrist false-start timestamps. The plain template kept the six original wrist false starts and added one.

Across the 513 scoreable evaluation decisions, the 90th-percentile absolute score change was 0.0071 for the plain model and 0.0589 for the quality model. Their maximum changes were 0.385 and 0.250. Three evaluation decisions crossed from positive to negative in the plain model. The quality model had nine such crossings and one crossing from negative to positive. Both arms retained the same pre-onset false-start timestamps for the ten episodes eligible for that diagnostic.

The result quantifies sensitivity to the specified filter and sampling grid on the original recordings. A prospective cohort with native 50 Hz signals is a separate test of transfer across acquisition hardware, wear conditions, participants, and reference labels.

## Reproduce

Use the [development clock archives](clock-study.md#reproduce), its lock, the fixed quality-study aggregate, and the original local PTT WFDB source files.

```powershell
python -m ppg_af.af_wrist_sampling_stress `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --quality-receipt experiments/af-wrist/quality-study-results.json `
  --ptt-dir data/ptt-raw --output runs/sampling-stress-results.json
```

The [study driver](../../ppg_af/af_wrist_sampling_stress.py) verifies the native scores and replays the same fixed thresholds after resampling. The [aggregate report](sampling-stress-results.json) includes score shifts for evaluation, wrist monitoring, predecessor history, and original stable fitting scopes.
