# Split sensitivity of quality-aware wrist AF classification

The [quality study](quality-study.md) observed 14 detections and four wrist false starts on the original development folds. This follow-up tests whether that result persists when the same ten participants are grouped differently. The five seeds are 2026 through 2030. Seed 2026 is the original protocol. No additional participants are used.

For each seed, fit the original seven-feature template and the fixed quality-interaction model at logistic C=1. Form identical participant folds for both models and select each fold's threshold from its inner held-out predictions. All seeds keep the same 17 complete AF episodes, 2,790 ECG-observed wrist seconds, decision clock, and eligibility. The program checks that paired models use the same fitting and scoring support for each seed.

| Fold seed | Original detections | Original false starts | Quality-interaction detections | Quality-interaction false starts |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12/17 | 6 | 14/17 | 4 |
| 2027 | 3/17 | 4 | 12/17 | 2 |
| 2028 | 8/17 | 5 | 13/17 | 4 |
| 2029 | 11/17 | 6 | 5/17 | 1 |
| 2030 | 1/17 | 2 | 1/17 | 0 |

The quality-interaction model reduced false starts for all five seeds and equaled or increased detections in four. The absolute result was unstable. Its recall ranged from 1/17 to 14/17 and met the 80% target only on the original split. Each seed reuses the same underlying recordings, so these results describe partition sensitivity within one small cohort.

## Cutoff diagnostic

After observing the split sensitivity, replay the same held-out scores at one preselected cutoff of 0.5. This diagnostic removes only threshold selection. It uses the same starts, denominators, scoreable cells and alert history. The original template detected 9–12 episodes with 3–5 false starts across seeds. The quality-interaction model detected 8–11 with 1–3 false starts. Its range narrowed from 1–14 detections, but neither model reached 80% at 0.5. In the quality-interaction model, the calibrated cutoffs marked 96–270 of the same 513 evaluation decisions positive across the five seeds. The fixed cutoff marked 300–308 positive. This identifies sensitivity to the cutoff policy in addition to the change in fitting groups.

## Reproduce

Use the locked development archives from the [decision cadence study](clock-study.md#reproduce).

```powershell
python -m ppg_af.af_wrist_quality_stability `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --output runs/quality-stability-results.json
```

The [split-sensitivity driver](../../ppg_af/af_wrist_quality_stability.py) reuses the original fold-local calibration and ECG event scoring. The [aggregate report](quality-stability-results.json) includes each partition's paired episode, burden, and pre-onset results.
