# Paired uncertainty for the fixed quality comparison

The original seed-2026 participant folds yielded 12 of 17 complete AF episodes and six wrist false starts for the template ECG baseline. The quality-interaction model detected 14 of 17 and produced four false starts over the same 2,790 seconds of ECG-observed wrist time. [Fixed quality comparison](quality-study.md)

Reproduce both held-out replays and reconcile episode detections, false starts, and observed wrist time by participant. In each of 10,000 draws, sample ten participant clusters with replacement using seed 20261124. Apply the same draw to both arms. Recalculate pooled sensitivity and false starts per observed wrist hour, then take the 2.5th and 97.5th percentiles of the paired differences. Model parameters, cutoffs, and alerts stay fixed during resampling.

| Quality interactions minus baseline | Observed difference | Descriptive 95% percentile interval | Valid draws |
| --- | ---: | ---: | ---: |
| Complete-episode sensitivity | +11.76 percentage points | 0.0 to +15.0 | 9,942 |
| False starts per ECG-observed wrist hour | −2.58 | −8.0 to 0.0 | 10,000 |

Four participants supplied complete episodes and one contributed 13 of 17. Exactly 58 draws sampled only participants without complete episodes. Those draws have no sensitivity denominator. Every participant supplied wrist ECG exposure, so all draws contribute to the false-start interval. Both intervals touch zero. The intervals describe variation from resampling these fixed development predictions. The [split-sensitivity study](quality-stability.md) separately measures variation under different participant partitions.

## Reproduce

Prepare the [original development clock inputs](clock-study.md#reproduce) and supply the original quality-study aggregate as a receipt. The command verifies both held-out summaries before running the resampling.

```powershell
python -m ppg_af.af_wrist_quality_uncertainty `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --quality-receipt experiments/af-wrist/quality-study-results.json `
  --output runs/quality-uncertainty-results.json
```

The [aggregate report](quality-uncertainty-results.json) includes the exact denominators and valid-draw counts. The [study driver](../../ppg_af/af_wrist_quality_uncertainty.py) reuses the original participant grouping and scoring functions.
