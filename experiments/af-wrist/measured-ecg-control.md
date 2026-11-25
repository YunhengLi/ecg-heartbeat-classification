# Measured ECG and AF event detection on a common clock

This study compares the original template ECG with simultaneously measured native ECG using the same seven interval features, class-balanced logistic regression at C=1, five participant folds, and fold-local cutoff selection. It asks whether discrimination on stable AF windows is enough to detect the onset of a complete episode.

## Method

Read the native 500 Hz ECG voltage over the same trailing 30 seconds used by each PPG decision. The feature calculation detects beats from voltage and does not use the annotated R peaks. The reference labels define training eligibility and event scoring. The common footprint retains all 114 fitting rows and 513 evaluation rows from the existing PPG comparison. Two of 175 preceding history rows lacked locally cached ECG voltage and were unavailable for both routes in this comparison. The real ECG route used no extra monitoring time.

## Results

| Measure | PPG-derived template ECG | Measured ECG |
| --- | ---: | ---: |
| Pooled ROC AUC on held-out stable windows | 0.819 | 0.986 |
| Complete AF episodes detected with fold-selected cutoffs | 12/17 | 3/17 |
| Wrist false starts with fold-selected cutoffs | 6 | 0 |
| Complete episodes detected with a fixed 0.5 cutoff | 12/17 | 2/17 |
| Wrist false starts with a fixed 0.5 cutoff | 4 | 0 |

Stable-window ROC AUC uses 34 AF and 80 non-AF windows labeled from the reference. All complete episodes still contribute to the event denominator. In the measured ECG route, thirteen missed episodes had in-episode scores below the selected cutoff and one had no score. The original template result reproduced on the common footprint. The fixed 0.5 diagnostic was added after observing the event result. Neither route reached the 80% target in any inner calibration fold.

The model classified stable measured ECG windows well on these participants. With the same seven features and alert policy, that strength did not translate to reliable starts on the episode clock. The result concerns the tested model and scoring protocol, with 17 complete episodes from four participants. A waveform or window-classification metric alone is insufficient to estimate alert performance.

## Reproduce

Use the staged development archives and the pinned AFPPGECG v6 inventory. Supply locally available complete ECG files or sparse source ranges.

```powershell
python -m ppg_af.af_wrist_measured_ecg `
  --source-inventory data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --output runs/measured-ecg-control-results.json
```

The [reference driver](../../ppg_af/af_wrist_measured_ecg.py) aligns the voltage features to the native clock and verifies the paired scoring support. The [aggregate report](measured-ecg-control-results.json) includes the missing-score audit and fixed-cutoff diagnostic.
