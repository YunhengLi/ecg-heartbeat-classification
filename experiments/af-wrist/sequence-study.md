# Chronological pulse context for wrist AF alerts

The [independent error audit](template-independent-error-audit.md) identified alerts that started before a complete AF episode and persisted into it. This development comparison asks whether the order of short-interval PPG features helps the classifier place alerts closer to the change in rhythm.

## Method

Split each existing trailing 30-second PPG input into three adjacent 10-second bins. Extract the median pulse interval, normalized interval IQR, and normalized successive-interval RMSSD in each bin. An indicator records whether at least two adjacent intervals support those estimates. Every decision uses its own archived PPG core. The calculation excludes archive padding and needs no samples from another decision row.

Compare the original seven template ECG features with two additions of 12 PPG features. The chronological model keeps the bins in measurement order. The order-invariant control sorts each row's identical bin tuples by RMSSD, IQR, median interval and support. This preserves the measured values and their within-bin associations while removing their time positions. Bins without enough support use an explicit zero support flag. All three bins were supported in 112 of 114 fitting rows, 510 of 513 scoreable evaluation rows, and 171 of 175 scoreable prehistory rows.

Use the same five outer participant folds, inner cutoff search, eligible stable fitting labels, immediate alert rule, and 10-second decision cadence. Check fit and score support against the original classifier. Compare definite in-episode alert starts, false starts over the same 2,790 ECG-observed wrist seconds and paired first-alert delay. A separate diagnostic counts definite starts in the 30 seconds before complete episodes when that time is observed and ECG confirms non-AF throughout.

## Fixed-C results

| Model, logistic C=1 | Complete episodes detected | Wrist false starts | Pre-onset non-AF starts |
| --- | ---: | ---: | ---: |
| Original template ECG | 12/17 | 6 | 3 |
| Template plus order-invariant pulse bins | 9/17 | 5 | 2 |
| Template plus chronological pulse bins | 12/17 | 9 | 7 |

Compared with the original, chronological input gained three detections and lost three. For the nine episodes both detected, median paired delay change was zero seconds. The chronological model matched pooled sensitivity and increased both wrist false starts and pre-onset starts. The order-invariant control reduced wrist false starts by one while losing three net detections.

## Follow-up with inner-selected regularization

After seeing the fixed-C results, test whether stronger or weaker regularization can help. For each model and outer fold, the existing inner-validation rule chooses C from 0.1, 1 and 10 along with its cutoff. All results are reported for the same outer folds.

| Model, C selected inside each fold | Complete episodes detected | Wrist false starts | Pre-onset non-AF starts |
| --- | ---: | ---: | ---: |
| Original template ECG | 13/17 | 6 | 1 |
| Template plus order-invariant pulse bins | 8/17 | 4 | 2 |
| Template plus chronological pulse bins | 12/17 | 9 | 7 |

The original template reproduced the previously reported 13/17 regularization result. Neither pulse-bin variant improved the sensitivity and false-alert trade-off in this study. The small, previously analyzed development cohort supports exploratory comparisons. A fresh paired wrist cohort is required to assess revised models independently.

## Reproduce

Run from the repository root with the staged archives from the [decision cadence study](clock-study.md#reproduce).

```powershell
python -m ppg_af.af_wrist_sequence_study `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --output runs/sequence-study-results.json
```

The [causal bin builder](../../ppg_af/af_wrist_sequence_features.py) checks archive alignment and builds both views. The [study driver](../../ppg_af/af_wrist_sequence_study.py) verifies fold and score availability and evaluates both variants. The [aggregate results](sequence-study-results.json) include the paired comparisons and calibration counts.
