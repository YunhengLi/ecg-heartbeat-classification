# PPG quality features for wrist AF classification

The [development failure audit](failure-audit.md) found a median PPG quality of 0.416 at template ECG false starts and 0.585 across scored non-AF wrist decisions. The [independent error audit](template-independent-error-results.json) reported corresponding medians of 0.382 and 0.441. This study tests whether the classifier can use that signal while retaining the same inputs, eligibility, and alert policy.

## Method

Use each trailing 30-second PPG window to obtain the existing quality score. It multiplies the fraction of usable samples after plateau masking by the fraction of signal power in 0.25–8 Hz. Recalculate the quality from native PPG to check the cached value and verify row alignment in the fit, evaluation and preceding history archives. All 881 rows with original shared features have a verified quality value. No lower-quality row is excluded from scoring.

Compare three standardized, class-balanced logistic classifiers at C=1 on the original five outer participant folds with the original nested cutoff selection.

1. Reproduce the original seven template ECG interval features and held-out results.
2. Append quality as an eighth feature.
3. Append quality and its products with normalized RMSSD and pNN50. The products let the contribution of beat irregularity vary with PPG quality.

The fixed comparison keeps the original training rows, evaluation denominators, 10-second decision stride, immediate alert policy and causal predecessor replay. The 17 complete episodes and 46.5 ECG-observed wrist minutes are identical across branches. The pre-onset diagnostic uses the ten episodes with a fully observed non-AF lookback. Persistence counts false starts that remained active during a full 30 seconds of continuously observed non-AF ECG.

## Fixed C=1 results

| Classifier | Episodes detected | Wrist false starts | Persisted ≥30 s in non-AF | Pre-onset starts | Median delay among detections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original template | 12/17 (70.6%) | 6 | 3 | 3 | 20.524 s |
| Template + quality | 13/17 (76.5%) | 5 | 2 | 3 | 20.436 s |
| Template + quality and interactions | 14/17 (82.4%) | 4 | 2 | 3 | 20.524 s |

Quality alone gained one detection and retained all 12 original detections. Quality with interactions gained three and lost one. Both had a median paired delay change of zero for episodes detected by both the original and candidate. The interaction model exceeded the 80% reference target across the pooled held-out folds. Its sensitivity was 11.8 percentage points above the original, beyond the protocol's 10-point matching tolerance. All five inner cutoff searches for each fixed classifier fell below the target. The pooled result comes from 17 episodes in the existing development cohort. It remains a candidate result that needs assessment on newly held-out participants.

## Regularization follow-up

The regularization follow-up was specified after observing the fixed-C results. It compares quality alone with the original feature set using C ∈ {0.1, 1, 10}. Each outer fold selects C and its cutoff using only the existing inner development predictions and calibration objective. The fixed C=1 interaction result above remains the primary result for that candidate.

| Classifier with inner-selected C | Episodes detected | Wrist false starts | Persisted ≥30 s in non-AF | Pre-onset starts | Median delay among detections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original template | 13/17 (76.5%) | 6 | 3 | 1 | 24.714 s |
| Template + quality | 13/17 (76.5%) | 5 | 2 | 3 | 20.436 s |

Both fits selected C=1 in two outer folds and C=10 in three. Two quality-only inner cutoff searches met the target, compared with zero for the original feature set. The tuned quality model gained two episode detections and lost two. For the 11 episodes detected by both, the median paired delay change was zero. The quality feature reduced wrist false starts by one at the same pooled sensitivity and increased pre-onset starts by two in the separate diagnostic. The held-out sensitivity was 76.5% for both.

## Reproduce

Use the locked development recordings and caches in the [decision clock study](clock-study.md#reproduce). The evaluator runs on the development archives.

```powershell
python -m ppg_af.af_wrist_quality_study --development-raw $development --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results experiments/af-wrist/clock-study-results.json --output runs/quality-study-results.json
```

The [benchmark driver](../../ppg_af/af_wrist_quality_study.py) verifies quality against native PPG and applies the original grouped fit. The [aggregate report](quality-study-results.json) preserves the paired outcomes and both calibration stages.
