# An onset-aware cutoff objective for wrist AF

The fixed quality-interaction model detected 14 of 17 complete AF episodes and produced four false wrist starts. It also had three alerts that started shortly before AF while ECG still showed non-AF. This exploratory follow-up tests whether the threshold search can reduce both types of false starts. The objective was specified after the [quality study](quality-study.md).

## Method

1. Refit the same C=1 quality-interaction model once with the original five outer participant folds, up to three inner folds, and seed 2026. Both cutoff objectives receive the same inner held-out scores. Keep the 30-second trailing PPG windows, 10-second decision cadence, immediate alert policy and predecessor replay.
2. Evaluate the same threshold candidates for both objectives. When any cutoff meets the 80% target, minimize alert burden and then prefer higher sensitivity. When no cutoff meets 80%, maximize attainable sensitivity and use burden to break a tie. The original objective measures false starts on the prescheduled wrist monitoring blocks. The alternate objective also charges one for a definite non-AF start in the 30 seconds before a complete AF episode with continuously observed non-AF ECG throughout that interval.
3. Count an alert once in the alternate objective if the same participant and timestamp appear in both scopes. The actual wrist false-start rate still uses the original wrist monitoring exposure. The extra episode clips contribute time only to the episode evaluation, and the experimental combined burden is a count of distinct starts. Starts at onset use the ECG state immediately before that timestamp, consistent with the paired diagnostic.

## Results

The adjusted objective chose a different cutoff in two of five outer folds. In the inner predictions for each of those folds, the new cutoff reduced pre-onset starts by one at the same sensitivity and wrist false-start count. All five inner calibrations fell below 80% under both objectives. The overlapping inner folds use some of the same participants, so their counts describe separate calibration selections.

| Held-out development outcome | Wrist-only calibration | Onset-aware calibration |
| --- | ---: | ---: |
| Complete episodes detected | 14/17 (82.4%) | 14/17 (82.4%) |
| Wrist false starts in 46.5 ECG-observed minutes | 4 | 4 |
| Pre-onset false starts across 10 eligible episodes | 3 | 3 |
| Exact overlap between wrist and pre-onset starts | 0 | 0 |
| Distinct false starts across both scopes | 7 | 7 |
| Wrist false starts active ≥30 seconds in verified non-AF | 2 | 2 |
| Median first-alert delay | 20.524 s | 20.524 s |

All 14 episode detections and delays were identical between the two objectives. The four wrist false starts and the three pre-onset starts also matched at the same timestamps. Of the seven episodes excluded from the pre-onset diagnostic, three lacked continuous ECG for the preceding 30 seconds and four had a preceding AF interval. The change improved the inner tie-breaking objective in two folds and gave identical observed held-out outcomes. Keep the original wrist-only objective for this model's next evaluation.

## Reproduce

Prepare the locked archives in the [quality study](quality-study.md#reproduce). This comparison uses development archives and calibration predictions only.

```powershell
python -m ppg_af.af_wrist_onset_calibration --development-raw $development --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results experiments/af-wrist/clock-study-results.json --quality-results experiments/af-wrist/quality-study-results.json --output runs/onset-calibration-results.json
```

The [calibration driver](../../ppg_af/af_wrist_onset_calibration.py) evaluates both objectives on shared scores. The [pre-onset metric](../../ppg_af/af_wrist_transition_metrics.py) enforces ECG coverage and counts unique events across scopes. The [aggregate JSON](onset-calibration-results.json) includes the inner and held-out outcomes.
