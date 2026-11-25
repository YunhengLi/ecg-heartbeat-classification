# Transfer an ICU PPG score to wrist AF detection

The wrist development cohort has 34 stable AF fitting windows. This experiment tests whether a larger set of PPG windows with record-level rhythm labels can improve episode detection on the same wrist development benchmark. It compares a frozen external score directly and a wrist-trained model that receives the frozen external score as a feature.

## Source model

Use the two checksummed MAT files in [MIMIC PERform AF v2.0](https://zenodo.org/records/15906524). They contain one 20-minute clinical recording each from 19 participants with an AF record label and 16 with a non-AF record label. The rhythm status labels the recording. No source onset or offset labels enter the experiment.

Split each 125 Hz bedside PPG recording into nonoverlapping 30-second windows. The release paper describes the recordings as mostly fingertip PPG. Keep finite signals with supported pulse intervals and resample each window individually to 100 Hz. Reuse the nine direct PPG features from the wrist study. All 35 source participants supplied usable windows, with 741/760 AF-labeled and 601/640 non-AF-labeled windows retained. A standardized logistic model with C=1 gives equal weight to each rhythm class and to each participant within its class. Its five-fold participant-held-out source record AUROC was 0.951, using the median of window scores in each record. [Source study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9393905/)

## Wrist comparisons

Use the [locked 10-second wrist grid](clock-study.md) and reproduce both wrist-only baselines. Every route shares the same 30-second trailing PPG windows, feature eligibility, replayed predecessor state and five outer participant folds. Report new alerts in the same 17 complete ECG-confirmed episodes and false starts during the same 46.5 minutes of ECG-observed wrist monitoring.

- **Frozen source score.** Only the ICU recordings fit the coefficients and standardization. For each wrist fold, the other wrist participants calibrate its cutoff. The held-out wrist participants supply neither features for fitting nor labels for calibration.
- **Template fusion.** Add the frozen source logit to the original seven template ECG features. Fit the standardized wrist logistic model at C=1 and choose its cutoff through the original nested participant-held-out calibration.
- **Sensitivity matching.** In an exploratory second operating point, use the original template model's inner validation sensitivity on the same wrist training participants as the target. Minimize false starts among cutoffs whose validation sensitivity is within ten percentage points, with closest sensitivity as fallback when no cutoff is within tolerance. Check the achieved sensitivity difference separately on the outer-held participants.

## Results

| Model | Cutoff objective | Complete AF detected | Wrist false starts |
| --- | --- | ---: | ---: |
| Wrist direct PPG | 80% target | 7/17 (41.2%) | 4 |
| Wrist template ECG | 80% target | 12/17 (70.6%) | 6 |
| Frozen source direct PPG | 80% target | 14/17 (82.4%) | 9 |
| Wrist template ECG plus frozen source logit | 80% target | 11/17 (64.7%) | 4 |
| Frozen source direct PPG | Matched template sensitivity | 14/17 (82.4%) | 10 |
| Wrist template ECG plus frozen source logit | Matched template sensitivity | 11/17 (64.7%) | 4 |

At the original objective, the frozen source score gained eight episodes and lost one against wrist direct PPG. It crossed the sensitivity target across the held-out development folds with five more false starts. Of the six episodes detected by both, its paired median delay was 10 seconds longer. Relative to the template reference, fusion gained three detections, lost four and produced two fewer false starts. Its paired median delay was five seconds shorter over the eight shared detections. In the ten episodes with continuously verified pre-onset non-AF ECG, fusion reduced the pre-onset starts from three to zero.

Both candidates found cutoffs within the 10-point tolerance in all five training folds. On the held-out participants, the frozen source score differed from the template by 11.8 percentage points, beyond that tolerance. Fusion was 5.9 points lower than the template, within tolerance, with two fewer false starts. Fusion returned the same reported counts and delay summaries under both objectives.

The source data provide evidence about cross-setting transfer from clinical bedside PPG to wrist PPG. The wrist comparison is exploratory and uses ten development participants. The independent wrist cohort generated zero predictions for this study. The output contains only aggregate metrics.

## Reproduce

Run this comparison after preparing the development artifacts in the [decision cadence study](clock-study.md#reproduce). The source directory holds `mimic_perform_af_data.mat` and `mimic_perform_non_af_data.mat` from the pinned release. Pass the original development lock and cadence report.

```powershell
python -m ppg_af.af_wrist_mimic_transfer `
  --source data/mimic-perform-af-v2 `
  --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 `
  --stride-features runs/wrist-stride10-features `
  --lock $lock `
  --clock-results experiments/af-wrist/clock-study-results.json `
  --output runs/mimic-transfer-results.json
```

[`_source_features`](../../ppg_af/af_wrist_mimic_transfer.py) stages the source windows. [`_calibrate`](../../ppg_af/af_wrist_mimic_transfer.py) runs the two cutoff objectives on the same training participants. The [aggregate results](mimic-transfer-results.json) include paired episode and pre-onset changes.
