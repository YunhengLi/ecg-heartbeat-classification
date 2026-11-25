# AF episode sensitivity, false alerts, and monitorable wear

The original wrist PPG template and its low-quality confirmation rule were replayed with the same model, score threshold, and ECG decision clock. Confirmation reduced false starts by 2.62 per ECG-observed hour and retained 98.26% of ECG-observed wear time. It detected 151 of 260 complete episodes, below the existing 80% sensitivity target. The evaluation uses previously reviewed participants from AFPPGECG v6. [Aggregate results](confirmation-evaluation-results.json)

## Method

1. Use the frozen template classifier every 10 seconds on the trailing 30 seconds of wrist PPG. When current optical quality falls below 0.40, require a positive score at the previous decision as well. A missing required score makes the current decision unavailable. [Confirmation rule](../../ppg_af/af_wrist_quality_confirmation.py)
2. Score complete AF episodes on the historical ECG-labeled grids. Use the two disjoint, outcome-blind additional wear schedules for the pooled alarm and coverage estimates. Keep all ECG-observed time in the denominator. An available decision applies prospectively to the following interval. Each block begins without a forward state for the coverage calculation. [Wear scheduling](half-hour-monitor-study.md)
3. Resample entire participants 10,000 times, using the same draws for both routes. Combine each person's wear across both schedules before resampling. No new thresholds were fitted. Use the existing 80% episode sensitivity target and a ±10 percentage-point similarity margin. For this retrospective check, use a provisional minimum of 90% observed-time coverage and a maximum of five percentage points of coverage loss. Apply the coverage limits to the respective 95% interval lower bounds. [Evaluation code](../../ppg_af/af_wrist_confirmation_evaluation.py)

## Results

| Outcome | Original template | Low-quality confirmation | Paired change (95% participant interval) |
| --- | ---: | ---: | ---: |
| Historical complete-episode sensitivity | 155/260 (59.62%) | 151/260 (58.08%) | −1.54 pp (−2.82 to +2.70) |
| False starts on 985.5 ECG-observed wear minutes | 253 (15.40/h) | 210 (12.79/h) | −2.62/h (−3.89 to −1.50) |
| Monitorable fraction of the same wear | 98.34% | 98.26% | −0.085 pp (−0.255 to 0.000) |
| Monitorable fraction of all 1,000 scheduled minutes | 96.92% | 96.83% | 50 fewer seconds |

The five-minute blocks contributed 44 versus 41 false starts. The half-hour blocks contributed 209 versus 169. Both schedules have zero adjudicated complete AF episodes and use the same 25 participants. The historical scheduled wear separately yielded 45 versus 39 false starts in 125 ECG-observed minutes. The episode sample had 16 participants with complete AF, with 157 of 260 episodes from one person. Equally weighted participant sensitivity was 55.65% versus 55.87%. Six episodes were gained and ten were lost. Median detection delay among all detections was 22.29 seconds for the original and 29.27 seconds for confirmation. [Earlier replay](quality-confirmation-study.md) · [Half-hour blocks](half-hour-monitor-study.md)

The similarity, false-start reduction, and provisional coverage checks passed on this dataset. Both absolute sensitivity estimates missed the 80% target, and 12.79 false starts per observed hour remains the measured candidate burden. A new wrist cohort with complete ECG-adjudicated AF episodes and representative wear is needed to test generalization. Pulsewatch requires provider approval and documents 30-second rhythm adjudication. Precise episode boundaries need confirmation. [Independent validation plan](prospective-validation-plan.md) · [Pulsewatch data access](https://pmc.ncbi.nlm.nih.gov/articles/PMC13654948/)

The Multisite-PPG P1 data intake checked the two original waveform archives and the authors' aligned 8-second wrist windows. The original timestamps cover 10.59 hours of shared recording span, with 550 of 634 one-minute windows passing the finite paired signal and timestamp gap checks. The author clock reconstruction agreed with the numeric ECG timestamps to within 1 ms. In the official 8-second windows, simple ECG and PPG peak detection and one-to-one matching within 65 ms gave mean beat F1 of about 0.16 at the best fixed positive lags. P1 has not passed intake for beat-supervised reconstruction. This one-person check depends on the peak detectors and P1 has no adjudicated AF episodes. [Multisite-PPG data and preprocessing](https://huggingface.co/datasets/snowballlab/Multisite-PPG)

## Reproduce

From the repository root, use the already staged AFPPGECG and PTT calibration archives. Set `RAW` to the private wrist archive root, `PTT` to the local PTT recordings, and `REPORT` to a new output path.

```bash
python -m ppg_af.af_wrist_confirmation_evaluation \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --prior-inputs "$RAW/template-independent-prepared" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --historical-results experiments/af-wrist/template-independent-results.json \
  --prior-monitor-plan "$RAW/retrospective-monitoring/plan-private.json" \
  --prior-monitoring-cache "$RAW/retrospective-monitoring/native" \
  --prior-monitoring-results experiments/af-wrist/retrospective-monitor-results.json \
  --plan "$RAW/retrospective-monitoring/half-hour-plan-private.json" \
  --monitoring-cache "$RAW/retrospective-monitoring/half-hour-native" \
  --half-hour-results experiments/af-wrist/half-hour-monitor-results.json \
  --ptt-dir "$PTT" --min-coverage 0.90 --max-coverage-loss 0.05 \
  --output "$REPORT"
```
