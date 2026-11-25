# Test alert entry and release on the frozen wrist AF scores

The historical audit found that 62 of 109 missed episodes already had an active alert at ECG onset and a positive decision during the episode. This comparison isolates how alert entry and release change sensitivity, the number of false starts, and time spent in a positive state. [Miss audit](short-rescue-study.md)

## Method

1. Keep the published 30-second pulse-template classifier, ten-second decision cadence, thresholds, and low-quality confirmation as the reference. Use the existing fold-specific model and cutoff on each held-out development participant. Use the fixed out-of-fit archival model and cutoff for the historical replay.
2. Test two fixed changes separately and together. The entry rule starts immediately at a score at least 0.10 above cutoff. A positive below that margin requires a preceding positive. The release rule keeps an active alert through one observed negative within 0.10 of the cutoff. A second consecutive shallow negative, a clear negative, or a missing score closes the alert. Both rules use the quality-filtered scores. [State replay](../../ppg_af/af_wrist_alert_state.py)
3. Replay only measured preroll and the original evaluation cells. Keep all ECG-observed wear in the denominator, including abstentions. Count definite new starts during a complete ECG-labeled episode. Separately count ECG-confirmed non-AF starts in the 30 seconds leading into an episode and forward non-AF time with a positive state. Use the two disjoint additional-wear schedules for pooled burden and coverage. Draw the same 25 participants with replacement for 10,000 paired resamples. [Study runner](../../ppg_af/af_wrist_alert_state_study.py)

## Results

| Fixed rule | Historical complete AF detected | False starts on 985.5 observed minutes | Starts per observed hour | Non-AF time with positive state | Starts just before AF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality confirmation | 151/260 (58.08%) | 210 | 12.79 | 28.40% | 39 |
| Stricter entry | 148/260 (56.92%) | 189 | 11.51 | 27.35% | 35 |
| Slower release | 148/260 (56.92%) | 195 | 11.87 | 29.69% | 38 |
| Both changes | 146/260 (56.15%) | 176 | 10.72 | 28.60% | 35 |

The 30-second pre-onset comparison had ECG-verified non-AF support for 184 of 260 complete episodes. All routes retained 58,100 monitorable seconds out of the same 59,130 ECG-observed seconds (98.26%). The combined rule changed sensitivity by −1.92 percentage points (95% participant interval −3.64 to 0.00) and false starts by −2.07 per observed hour (−3.28 to −1.03). It gained four and lost nine detections. Slower release alone saved 15 starts and increased the non-AF time in a positive state by 700 seconds. Starting and remaining active measure different aspects of burden. [Aggregate results](alert-state-results.json)

| Participant split | Quality confirmation | Stricter entry | Slower release | Both changes |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12/17, 5 false starts | 4/17, 4 | 11/17, 5 | 4/17, 4 |
| 2030 | 1/17, 1 false start | 1/17, 1 | 1/17, 1 | 1/17, 1 |

The development comparisons each used the same 46.5 ECG-observed wrist minutes, with 77.78% forward score coverage in every arm. The stricter entry lost eight detections in the original development split, despite a much smaller historical loss. All retrospective sensitivities were below the 80% development target. Keep the original quality confirmation as the reference. This study uses the existing AFPPGECG participants and established wear schedules.

## Reproduce

From the repository root, point `RAW` to the local AFPPGECG archive root and `PTT` to the staged PTT recordings. Use a fresh output path for `REPORT`.

```bash
python -m ppg_af.af_wrist_alert_state_study \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --quality-results experiments/af-wrist/quality-study-results.json \
  --development-confirmation-results experiments/af-wrist/quality-confirmation-results.json \
  --historical-confirmation-results experiments/af-wrist/confirmation-retrospective-results.json \
  --prior-inputs "$RAW/template-independent-prepared" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --historical-results experiments/af-wrist/template-independent-results.json \
  --prior-monitor-plan "$RAW/retrospective-monitoring/plan-private.json" \
  --prior-monitoring-cache "$RAW/retrospective-monitoring/native" \
  --plan "$RAW/retrospective-monitoring/half-hour-plan-private.json" \
  --monitoring-cache "$RAW/retrospective-monitoring/half-hour-native" \
  --half-hour-results experiments/af-wrist/half-hour-monitor-results.json \
  --ptt-dir "$PTT" --output "$REPORT"
```
