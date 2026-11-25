# Missed AF episodes and a short-window alert trigger

The frozen wrist study reduced false starts with low-quality confirmation and detected 151 of 260 complete AF episodes. This follow-up attributes the remaining misses and tests a short-window trigger on the ten original development participants. [Frozen evaluation](confirmation-evaluation.md)

## Historical miss audit

Replay the original and quality-confirmed policies on the same sealed ECG intervals and PPG score clock. A detection requires a definite new alert during the ECG episode. Trace the alert state from the available PPG history. Count a carried alert when it was active at ECG onset, remained positive at an in-episode decision, and generated no new start during the episode. [Miss audit code](../../ppg_af/af_wrist_confirmation_miss_audit.py)

| Complete episode outcome | Original template | Low-quality confirmation |
| --- | ---: | ---: |
| Detected | 155 | 151 |
| Carried alert with an in-episode positive | 66 | 62 |
| All in-episode scores below cutoff | 32 | 40 |
| No in-episode decision endpoint | 5 | 5 |
| No score with flat PPG | 2 | 2 |

Confirmation recovered six carried-alert misses and lost ten detections. Eight of the losses had all in-episode positives removed and two became carried alerts. There were no ambiguous in-episode starts in either replay. In the original route, 19 of the 66 carried misses had a traceable ECG-confirmed non-AF alert origin and 45 were active when the replayed segment began. Continuous independent recordings can resolve more of that history. [Audit results](confirmation-miss-audit-results.json) · [Original error study](template-independent-error-audit.md)

## Development comparison

Fit 10-second and 30-second pulse-template logistic models with the same stable training data, five outer participant folds and up to three inner calibration folds. Keep the published low-quality confirmation as the baseline. When its decision is negative, permit a short-window positive if optical quality is at least 0.40, its prior short score was observed and below its cutoff, and the current short score is at least 0.10 above that cutoff. The 0.10 margin is fixed. Both threshold selections use only the training side of the outer fold. The two triggers run through the same alert state and use the same ECG exposure. [Short-trigger code](../../ppg_af/af_wrist_short_rescue.py)

| Participant split | Complete episodes detected | False starts | Monitorable fraction | Detections brought forward |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12/17 → 12/17 | 5 → 5 | 77.78% → 77.78% | 2 |
| 2030 | 1/17 → 1/17 | 1 → 1 | 77.78% → 77.78% | 1 |

The short trigger changed five evaluation decisions in the original split and one in the alternate split. It gained and lost zero episode detections in both. Median delay among the 12 detections in the original split moved from 20.52 to 15.76 seconds. The single alternate-split detection moved from 33.53 to 3.53 seconds. The 46.5 ECG-observed minutes are shared across both splits. The rule preserved 2,170 monitorable seconds of 2,790. Its sensitivity was below 80% in both splits, so the rule remains exploratory. [Development results](short-rescue-results.json)

The next independent comparison requires continuous ECG and wrist PPG, ECG-defined episode boundaries, PPG wear and noise masks, and enough preroll to reconstruct alert state. The Basel AF Burden Trial documents exact ECG episode timing in a 92-person paired cohort. Its one-minute reference definition and grouping of AF with flutter require clarification before comparing to the current endpoint. [External source review](external-wrist-validation-sources.md)

## Reproduce

Set `RAW` to the staged private AFPPGECG archive root and `PTT` to the local PTT recordings. Supply fresh output paths for `AUDIT` and `RESCUE`.

```bash
python -m ppg_af.af_wrist_confirmation_miss_audit \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --prior-inputs "$RAW/template-independent-prepared" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --historical-results experiments/af-wrist/template-independent-results.json \
  --confirmation-results experiments/af-wrist/confirmation-retrospective-results.json \
  --baseline-audit experiments/af-wrist/template-independent-error-results.json --output "$AUDIT"

python -m ppg_af.af_wrist_short_rescue \
  --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --lock "$RAW/holdout-window-seeded-lock.json" --ptt-dir "$PTT" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --quality-results experiments/af-wrist/quality-study-results.json \
  --confirmation-results experiments/af-wrist/quality-confirmation-results.json --output "$RESCUE"
```
