# Paired participant uncertainty in the retrospective wrist replay

Both quality variants detected more AF episodes in the pooled retrospective results. Giving each of the 16 participants with complete AF episodes equal weight lowered recall from 55.7% for the historical template to 50.7% for quality alone and 51.2% for quality with interactions. Both variants also had higher observed false-alert rates, with 95% paired intervals spanning decreases and increases.

## Paired differences

| Frozen variant versus historical template ECG | Pooled sensitivity change, percentage points (95% interval) | False starts per ECG-observed wrist hour change (95% interval) | Participant-macro recall change, percentage points |
| --- | ---: | ---: | ---: |
| PPG quality | +3.85 (−4.72 to +13.83) | +3.84 (−1.92 to +10.56) | −4.90 |
| PPG quality and interactions | +5.00 (−4.72 to +18.31) | +1.44 (−4.32 to +6.72) | −4.47 |

The observed sensitivity differences fall inside the original 10-percentage-point margin. Their intervals extend above that margin. The false-alert intervals include both benefit and harm. These estimates come from the previously evaluated cohort.

| Model | Participant with most episodes | Other 15 participants, pooled | Other 15 participants, macro recall |
| --- | ---: | ---: | ---: |
| Historical template ECG | 107/157 | 48/103 | 54.8% |
| Template ECG + quality | 109/157 | 56/103 | 49.5% |
| Template ECG + quality and interactions | 109/157 | 59/103 | 50.0% |

The largest contributor supplied two extra net detections for each variant. That accounted for 20.0% of the quality-only gain and 15.4% of the interaction gain. The same contributor accounted for 15 of 28 newly detected episodes with quality alone and 16 of 32 with the interaction variant, along with 13 and 14 lost detections, respectively. For each variant, five participants gained detections, three lost detections and eight had equal counts. Macro recall accounts for the proportion gained or lost within each participant.

## Method and reproduction

1. Verify the sealed archives and original model locks, then reconcile all three frozen replays with their aggregate results. Fit parameters, cutoffs, the score gate and the immediate alert policy come from the original replay. [Retrospective methods and primary results](quality-retrospective-study.md)
2. Aggregate complete episodes, detections, false starts and ECG-observed wrist time by participant in memory. In each of 10,000 bootstrap draws, resample the same 25 participants with replacement for all models. Compute each variant's change in pooled episode sensitivity and wrist false-start rate, then take the 2.5th and 97.5th percentiles. All 10,000 paired draws had usable denominators for both endpoints. [Audit code](../../ppg_af/af_wrist_quality_retrospective_uncertainty.py)
3. Average each participant's episode recall over the 16 participants with complete AF episodes. Compare paired gains and losses for the largest episode contributor and the other 15, then write only aggregate counts. [Aggregate results](quality-retrospective-uncertainty-results.json)

From the repository root, use the verified local archive path in `RAW`, the existing private retrospective variant lock in `PRIVATE_LOCK` and a fresh aggregate output path in `REPORT`.

```bash
python -m ppg_af.af_wrist_quality_retrospective_uncertainty \
  --retro-results experiments/af-wrist/quality-retrospective-results.json \
  --protocol-lock "$RAW/holdout-window-seeded-lock.json" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --variant-lock "$PRIVATE_LOCK" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --sealed-inputs "$RAW/template-independent-prepared" --output "$REPORT"
```
