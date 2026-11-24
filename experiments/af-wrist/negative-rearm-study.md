# Require two negative decisions before a new alert

The historical wrist audit recorded 24 false starts within 30 seconds of a negative reset. The score in these cases had crossed below the threshold and then returned above it. This development replay checks whether suppressing new starts after a single negative reduces the alert burden. [Historical error audit](template-independent-error-audit.md)

## Method

The original policy can issue a new alert on the first positive after a negative. The candidate reopens that possibility after two consecutive observed negative decisions. An unavailable decision clears the state in both policies. Any positive before rearming consumes the short negative streak. Unknown predecessor history propagates possible states and can leave a start ambiguous. Existing classifier predictions and PPG coverage remain intact. [`apply_negative_rearm`](../../ppg_af/af_wrist_history.py)

Replay the plain template and the quality-interaction model on the same ten people and original 10-second decision clock. For each model and fold, use the already-defined immediate-policy score and cutoff. The comparison holds all 17 complete episodes and 46.5 ECG-observed wrist minutes fixed. It verifies that every definite candidate start was also an immediate start. Evaluate the original partition and the previously reported weak partition as a prespecified stress comparison. [Quality split sensitivity](quality-stability.md)

## Results

| Participant partition | Model | Immediate | Two negative decisions to rearm | Difference in detections within 10 percentage points |
| --- | --- | --- | --- | --- |
| 2026 | Plain template | 12/17 and 6 false starts | 11/17 and 5 | Yes |
| 2026 | Template with quality interactions | 14/17 and 4 | 12/17 and 2 | No |
| 2030 | Plain template | 1/17 and 2 | 1/17 and 1 | Yes |
| 2030 | Template with quality interactions | 1/17 and 0 | 1/17 and 0 | Yes |

The candidate saved one false start at the cost of one true detection for the original plain model. With quality interactions it saved two false starts and lost two detections, a sensitivity decrease of 11.8 percentage points. No rearmed model reached 80% sensitivity. These development findings support retaining the immediate policy as the comparison baseline. [Aggregate results](negative-rearm-results.json)

## Recalibrate inside the training groups

After the fixed-cutoff test, reuse the same inner held-out model scores and search candidate cutoffs under the new alert policy. One rule seeks the 80% episode-sensitivity target. The other first restricts the cutoff to within ten percentage points of immediate-policy sensitivity on the same training groups and then minimizes false starts. When the matching range has no eligible cutoff, choose the closest sensitivity. Both recalibrated rules reproduced every outer detection and wrist false-start count in the preceding table. All 20 inner target searches fell short of 80%. Neither threshold search recovered the two detections lost by the original quality model. [Recalibration results](negative-rearm-calibration-results.json)

## Reproduce

Use the original local wrist development archives in `RAW`.

```bash
python -m ppg_af.af_wrist_negative_rearm --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --quality-results experiments/af-wrist/quality-study-results.json \
  --output runs/negative-rearm-results.json
```

[`compare_rearming`](../../ppg_af/af_wrist_negative_rearm.py) checks fixed scores, cutoffs, and paired alert changes.

To reproduce recalibration, add `--recalibrate` and choose a fresh output path.
