# VitalDB source training and wrist transfer

This experiment tests whether using ectopy as a non-AF training example helps a PPG rhythm score distinguish irregular rhythms. Both classifiers learn from surgical plethysmography with ECG rhythm labels. The source combines atrial fibrillation and atrial flutter under `AFIB/AFL`. The separate wrist comparison uses ECG-confirmed AF events. [VitalDB clock preflight](vitaldb-preflight.md)

## Method

1. Fix the 30-case source cohort before reading waveforms. Use six distinct participants with both AF or flutter and sinus windows, 12 with AF or flutter and without a qualifying sinus window, and 12 with sinus and ectopy and without a source AF or flutter label. Source windows span 30 seconds, end on multiples of 30 seconds, and fit inside one clean reference rhythm. The preflight's case ordering keeps repeat participant keys together.
2. Use both native ECG and PPG on the verified `EVENT` clock. Require at least 95% of the clock-check annotations in each case to have a detected ECG peak within 150 ms. The 30 cases passed. The score uses nine PPG features on each trailing window, resampled from 500 Hz to 100 Hz. ECG supplies the training rhythm and clock verification.
3. Fit two class- and participant-balanced logistic regressions with `C=1`. Retain up to 32 evenly spaced AF or flutter and sinus training windows per participant and rhythm. The two arms share the same 350 positive examples and 248 negatives per fit across the full source cohort. The ectopy arm replaces 35 of the 248 sinus examples using candidates from the same participant. Remove constant features and fit the scaler inside each source fit.
4. Score source windows on five participant-held-out folds. Use the same evaluation windows for both arms, with equal participant weight within each reference class for AUC. Refit each source score on all source training rows and freeze it before applying it to the same ten wrist development participants. In the wrist replay, calibrate a source-only score and separately add its logit to the seven template ECG features. The cutoffs and fusion fits use only the training side of each original outer group.

## Surgical source results

The source supplied 377 AF or flutter, 248 sinus, 40 patterned atrial ectopy, and 21 patterned ventricular ectopy windows with supported PPG. AUC compares ranking of stable windows. An analyzed window from the same participant is never split between the fitting and held-out sides of a source fold.

| Source training negatives | AF or flutter vs sinus | AF or flutter vs patterned ectopy |
| --- | ---: | ---: |
| 248 sinus | 0.979 | 0.620 |
| 213 sinus + 35 ectopy | 0.981 | 0.736 |

## Wrist development results

The comparisons share all 17 complete ECG-confirmed AF episodes and 46.5 minutes of ECG-observed wrist time. Each cell reports episodes detected and wrist false starts. The matched-sensitivity cutoffs were sought on training groups. Compare the actual held-out sensitivity separately.

| Route | 80% calibration target | Calibration matched to template |
| --- | --- | --- |
| Original wrist direct PPG | 7/17 and 4 | — |
| Original wrist template ECG | 12/17 and 6 | — |
| Source-only PPG, sinus negatives | 8/17 and 4 | 8/17 and 4 |
| Source-only PPG, ectopy negatives | 2/17 and 6 | 2/17 and 6 |
| Template + sinus-negative source score | 8/17 and 3 | 10/17 and 4 |
| Template + ectopy-negative source score | 9/17 and 5 | 9/17 and 5 |

Replacing sinus training windows improved AF-or-flutter versus ectopy ranking on held-out surgical participants. Wrist transfer showed the opposite result for source-only scoring. The closest fusion result lost two of 17 events and avoided two false starts. Its 11.8-point sensitivity loss exceeded the prespecified 10-point tolerance. These are development comparisons, and surgical plethysmography and AF-or-flutter labels differ from the wrist AF target. [Aggregate results](vitaldb-transfer-results.json)

## Reproduce

Prepare the public sources from the [VitalDB preflight](vitaldb-preflight.md). Generate the transfer plan locally and save each selected `https://api.vitaldb.net/{case}.vital` as `{case}.vital` in `NATIVE`. `RAW` contains the original private wrist development inputs. [Native waveform API](https://api.vitaldb.net/)

```bash
python -m ppg_af.af_vitaldb_transfer --annotations-dir "$ANNOTATIONS" \
  --track-list "$TRACKS" --private-plan --output runs/vitaldb-transfer-private-plan.json

python -m ppg_af.af_vitaldb_transfer --annotations-dir "$ANNOTATIONS" \
  --track-list "$TRACKS" --native-cache "$NATIVE" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --output runs/vitaldb-transfer-results.json
```

[`training_rows` and `source_models`](../../ppg_af/af_vitaldb_transfer.py) preserve positive examples and replace negative examples within participants. [`evaluate`](../../ppg_af/af_wrist_mimic_transfer.py) reuses the locked development calibration and comparison code for both source arms.
