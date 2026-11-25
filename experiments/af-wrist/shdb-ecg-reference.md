# Continuous AF alerts from measured Holter beat intervals

SHDB-AF supplies long ECG recordings, participant-level grouping, separate measured R-peak annotations, and clinician-reviewed rhythm transitions. It lets us test how input length and the alert objective affect event sensitivity with measured ECG timing. The study uses ECG annotations as inputs and targets. [SHDB-AF v1.0.1](https://physionet.org/content/shdb-af/1.0.1/) · [Scientific Data publication](https://doi.org/10.1038/s41597-025-04777-4)

## Verify coverage before fitting

Use the release's 98 annotated recordings from 93 participants. Stage and verify the `.hea`, `.qrs` and `.atr` files against the release SHA-256 catalog. Keep only the recording-to-participant grouping fields from the metadata. The released headers and annotations share a 200 Hz clock. De-duplicate 11 identical peak timestamps, leaving 10,349,722 unique peaks. The parser distinguishes AF from the labeled other rhythms and omits three spans with the undocumented `(AB` rhythm mark. `(N` covers the source's remaining rhythms.

Score on a fixed ten-second decision clock and start both feature routes at a 30-second decision. Across 838,859 scheduled decisions, 838,489 had known ECG rhythm. Both horizons could compute features in 837,911 of those reference windows. Keep complete episode coverage even when a score is unavailable. Treat AF already active at the start or continuing beyond the recording as censored. There are 767 complete episodes, 309 shorter than 30 seconds. At least one usable decision endpoint occurred in 720 complete episodes. [Reference preflight](shdb-ecg-preflight-results.json)

## Compare cadence, input length and cutoff selection

Train the same regularized logistic model on 10-second, 30-second, or concatenated 10- and 30-second measured ECG interval features. Use the same stable reference fitting rows, capped at 120 per participant and rhythm, and score all routes on the same available decision rows. Divide by participant into five held-out folds. Within each fold, reserve separate fitting and calibration participants. Choose thresholds according to episode sensitivity and false starts on calibration participants. Also select thresholds for shorter and combined inputs within ten points of the 30-second model's calibration sensitivity when possible.

| Feature route and cutoff objective | Complete episodes detected | Sensitivity | False starts per ECG hour | Median detection delay |
| --- | ---: | ---: | ---: | ---: |
| 10 seconds, target 80% | 443/767 | 57.8% | 4.04 | 14.1 s |
| 30 seconds, target 80% | 289/767 | 37.7% | 2.00 | 28.9 s |
| Combined, target 80% | 291/767 | 37.9% | 1.86 | 29.7 s |
| 10 seconds, match 30-second calibration | 464/767 | 60.5% | 2.35 | 13.6 s |
| Combined, match 30-second calibration | 249/767 | 32.5% | 0.92 | 33.8 s |

The shorter window had lower stable-window AUC than the 30-second window, 0.946 versus 0.956, and more complete episode detections. At the primary cutoffs it detected 88 of 309 short episodes versus 15 with the long window. The 10-second route's training-matched calibration exceeded the 30-second held-out result by 22.8 points, outside the tolerance. The combined matched route lowered the held-out false rate by 1.08 per ECG hour and had 5.2 points lower sensitivity. All rates share 2,329.14 ECG-observed hours. All route estimates stayed below 80% episode sensitivity. [Grouped results](shdb-alert-results.json)

## Train on the first available post-onset input

On the same folds, choose the first available decision within 20 seconds of each eligible complete AF onset with 20 seconds of verified prior non-AF. Match those positives to each person's stable AF windows. Retain the same non-AF fitting rows and train on the same 10-second interval representation. Each fit has 245–369 matched positives from 36–39 people. Compare thresholds targeting 80% and thresholds selected at similar sensitivity on the calibration participants. [Onset code](../../ppg_af/af_shdb_onset.py)

| Fitting positives and cutoff | Detected | False starts per ECG hour | Median delay |
| --- | ---: | ---: | ---: |
| Stable AF, target 80% | 458/767 | 4.08 | 13.0 s |
| First post-onset input, target 80% | 427/767 | 5.61 | 11.4 s |
| First post-onset input, matched calibration | 423/767 | 5.34 | 11.5 s |

Training with early positives shortened the median delay among the events it detected. It also lowered total sensitivity and increased false starts across the same exposure. [Onset results](shdb-onset-results.json)

## Calibrate duration and alerts separately

Reuse the same five held-out groups, stable training examples and 10-second ECG scores from the original episode comparison. A decision becomes available at the end of its input window and applies to the following ten seconds. The first bin of each recording has no state. Measure exact AF overlap within bins and exclude time with no verifiable ECG rhythm. Keep the same threshold candidates and select one using the original alert objective and another using time sensitivity minus the non-AF false-positive time fraction on calibration participants. [State duration code](../../ppg_af/af_state.py) · [Threshold comparison](../../ppg_af/af_shdb_state_calibration.py)

| Calibration objective | AF time detected | Non-AF time positive | New alerts during complete AF | False starts per ECG hour | Median detected delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original episode target | 46.0% | 3.17% | 443/767 | 4.04 | 14.0 s |
| Balanced causal state | 90.5% | 16.20% | 431/767 | 13.30 | 9.0 s |

Both use 2,329.14 ECG-observed hours and 99.92% scoreable causal state coverage. The state-based objective covered more annotated AF time and detected more short episodes, 130 versus 88 of 309. It detected fewer episodes overall and produced more false starts. The reference AF fraction was 19.57% of scoreable time. Positive state occupied 11.56% of time at the original cutoff and 30.77% at the state-balanced cutoff. [Threshold results](shdb-state-results.json)

For a separate estimate of time in AF, average the model's probability across each participant's scoreable time. Fit a logistic map of the original log odds and a monotone isotonic map on the same separate calibration participants. Exact AF overlap supplies the target fraction for each bin. Use the duration-weighted Brier score to judge probability error for individual seconds. Keep episode cutoff comparisons separate. [Probability calibration code](../../ppg_af/af_shdb_state_probability.py)

| Probability estimate | Brier score | Estimated AF time | Median absolute participant error |
| --- | ---: | ---: | ---: |
| Original score | 0.0922 | 27.41% | 8.42 points |
| Logistic calibration | 0.0785 | 19.50% | 7.08 points |
| Isotonic calibration | 0.0795 | 19.65% | 7.22 points |

The observed AF time was 19.57% across 8,378,150 scoreable seconds. The logistic map brought pooled estimated duration close to this value and reduced median participant error. Individual error remains substantial. The improvement uses measured ECG intervals and separate calibration participants. It provides a directly measured reference for a future wrist probability calibration comparison. [Probability results](shdb-state-probability-results.json)

## Reproduce

```bash
python -m ppg_af.af_shdb_stage --cache-dir "$PRIVATE_SHDB/annotations"
python -m ppg_af.af_shdb_ecg --annotations-dir "$PRIVATE_SHDB/annotations" \
  --feature-dir "$PRIVATE_SHDB/features" --output runs/shdb-preflight.json
python -m ppg_af.af_shdb_alerts --annotations-dir "$PRIVATE_SHDB/annotations" \
  --feature-dir "$PRIVATE_SHDB/features" --preflight runs/shdb-preflight.json \
  --output runs/shdb-alerts.json
```

Install the wrist requirements, `requests`, and `wfdb==4.3.1`. The inputs are R peaks detected on the measured Holter ECG. The reference helps isolate the alert design and input timing. [Source verification](../../ppg_af/af_shdb_stage.py) · [Clock and interval features](../../ppg_af/af_shdb_ecg.py) · [Grouped fit and scoring](../../ppg_af/af_shdb_alerts.py)

Run `ppg_af.af_shdb_onset` with the same `--annotations-dir`, `--feature-dir`, and a new `--output` to reproduce the matched onset training comparison.

Run `ppg_af.af_shdb_state_calibration` with the same annotation and feature directories, `--alert-results experiments/af-wrist/shdb-alert-results.json`, and a new `--output` to compare the cutoff objectives. Run `ppg_af.af_shdb_state_probability` with the same two directories and a new `--output` to compare the probability maps.
