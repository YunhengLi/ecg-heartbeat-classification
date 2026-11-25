# Verify optical pulse peaks across two filters

In longer wrist recordings, false AF starts aligned with increased optical interval irregularity while native ECG timing remained similar to matched negatives. We tested whether corroborating each pulse in a second frequency view reduces that error and improves alerting. [False-start timing audit](false-start-audit.md)

## Method

Use the original broad 0.4–8 Hz peak detector as the direct pulse control. Check each peak for a narrow 0.4–4 Hz peak within 80 ms. An interval qualifies only when both adjacent broad-band peaks are corroborated. An unverified peak invalidates the intervals on both sides, so it cannot create a new long interval by joining its neighbors. Keep the original quality score and use the same AF folds, classifiers and calibration. [Peak matching and interval logic](../../ppg_af/af_wrist_pulse_consensus.py)

Separately compare each method with native ECG R–R timing on the 223 previously verified paired 30-second windows. Summarize the absolute error in the sum of normalized RMSSD and pNN50 by reference rhythm.

## Results

| Paired reference | Template error | Broad-band pulse error | Two-filter pulse error | Two-filter median retained interval fraction |
| --- | ---: | ---: | ---: | ---: |
| 77 stable AF windows | 0.202 | 0.199 | 0.188 | 86.0% |
| 146 stable non-AF windows | 0.387 | 0.370 | 0.363 | 100.0% |
| All 223 windows | 0.278 | 0.274 | 0.279 | 94.3% |

All 881 originally scoreable AF input rows retained sufficient confirmed peaks to compute interval features. In the original partition, broad-band pulse and quality features gave 12/17 detections with three false starts. Two-filter agreement gave 9/17 with three. The original quality template gave 14/17 with four. On the weak partition all three yielded 1/17 with zero. Rejecting intervals reduced the median error within each stable reference rhythm and still lost AF detections. [Aggregate results](pulse-consensus-results.json)

Run `python -m ppg_af.af_wrist_pulse_consensus` with the original development archives, locks, PTT source and verified waveform arguments from the [native beat study](native-neural-beat-study.md#reproduce). The broad-band control exactly reproduces the original direct PPG interval features before profiling the effect of peak agreement.
