# Wrist PPG under exercise motion

An independent device and activity stress test replayed the frozen wrist pipeline on 19 public Shimmer recordings from eight participants. The optical channel was recorded on the wrist, and a chest ECG supplied hand-marked R peaks. Walking, running, and two cycling conditions were included. The release supplies beat locations and activity labels. [PhysioNet release](https://physionet.org/content/wrist/1.0.0/) · [Jarchi and Casson, Data, 2017](https://doi.org/10.3390/data2010001)

## Method

1. Verify the release checksums and retain its full ECG clock. Unwrap the Shimmer sample times and set the first PPG sample to the start of each released exercise segment. The sources use independent device clocks and the original segmentation limits the precision of cross-device alignment. The publisher low-pass filtered the cycling PPG. These conditions support a window-level heart-rate comparison.
2. Every 10 seconds, take the trailing 30 seconds. Interpolate to a uniform 256 Hz grid and use anti-aliasing resampling to 100 Hz from samples strictly before each decision. Leave any window with a gap over 100 ms unavailable. The gap limit was set after inspecting the published timestamp spacing and before extracting heart-rate errors.
3. Estimate heart rate as 60 divided by median inter-beat interval for direct PPG, pulse-template ECG, and measured ECG. Score against the same statistic from the hand-marked R peaks. Use matched windows for route comparisons. Replay the original frozen template AF model and both frozen quality variants, using their development cutoffs. Treat the initial positive state as uncertain because pre-activity PPG history is absent.
4. In a follow-up after the frozen replay, test one fixed motion subtraction method after observing the baseline errors. Inside each already-observed window, regress 0.4–8 Hz PPG against three accelerometer and three gyroscope axes at five offsets between −250 and +250 ms. Standardize the lagged predictors and use a fixed 0.10 ridge penalty on the per-sample loss. Subtract only the estimated motion component. ECG labels are used to measure the error before and after subtraction. The same pulse detector runs in both arms.

## Heart rate

Of 632 scheduled windows, 591 had supported PPG and all three estimates. Mean absolute error is in beats per minute, computed on those paired windows. The motion-filter column uses the same windows as direct PPG in each row.

| Activity | Paired windows | Direct PPG | Template ECG | Measured ECG | Direct PPG after motion subtraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 591 | 25.62 | 25.68 | 7.15 | 38.13 |
| Walk | 185 | 40.16 | 40.28 | 9.56 | 67.37 |
| Run | 124 | 33.40 | 33.37 | 18.89 | 43.51 |
| Low-resistance bike | 161 | 15.37 | 15.18 | 0.29 | 22.00 |
| High-resistance bike | 121 | 9.04 | 9.47 | 0.58 | 9.35 |

Measured ECG had a 0.47 bpm median absolute error over paired windows. Its mean was higher because some exercise windows also challenged the simple ECG peak detector. Out of 465 paired PPG windows with quality at least 0.90, 326 direct PPG estimates and 324 template estimates missed the reference by more than 10 bpm. The present quality score rewards in-band energy, which exercise motion can also produce. The linear motion removal increased mean error for each activity and worsened 390 of 591 paired windows. A movement reference can share frequencies with a real pulse, so linear subtraction can also remove cardiac information.

## Frozen AF score behavior

| Template model | Positive decisions out of 591 scored | Definite starts | Starts after unavailable PPG | Possible starts at recording boundaries |
| --- | ---: | ---: | ---: | ---: |
| Historical | 591 | 14 | 14 | 17 |
| With quality | 573 | 24 | 13 | 16 |
| With quality and interactions | 525 | 40 | 12 | 13 |

The historical classifier stayed positive on every scoreable decision, so it produced fewer starts even with a larger positive fraction. Every definite historical start followed an unavailable window. The absence of ECG rhythm intervals prevents estimating AF sensitivity or false alerts per monitored hour on this source. Windows overlap and several activities came from the same participants, so the results are descriptive. [Aggregate output](exercise-stress-results.json)

## Compare spectral heart-rate estimates

After the linear subtraction result, we tested a fixed alternative on the same data. Estimate the PPG spectrum and its coherence with the six motion axes using overlapping eight-second segments within each trailing window. Search from 0.75 to 3.5 Hz. The motion-weighted score multiplies PPG power by one minus the highest squared coherence with any axis at that frequency. The formula does not fit to the reference ECG. This follow-up uses the same eight participants and 591 paired windows across all four activities. [Frequency coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html)

| Activity | Direct PPG interval MAE | PPG spectrum peak MAE | Motion-weighted spectrum peak MAE |
| --- | ---: | ---: | ---: |
| All | 25.62 | 41.51 | 19.07 |
| Walk | 40.16 | 46.33 | 22.77 |
| Run | 33.40 | 48.09 | 30.67 |
| Low-resistance bike | 15.37 | 29.38 | 14.10 |
| High-resistance bike | 9.04 | 43.51 | 8.14 |

The motion-weighted estimator gave a 3.57 bpm median error instead of the direct interval estimator's 18.20 bpm. Among the same 465 high-quality windows, large errors dropped from 326 to 185. Mean error remained high for walking and running, and the spectrum yields an average frequency without a beat timeline. [Spectral results](exercise-spectral-results.json)

## Reproduce

Use the published exercise release unpacked into `EXERCISE`, the original PTT template data in `PTT`, and the private frozen wrist locks from the [quality retrospective study](quality-retrospective-study.md) in `RAW`. Install `wfdb` alongside `requirements-af-wrist.txt` for this runner.

```bash
python -m ppg_af.af_wrist_exercise --exercise-dir "$EXERCISE" --ptt-dir "$PTT" \
  --protocol-lock "$RAW/holdout-window-seeded-lock.json" \
  --baseline-lock "$RAW/template-independent-model-lock.json" \
  --variants-lock "$RAW/quality-retrospective-model-lock-private.json" \
  --output runs/exercise-stress-results.json
```

[`clocked_window` and `remove_linear_motion`](../../ppg_af/af_wrist_exercise.py) define the trailing window and the single motion experiment. Only aggregate results are exported.

Add `--spectral-followup` and use a fresh output path to run [`spectral_heart_rate`](../../ppg_af/af_wrist_exercise.py).
