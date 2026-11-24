# Calibrate expected AF time from frozen wrist scores

The original wrist alert policy reported a positive state for 32.65% of scoreable half-hour monitoring time. The ECG reference was in AF for 4.06% of the same time. We checked whether probability calibration can reduce this duration bias while preserving rhythm discrimination. The half-hour times were already scored for alerts and retain their original out-of-fit participant roster. [Half-hour wrist replay](half-hour-monitor-study.md)

## Method

1. Keep the original plain-template PPG model fixed. Use only the original five separate calibration participants to fit either probability map. The maps regress exact next-bin ECG AF fractions on the original log odds. The first map uses the original wrist monitoring clocks. The second adds those same participants' disjoint, outcome-blind five-minute development monitoring blocks. All original alert scores and their alert cutoff remain available. [Measured ECG probability comparison](shdb-ecg-reference.md#calibrate-duration-and-alerts-separately)
2. Compare both maps with the original model probabilities and with a constant equal to the observed AF fraction of each calibration input. Score the original half-hour roster without applying new thresholds. Start each block without a prediction and retain only forward bins with a score and an ECG reference. The output contains 44,140 scoreable seconds, of which 1,790 seconds were AF. All AF time in the scoreable test came from one participant. [Probability calibration code](../../ppg_af/af_wrist_state_probability.py)

The original calibration wear contained 1,450 scoreable seconds, including 290 AF seconds from one person. With the added time, it contained 5,480 seconds, including 870 AF seconds from two people. Both calibration sets had higher observed AF fractions than the test.

## Results

| Probability estimate | Duration Brier score | Estimated AF time | Mean probability in AF | Mean probability outside AF |
| --- | ---: | ---: | ---: | ---: |
| Uncalibrated wrist score | 0.2304 | 32.03% | 44.64% | 31.50% |
| Logistic map, original calibration | 0.0531 | 3.94% | 4.09% | 3.94% |
| Logistic map, original plus extra monitoring | 0.0604 | 10.36% | 12.50% | 10.27% |
| Constant, original calibration AF fraction | 0.0643 | 20.00% | 20.00% | 20.00% |
| Constant, expanded calibration AF fraction | 0.0529 | 15.88% | 15.88% | 15.88% |

The original calibration map matched the pooled AF fraction more closely, but assigned almost the same probabilities during AF and non-AF time. Adding monitored calibration time gave more separation and worse Brier error than its constant comparator. The narrow AF support limits inference on either map. The result shows why pooled duration bias and individual rhythm discrimination both matter. [Aggregate results](wrist-state-probability-results.json)

## Reproduce

Run `ppg_af.af_wrist_state_probability` with the locked development archives, the prior five-minute plan and sealed half-hour plan, and monitoring inputs as listed in the [half-hour study](half-hour-monitor-study.md#reproduce). Pass `--half-hour-results experiments/af-wrist/half-hour-monitor-results.json`, `--extension-plan` and `--extension-cache` for the [development monitoring extension](monitor-extension-study.md), and a fresh `--output`. Use the same fixed PTT calibration and historical model lock.
