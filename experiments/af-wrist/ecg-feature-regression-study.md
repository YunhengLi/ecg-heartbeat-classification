# Predict measured ECG interval statistics from wrist PPG

We asked whether learning seven ECG timing features from paired ECG and PPG would help AF alerts. The prediction target was the ECG interval representation used by the classifier. The earlier waveform network struggled to locate measured beats. [TCN waveform comparison](native-tcn-study.md)

## Method

Use 223 verified stable wrist PPG and ECG pairs from nine development participants. Build targets from the native 500 Hz ECG beat annotation clock. Extract seven original pulse-template statistics, nine direct optical features and the PPG quality scalar from each input. Each fit standardizes its optical inputs and learns a multi-output ridge regression with alpha 8. Clip estimates to the range of the corresponding measured feature in that fit.

First evaluate the seven predictions against ECG on five held-out groups. For alerts, fit a separate regressor inside every inner or outer classification fit. Its paired examples come only from the fitting participant roster. Compare replacing the seven template timing columns and adding the seven estimates while retaining all original columns. Both variants and the quality-interaction reference use the original clock, stable labels, window coverage, logistic regularization and grouped calibration. There are 20 regression fits per candidate and grouping. [Regression and fold checks](../../ppg_af/af_wrist_ecg_feature_regression.py)

| Held-out median absolute error in combined irregularity | Pulse template | Regressed ECG statistics |
| --- | ---: | ---: |
| All 223 paired windows | 0.277 | 0.379 |
| 77 stable AF windows | 0.198 | 0.390 |
| 146 stable non-AF windows | 0.387 | 0.377 |

The combined statistic is normalized successive interval RMSSD plus pNN50. The median ECG interval error also rose from 40 to 130 ms across all pairs. The small decrease in non-AF irregularity error came with larger errors on AF windows.

| Alert representation | 2026 complete AF | 2026 false starts | 2030 complete AF | 2030 false starts |
| --- | ---: | ---: | ---: | ---: |
| Quality-interaction template | 14/17 | 4 | 1/17 | 0 |
| Replace template timing features | 10/17 | 1 | 1/17 | 0 |
| Add regressed timing features | 11/17 | 2 | 1/17 | 0 |

The 2026 candidates lost 23.5 and 17.6 percentage points of sensitivity, outside the locked ten-point tolerance. Both stayed at one detected event under the alternate grouping. The lower false-start counts support keeping the learned features as a development result and retaining the existing template and confirmation comparison. [Aggregate results](ecg-feature-regression-results.json)

## Reproduce

Run `ppg_af.af_wrist_ecg_feature_regression` with the locked development inputs from the [quality confirmation study](quality-confirmation-study.md#reproduce), the `--ptt-dir` used to fit the pulse template, and the `--waveform-pairs` and `--waveform-results` inputs used in the [native beat transfer study](native-neural-beat-study.md). Set `--output` to a new path.
