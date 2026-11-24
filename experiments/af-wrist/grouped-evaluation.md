# Participant-held-out AF screening comparison

The experiment compares three routes on synchronized AFPPGECG v6 wrist recordings. Every route receives the same 30-second PPG window. Reference ECG supplies the training labels and episode boundaries. [Source dataset](https://doi.org/10.5281/zenodo.23921320)

| Route | Classifier input |
| --- | --- |
| Direct wrist PPG | Nine pulse timing and spectral features |
| Pulse-template ECG | Seven beat-interval features from PPG-timed templates |
| Frozen RDDM ECG | The same seven features after RDDM generation and PPG-guided timing correction |

The template and timing correction use only the external 22-subject sitting fingertip PTT calibration set. The released RDDM model stays frozen. Each reconstruction sees 30 seconds of PPG preceded by two seconds of reflected input to fit the model's four-second blocks. A decision at the 30-second boundary uses samples available by that time. Each classifier is a standardized, class-weighted logistic regression. [PTT calibration source](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) · [RDDM checkpoint setup](../rddm/README.md)

## Splits and operating point

The default is five participant-held-out folds with up to three participant-level inner folds and seed 2026. Scaling and classifier fitting use training rows only. The inner models produce calibration predictions on their own held-out people. For each route, choose a threshold targeting at least 80% complete-episode sensitivity and minimizing validation false-alert starts. Freeze the threshold and use it on the outer test participants. Classifier scores remain usable for window metrics when a fold lacks support for episode threshold calibration.

The primary analysis uses shared feature support. A 30-second window with missing PPG or without usable beat intervals on any route leaves all three routes unavailable. It stays on the timeline. The result also records how many wrist windows each route could score before the shared requirement.

The training sampler uses ECG and PPG. It skips acceleration samples, so differences in the auxiliary channel's duration do not remove an otherwise usable ECG and PPG pair.

## Monitoring and episode samples

- **Calendar coverage.** A seeded 150-second block per participant is sampled on the ECG clock using timestamps alone. Keep time outside active PPG segments in the denominator.
- **Wrist-monitoring false alerts.** A separate seeded 300-second block is drawn from wrist recording timestamps without using labels or waveform values. Retain gaps inside the block. Report false alert starts and ECG-observed time in these sampled blocks. Independent blocks reset alert state at the start.
- **Episode sensitivity.** Use fully observed episodes in the monitoring blocks and up to two additional short AF episodes with context per participant. A complete selected run fits within 480 seconds. Keep clipped episodes separate. The extra episode windows contribute sensitivity observations without adding time to the false-alert denominator.
- **Classifier fitting.** Sample up to two stable windows per rhythm per participant in addition to monitoring time. Exclude the separately selected episode runs from fitting. The auxiliary window analysis evaluates discrimination and the false-positive rate on stable selected windows under thresholds chosen on inner folds. Its class proportions reflect the sampling design.

An alert starts at the end of the first positive window. Consecutive positives count as one alert. A complete AF episode is detected when a new alert starts while it is active. The comparison flags an episode result as comparable only when the outer test sensitivity range is within 10 percentage points, each route reaches at least 70% held-out sensitivity, and the validation target was met in every fold. The short wrist blocks support observed counts and a rate per sampled hour. Longer continuous wrist recordings are needed for a daily false-alert estimate.

## Recorded pilot

The paced retrieval recovered eight protocol-complete participants out of 35 paired people in the release. Five outer folds assessed 172 total windows. Five participants supplied eligible AF fitting windows, and there were three complete reference AF episodes. The wrist-monitoring blocks supplied 36.5 minutes of ECG reference time with 75.3% shared analyzable coverage.

| Route | Complete AF episodes detected | False alerts during wrist monitoring | Folds meeting the validation target |
| --- | ---: | ---: | ---: |
| Direct wrist PPG | 0/3 | 3 | 1/5 |
| Pulse-template ECG | 0/3 | 3 | 1/5 |
| RDDM ECG | 0/3 | 2 | 1/5 |

The episode comparison is unresolved at the target sensitivity. Each route could meet the 80% calibration target in one fold. The other folds used the best validation sensitivity available from their candidate thresholds. The RDDM route recorded one fewer false alert, and all three routes missed all three complete episodes.

The auxiliary window comparison applies thresholds calibrated against stable window sensitivity. It pools held-out scores and counts from the five outer folds.

| Route | Window AUROC | AF windows detected | Non-AF windows flagged |
| --- | ---: | ---: | ---: |
| Direct wrist PPG | 0.805 | 23/27 | 11/67 |
| Pulse-template ECG | 0.821 | 24/27 | 11/67 |
| RDDM ECG | 0.658 | 19/27 | 43/67 |

Direct PPG and the template were within four percentage points of each other in window sensitivity and tied at 11 false-positive windows. The RDDM sensitivity was outside the ten-point tolerance. These window counts describe the selected windows and their validation-calibrated thresholds. [Aggregate results and fold calibration](grouped-results.json)

## Reproduce

Install `requirements-af-wrist.txt` and follow the [RDDM setup](../rddm/README.md) to obtain the pinned upstream model, weights, and PTT records. Use the public record file prefixes listed in the [recorded results](grouped-results.json) to reproduce this subset. The same commands accept the full paired v6 cohort. Stage one archive per participant. Keep the inputs and feature caches outside Git.

```powershell
python -m pip install -r requirements-af-wrist.txt
python -m ppg_af.af_wrist_cohort --ecg data/NNN_ECG.mat --ppg data/NNN_PPG.mat --output data/wrist-staged/NNN.npz --monitor-seconds 150 --wear-seconds 300 --fit-per-class 2 --episode-count 2 --max-episode-run-seconds 480
python -m ppg_af.af_wrist_features --cohort data/wrist-staged --output data/wrist-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda --batch-size 12
python -m ppg_af.af_wrist_train --features data/wrist-features --monitor-kind monitor_wear --folds 5 --inner-folds 3 --target-sensitivity 0.8 --sensitivity-tolerance 0.1 --output runs/wrist-grouped-results.json
```

The inference command also supports `--device cpu`. Use a fresh feature directory when changing the model weights, PTT calibration data, or inference settings.
