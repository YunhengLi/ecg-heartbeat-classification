# Alert history and decision cadence

This experiment tests whether the apparent gains from confirmation survive replaying the alert state across the start of each evaluation block. It then tests whether making a decision every 10 seconds helps when all three routes still use the previous 30 seconds of wrist PPG.

The same 10 development participants, feature version, and AF episode boundaries were used throughout. The fixed split reused the frozen models from five fit participants and evaluated them on five calibration participants. The grouped comparison used five outer participant folds and up to three inner folds for calibration. Preceding and added decision rows were used for scoring only. The fit rows, labels, and fold seed stayed fixed. The 25 independent participants did not enter this analysis.

## Methods

1. Recover the 90 seconds before each evaluation block on the original PPG clock. Complete, non-flat PPG was available for 75 of 102 preceding windows across 34 blocks. The 27 remaining windows comprised 18 outside wrist recording, seven flatlines, and two with non-finite samples. All 102 were resolved from the staged windows and cached source waveforms. Timing and ECG metrics still start at the original cutoff.
2. Run the same 30-second decision schedule with its predecessor state restored. Positive decisions already sustaining an alert do not start a new alert at the cutoff. A negative decision, a known signal gap, or model abstention resets the streak. An unknown source window preserves the possible states separately until the decisions resolve them. Across both schedules, the cutoff state was resolved for all reported comparisons.
3. Build the 10-second decision schedule with a trailing 30-second input. Verify continuity of native PPG samples before joining inputs across 30-second archive rows. Reuse features at matching decision times and extract features for the additional times. The three routes had 342 additional scoreable evaluation inputs and 100 additional scoreable history inputs. All 235 original evaluation endpoints and 102 original history endpoints were retained with identical feature arrays.
4. Compare immediate alerts and 30-second confirmation. The latter requires two consecutive positive decisions at 30-second cadence and four at 10-second cadence. Both rules confirm 30 seconds after the first positive if the sequence remains positive. Adjacent 10-second decisions use overlapping PPG samples.
5. Apply each route and rule's previous threshold to both schedules, then recalibrate for each schedule on the same five calibration participants. In the nested comparison, each outer test participant uses only the threshold learned within that fold's training participants. Threshold search includes scores from the predecessor windows and uses the existing target of 80% complete-episode sensitivity, then minimizes wrist false alert starts among thresholds meeting it.

Each schedule covers 6,720 seconds with ECG reference across all evaluation blocks and preserves the same 17 complete and 10 partial AF episodes. The grouped false-alert counts use 2,790 seconds of ECG-observed time in the prescheduled wrist monitoring blocks, including 2,040 seconds of non-AF time. Those counts exclude episode-enriched blocks. The fixed calibration split has 1,500 seconds of ECG-observed wrist monitoring, including 1,200 seconds of non-AF time. Each 10-second cell inherits the reference validity of its original 30-second interval to keep exposure comparable.

## Results

### Effect of restoring predecessor state

The previous thresholds for 30-second confirmation detected 13 of 17 episodes on the fixed calibration split for each route when each block started with an empty alert state. With the same models, cutoffs, and blocks, restoring predecessor state changed that to 2 for direct PPG, 2 for template ECG, and 0 for RDDM ECG. False alert starts changed from 3 to 1, 3 to 1, and 4 to 0, respectively. With immediate alerts at the previous thresholds, the change was smaller. Direct PPG detected 9 → 9 episodes with 4 → 3 false alerts, template ECG 11 → 10 with 4 → 3, and RDDM ECG 10 → 10 with 9 → 8.

The endpoint credits a new alert that starts during a complete AF episode. An alert already active before the episode does not count as a new detection. This explains why carrying state can lower both episode detection and false-alert counts. The previous state-free replay could count a streak already in progress as new inside a clip.

### Nested participant-level cross-validation

Each result is **complete episodes detected out of 17 / wrist false alert starts**. The older results are from the [alert timing study](alert-timing-study.md).

| Alert policy | Signal route | 30 s, empty initial state | 30 s, replayed state | 10 s, replayed state |
| --- | --- | ---: | ---: | ---: |
| Immediate | Direct PPG | 9 / 6 | 7 / 4 | 7 / 4 |
| Immediate | Template ECG | 10 / 7 | 10 / 6 | 12 / 6 |
| Immediate | RDDM ECG | 3 / 9 | 3 / 9 | 8 / 20 |
| 30 s confirmation | Direct PPG | 12 / 3 | 7 / 2 | 7 / 3 |
| 30 s confirmation | Template ECG | 10 / 2 | 0 / 1 | 5 / 3 |
| 30 s confirmation | RDDM ECG | 13 / 7 | 2 / 7 | 4 / 6 |

With replayed history, template ECG at 10-second cadence and immediate alerting detected the most episodes in held-out development folds, 12 of 17 (70.6%), with six false starts. At equal sensitivity for direct PPG, 30-second decisions with confirmation produced two false starts versus four with immediate alerting. Both detected seven episodes. Across routes at 10-second cadence and immediate alerting, RDDM detected eight, one more than direct PPG, with 20 false starts versus four. These comparisons are below the 80% target. The shorter decision interval also enables more classifications and can turn brief score changes or gaps into separate alert starts.

### Fixed-split recalibration

On the five calibration participants, 10-second immediate RDDM was the only replayed route and policy that reached the 80% target after recalibration. It detected 14 of 17 with 17 false starts. Direct PPG and template ECG each detected 12 of 17 with five false starts at the same cadence using their own recalibrated thresholds. The RDDM calibration result fell to 8 of 17 in the outer folds. There is no comparison of false-alert burden at the prespecified 80% sensitivity with held-out participants.

The prescheduled wrist time is short and distributed across participants, so the outcome is reported as observed false starts rather than extrapolating to a day of use. A duration-matched confirmation rule still sees different score sequences on the finer grid. Full fixed-cutoff, recalibrated, and nested results, including event delays and cutoff-state uncertainty, are in [the aggregate JSON](clock-study-results.json).

## Reproduce

Use the frozen development cohort and lock described in the [window-seeded evaluation](window-seeded-evaluation.md#reproduce). The inputs include that cohort's raw decision-window archives and features, the paired AFPPGECG v6 recordings, and the same PTT calibration and upstream RDDM weights used for the original features. Run the commands from the repository root with the [wrist and RDDM dependencies](README.md#code-and-commands) installed.

Use [the prehistory staging module](../../ppg_af/af_wrist_preroll.py) to read the preceding native PPG and [the stride module](../../ppg_af/af_wrist_stride.py) to build the finer decision grid. [The alert replay](../../ppg_af/af_wrist_history.py) tracks active confirmation at each cutoff. [The comparison driver](../../ppg_af/af_wrist_clock_study.py) freezes or recalibrates cutoffs and scores the grouped test folds.

```powershell
$development = 'data/wrist-development'
$developmentFeatures = 'data/wrist-development-features'
$lock = 'runs/wrist-window-seeded-lock.json'

python -m ppg_af.af_wrist_preroll --cohort $development --mat-dir data/afppgecg-v6 --lock $lock --output runs/wrist-history
python -m ppg_af.af_wrist_features --cohort runs/wrist-history --output runs/wrist-history-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda
python -m ppg_af.af_wrist_stride --cohort $development --preroll runs/wrist-history --lock $lock --output runs/wrist-stride10 --feature-output runs/wrist-stride10-features --cohort-features $developmentFeatures --preroll-features runs/wrist-history-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda
python -m ppg_af.af_wrist_clock_study --development-features $developmentFeatures --history-raw runs/wrist-history --history-features runs/wrist-history-features --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --previous-confirmation experiments/af-wrist/alert-timing-results.json --output runs/clock-study-results.json
```

The scoring command checks that the lock's SHA-256 matches the threshold report. If the local lock was regenerated, regenerate the previous thresholds with [the alert timing study command](alert-timing-study.md#reproduce), then supply that report as `--previous-confirmation`.
