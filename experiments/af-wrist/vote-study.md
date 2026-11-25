# Two-of-three alerting for RDDM ECG

The [failure audit](failure-audit.md) found that 14 of 20 immediate RDDM false starts at 10-second cadence followed a single positive decision. This study tests whether requiring a second positive within the previous 20 seconds reduces false starts while preserving episode detection.

## Method

Each decision still uses a trailing 30-second PPG input. The rule opens an alert at a positive decision if at least one of the preceding two decisions was also positive. For example, positive, negative, positive opens on the third decision. A negative ends an active alert and can occupy one place between two positives. A missing decision clears the vote history and ends the alert. The same preceding PPG decisions initialize the alert state before each scored block.

The comparison uses five participant-held-out development folds and the original feature, fit and ECG reference rows. Outer test scores come from a model fitted on other participants. Multiple cutoff policies reuse the same scores and training-fold calibration predictions.

1. Reproduce the immediate-alert cutoff for each outer fold and apply it to both rules. This isolates what the new rule does to the same decisions.
2. Select a voting cutoff inside each outer fold's training participants using the original 80% complete-episode sensitivity target. The search minimizes false starts among cutoffs meeting the target and otherwise retains the best achievable sensitivity.
3. Select another voting cutoff in the same inner predictions to match each immediate rule's achieved inner sensitivity within 10 percentage points. Among cutoffs in that band, use the one with fewest false starts. If no cutoff is in the band, use the closest sensitivity and report it as outside the band. The held-out results show whether the match transfers to new participants.

The 17 complete AF episodes come from all original evaluation scopes. False starts come from the same 46.5 minutes of ECG-observed prescheduled wrist monitoring, including 34 minutes of non-AF time. Look for a definite new alert during an episode and measure delay from the reference onset. When comparing timing, pair delays for episodes detected by both rules. False starts match only at the same decision time for the same participant.

## Results

| 10-second policy | Complete episodes detected | Wrist false starts | Median additional delay on shared RDDM detections |
| --- | ---: | ---: | ---: |
| RDDM immediate | 8/17 (47.1%) | 20 | Reference |
| RDDM two-of-three, fixed cutoffs | 6/17 (35.3%) | 11 | 15 s |
| RDDM two-of-three, recalibrated to 80% target | 6/17 (35.3%) | 11 | 15 s |
| RDDM two-of-three, matched inner sensitivity | 6/17 (35.3%) | 11 | 15 s |
| Template ECG immediate, prior reference | 12/17 (70.6%) | 6 | — |

At the existing cutoffs, the two-of-three rule reduced false starts from 20 to 11 and detections from 8 to 6. It removed nine exact false starts, all single-positive starts in the previous audit. Both lost episodes had only one positive in-episode decision. Among the six episodes detected by both rules, four started later and two kept the same start. The paired median increase in delay was 15 seconds. The median among all detected episodes fell from 29.2 to 25.6 seconds because the set of detected episodes changed.

The two recalibration objectives selected identical voting cutoffs in all five folds. The vote reached the 80% inner target in one fold, compared with three for immediate RDDM. All five matched calibrations were inside the inner sensitivity band. On the outer participants both approaches lost the same two episodes and picked up no new ones. The held-out sensitivity gap was 11.8 percentage points, beyond the prespecified 10-point tolerance. After recalibration nine false starts matched immediate RDDM exactly, while 11 were only in the immediate policy and two only in the vote policy.

The vote reduced observed false starts at the cost of lower sensitivity and later alerts on shared episodes. The development results do not support adopting it as an improvement at comparable sensitivity. Template ECG at 10-second cadence remains the development benchmark. A separate score analysis can investigate its near-cutoff misses while preserving the original episode endpoint.

## Reproduce

Use the locked development features, 10-second grid and clock report from the [decision cadence study](clock-study.md#reproduce). The command refits the RDDM AF classifier on the original folds and checks that its immediate-alert episode and false-start results reproduce before evaluating voting. It reads only the development participants. The 25 independent participants remain reserved.

```powershell
python -m ppg_af.af_wrist_vote_study --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results runs/clock-study-results.json --output runs/vote-study-results.json
```

[`apply_two_of_three_alerts`](../../ppg_af/af_wrist_vote.py) replays the causal vote. [`fit_comparison`](../../ppg_af/af_wrist_vote_study.py) sets thresholds on training participants. [`compare_alert_runs`](../../ppg_af/af_wrist_vote_compare.py) pairs episode outcomes and false starts on the ECG clock. The [aggregate results](vote-study-results.json) include calibration status and event-level changes as counts.
