# Require a second positive only for lower-quality PPG

The false-start audit found increased optical interval variation in apparent non-AF alerts. Quality-aware classifiers also tended to start more false alerts at lower optical quality. We tested whether confirmation should depend on the optical input at the current decision. [Timing audit](false-start-audit.md)

## Method

Keep every raw AF probability. When current PPG quality is below 0.40, require both the current probability and the preceding raw probability to exceed the same cutoff. A missing preceding probability makes that low-quality decision unavailable. At higher quality, use the current probability and the original immediate alert rule. The two decisions are ten seconds apart and share 20 seconds of their 30-second optical input. The quality boundary came from the descriptive development monitoring and false-start review.

Compare the conditional rule with the immediate baseline and a rule requiring the preceding positive at all qualities. Test the original cutoffs and cutoffs recalibrated inside each training fold. Verify that the classifiers' probabilities remain identical after changing cutoff selection. [Quality confirmation code](../../ppg_af/af_wrist_quality_confirmation.py)

## Grouped development results

| Grouping and model | Immediate | Lower-quality confirmation at fixed cutoff | Lower-quality confirmation recalibrated |
| --- | --- | --- | --- |
| 2026 plain template | 12/17 detections, 6 false starts | 12/17, 5 | 10/17, 6 |
| 2026 quality interactions | 14/17, 4 | 12/17, 4 | 2/17, 1 |
| 2030 plain template | 1/17, 2 | 1/17, 1 | 1/17, 1 |
| 2030 quality interactions | 1/17, 0 | 1/17, 0 | 1/17, 0 |

Fixed conditional confirmation on the plain template preserved the original total detections and avoided one false start in both development groupings. Calibration under the new rule did not improve the primary held-out result. [Grouped results](quality-confirmation-results.json)

## Retrospective check with a fixed plain template

Replay only the original frozen plain template and original cutoff on both previously scored sets of times from the 25-person roster. None of these participants were used for classifier training or calibration. The previous false-start audit helped motivate this rule, so these outcomes are retrospective. [Replay code](../../ppg_af/af_wrist_confirmation_retrospective.py)

| Scoring set | Original detections | Confirmed detections | Original false starts | Confirmed false starts |
| --- | ---: | ---: | ---: | ---: |
| Historical 260 episodes and 125 ECG-observed wrist minutes | 155/260 | 151/260 | 45 | 39 |
| Additional 241.5 ECG-observed wrist minutes | — | — | 44 | 41 |

Historical sensitivity decreased by 1.54 percentage points, within the preset ten-point tolerance. Six episodes were gained and ten lost. Of 145 episodes detected by both, 68 began alerting later, two earlier and 75 at the same time. The historical false-start rate changed by −2.88 per hour, with a paired participant interval from −5.28 to −0.96. The additional time change was −0.75 per hour with interval −1.97 to +0.51. Historical score coverage declined by 127 evaluation or predecessor decisions. All additional monitoring decisions remained scoreable. The historical sensitivity was 58.1% for the candidate. The data support follow-up of this decision rule while the 80% target and prospective validation remain unresolved. [Retrospective results](confirmation-retrospective-results.json)

## Reproduce

Run `ppg_af.af_wrist_quality_confirmation` on the original wrist archive inputs and pinned clock results from the [pulse consensus study](pulse-consensus-study.md). The retrospective runner additionally uses the original template model lock and sealed inputs from the [earlier quality replay](quality-retrospective-study.md), plus the staged plan, cache, result and PTT data from the [additional monitoring study](retrospective-monitor-study.md). Each runner accepts a fresh `--output`. Model locks and exact participant times remain local.
