# Ten-second PPG-derived ECG on the historical wrist cohort

The historical duration audit found that the 30-second template missed all 12 complete AF episodes of ten seconds or less. This experiment tests a model with a shorter PPG input on the same episodes and longer representative wrist wear. [Episode duration audit](confirmation-subgroups-study.md)

## Method

1. Rebuild template ECG timing features from the final ten seconds of each original trailing PPG window. Keep the external PTT-derived pulse template, rate, feature definitions and ten-second decision clock.
2. Fit standardized, class-balanced logistic regression with `C=1` on the same 43 stable inputs from five original participants. Freeze two short-model thresholds using the separate five original calibration participants. The target rule chooses the fewest false starts among cutoffs reaching 80% sensitivity. The matched rule chooses the fewest false starts at the closest sensitivity to the original 30-second model on that development calibration clock.
3. Apply both frozen shorter-input thresholds to the same historical 25-person roster. Compare only rows where both pulse lengths can be scored and retain the complete ECG reference. Repeat with the original adjacent-score rule using the original 30-second optical quality below 0.40. This preserves the confirmation condition while changing the rhythm model's input length.
4. Apply the same frozen models, thresholds and rules to the 25 sealed half-hour blocks. Compare new starts, scoreable coverage and confirmed rhythm time. Group bootstrap uncertainty by participant.
5. For the matched short model with quality confirmation, add a fixed rearming rule that waits for two observed negative decisions before counting another start. An unavailable input retains the original reset behavior. Reuse every score and cutoff from that short model.

The [historical runner](../../ppg_af/af_wrist_short_retrospective.py) freezes both candidate cutoffs before reading the historical inputs. The [half-hour runner](../../ppg_af/af_wrist_short_monitor.py) uses the same fitting function and replays the same ECG clock.

## Results

All 881 originally scoreable development windows also supported ten seconds. In calibration, the original 30-second model detected 12/17 with five false starts. The 10-second target and matched cutoffs detected 14/17 with nine starts and 12/17 with ten starts.

| Frozen input and rule | Historical complete AF detected | Historical false starts in 125 minutes | Half-hour false starts in 744 minutes |
| --- | ---: | ---: | ---: |
| 30 seconds, original cutoff | 155/260 | 47 | 209 |
| 10 seconds, target cutoff | 197/260 | 77 | 353 |
| 10 seconds, matched cutoff | 199/260 | 69 | 350 |
| 30 seconds, adjacent optical-quality confirmation | 151/260 | 41 | 170 |
| 10 seconds, target and adjacent confirmation | 179/260 | 58 | 262 |
| 10 seconds, matched and adjacent confirmation | 175/260 | 50 | 251 |
| 10 seconds, matched and confirmed with two-negative rearm | 166/260 | 44 | 196 |

There were 21 historical evaluation and nine history rows where the 30-second model had support and the shorter one did not. The matched 30-second control has two more false starts than the original full-support 45 because unavailable decisions reset the alert state. The common shorter-input support removed 28 rows out of 4,474 scoreable half-hour decisions. [Historical results](short-retrospective-results.json) · [Half-hour results](short-monitor-results.json)

The target cutoff recovered two of 12 episodes up to ten seconds. Its paired overall historical sensitivity increase was 16.2 percentage points, with a participant-bootstrap interval of 11.3 to 28.5 points. The matched cutoff recovered one of the 12 shortest episodes. It increased historical sensitivity by 16.9 points while adding 10.56 false starts per observed hour. The matched cutoff produced a different sequence of new starts, so a lower calibration sensitivity did not imply a lower historical episode detection count.

Even with adjacent optical-quality confirmation, the matched short model produced 81 more false starts than its 30-second comparator in the half-hour blocks. That corresponds to +6.53 starts per ECG-observed hour, with a 95% paired interval from +2.17 to +11.30. The blocks had no complete AF episodes. Observed non-AF time positive fell from 28.6% to 19.3% while starts increased. This distinction matters when deciding whether a model is appropriate for continuous wear.

The two-negative rearm reduced short-model false starts by 55 in the half-hour blocks, or 4.44 per ECG-observed hour with a participant interval of 2.33 to 6.75 fewer. Historical sensitivity decreased from 175/260 to 166/260. Compared with the 30-second confirmed route, the rearmed short route still had 26 more half-hour false starts, or +2.10/h, with an interval from −0.88 to +5.27. The two strategies require a new cohort with complete episodes and representative wear time to compare this trade-off.
