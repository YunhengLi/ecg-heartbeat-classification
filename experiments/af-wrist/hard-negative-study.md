# Add irregular but ECG-confirmed non-AF wrist windows

The additional development monitoring supplies new non-AF optical signals from the original participants. This study tests whether training on the cases with the most irregular template intervals reduces false alerts. The validation and calibration stay on the original development clock. [Source monitoring replay](monitor-extension-study.md)

## Method

Divide each prescheduled five-minute block into two equal time bins. Within a bin, require the full 30-second PPG input to be scoreable and all three 10-second ECG reference cells to be observed without overlapping an annotated AF episode. Rank candidates by the sum of normalized RMSSD and pNN50 from the template intervals. Pair the highest-ranked eligible window with a fixed-seed random nonoverlapping negative from the same person and time bin. Each arm adds 42 windows from nine people. The median selection score is 1.201 for the random arm and 1.801 for the irregular arm.

Refit the original quality-interaction logistic classifier. Run the baseline, random-negative arm, and irregular-negative arm on the same two original participant groupings. In every inner and outer fit, add only a fitting participant's new negatives. Keep all original AF training rows and evaluate on the original held-out inputs. The additional monitoring is now a training source for this experiment. [Training and selection code](../../ppg_af/af_wrist_hard_negatives.py)

## Results

| Participant partition | Original quality template | Matched random negatives | High-irregularity negatives |
| --- | --- | --- | --- |
| 2026 | 14/17 detections, 4 false starts | 9/17, 1 | 10/17, 2 |
| 2030 | 1/17, 0 | 2/17, 2 | 1/17, 0 |

Both added-negative arms lost more than ten sensitivity points on the original split. The irregular arm retained one more detection than the matched random arm and added one false start. The second grouping retained its weak baseline with irregular training. [Aggregate results](hard-negative-results.json)

Run `python -m ppg_af.af_wrist_hard_negatives` with the private monitoring plan and cache, PTT calibration inputs, and the same `--lock`, development archives, clock and quality receipts as the [monitoring study](monitor-extension-study.md#reproduce). The runner uses `_fit_expanded` to keep supplemental data in the right folds.
