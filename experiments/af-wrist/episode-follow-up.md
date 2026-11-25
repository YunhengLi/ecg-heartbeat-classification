# Missed AF episodes and the independent wrist test

This follow-up replays the three complete ECG-confirmed episodes from the eight-participant pilot, then freezes a classifier and its alert threshold for each route before applying them to unused AFPPGECG participants. [Original grouped pilot](grouped-evaluation.md) · [AFPPGECG v6 source](https://doi.org/10.5281/zenodo.23921320)

## Why the pilot missed the episodes

The replay rebuilt the five participant-held-out fits from the original feature archives and applied the recorded fold thresholds. It recovered the recorded alert counts and all three missed episodes. The diagnosis was the same for direct PPG, pulse-template ECG, and RDDM ECG.

| Complete episodes | Shared feature support | What happened |
| ---: | --- | --- |
| 2 | All six ECG-valid decision windows had scores | Both were in one held-out fold. Its calibration reference had one complete episode with flat PPG. Every candidate threshold produced zero episode detections in calibration. The tie-break chose the no-alert threshold above 1. |
| 1 | All 11 ECG-valid decision windows had numerically complete PPG with zero peak-to-peak amplitude | The waveform was flat, so none of the routes could extract usable pulse or beat features. |

The replay traces one failure to unusable PPG and two to threshold calibration. With only two episodes that can produce scores, sensitivity on all three reference episodes has a ceiling of 66.7% under the shared feature rule. The flatline remains in the denominator. [Recorded replay](episode-replay-results.json)

## Independent test protocol

The original eight participants form the development cohort. Split them into disjoint fitting and threshold-calibration groups using their ECG reference and feature availability. The calibration group includes all three complete development episodes. Fit each standardized logistic classifier on the fitting group. Choose alert thresholds in the calibration group with an 80% complete-episode sensitivity target and the existing false-alert objective. Freeze the models, thresholds, test roster, and sensitivity comparison before opening any new participant's test scores.

The 27 other paired participants form the eligible test roster. As in the original benchmark, wrist monitoring blocks come from recording timestamps without looking at rhythm labels. Use complete reference episodes with observable context for sensitivity and retain PPG gaps. Compare false alerts over the same ECG-observed wrist-monitoring time. Window-level classification remains a separate diagnostic outcome.

Timestamp screening can establish that an ECG-confirmed episode lies inside a wrist recording segment. Waveform-level availability and feature support require decoding the corresponding samples. Report those checks separately.

The freeze used five participants for fitting and three for threshold calibration. Every route detected two of the three complete calibration episodes. The remaining episode had no ECG-valid shared-score endpoint. This limits sensitivity on this calibration set to 66.7%. The three frozen episode thresholds therefore carry the status `target_not_met`.

## Recorded follow-up

The survey verified ECG annotations for 14 of the 27 unused participants and completed the annotation and timing checks for five. Four of those five contributed 29 candidate episodes under the existing duration and recording rules. Their count precedes waveform validation.

Source reads timed out or returned server errors on several requested ranges. The recovered scheduled runs supplied 43 windows from two new participants and two complete ECG-confirmed AF episodes. Both episodes had usable shared scores. The independent wrist blocks supplied ten minutes of ECG reference, including five minutes of non-AF rhythm, with full shared feature coverage.

| Route | Complete AF episodes detected | Wrist false alerts | Stable AF windows detected | Stable non-AF windows flagged |
| --- | ---: | ---: | ---: | ---: |
| Direct PPG | 0/2 | 2 | 12/15 | 2/21 |
| Pulse-template ECG | 1/2 | 3 | 12/15 | 2/21 |
| RDDM ECG | 2/2 | 2 | 3/15 | 11/21 |

Stable windows use their separately calibrated thresholds. The episode sensitivities differ substantially, and the calibration target was unmet. The short test leaves the matched-sensitivity false-alert comparison unresolved. [Aggregate holdout results and survey](independent-holdout-results.json)

## RDDM batch composition check

The author runner initializes its random seed once per input archive and draws diffusion noise as it processes successive chunks. A control reran the same five PPG windows in one episode run twice, once with the original preceding windows and once with the expanded archive. The three frozen classifiers used the same parameters in both conditions. RDDM feature vectors changed for all five windows, with a maximum absolute AF score difference of 0.268. The differences for direct PPG and template ECG were zero. The reported holdout counts use the expanded archive and its recorded inference configuration. [RDDM runner](../../ppg_af/rddm.py) · [Batch audit counts](independent-holdout-results.json)

## Run the replay and independent test

The recorded holdout on this page uses archive-seeded features from [the original follow-up revision](https://github.com/YunhengLi/ecg-heartbeat-classification/tree/6c797f7e28aaba8588c7aa96d853181766d7006b). The current wrist implementation uses the [window-seeded protocol](window-seeded-evaluation.md).

First use the [original staging and feature extraction commands](grouped-evaluation.md#reproduce) to prepare the development archives. Preserve the private model lock separately from the recordings and the results.

```powershell
python -m ppg_af.af_wrist_replay --features data/wrist-features --waveforms data/wrist-staged --expected experiments/af-wrist/grouped-results.json --output runs/episode-replay.json
```

Create a local JSON array of the 27 public record prefixes in the v6 paired inventory after removing the eight prefixes from [the grouped pilot](grouped-results.json). Stage test waveforms and features in separate directories using the same window settings. The cohort command accepts `--survey-only` for an annotation and timestamp check before loading waveform samples.

```powershell
python -m ppg_af.af_wrist_cohort --ecg data/NNN_ECG.mat --ppg data/NNN_PPG.mat --survey-only --wear-seconds 300 --max-episode-run-seconds 480 --output runs/wrist-support.json
python -m ppg_af.af_wrist_holdout freeze --development-features data/wrist-features --calibration-subject 005 --calibration-subject 012 --calibration-subject 039 --test-roster data/wrist-test-roster.json --output runs/wrist-model-lock.json
python -m ppg_af.af_wrist_holdout evaluate --lock runs/wrist-model-lock.json --test-features data/wrist-test-features --output runs/wrist-holdout.json
```
