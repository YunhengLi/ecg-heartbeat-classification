# Native ECG timing and wrist PPG on the same AF clock

This experiment measures how much of the wrist alert gap remains when the model receives beat times from actual ECG. It uses the same AFPPGECG v6 people, annotated episodes and decision times as the existing wrist study. [AFPPGECG v6 source](https://zenodo.org/records/23921320) · [Short-template results](short-retrospective-study.md)

## Method

1. Read the original 500 Hz ECG QRS annotations. Convert the archive's one-based sample indices and compute the existing seven timing features in trailing 10- and 30-second inputs. Require three measured beats and reject windows spanning a missing or unsupported RR interval. Stage only feature rows and verify their source sample endpoints against the sealed wrist archive.
2. Fit two standardized balanced logistic models on the original 43 stable examples. Four people contribute those examples from the five assigned fitting participants. Use the other five original development participants for threshold selection. For each ECG route, choose the original 80% objective and the closest development sensitivity to the frozen wrist template, with false starts used to break ties.
3. Apply the frozen scores on the same 260 complete historical events and 125 ECG-observed monitoring minutes. The primary comparison uses the intersection of both ECG windows and the original wrist PPG support. Repeat native ECG on all its eligible intervals to measure the effect of optical availability separately.

The [ECG staging code](../../ppg_af/af_wrist_ecg_timing_stage.py) preserves the exact input end. The [historical ECG replay](../../ppg_af/af_wrist_ecg_timing_replay.py) uses measured QRS times as features and adjudicated AF intervals as outcomes. This is a measured timing reference, with ECG beats provided by the release. An end-to-end raw-signal device would also require a validated beat detector.

## Historical results

The original wrist cutoff detected 12/17 episodes with five false starts in the development calibration group. The matched 10-second ECG cutoff also detected 12/17, with two false starts. Its 80% target cutoff detected 14/17 with five. The 30-second ECG route reached at most 10/17 with two false starts.

| Route | Historical detections on shared support | False starts over 125 minutes |
| --- | ---: | ---: |
| Original wrist PPG template | 155/260 | 45 |
| Native ECG 10 seconds, calibration-matched | 176/260 | 2 |
| Native ECG 10 seconds, 80%-target cutoff | 142/260 | 14 |
| Native ECG 30 seconds, closest available cutoff | 128/260 | 2 |

The matched 10-second ECG had an 8.1-point sensitivity increase. The paired 95% interval was −15.5 to +15.3 points across participants. It reduced false starts by 20.64 per ECG-observed hour, with a paired interval of 12.48 to 29.28 fewer starts per hour. The matched ECG route detected 178/260 on full native support with two false starts. The shared evaluation retained 5,414 scoreable cells, versus 5,424 for the original PPG and 5,453 for ECG. [Historical aggregate results](ecg-timing-replay-results.json)

## Continuous wrist blocks

Stage native QRS features on the 25 sealed half-hour blocks and their causal history. Reapply the frozen PPG and native ECG models. Compare the four routes on the 4,403 decisions where each can be scored. Verify that the two published optical replays reproduce on their own previously reported support before comparing the intersection.

| Route on common input support | False starts over 744 ECG-observed minutes | ECG-confirmed non-AF time positive | ECG-confirmed AF time positive |
| --- | ---: | ---: | ---: |
| Original 30-second PPG template | 211 | 31.3% | 46.7% |
| Matched 10-second PPG template | 350 | 23.6% | 32.8% |
| Matched 10-second native ECG | 74 | 2.1% | 0.0% |
| 30-second native ECG, closest available cutoff | 69 | 5.6% | 8.3% |

On the development calibration group, both matched 10-second models detected 12/17 episodes. On the half-hour blocks, the short ECG route produced 22.26 fewer false starts per ECG-observed hour than the short PPG route, with a 95% participant-bootstrap interval of 12.96 to 31.97 fewer. The blocks contained no complete AF episodes. One person supplied all ECG-confirmed AF time, and the frozen matched 10-second ECG model gave no positive decisions during that time. This comparison estimates false-start burden with limited evidence on sensitivity. [Continuous runner](../../ppg_af/af_wrist_ecg_timing_monitor.py) · [Continuous aggregate results](ecg-timing-monitor-results.json)
