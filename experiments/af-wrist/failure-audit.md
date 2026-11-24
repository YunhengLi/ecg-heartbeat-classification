# Where the wrist AF alerts fail

The audit traces all 12 configurations from the [decision cadence study](clock-study.md). It covers three PPG-derived routes, two decision cadences, and immediate alerting or 30-second confirmation. Every score is from an outer fold held out by participant, with its threshold selected inside the training folds. All configurations use the same trailing 30 seconds of PPG, ECG reference intervals, pre-block alert state and 10 development participants. The 25 independent participants stayed reserved.

## What was counted

The primary outcome remains a definite new alert that starts during a complete ECG-reference AF episode. An alert at the decision timestamp uses the ECG rhythm just before that timestamp. The audit separates an alert active at onset from a confirmed alert observed at a later in-episode decision. That distinction matters when a pre-onset alert clears at the first decision in AF.

Each of the 17 complete episodes is traced in every configuration. The 10 partial episodes stay outside the complete-episode denominator. A missed episode receives one primary attribution in priority order. The audit checks for uncertain starts and absence of decision endpoints, then an alert active at onset, score availability, cutoff crossing, and confirmation. A primary attribution describes an observed pattern, and lower-priority patterns can coexist.

False alert starts use only the prescheduled wrist-monitoring blocks, which contain 46.5 minutes of usable ECG reference, including 34 minutes of non-AF time. The audit checks how long each alert remains active after its start, what preceded a restart, and whether annotated AF appears within the trailing input. Duration is a lower bound for alerts still active at a block boundary. The heuristic PPG quality is compared only at scored decisions.

## Missed episodes

| Cadence | Alert rule | Route | Detected / 17 | Already active at onset | Below cutoff | Confirmation incomplete | No score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 30 s | Immediate | Direct PPG | 7 | 9 | 0 | 0 | 1 |
| 30 s | Immediate | Template ECG | 10 | 5 | 1 | 0 | 1 |
| 30 s | Immediate | RDDM ECG | 3 | 2 | 11 | 0 | 1 |
| 30 s | 30 s confirmation | Direct PPG | 7 | 7 | 1 | 1 | 1 |
| 30 s | 30 s confirmation | Template ECG | 0 | 13 | 1 | 2 | 1 |
| 30 s | 30 s confirmation | RDDM ECG | 2 | 13 | 0 | 1 | 1 |
| 10 s | Immediate | Direct PPG | 7 | 9 | 0 | 0 | 1 |
| 10 s | Immediate | Template ECG | 12 | 2 | 2 | 0 | 1 |
| 10 s | Immediate | RDDM ECG | 8 | 1 | 7 | 0 | 1 |
| 10 s | 30 s confirmation | Direct PPG | 7 | 7 | 1 | 1 | 1 |
| 10 s | 30 s confirmation | Template ECG | 5 | 7 | 0 | 4 | 1 |
| 10 s | 30 s confirmation | RDDM ECG | 4 | 1 | 0 | 11 | 1 |

Across 204 episode-configuration observations, 72 had a new in-episode alert. Of the 132 misses, 76 already had an alert active at onset, 24 stayed below the fold cutoff, 20 crossed it without reaching confirmation, and 12 produced no score. In 64 of the 76 onset-active misses, a confirmed alert was also active at an in-episode decision. The other 12 had no confirmed alert at any in-episode decision. Both onset-active misses in the best template ECG configuration fell in this second group. All four of its scored misses stayed below the fold cutoff during AF, with three scoring within 0.03 of it at their best in-episode decision. The fifth miss was the common flatline case.

One unique episode was missed by all 12 configurations. Every trailing PPG input for that episode was flat, including 11 decision endpoints on the 30-second schedule and 34 on the 10-second schedule. The other 16 unique episodes were detected by at least one configuration. Three episodes were shorter than the feature window. On each cadence, 13 of 17 episodes had a scoreable 30-second input wholly inside AF. The other scoreable episodes needed a window that mixed reference rhythms.

## False alert behavior

| Cadence | Alert rule | Route | False starts | Active <30 s | Active ≥30 s | Cut off before 30 s | One positive then reset |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 30 s | Immediate | Direct PPG | 4 | 0 | 3 | 1 | 1 |
| 30 s | Immediate | Template ECG | 6 | 0 | 4 | 2 | 1 |
| 30 s | Immediate | RDDM ECG | 9 | 0 | 8 | 1 | 4 |
| 30 s | 30 s confirmation | Direct PPG | 2 | 0 | 2 | 0 | 0 |
| 30 s | 30 s confirmation | Template ECG | 1 | 0 | 1 | 0 | 0 |
| 30 s | 30 s confirmation | RDDM ECG | 7 | 0 | 6 | 1 | 0 |
| 10 s | Immediate | Direct PPG | 4 | 2 | 1 | 1 | 2 |
| 10 s | Immediate | Template ECG | 6 | 2 | 3 | 1 | 1 |
| 10 s | Immediate | RDDM ECG | 20 | 16 | 3 | 1 | 14 |
| 10 s | 30 s confirmation | Direct PPG | 3 | 1 | 2 | 0 | 0 |
| 10 s | 30 s confirmation | Template ECG | 3 | 0 | 3 | 0 | 0 |
| 10 s | 30 s confirmation | RDDM ECG | 6 | 2 | 4 | 0 | 0 |

The 71 false starts are observations across configurations on the same monitoring periods. None had annotated AF in its trailing PPG input. At 10-second cadence, 14 of RDDM's 20 immediate false alerts followed a single positive decision that then reset. Fourteen also occurred within 30 seconds after a previous alert reset on a negative decision. These two counts can overlap. Template ECG at the same cadence produced one single-positive false start out of six, and three of its alerts stayed active for at least 30 seconds.

For the 10-second immediate template, median PPG quality was 0.416 at false starts and 0.585 across scored non-AF monitoring decisions. The quality measure is a waveform heuristic and this comparison is descriptive. Motion measurements were incomplete across the audited inputs, so these observations cannot establish motion as a cause. The clock and score traces identify alert-level failure patterns. They do not determine whether a specific reconstructed ECG beat caused a classification error.

## Implications for the next experiment

Keep the 10-second immediate template as a reference configuration. Its near-cutoff misses justify inspecting the score trajectories before onset and during AF while keeping the existing new-alert endpoint. RDDM's isolated positives make a shorter confirmation rule, such as two positives in three decisions at 10-second cadence, a candidate for a controlled comparison. First replay at identical cutoffs to isolate the effect of confirmation, then compare at similar fold-calibrated sensitivity. The single flatline episode needs a usable measurement.

## Reproduce

Prepare the same locked development inputs described in [the clock study](clock-study.md#reproduce). Run the audit with the original raw cohort, features, and both replay grids. The command refits the same nested folds and verifies the detected episode counts, false starts and median delay against the clock study before saving the aggregate.

```powershell
python -m ppg_af.af_wrist_failure_study --development-raw $development --development-features $developmentFeatures --history-raw runs/wrist-history --history-features runs/wrist-history-features --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results runs/clock-study-results.json --output runs/failure-audit-results.json
```

[`audit_events`](../../ppg_af/af_wrist_failure_events.py) traces score, active alert and reference state on each decision clock. [`load_input_evidence`](../../ppg_af/af_wrist_failure_inputs.py) inspects the whole trailing waveform for flats and gaps. [`audit_route`](../../ppg_af/af_wrist_failure_study.py) joins the two and reconciles them with the published counts. The [aggregate results](failure-audit-results.json) include miss attributions, alert persistence, score-to-threshold margins and signal quality.
