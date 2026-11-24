# Wrist PPG for atrial fibrillation screening

Does reconstructing an ECG-like waveform from wrist PPG improve atrial fibrillation (AF) screening when the optical signal is noisy? The first stage checks whether synchronized wrist PPG, reference ECG, rhythm labels, and wrist acceleration can support that comparison.

## Data and window rules

The AFPPGECG v6 release contains 39 ECG files and 35 matched wrist PPG files. Those 35 pairs total 56.48 GiB. The reader keeps green PPG at 100 Hz, reference ECG at 500 Hz, and three-axis wrist acceleration at 50 Hz. It places each wrist segment on the ECG clock using the release timestamps and preserves missing samples. [AFPPGECG v6 data and license](https://doi.org/10.5281/zenodo.23921320)

AF labels come from the release's ECG annotations. One annotation applies to the interval between its two successive R peaks. A stable 30-second window needs one rhythm throughout the window and a one-second margin on both sides. A window near a transition stays separate. Missing ECG samples, unsupported labels, and RR intervals longer than the pilot's three-second reference limit can leave time unscorable. A PPG gap counts against PPG availability even if the ECG label is usable.

The release records two synchronization methods. Beat-based synchronization is more precise and assumes a pulse delay. Coarse synchronization is suitable for this stable-window audit. An independently synchronized dataset is needed to validate absolute pulse delay or compare fine R-peak timing. [Dataset format and synchronization documentation](https://zenodo.org/records/23597128)

The wrist-motion proxy is the RMS of the three-axis acceleration after subtracting each axis median. PPG quality is a separate 0–1 heuristic based on sample availability, flat periods, and spectral power in the pulse band. Motion can fall within that band, so a quality score and an acceleration value provide different observations.

## Pilot results

The 22 selected 30-second windows from three participants had usable ECG labels. There were eight AF and 14 non-AF windows, with 19 beat-synchronized and three coarse-synchronized windows. All ECG windows were fully observed. One PPG window had only 63.4% of its samples, leaving about 11 seconds missing. Both rhythms appeared in examples with substantial wrist acceleration. [Aggregate pilot results](pilot-results.json)

Four feature routes were run on two separate 32-second inputs from one participant, one stable AF and one stable non-AF. Each route used the central 30 seconds for features. The template and the RDDM timing correction were calibrated on three independent fingertip PTT subjects. The RDDM checkpoint weights stayed frozen. [PTT data](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) · [RDDM setup](../rddm/README.md)

| Route | Feature count | Windows with usable interbeat intervals |
| --- | ---: | ---: |
| Native ECG reference | 7 | 2/2 |
| Direct wrist PPG | 9 | 2/2 |
| PPG to pulse-template ECG | 7 | 2/2 |
| PPG to frozen RDDM ECG with timing correction | 7 | 2/2 |

The clips were chosen using their reference labels to exercise the pipeline. AF classification, group cross-validation, episode sensitivity, and false alerts per day require a separate evaluation on held-out recordings. The finite feature values and usable interbeat intervals above establish that the paths executed on these samples.

## Code and commands

| Code | Role |
| --- | --- |
| [`afppgecg.py`](../../ppg_af/afppgecg.py) | Open the matching MAT files, map the clocks, and read native-rate samples and reference intervals. |
| [`af_wrist_audit.py`](../../ppg_af/af_wrist_audit.py) and [`wrist_audit.py`](../../ppg_af/wrist_audit.py) | Audit paired windows and report the reference state, signal coverage, quality, and motion. |
| [`af_wrist_smoke.py`](../../ppg_af/af_wrist_smoke.py) | Run the feature paths on two real windows and check that they supply usable interbeat intervals. |
| [`af_events.py`](../../ppg_af/af_events.py) | Convert consecutive positive decisions into one alert, score new alerts against ECG episodes, and track available time. |

Download matching v6 `*_ECG.mat` and `*_PPG.mat` files and run the audit from the repository root. Replace `NNN` and choose the zero-based PPG segment and its offset in seconds. `--slice` takes `SEGMENT:OFFSET_SECONDS:DURATION_SECONDS`. Use a duration that is a positive multiple of 30. To find segment bounds, open the pair with `load_subject` and inspect `subject.segments`.

```powershell
python -m pip install -r requirements-af-wrist.txt
python -m ppg_af.af_wrist_audit --ecg data/NNN_ECG.mat --ppg data/NNN_PPG.mat --slice 0:30:90 --output runs/wrist-audit.json
```

For the feature smoke test, install the RDDM dependencies and obtain the pinned model, weights, and at least three sitting PTT records using the [RDDM setup](../rddm/README.md). Supply two local `SEGMENT:OFFSET_SECONDS` arguments. Each offset starts a 32-second clip. Choose one stable AF window and one stable non-AF window with complete PPG.

```powershell
python -m ppg_af.af_wrist_smoke --ecg data/NNN_ECG.mat --ppg data/NNN_PPG.mat --window $afWindow $nonAfWindow --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --output runs/wrist-smoke.json
```

## Comparison on held-out participants

See [participant-held-out methods and reproduction commands](grouped-evaluation.md) for the training pipeline.

The [episode follow-up](episode-follow-up.md) traces the three pilot misses and defines the independent test on unused participants.

The [window-seeded evaluation](window-seeded-evaluation.md) records the RDDM repeatability fix and the expanded calibration and test protocol.

The [alert timing study](alert-timing-study.md) checks preceding waveform context and compares one- and two-window confirmation on development participants.

The [alert history and decision cadence study](clock-study.md) restores alert state before the scoring cutoff and compares 30-second and 10-second decisions using the same input length and ECG reference.

The [false-alert and missed-episode audit](failure-audit.md) traces the errors in all 12 development configurations to guide the next experiment.

The [two-of-three alert study](vote-study.md) tests a shorter confirmation rule on the same 10-second RDDM decision grid and separates fixed-cutoff effects from recalibration.

The [template ECG onset study](template-onset-study.md) compares pre-onset and in-episode scores, an added recent beat-timing feature, and fold-local regularization.

Use the same participant-level folds and candidate windows for all three PPG routes. Train each AF classifier on training participants, set its quality and decision thresholds using validation participants, and freeze those thresholds before opening the test predictions. Keep native ECG as a separate reference route. Report performance on shared eligible windows and report each route's coverage over ECG-annotated time.

For the continuous test, retain PPG gaps on the ECG clock. Report complete-episode sensitivity, false alerts per ECG-observed day at comparable sensitivity, and coverage by PPG quality and motion. Show clipped episodes separately. Select the continuous test runs without conditioning inclusion on a known AF episode so the false-alert denominator reflects monitored time.
