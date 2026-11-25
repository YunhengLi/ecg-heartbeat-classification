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

The [development research summary](research-summary.md) compares the major completed studies and links to their code, results and reproduction commands.

See [participant-held-out methods and reproduction commands](grouped-evaluation.md) for the training pipeline.

The [episode follow-up](episode-follow-up.md) traces the three pilot misses and defines the independent test on unused participants.

The [window-seeded evaluation](window-seeded-evaluation.md) records the RDDM repeatability fix and the expanded calibration and test protocol.

The [alert timing study](alert-timing-study.md) checks preceding waveform context and compares one- and two-window confirmation on development participants.

The [alert history and decision cadence study](clock-study.md) restores alert state before the scoring cutoff and compares 30-second and 10-second decisions using the same input length and ECG reference.

The [false-alert and missed-episode audit](failure-audit.md) traces the errors in all 12 development configurations to guide the next experiment.

The [two-of-three alert study](vote-study.md) tests a shorter confirmation rule on the same 10-second RDDM decision grid and separates fixed-cutoff effects from recalibration.

The [template ECG onset study](template-onset-study.md) compares pre-onset and in-episode scores, an added recent beat-timing feature, and fold-local regularization.

The [independent test preparation](template-holdout-preparation.md) freezes the fixed-split template classifier and audits source coverage on the reserved roster.

The [independent template ECG evaluation](template-independent-evaluation.md) applies the single frozen classifier to the recovered independent recordings.

The [independent error audit](template-independent-error-audit.md) attributes missed and false alerts and tests participant-balanced fitting on development data.

The [transition training study](template-transition-training.md) tests added ECG-verified onset examples on the original development folds.

The [data and evaluation foundation](data-evaluation-foundation.md) describes the paired inventory, split controls and score-free preflight.

The [chronological pulse study](sequence-study.md) compares ordered 10-second PPG bins and a control with the same bins sorted by measured features.

The [PPG quality study](quality-study.md) tests whether rhythm features are more useful when the classifier accounts for signal quality. The [split-sensitivity check](quality-stability.md) repeats the same model comparison across five development partitions.

The [uncertainty study](quality-uncertainty.md) estimates paired participant-level intervals for that fixed development comparison. The [quality retrospective replay](quality-retrospective-study.md) compares two frozen quality models on the previously audited 25-person cohort.

The [retrospective uncertainty study](quality-retrospective-uncertainty-study.md) compares participant-average sensitivity and paired changes in pooled sensitivity and false starts.

The [ICU transfer study](mimic-transfer-study.md) tests frozen source scores and feature fusion on the original wrist development folds.

The [source CNN study](mimic-cnn-study.md) trains a small neural model on the same source PPG windows and compares frozen transfer on the wrist clock.

The [onset-aware calibration study](onset-calibration.md) tests whether counting verified pre-onset false starts during cutoff selection changes held-out alerts.

The [stable training expansion](training-expansion.md) adds verified local ECG–PPG windows and checks the effect of a seventh AF training contributor across five development partitions.

The [measured ECG study](measured-ecg-control.md) compares stable-window discrimination and new-alert detection on the same development clock.

The [measured ECG context study](measured-ecg-context.md) and [pulse-template context study](template-context.md) hold the decision clock fixed while comparing 10- and 30-second inputs in each route.

The [native transition study](transition-native-study.md) verifies an additional donor's pre-onset and early-AF input pair and repeats the original onset training comparison.

The [dedicated onset classifier](onset-head-study.md) trains early AF as a separate start-detection task and compares it on two original participant splits.

The [lower-quality non-AF study](low-quality-training-study.md) compares alternative training windows on the same participant and time-bin assignments.

The [native wrist waveform study](waveform-study.md) compares external and locally fitted reconstruction on held-out participants and isolates the release's timing assumption.

The [fold-local template study](local-template-study.md) tests whether fitting the wrist waveform inside each training fold improves AF alerts.

The [ECG annotation transfer study](afdb-transfer-study.md) tests a frozen external rhythm model on measured and PPG-derived ECG timing features.

The [sampling stress study](sampling-stress.md) evaluates a synthetic 50 Hz bandwidth change with the original model cutoffs. The [extra episode replay](extra-episode-study.md) verifies a short new development event without changing the benchmark.

The [wrist exercise stress study](exercise-stress-study.md) compares heart-rate error, the native PPG quality score, and frozen alert behavior with an independent device and published ECG R peaks.

The [ScientISST MOVE study](scientisst-move-study.md) applies the fixed motion-coherence heart-rate estimator to a separate Empatica E4 cohort with synchronized chest ECG.

The [beat-timing study](beat-cnn-study.md) fits a compact neural network for ECG R-peak timing on participant-held-out Empatica E4 recordings and freezes it for transfer to manually marked Shimmer exercise recordings.

The [native neural beat study](native-neural-beat-study.md) compares the source-frozen neural beat features with the original AF classifier and evaluates wrist-specific fine-tuning on held-out native ECG pairs.

The [raw PPG CNN study](raw-cnn-study.md) compares a direct waveform AF classifier and template classifier trained on the same expanded wrist data.

The [native TCN study](native-tcn-study.md) compares waveform loss and QRS supervision on identical participant-held-out ECG pairs.

The [native motion study](motion-af-study.md) audits the original accelerometer clock and compares AF alerts and reconstruction before and after motion processing.

The [ambient-light study](ambient-light-study.md) tests whether a second native optical measurement improves AF alerting under the original evaluation protocol.

The [longer wrist monitoring study](monitor-extension-study.md) compares the two frozen template classifiers over additional times selected from the original development participants using only wear clocks.

The [hard-negative study](hard-negative-study.md) tests whether the most irregular non-AF inputs from the additional development monitoring improve held-out alerts. The [direct PPG fusion study](direct-fusion-study.md) adds pulse spectrum to the template and quality features under the same folds.

The [retrospective monitoring extension](retrospective-monitor-study.md) compares four frozen models over newly sampled monitoring time from 25 previously compared participants excluded from all fits.

The [false-start timing audit](false-start-audit.md) compares wrist-derived and native ECG interval irregularity at these starts against matched negative decisions.

The [pulse consensus study](pulse-consensus-study.md) verifies optical peaks across two filter bands, compares measured interval variation and replays the original AF folds.

The [PeriWave dual-wavelength study](periwave-study.md) checks whether separately measured red and infrared PPG corroborate individual pulse timing on a four-center surgical source.

The [Auricle study](auricle-study.md) estimates the relative timing of ear and finger signals before using them to corroborate pulses in later windows.

The [MSPTDfast v2 study](msptd-study.md) uses a published pulse detector to compare timing errors and AF alerts on the original wrist data. The [interval order study](interval-order-study.md) compares observed interval sequences with a distribution-only control on wrist alerts and on surgical rhythm ranking.

The [VitalDB preflight](vitaldb-preflight.md) verifies the source's labeled-window support and the original case clock for paired surgical ECG and plethysmography. The [source transfer study](vitaldb-transfer-study.md) compares sinus and ectopy training negatives on the same source and wrist participant splits.

The [negative rearming study](negative-rearm-study.md) checks whether allowing a new alert after two consecutive negative decisions changes held-out detections and false starts at fixed cutoffs.

The [quality-conditioned confirmation study](quality-confirmation-study.md) requires a preceding positive only at low current PPG quality and checks the fixed plain-template rule on both previously scored monitoring sets.

The [SHDB-AF measured ECG reference](shdb-ecg-reference.md) compares short and long interval inputs on the same continuous Holter clock and participant-held-out folds.

The [paired ECG feature regression study](ecg-feature-regression-study.md) learns measured ECG interval statistics from wrist PPG only inside the fitting participants. The [joint wrist context study](joint-context-study.md) fits 10-second and 30-second timing features together on the same decision grid.

The [independent validation plan](prospective-validation-plan.md) defines a frozen primary comparison and gives sample-size and exposure scenarios for sensitivity and false-alert burden.

The [GalaxyPPG study](galaxy-quality-study.md) checks the original PPG quality boundary against measured ECG interval errors on a Galaxy Watch and Empatica E4. The [pulse amplitude study](pulse-amplitude-study.md) adds wearable pulse prominence and its association with the preceding interval on the original AF folds.

The [half-hour monitoring study](half-hour-monitor-study.md) compares the two frozen alert policies on new wear times and adds causal rhythm-state duration metrics. The [GalaxyPPG motion study](galaxy-motion-quality-study.md) learns interval reliability from measured accelerometry on held-out people, including device transfer.

The [Holter state study](shdb-ecg-reference.md#calibrate-duration-and-alerts-separately) keeps the same measured ECG scores and compares episode thresholds, time-state thresholds and calibrated probabilities for estimating AF time.

The [wrist AF time probability study](wrist-state-probability-study.md) calibrates the frozen wrist scores on the original calibration participants and compares time estimates, rhythm separation and constant-prevalence controls on the sealed half-hour clocks.

The [within-person wrist ranking study](wrist-state-ranking-study.md) compares forward AF-time discrimination on the sealed historical replay with pooled and person-level AUC.

The [DeepBeat transfer study](deepbeat-transfer-study.md) applies the authors' two-output wrist model to the same participant groups and checks the effect of preprocessing. The [open ICU clock audit](bashar-mimic-offset-study.md) links reviewed AF annotations to paired bedside ECG and plethysmography near qualified offsets.

The [weak-pulse study](weak-pulse-study.md) changes peak prominence and tests a gap-only recovery rule against paired wrist ECG and held-out AF alerts. The [frozen motion transfer study](motion-quality-transfer-study.md) applies Galaxy and E4 timing-reliability models to an independent Shimmer exercise dataset.

The [confirmation spacing study](confirmation-spacing-study.md) compares overlapping and disjoint PPG confirmation inputs. Its separate episode audit uses the ECG clock to count exactly where score availability and input duration permit detection.

The [native half-hour motion audit](half-hour-motion-audit.md) uses nearby same-block controls with ECG-confirmed non-AF to compare optical spectra and measured wrist motion at false starts.

The [PaPaGei-S transfer study](papagei-transfer-study.md) tests public pretrained and random frozen encoders with the same nested wrist AF heads. The [entropy confirmation study](entropy-confirmation-study.md) changes the gate that requests the existing adjacent positive score.

The [wearer score study](personal-score-study.md) evaluates fixed and rolling PPG-only baselines after a common warm-up. The [episode subgroup audit](confirmation-subgroups-study.md) reports detection by the duration and contribution of historical complete ECG reference episodes.

The [short-template historical study](short-retrospective-study.md) compares ten- and 30-second PPG inputs on the same 260 episodes and half-hour wear blocks. It reports a cutoff matched on development sensitivity as well as a target-sensitivity cutoff.

The [paired-source training study](paired-source-study.md) tests shared ECG and PPG representation training with a within-person shifted control. The [optical beat opportunity audit](beat-opportunity-study.md) compares candidate beat matching with a density-matched timing rotation.

The [same-clock native ECG study](ecg-timing-replay-study.md) compares measured ECG QRS timing with PPG-derived timing across 260 complete historical episodes and 744 ECG-observed wrist minutes.

The [joint confirmation evaluation](confirmation-evaluation.md) reports paired AF episode sensitivity, false alert starts, and monitorable wear for the same two frozen wrist policies.

Use the same participant-level folds and candidate windows for all three PPG routes. Train each AF classifier on training participants, set its quality and decision thresholds using validation participants, and freeze those thresholds before opening the test predictions. Keep native ECG as a separate reference route. Report performance on shared eligible windows and report each route's coverage over ECG-annotated time.

For the continuous test, retain PPG gaps on the ECG clock. Report complete-episode sensitivity, false alerts per ECG-observed day at comparable sensitivity, and coverage by PPG quality and motion. Show clipped episodes separately. Select the continuous test runs without conditioning inclusion on a known AF episode so the false-alert denominator reflects monitored time.
