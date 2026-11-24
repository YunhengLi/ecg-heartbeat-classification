# Native wrist motion and AF

The same motion-coherence weighting that lowered mean heart-rate error in two activity datasets was tested on the original wrist AF participants. A window's strongest frequency summarizes rate. AF alerts and ECG reconstruction also depend on the timing of individual beats. The experiments below score those endpoints separately. [Shimmer exercise](exercise-stress-study.md) · [Empatica E4](scientisst-move-study.md)

## Align acceleration to the original PPG

Use the 50 Hz native three-axis acceleration and 100 Hz green PPG from each original wrist packet. Reproduce every input PPG sample against the known native segment before reading acceleration. A per-file receipt binds the new acceleration to the same PPG row indices and core digest. The source yielded 832 verified acceleration windows out of 881 originally scoreable rows, including 210 of 240 scoreable routine wrist decisions. Nine people supplied verified motion. Native segment duration and sample counts disagree in the 49 remaining rows. Those PPG rows remain in the score and exposure denominators.

## Compare AF models on the existing clock

Fit the original template quality model, a control with a binary observed-motion flag, and a model with that flag plus residual PPG band power after motion weighting. Residual power is retained power divided by total power from 0.75 to 3.5 Hz. For the 49 windows without verified motion, use zero for the flag and the residual feature. This control measures how much the signal changes results beyond availability alone. All arms use the same ten people, five grouped outer folds, 30-second PPG input, ten-second decision spacing and immediate alert policy. Cutoffs use inner held-out predictions only.

In a separate experiment, use the square root of each coherence power weight to attenuate the PPG Fourier amplitude. Keep phase and use only samples in the trailing window, with two seconds of reflection for the transform. Apply the unchanged PTT pulse template to the weighted PPG. Replace its interval features and recompute their products with the original native PPG quality score. Fit and calibrate the classifier again inside each original group. Each of the 832 filtered inputs kept usable features, and the other 49 use the original template features.

| Original partition | Complete AF episodes detected | Wrist false starts |
| --- | ---: | ---: |
| Original template with quality interactions | 14/17 | 4 |
| Add the presence of acceleration | 13/17 | 4 |
| Add residual power and acceleration presence | 8/17 | 3 |
| Reconstruct from motion-filtered PPG and refit | 12/17 | 4 |

The last two arms exceeded the prespecified ten-point sensitivity loss relative to the original quality model. All four arms gave 1/17 and zero false starts on the previously identified weak partition. Keep both 17-episode results in view. They use two groupings of the same ten people and 46.5 minutes of ECG-observed wrist time. [Motion quality results](motion-quality-results.json) · [Reconstructed template replay](motion-reconstruction-results.json)

## Compare ECG beat timing on stable paired windows

Of 114 original stable ECG–PPG pairs, 102 from nine people had verified acceleration. For both arms, use one fixed external template shape and the archive's documented 300 ms delay on beat-synchronized ECG. Coarse synchronization retains the original PTT delay. Compare 285,600 common sample points and 3,187 reference QRS locations after trimming the window edges. Both inputs and references come from the fixed development archive.

| Input to the fixed template | QRS F1 within 100 ms | Median absolute QRS timing error | Normalized MAE |
| --- | ---: | ---: | ---: |
| Original PPG | 0.666 | 27 ms | 0.2530 |
| Phase-preserving motion filtering | 0.648 | 30 ms | 0.2541 |

In the 42 windows where weighting removed at least half the PPG-band power, QRS F1 fell from 0.696 to 0.664. The weighted heart-rate estimator can be studied on its own. The tested motion features and filter did not improve either of the wrist AF and ECG endpoints. [Waveform results](motion-waveform-results.json)

## Reproduce

Stage native motion using the paired MAT sources and the original raw PPG and feature files. The portable staging module accepts local MAT files or seekable byte-range inputs through [`stage_person`](../../ppg_af/af_wrist_acceleration_stage.py). The following commands use `SOURCE` for the original local MAT files, `RAW` for the original protected development archives and `MOTION` for a new staging directory.

```bash
python -m ppg_af.af_wrist_acceleration_stage --source-dir "$SOURCE" --output "$MOTION" \
  --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --lock "$RAW/holdout-window-seeded-lock.json"
```

Use the same four development archive arguments and `--lock` with `python -m ppg_af.af_wrist_motion_quality` or `python -m ppg_af.af_wrist_motion_reconstruction`. Also supply `--acceleration-sidecars "$MOTION"`, `--clock-results experiments/af-wrist/clock-study-results.json`, `--quality-results experiments/af-wrist/quality-study-results.json`, and a new `--output` path. The reconstruction runner additionally takes `--ptt-dir`.

For `python -m ppg_af.af_wrist_motion_waveform`, supply the first two development archive arguments, `--acceleration-sidecars "$MOTION"`, `--waveform-pairs` from the [verified waveform study](waveform-study.md), `--waveform-results experiments/af-wrist/waveform-study-results.json`, `--ptt-dir` and `--output`. [`residual_pulse_power`](../../ppg_af/af_wrist_motion_quality.py) extracts the motion feature, [`motion_filtered_ppg`](../../ppg_af/af_wrist_motion_waveform.py) applies the phase-preserving weights, and [`transformed_intervals`](../../ppg_af/af_wrist_motion_reconstruction.py) verifies feature coverage before AF scoring.
