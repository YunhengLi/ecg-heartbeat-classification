# Transfer the motion-weighted heart-rate estimate to Empatica E4

We applied the estimator from the Shimmer exercise study to a separate wrist device and activity cohort. The ScientISST MOVE release provides 64 Hz Empatica E4 wrist PPG, 32 Hz wrist accelerometry and two simultaneous 500 Hz chest ECG channels. The motion weight uses the three available axes. The spectrum settings, weighting rule, and 30-second input remain fixed. [Original exercise study](exercise-stress-study.md) · [ScientISST MOVE 1.0.1](https://doi.org/10.13026/hyxq-r919)

## Align and verify without using PPG performance

1. Verify the source checksums for 17 chest and 16 E4 files. Require at least two common positive-duration activity annotations whose estimated wrist-to-chest offset agrees within 2 ms. Check that the offset agrees with the coarse EDF header within one second. Fourteen people supplied verified pairs. One had no E4 file and two lacked common anchors. The reader discarded the incomplete final data record in 13 files.
2. Schedule 30-second windows every ten seconds within the chest–wrist overlap of shared annotated activities. For a reference estimate, require the gel and dry chest ECG pulse rates to agree within 5 bpm. In addition, at least 80% of peaks in both directions must have a counterpart within 100 ms. The gel-electrode median interval supplies the reference BPM when the gate passes. This is an automated consensus reference.
3. Resample each trailing PPG and motion window to 100 Hz with no samples from after its decision. Compare the median PPG interval, the strongest PPG frequency, and the strongest frequency after motion-coherence weighting. The estimator does not fit to this cohort or use ECG at inference. Use the same paired windows for the three results.

## Results

The ECG consensus supplied 1,813 of 3,026 scheduled windows. All three PPG estimates and the reference were available on 1,809 windows from 14 people. Most of the 1,217 excluded windows failed the ECG consensus, particularly during running. Running provided 1,254 of the analyzed windows.

| Activity | Paired windows | Direct PPG interval MAE | PPG spectrum peak MAE | Motion-weighted peak MAE |
| --- | ---: | ---: | ---: | ---: |
| All | 1,809 | 23.94 | 32.89 | 14.81 |
| Run | 1,254 | 28.58 | 37.47 | 13.12 |
| Walk before run | 174 | 9.66 | 18.18 | 16.47 |
| Walk after run | 176 | 16.77 | 27.10 | 17.80 |
| Gesticulate | 104 | 12.90 | 25.00 | 25.31 |
| Lift | 55 | 17.69 | 25.78 | 21.43 |

Across paired windows, median error fell from 19.64 to 1.97 bpm, and equal-participant mean error fell from 22.39 to 15.82 bpm. Nine of 14 people had lower mean error. The weighted method reduced running error and raised mean error for some hand movements and walking. The result concerns average heart rate in the observed activities. AF alerts and beat-by-beat timing require separate reference endpoints. [Full aggregate results](scientisst-move-results.json)

## Reproduce

Unpack the [PhysioNet release](https://physionet.org/content/scientisst-move-biosignals/1.0.1/) into `MOVE`. Install `edfio==0.4.18` alongside the wrist research dependencies.

```bash
python -m ppg_af.af_wrist_move --move-dir "$MOVE" --output runs/scientisst-move-results.json
```

[`activity_clock` and `consensus_heart_rate`](../../ppg_af/af_wrist_move.py) define the reference. [`spectral_heart_rate`](../../ppg_af/af_wrist_exercise.py) applies the fixed frequency weighting.
