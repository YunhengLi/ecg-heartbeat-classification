# Compare waveform and QRS supervision on native wrist ECG pairs

Individual R-peak recovery and whole-waveform reconstruction are different endpoints. We tested whether waveform supervision plus a stronger QRS loss improves both on the same paired native wrist inputs. [Paired waveform data](waveform-study.md)

## Design

Use the 223 verified native 30-second signal pairs and the original five outer participant groups. In every fold, reserve separate development participants for early stopping. Fit the local template and both temporal convolutional networks on the same remaining wrist participants. The template with the release-delay rule is frozen from the existing comparator.

The first TCN uses a smooth L1 waveform loss. The second adds a weighted ECG R-peak likelihood loss and quadruples the waveform loss around annotated QRS centers. Both use the same eight channels, optimizer, initial seed and up to 12 epochs in each fold. Each input contains only wrist PPG and its available-sample mask. The output is a normalized ECG trace; the QRS variant also emits a separate timing score. [Temporal model and loss](../../ppg_af/temporal.py) · [Wrist runner](../../ppg_af/af_wrist_native_tcn.py)

## Results

All 223 windows from nine people supplied common evaluation support. After excluding one second at each edge and taking the common waveform support, 99.57% of the interior samples remained. Beat targets were mapped to the nearest 100 Hz sample. There were 7,502 reference beats, scored one to one with a 100 ms tolerance. [Aggregate results](native-tcn-results.json)

| Route | Normalized waveform MAE | Median waveform correlation | F1 from predicted waveform |
| --- | ---: | ---: | ---: |
| Zero waveform | 0.21529 | — | — |
| Template with released clock timing | 0.24729 | −0.014 | 0.647 |
| Fold-local template | 0.22751 | 0.018 | 0.671 |
| Waveform-only TCN | 0.21607 | 0.001 | 0.070 |
| QRS-weighted TCN | 0.21624 | 0.007 | 0.075 |

The weighted model's separate timing head yielded F1 of 0.126. The ECG detector on the measured trace yielded 0.957. Median waveform correlation remained near zero across the reconstructed traces. Normalized amplitude MAE alone favored predictions close to the baseline level, while template timing yielded more correctly located beats. The evaluated TCN variants provided no improvement over the fold-local template in either beat matching or waveform correlation. [Waveform and beat metrics](../../ppg_af/tcn_metrics.py)

Run `python -m ppg_af.af_wrist_native_tcn` with the same wrist archives and waveform verification arguments as the [native beat transfer study](native-neural-beat-study.md#reproduce). The published comparison ran on CPU with `--threads 4`.
