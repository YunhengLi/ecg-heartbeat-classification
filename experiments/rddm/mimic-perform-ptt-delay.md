# A local delay restores beat matching for a pooled ECG template

The pulse-template model reconstructs ECG by detecting PPG pulses, subtracting a fitted delay, and placing a representative ECG beat at each estimated time. This experiment tests whether the drop in beat matching after pooling healthy and ICU recordings came from the pooled delay or the pooled beat shape. It uses the same held-out ICU windows as the [original waveform benchmark](mimic-perform-waveform.md).

## Method

1. Prepare one midpoint 12-second window per participant and activity from the [PhysioNet PTT PPG v1.1.0 release](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/). The 22 participants contribute 66 paired windows from sitting, walking, and running. Use green fingertip PPG `pleth_3` and the existing 500-to-125 Hz preparation. The [MIMIC PERform AF v2.0 release](https://zenodo.org/records/15906524) supplies 676 eligible paired ICU windows at 125 Hz.
2. Reuse the five outer MIMIC patient folds stratified by record AF status and seeded with 2026. Fit the local template on the other 28 MIMIC patients in each fold. Pool those same patients with the 22 PTT participants for the joint template. Each contributor supplies one normalized delay histogram and one representative beat where available. The model averages the histograms and takes the samplewise median of the representative beats.
3. Copy the joint template and replace only its delay with the delay fitted on the MIMIC training patients. This preserves the exact joint waveform shape. Use held-out PPG to generate every forecast, with no test ECG alignment or adjustment.
4. Trim one second from each edge and score the intersection of sample support across all three forecasts. Use ECG detector-derived reference beats and match peaks one to one within 100 ms. Report the original MIMIC-only model’s score on its own support for reproduction.

## Results

The comparison used 676 held-out windows from 35 MIMIC patients and 99.7982% of the trimmed sample positions. All models shared 9,977 reference beats. The local fits selected delays of 24–40 ms across folds. The pooled model selected 176 ms on all five folds.

| Template | Beat F1 | Matched beats | Median consecutive RR error | Median Pearson r | Relative MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIMIC shape and delay | 0.44577 | 4,424 | 8 ms | −0.02625 | 0.18543 |
| Pooled shape and delay | 0.27148 | 2,695 | 32 ms | 0.00048 | 0.19294 |
| Pooled shape with MIMIC delay | 0.44572 | 4,424 | 8 ms | −0.03448 | 0.18777 |

The locally fitted delay recovered nearly all of the lost beat F1 with the pooled shape fixed. F1 rose from 0.375 to 0.385 in AF records and from 0.130 to 0.528 in non-AF records. The corrected template had higher waveform error than the local template. All three had higher relative MAE than the zero waveform, which scored 0.15134 on the common samples. The benefit here is confined to beat timing in the pooled model. This ICU waveform comparison uses record-level rhythm labels and assesses no wrist AF alert outcomes.

## Reproduce

Use Python 3.11 or later with the dependencies for the [PTT waveform benchmark](README.md#2-run-the-external-ptt-benchmark). Prepare its local PTT WFDB directory and both pinned [MIMIC MAT files](mimic-perform-waveform.md#reproduce). Run from the repository root.

```powershell
python -m ppg_af.mimic_perform_ptt_delay --mimic-data data/mimic-perform-af --ptt-data data/ptt-raw --output runs/mimic-perform-ptt-delay-results.json
```

The [experiment driver](../../ppg_af/mimic_perform_ptt_delay.py) fits and replays the three templates. The [aggregate report](mimic-perform-ptt-delay-results.json) includes per-fold delays, common support, and metrics by record status.
