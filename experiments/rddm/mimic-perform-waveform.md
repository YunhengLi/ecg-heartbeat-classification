# Patient-held-out waveform baselines on MIMIC PERform AF

This experiment compares two lightweight PPG-to-ECG reconstructions against simultaneously recorded ECG in an ICU cohort. It tests whether a model that reconstructs more of the ECG waveform also preserves more beat timings. The source includes 20-minute ECG and PPG recordings at 125 Hz from 19 patients with AF and 16 without AF. [MIMIC PERform v2.0 release](https://zenodo.org/records/15906524) · [Dataset documentation](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/)

## Method

1. Check both published MAT-file hashes and the patient count, signal rate, paired lengths, and record status. Schedule one 12-second window at each minute midpoint. This yields 700 planned pairs. Exclude windows with nonfinite samples or a fully flat signal. The final sample contains 676 pairs from all 35 patients, including 371 from AF records and 305 from non-AF records.
2. Split patients into five folds, stratified by record AF status, with seed 2026. Fit the ridge model and the generic pulse template on the other 28 patients for each test fold. The ridge model combines delayed PPG taps and slope features. The template learns a representative ECG beat and typical delay from pulse arrival to ECG, then places that beat shape using the held-out PPG pulses.
3. Score both predictions on the native clock after trimming one second at each edge. Use the sample intersection with a measured ECG and a prediction from each model. Compare waveform Pearson correlation and normalized absolute error on those samples. Match detected peaks one to one within 100 ms. The peak reference comes from the same detector run on measured ECG. The source has whole-record rhythm labels and the peak metric is detector-based.

## Results

Both models produced predictions over 99.81% of eligible interior samples in common. The waveform and beat metrics favor different models.

| Method | Median waveform Pearson r | Relative MAE | Detector-based beat F1 | Median consecutive RR error |
| --- | ---: | ---: | ---: | ---: |
| Delayed ridge | 0.142 | 0.156 | 0.265 | 32 ms |
| Pulse template | −0.026 | 0.185 | 0.446 | 8 ms |
| Zero waveform | — | 0.151 | — | — |

The template matched 4,425 of the 9,978 reference peaks. Ridge matched 1,972. The RR errors use intervals whose two consecutive peaks both matched, yielding 2,723 intervals for the template and 403 for ridge. In AF records the ridge and template beat F1 values were 0.296 and 0.385. Their waveform correlations were 0.048 and −0.023. Both methods had higher overall relative MAE than a zero waveform on the same samples.

These are pooled, descriptive results on fixed windows. Ridge and template use offline PPG context within each window. Episode onset, false alerts and delay require continuous wrist data with ECG episode boundaries. The RDDM paper lists MIMIC-AFib as a training source and cites the same Bashar study referenced by the PERform curation. The published checkpoint's patient-level split has not been verified here, so this experiment benchmarks models fitted from scratch on the stated folds. [RDDM experiment setup](https://arxiv.org/html/2308.13568#S4.SS1) · [PERform curation](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/)

## Reproduce

Download `mimic_perform_af_data.mat` and `mimic_perform_non_af_data.mat` from the pinned [Zenodo release](https://zenodo.org/records/15906524) into `data/mimic-perform-af`. From the repository root, use Python 3.11 or later with the NumPy, SciPy and scikit-learn dependencies from `requirements-af-wrist.txt`.

```powershell
python -m ppg_af.mimic_perform_waveform --data data/mimic-perform-af --output runs/mimic-perform-waveform-results.json
```

The [benchmark driver](../../ppg_af/mimic_perform_waveform.py) assembles the paired windows and applies the folds. The [waveform scoring code](../../ppg_af/tcn_metrics.py) scores the common sample support and matched beats. The [aggregate report](mimic-perform-waveform-results.json) records counts and metrics for all windows and each record-status group.
