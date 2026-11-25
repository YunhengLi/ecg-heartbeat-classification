# Transfer AF beat timing from ECG annotations to wrist monitoring

This experiment fits a classifier on the MIT-BIH Atrial Fibrillation Database, then applies it to the original wrist development folds. It compares transfer onto measured ECG and onto ECG estimated from wrist PPG. The source is PhysioNet AFDB v1.0.0, released under the Open Data Commons Attribution License v1.0. Cite Moody and Mark's 1983 study and PhysioNet when using the source. [AFDB release and citation](https://physionet.org/content/afdb/1.0.0/) · [ODC-By license](https://physionet.org/content/afdb/view-license/1.0.0/)

## Method

1. Verify the original AFDB roster and SHA-256 manifest. The pilot retrieved 2,502,477 bytes of annotations, headers and documentation from the official release. The Boston AFDB and the Vilnius wrist recordings come from separate reported acquisition cohorts. Their anonymized releases provide no shared identifier for direct patient linkage. [AFDB acquisition](https://physionet.org/content/afdb/1.0.0/) · [Wrist study source](https://zenodo.org/records/23921320)
2. Take a 30-second input every 60 seconds on each 250 Hz annotation timeline. Require the same manual rhythm label across the input and one-second flanks. Use explicit `(AFIB` as AF and `(N` as the other-rhythm group. Exclude atrial flutter, junctional rhythm, transitions, documented unreadable tape blocks, and sparse beat support. The `(N` code denotes the release's other rhythms. The source contains 5,452 stable AF windows and 8,774 stable `(N` windows across 25 recording participants.
3. Compute the seven existing interval features using only `.qrs` timestamps within each source window. The `.atr` rhythm annotations supply labels. Current `.qrs` ticks come from an automated beat detector and carry generic beat labels. Apply the same `.qrs` provenance to every record. Two records have annotations without voltage. Their last available beat bounds the scan. The feature measuring valid interval fraction is constant in the eligible source windows because they use an all-present sample mask. The other six features can vary. [AFDB beat and rhythm annotation documentation](https://physionet.org/content/afdb/1.0.0/)
4. Fit one standardized C=1 logistic model on source data and balance each participant's contribution within each rhythm. Freeze the scaler and coefficients. On both target inputs, preserve the original five outer participant groups, the seven-feature order, the shared ECG and PPG footprint and the immediate alert policy. Choose the source model's cutoff on only the training participants of each outer wrist fold. Keep the original nested cutoff selection for the wrist-trained comparators and reconcile their results with the [measured ECG control](measured-ecg-control.md).

## Results

The comparison retained **513 evaluation decisions**, **173 scoreable history decisions**, **114 stable fitting rows** and the same ECG reference exposure. The stable-window ROC AUC and complete-episode alert metrics measure different outcomes.

| Target input | Model fit | Stable-window ROC AUC | Complete AF episodes detected | Wrist false starts in 46.5 minutes | Pre-onset starts |
| --- | --- | ---: | ---: | ---: | ---: |
| Measured ECG | Wrist development | 0.9860 | 3/17 | 0 | 2 |
| Measured ECG | Frozen AFDB | 0.9474 | 4/17 | 0 | 1 |
| PPG-template ECG | Wrist development | 0.8188 | 12/17 | 6 | 3 |
| PPG-template ECG | Frozen AFDB | 0.8235 | 4/17 | 2 | 1 |

On measured ECG the transferred model gained one detection, and median delay changed from 13.53 to 14.52 seconds. Three of its five training-group cutoff searches reached the 80% sensitivity target. The held-out sensitivity was 23.5%. On template ECG the transfer lost eight detections and gained none. Median delay changed from 20.52 to 47.95 seconds. The sensitivity dropped by 47.1 percentage points, beyond the ten-point tolerance for comparing false burden at similar sensitivity. The original wrist-trained classifier remains the comparator.

## Reproduce offline

Put `RECORDS`, `SHA256SUMS.txt`, `notes.txt`, `ANNOTATORS`, and each listed record's current `.hea`, `.atr`, and `.qrs` files in `data/afdb-1.0.0-annotations`. Save the [official license page](https://physionet.org/content/afdb/view-license/1.0.0/) there as `license.html`. Use the [same wrist archives](measured-ecg-control.md#reproduce) as the measured ECG control and run with the NumPy, SciPy, scikit-learn and WFDB dependencies from `requirements-af-wrist.txt` and `requirements-rddm.txt`.

```powershell
python -m ppg_af.af_wrist_afdb_transfer `
  --afdb-dir data/afdb-1.0.0-annotations `
  --source-inventory data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --measured-control-results experiments/af-wrist/measured-ecg-control-results.json `
  --output runs/afdb-transfer-results.json
```

The [benchmark code](../../ppg_af/af_wrist_afdb_transfer.py) reads local files, validates the source checksums and fits and calibrates the models. The [aggregate JSON](afdb-transfer-results.json) records coverage, diagnostics and paired changes.
