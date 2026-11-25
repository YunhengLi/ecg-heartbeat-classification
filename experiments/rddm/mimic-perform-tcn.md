# Small neural ECG reconstruction on MIMIC PERform AF

This pilot tests whether a compact temporal CNN can recover ECG waveforms and beat timing from paired ICU PPG. It also tests whether emphasizing the short region around each ECG R peak improves the waveform enough for a beat detector to recover more beats. The [paired waveform benchmark](mimic-perform-waveform.md) supplies the delayed ridge, pulse template, and fixed patient folds.

## Method

1. Verify both [MIMIC PERform v2.0](https://zenodo.org/records/15906524) MAT-file checksums and reuse the same 676 paired 12-second windows from 35 patients. Their ECG and PPG share the source's 125 Hz grid. The source gives a rhythm status for the entire record. The R-peak targets come from running the existing beat detector on the measured ECG, so the peak comparisons use detector-derived references.
2. Hold out the same seven patients per outer fold. Split the other 28 patients into 22 for fitting and six for epoch selection, stratified by record AF status. The two CNN variants share these assignments, initialization seeds and minibatch order. The ridge and pulse-template baselines reproduce their original fits on all 28 outer-training patients. No held-out ECG enters fitting, epoch selection, prediction, or alignment.
3. Train an eight-channel dilated temporal CNN with AdamW, batches of 32 windows and up to eight epochs. Select the minimum validation loss, with three epochs of patience. The waveform-only model minimizes smooth L1 error. The QRS-weighted model multiplies the waveform error weight by five within four samples of a detector-derived R peak. Both use waveform loss only. The fixed training budget and small networks make this an exploratory comparison.
4. Score all waveforms together on the native clock after trimming one second from each edge. Use only the common support. Compare normalized mean absolute error, median Pearson correlation and one-to-one beat matches within 100 ms. The CNN sees the whole PPG window for normalization and uses centered convolutions for prediction, so this is an offline reconstruction benchmark.

## Results

Both CNNs produced one finite held-out forecast for each of the 676 windows. The four methods shared 99.81% of the interior samples. The original ridge and pulse-template metrics reproduced exactly overall and within both record-status groups.

| Method | Median waveform Pearson r | Relative MAE | Detector-based beat F1 | Matched beats |
| --- | ---: | ---: | ---: | ---: |
| Delayed ridge | 0.142 | 0.156 | 0.265 | 1,972 |
| Pulse template | −0.026 | 0.185 | 0.446 | 4,425 |
| Waveform-only CNN | −0.010 | 0.153 | 0.249 | 1,888 |
| QRS-weighted CNN | −0.046 | 0.155 | 0.204 | 1,493 |
| Zero waveform | — | 0.151 | — | — |

The waveform-only CNN had lower relative MAE than the ridge and template yet still had higher error than the zero waveform. Ridge retained the highest waveform correlation and the pulse template retained the highest beat F1. Emphasizing samples near the reference peaks reduced the CNN's beat F1 in both AF records (0.243 to 0.208) and non-AF records (0.257 to 0.199). In this pilot, extra waveform penalty near detected R peaks did not improve beat recovery. The beat metrics use 9,978 detector-derived reference peaks and describe the pooled fixed windows. They do not measure wrist AF alerts or performance on manually verified beat annotations.

## Reproduce

From the repository root, use Python 3.11 or later with NumPy, SciPy, scikit-learn and a CPU installation of PyTorch. Place the two pinned MAT files in `data/mimic-perform-af` as in the [waveform benchmark](mimic-perform-waveform.md#reproduce).

```powershell
python -m ppg_af.mimic_perform_tcn --data data/mimic-perform-af --output runs/mimic-perform-tcn-results.json --threads 2
```

The [experiment driver](../../ppg_af/mimic_perform_tcn.py) assigns patient folds and compares the neural waveforms with the original baselines. [`fit_tcn`](../../ppg_af/temporal.py) fits both models. [`score_group`](../../ppg_af/tcn_metrics.py) compares the shared waveform samples and beat timings. The [aggregate report](mimic-perform-tcn-results.json) records support and metrics for each record-status group and the validation loss by epoch.
