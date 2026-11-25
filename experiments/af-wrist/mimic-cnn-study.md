# Frozen MIMIC PPG rhythm CNN on wrist development data

The [logistic transfer comparison](mimic-transfer-study.md) found higher episode sensitivity with a frozen ICU PPG model and more wrist false starts. This fixed experiment checks whether learning directly from pulse waveforms improves that transfer result.

## Method

1. Reuse the [MIMIC PERform AF v2.0 release](https://zenodo.org/records/15906524), its two file checksums, and the published logistic model's exact 1,342 eligible windows. The 35 source participants contributed 741 AF-record windows and 601 non-AF-record windows. Each nonoverlapping 30-second window inherits its record's AF status. Resample each window from 125 Hz to 100 Hz and keep its endpoint fixed.
2. Scale each input by subtracting its median, dividing by the 95th percentile absolute centered amplitude, and clipping to ±8. Train one 4,589-parameter CNN with four one-dimensional convolutions, temporal mean and variance pooling, and a single AF output. Use seed 2026, 12 AdamW epochs, a learning rate of 0.001, and batch size 64. Each class has equal total loss weight, and each participant has equal weight within their class. All weights are learned from MIMIC.
3. Freeze the network. Score only the original shared development rows, including 513 evaluated decisions and 175 predecessor cells from the ten wrist participants. Use the same 30-second trailing input, 10-second decision clock, and predecessor replay. In each outer fold, calibrate the frozen score using that fold's wrist training participants and score its held-out participants. Use both the existing 80% episode target and the threshold rule that targets sensitivity near the template on the calibration roster.
4. Refit and verify the original wrist template and frozen ICU logistic results. Keep the same 17 complete ECG-confirmed episodes and 46.5 minutes of ECG-observed wrist exposure for all comparisons. The training report contains aggregate results. The independent wrist cohort generated zero predictions.

## Results

| Model | Threshold policy | Detected AF episodes | Wrist false starts |
| --- | --- | ---: | ---: |
| Wrist template | Original template policy | 12/17 | 6 |
| Frozen source logistic | 80% target | 14/17 | 9 |
| Frozen source CNN | 80% target | 4/17 | 23 |
| Frozen source logistic | Match template in calibration | 14/17 | 10 |
| Frozen source CNN | Match template in calibration | 4/17 | 17 |

The CNN's weighted source training loss fell from 0.694 to 0.403. It reached the 80% sensitivity target on four of the five wrist training rosters, then detected 4 of 17 held-out episodes under each policy. Those four were among the episodes already detected by both baselines. Against the frozen logistic at the 80% policy, the CNN lost ten detections and added fourteen wrist false starts. The held-out sensitivity gaps exceeded the ten-percentage-point matching tolerance under both policies. This fixed CNN did not improve the ICU-to-wrist comparison.

## Reproduce

Use the pinned local MIMIC MAT files described in the [transfer study](mimic-transfer-study.md#reproduce) and the private development archives prepared for the [decision cadence study](clock-study.md#reproduce). The verified run used Windows AMD64, Python 3.14.7, PyTorch 2.10.0+cpu, NumPy 2.5.3, SciPy 1.18.1, and scikit-learn 1.9.1. It uses two CPU threads and deterministic PyTorch operations. The [aggregate report](mimic-cnn-results.json) records the dependency versions and the complete learning curve.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python -m ppg_af.af_wrist_mimic_cnn `
  --source data/mimic-perform-af --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock `
  --clock-results experiments/af-wrist/clock-study-results.json `
  --transfer-results experiments/af-wrist/mimic-transfer-results.json `
  --output runs/mimic-cnn-results.json
```

The [CNN driver](../../ppg_af/af_wrist_mimic_cnn.py) fits the source network and reuses the published calibration helpers. The [source loader](../../ppg_af/af_wrist_mimic_transfer.py) applies one eligibility rule to the waveform and feature models.
