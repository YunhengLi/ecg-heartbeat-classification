# Correcting RDDM beat timing with PPG

## Question

Can we move existing beats closer to the right times by using timing information in the input PPG? The released RDDM checkpoint found 211 of the 940 annotated beats within 100 ms on the original PTT scoring support. Increasing the tolerance to 200 ms increased matches to 486. That is evidence of timing disagreement, alongside beat-count differences. Wider matching can also pair adjacent beats. [Frozen checkpoint benchmark](ptt-results.json)

## Method and reasoning

The trainable part is a small calibration model. For each of five folds, it learns a single delay from the timing difference between PPG pulse peaks and manually annotated ECG R peaks. It pools candidate delay histograms from the training subjects, weighting subjects equally, and chooses the smoothed mode. The delays learned across folds were 176–184 ms on this dataset.

At test time, subtract the learned delay from detected PPG pulse times to estimate R-peak times. For each four-second generated block, choose a single translation within ±400 ms that aligns its detected peaks with those estimates. The search uses a 50 ms matching tolerance and needs at least two matched anchors to apply a shift. It favors lower timing error and smaller shifts when match counts tie. The operations preserve the samples and the spacing between beats within each block. The real test ECG enters only when computing the scores.

```python
delay = fit_pulse_delay(training_subject_histograms, sample_rate)
corrected, shifts = correct_timing(ppg, frozen_rddm_waveform, delay, sample_rate)
```

- [`pulse_timing.py`](../../ppg_af/pulse_timing.py) fits the delay and chooses the shifts.
- [`rddm_timing.py`](../../ppg_af/rddm_timing.py) applies the calibration to the held-out subjects and scores all three methods.
- [`timing_diagnostics.py`](../../ppg_af/timing_diagnostics.py) reports timing offsets and unmatched peaks at two tolerances.

## Comparison

The paired comparison reuses the same 66 windows from 22 people and derives the corrected waveform from the exact frozen output. Each method is scored on the same samples, trimming the first and last second and the 400 ms on each side of every internal block boundary. The common support covers 84% of the previously scored time and contains 791 reference peaks. The raw historical F1 of 0.231 uses a wider scoring region. [Dataset](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) · [Detailed results](timing-results.json)

| Method | R-peak F1 | Matched R peaks | Median matched timing error | Median matched R–R error | Median waveform correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen RDDM | 0.220 | 169 | 48 ms | 40 ms | 0.009 |
| Frozen RDDM + PPG timing correction | 0.608 | 472 | 32 ms | 32 ms | 0.031 |
| PTT-trained pulse template | 0.882 | 708 | 16 ms | 8 ms | 0.185 |

The correction shifted 177 of 198 blocks. It improved peak F1 in sitting, walking and running. The timing error is conditional on matched beats. The R–R metric uses consecutive matched intervals, so its support increased from 60 to 259 intervals after correction. The detected beat count can change when a shift moves a beat across the common scoring boundary or changes the detector's behavior near joins.

This experiment shows a timing benefit from a PPG-based correction for frozen RDDM on the previously inspected PTT cohort. Overall waveform correlation remained low and the pulse template kept the highest R-peak F1. Independent validation is still needed. AF performance is a separate outcome that requires a cohort with adjudicated rhythm labels.

## Reproduce the run

Use the author checkout, weights, and raw records from the [RDDM setup guide](README.md). Run from the repository root:

```powershell
python -m ppg_af.rddm_timing --upstream rddm-upstream --weights rddm-weights --ptt-data data/ptt-raw --output runs/rddm-timing.json
```

The aggregate JSON records the checkpoint hashes, seeds, learned delays, shift distribution, per-activity comparison, and signed timing diagnostics. Every subject's windows are kept in the same fold. Validation and test subjects contribute no data to the fitted delay for their fold.
