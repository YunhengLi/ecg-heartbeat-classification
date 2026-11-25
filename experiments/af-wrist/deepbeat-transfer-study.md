# Transfer the published DeepBeat classifier to the original wrist folds

DeepBeat predicts AF and optical signal quality from PPG. We used the authors' pretrained multitask model as a direct optical route beside the local pulse template. The checkpoint contains 111,629 parameters and separate quality and rhythm outputs. The experiment uses citable public weights and the project's wrist data. [DeepBeat paper](https://doi.org/10.1038/s41746-020-00320-4) · [Authors' pinned release](https://github.com/AshleyLab/deepbeat/blob/f54004a80eaca99460d1e171c8ec203e182dacb9/deepbeat.h5)

## Method

1. Load the verified HDF5 weights with Keras 3.12.0 on the PyTorch backend. Use the last 25 seconds of each shared original PPG input and resample to 32 Hz. Use the project's 0.4–8 Hz optical band because the paper does not specify its passband. The first run scaled to [0, 1] after filtering. After the quality output assigned all samples to noise, repeat with scaling before filtering, following the order described in the paper. Apply both orders to 22 independent sitting PTT records as an input control.
2. Keep the original two-way AF classification rule. Use the published excellent-quality gate as a separate comparison and retain the lost score coverage. For another arm, choose a threshold on the other participant groups with the same 80% episode sensitivity target used by the local project. The external weights stay frozen. [DeepBeat evaluation code](../../ppg_af/af_wrist_deepbeat.py)
3. Reproduce the initial 30-second plain template. Fit a second template using the same 25 seconds as DeepBeat, with original training windows, participant splits and nested calibration. Use the same ten-second decision times and ECG episode labels for every comparison.

## Results

| Route | Original grouping detected | Original grouping false starts | Alternate grouping detected | Alternate grouping false starts |
| --- | ---: | ---: | ---: | ---: |
| 30-second pulse template | 12/17 | 6 | 1/17 | 2 |
| 25-second pulse template | 10/17 | 7 | 1/17 | 2 |
| DeepBeat, scale after filtering, published rule | 5/17 | 6 | 5/17 | 6 |
| DeepBeat, scale after filtering, training-selected threshold | 2/17 | 13 | 5/17 | 9 |
| DeepBeat, scale before filtering, published rule | 4/17 | 3 | 4/17 | 3 |
| DeepBeat, scale before filtering, training-selected threshold | 12/17 | 9 | 12/17 | 9 |

Scaling after filtering yielded 0.776 stable-window ROC AUC and classified all 881 wrist inputs and 22 sitting controls as noise. Scaling before filtering yielded 0.491 stable-window AUC and sent every input and sitting control to the excellent class. The excellent-only gate therefore removed all 513 scoreable evaluation decisions under the first processing order and retained all of them under the second. These extreme quality outputs demonstrate sensitivity to input treatment on our data. The before-filter arm with separately selected thresholds matched primary episode sensitivity of the 30-second template and made three additional false starts. [Aggregate results](deepbeat-transfer-results.json)

## Reproduce

Install Keras 3.12.0 with the PyTorch backend, the original wrist requirements and the PTT data. Use the same development paths described in the [quality confirmation study](quality-confirmation-study.md). Pass `--confirmation-results experiments/af-wrist/quality-confirmation-results.json`, the original `--ptt-dir`, the downloaded `--weights`, and a new `--output` to `ppg_af.af_wrist_deepbeat`. The weight SHA-256 is `6df126f88855388f8706ea6e158b80902752a019b1f3c26d3ceab22991e4eee3`. The local port uses open weights with documented input assumptions. The original model's own training data and supplementary preprocessing would be needed to verify identical author inputs. [Original code](https://github.com/AshleyLab/deepbeat/tree/f54004a80eaca99460d1e171c8ec203e182dacb9)
