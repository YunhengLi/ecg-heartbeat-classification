# Individual ECG beat timing from wrist PPG

The spectral approach estimates an average heart rate over a 30-second window. We trained a compact one-dimensional convolutional network to ask whether the input also supports estimating the timing of individual ECG R peaks. This experiment evaluates beat timing on exercise and activity recordings. [ScientISST MOVE](https://physionet.org/content/scientisst-move-biosignals/1.0.1/) · [Shimmer exercise release](https://physionet.org/content/wrist/1.0.0/)

## Method

1. Use nonoverlapping 30-second intervals and the source activity anchors to align the ScientISST MOVE wrist PPG and chest ECG. Require gel and dry electrodes to agree on rate and at least 80% of peak locations within 100 ms in both directions. Use gel peak times as automatic training targets. There are 610 paired windows from 14 people. Both chest channels agree in 611 of the 1,010 scheduled windows, and one lacks finite paired wrist input.
2. Resample inputs to 100 Hz. Train each network to produce an R-peak likelihood at each sample using a 30 ms Gaussian target around each reference beat and a participant-balanced binary loss. One model uses PPG only. The second adds the filtered acceleration magnitude. Each fit uses 20 epochs and the same optimization recipe. Randomly invert PPG polarity during training to reduce reliance on the optical sign convention.
3. Hold out entire ScientISST MOVE participants across five folds. Then train on all source participants and apply the models to eight disjoint Shimmer exercise participants using manual ECG R-peak marks. Keep the classifier and the peak probability threshold fixed for transfer. Compare both with pulse peaks shifted back by the same 300 ms used in the archived wrist timing check.
4. Predict using the signal collected inside the whole 30-second input. Exclude one second on each edge and match peaks one to one on the released clock within 100 ms. Report pooled detection F1 and mean F1 across people. The processing uses later samples within the observed window to estimate earlier beat positions.

## Results

| Cohort and route | Matched / reference beats | F1 | Mean participant F1 | Median timing error on matched beats |
| --- | ---: | ---: | ---: | ---: |
| MOVE fixed 300 ms delay | 16,248 / 38,937 | 0.456 | 0.435 | 50 ms |
| MOVE PPG CNN | 19,668 / 38,937 | 0.476 | 0.446 | 50 ms |
| MOVE PPG + acceleration CNN | 20,360 / 38,937 | 0.489 | 0.469 | 50 ms |
| Shimmer fixed 300 ms delay | 3,501 / 9,563 | 0.340 | 0.363 | 47.5 ms |
| Shimmer frozen PPG CNN | 5,021 / 9,563 | 0.422 | 0.461 | 45.6 ms |
| Shimmer frozen PPG + acceleration CNN | 4,930 / 9,563 | 0.394 | 0.392 | 51.6 ms |

Shimmer supported 203 of 218 scheduled windows across 19 activity recordings. The PPG-only CNN increased transfer recall from 0.366 to 0.525 and precision from 0.318 to 0.352. Median RR interval error on consecutively matched beats increased from 47.5 to 49.6 ms. The acceleration model's running F1 fell from 0.534 to 0.393 relative to the PPG-only CNN. This experiment leaves waveform shape and AF sensitivity outside its endpoints. [Aggregate results](beat-cnn-results.json)

### Training-only fixed-delay comparison

We also searched delays from 0 to 600 ms in 20 ms steps, selecting the highest mean participant F1 on the original fitting groups. Two folds chose 200 ms and three chose 240 ms. Fitting all MOVE participants selected 240 ms, which was then frozen for Shimmer. Pooled F1 increased from 0.456 to 0.464 on held-out MOVE and from 0.340 to 0.345 on Shimmer. The PPG-only CNN retained a 0.077 pooled F1 advantage on Shimmer. Its 49.6 ms median RR interval error remained above the calibrated baseline's 48.1 ms. [Delay calibration code](../../ppg_af/af_wrist_beat_delay.py) · [Aggregate results](beat-delay-results.json)

## Reproduce

Install the AF wrist dependencies, PyTorch, `edfio==0.4.18` and `wfdb==4.3.1`. The published results use PyTorch 2.10.0 with CUDA 12.8. The source directories must contain the corresponding published checksum lists.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m ppg_af.af_wrist_beat_cnn \
  --move-dir "$MOVE" --shimmer-dir "$SHIMMER" --device cuda \
  --output runs/beat-cnn-results.json
```

The same command can run `ppg_af.af_wrist_beat_delay` with the two source directories and its own `--output`. That runner omits `--device`.

[`BeatCNN`, `reference_targets` and `evaluate`](../../ppg_af/af_wrist_beat_cnn.py) define the architecture, soft beat targets, participant folds, and frozen transfer.
