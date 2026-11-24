# Train a raw PPG AF classifier on the same expanded wrist data

This comparison tests a direct optical waveform classifier against a template classifier on the same paired wrist training windows. The source-only rhythm CNN used a different cohort, so its earlier transfer result could not isolate the model family. [Source transfer study](mimic-cnn-study.md)

## Design

Use the original stable wrist fitting windows and all 109 disjoint supplemental windows, comprising 43 AF and 66 non-AF. Keep the same people, fold labels, scoreable inputs, ten-second decision cadence and nested cutoff selection. For each of two grouped partitions, fit 20 CNNs across the outer test and inner calibration sets. The identical data expansion is applied to the quality-interaction template control.

The 5,857-parameter CNN reads standardized PPG from the trailing 30 seconds. Its convolutions summarize local pulse shape and longer interval variation. Temporal mean and standard deviation feed an AF probability head. The fixed training recipe uses 18 epochs and a class-balanced binary loss. The grouped fitting callback receives the applicable training labels and held-out input arrays without held-out labels. [CNN architecture](../../ppg_af/af_wrist_raw_cnn.py) · [Grouped model fitting](../../ppg_af/af_wrist_cv.py)

## Results

| Participant partition | Original quality template | Expanded quality template | Expanded raw PPG CNN |
| --- | --- | --- | --- |
| 2026 | 14/17 detections, 4 false starts | 11/17, 1 | 1/17, 3 |
| 2030 | 1/17, 0 | 12/17, 2 | 1/17, 4 |

The CNN detected one episode in both groupings and had more false starts than the control trained on the same data. Results support the template for this sample size and training recipe. [Aggregate results](raw-cnn-results.json)

To reproduce, run `python -m ppg_af.af_wrist_raw_cnn` with the original lock, clock, quality and four development archive arguments from the [motion study](motion-af-study.md#reproduce). Also supply `--waveform-pairs`, `--waveform-results`, `--ptt-dir` and a fresh `--output`. The published results use `--device cuda` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
