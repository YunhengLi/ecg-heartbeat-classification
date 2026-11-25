# Ambient light and wrist AF detection

The AFPPGECG release includes a 100 Hz ambient optical channel with the green wrist PPG. It provides a separate measurement of background optical variation. We tested whether frequency information shared with that channel helps the template ECG classifier distinguish AF. [Original AFPPGECG release](https://zenodo.org/records/23921320)

## Method

1. Reproduce each of 881 originally scoreable green PPG inputs from the native MAT files. Read the ambient input at the same indices only after confirming that both channel lengths agree. All 881 rows from nine scoreable participants supplied finite and nonconstant ambient data. The portable runner verifies the green PPG digest and decision row numbers before using each paired ambient clip.
2. Estimate squared coherence between green and ambient optical channels on the same trailing 30 seconds. Use the 0.75–3.5 Hz band from the motion experiments. The added feature is green PPG power remaining after weighting by one minus coherence, divided by original in-band power. Median retained power was 0.817. This is a feature derived from shared frequency content.
3. Compare the original template quality interactions with the same model plus ambient coherence. Use `C=1`, identical PPG inputs, the same participant splits and training-only cutoff selection. Ambient availability is constant on scoreable windows, so it provides no variation in this comparison.

## Results

| Participant partition | Original quality model | Model with ambient coherence |
| --- | --- | --- |
| 2026 | 14/17 complete detections and 4 wrist false starts | 2/17 and 1 |
| 2030 | 1/17 and 0 | 1/17 and 0 |

The original partition lost 12 AF detections and avoided three false starts over the same 46.5 minutes of ECG-observed wrist monitoring. This loss exceeded the prespecified ten-percentage-point sensitivity tolerance. The weak partition retained its earlier result. The evaluated feature adds no support for choosing this model as the alert baseline. [Aggregate results](ambient-light-results.json)

## Reproduce

The original wrist development files are in `RAW`, the original native MAT files in `SOURCE`, and the private staged ambient files in `AMBIENT`. Omit `--source-dir` when matching files are already staged.

```bash
python -m ppg_af.af_wrist_ambient --source-dir "$SOURCE" \
  --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --ambient-sidecars "$AMBIENT" --lock "$RAW/holdout-window-seeded-lock.json" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --quality-results experiments/af-wrist/quality-study-results.json \
  --output runs/ambient-light-results.json
```

[`NativeAmbient`, `ambient_residual_power` and `evaluate_ambient`](../../ppg_af/af_wrist_ambient.py) define the shared native clock, ambient optical feature, and participant-level AF comparison.
