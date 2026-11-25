# Data and evaluation foundation for wrist AF experiments

This protocol makes the current dataset and benchmark explicit before testing larger models. It uses the [fixed development protocol](window-seeded-evaluation.md) and the [independent reference audit](template-independent-reference-results.json).

## Data inventory

For each source, record the release, acquisition setting, sensor locations, sampling rates, synchronization method, and available reference labels. Track the three useful input types separately. ECG-only recordings can support ECG pretraining. PPG-only recordings can support PPG pretraining. Paired training examples for waveform reconstruction use ECG and PPG recorded from the same participant over the same time interval.

The current AFPPGECG v6 release contains 39 ECG recordings and 35 matched PPG recordings. Ten paired participants form the development set. Five are designated for fixed fitting and five for fixed calibration, and all ten can enter grouped development cross-validation. Twenty-five others form the published independent evaluation. Its outputs have been opened for the [error audit](template-independent-error-audit.md). All 35 paired participants are assigned.

## Checks before a model run

- Verify the source and private protocol digests and participant rosters. Keep a person's original recordings, overlapping windows and supplemental training examples in the same fold.
- Compare raw and feature clocks and ECG reference labels for the same decision rows. Each decision ends on the prespecified 10-second grid and reads only the preceding 30 seconds of PPG. Prehistory initializes state and contributes zero scored exposure.
- Record raw PPG gaps, unavailable model features and unknown alignment separately. Complete ECG-confirmed AF episodes remain in the sensitivity denominator when the model has no usable PPG input.
- Separate complete episodes from partial episodes. Count new in-episode starts for sensitivity and use the prescheduled wrist-monitoring blocks alone for the false-start denominator. Report alert delay alongside both outcomes. The 80% development sensitivity target and the ten-percentage-point matching tolerance are defined in the existing protocol.

| Reference coverage | Development | Published independent |
| --- | ---: | ---: |
| Participants | 10 | 25 |
| Complete AF episodes | 17 | 260 |
| Partial AF episodes | 10 | 82 |
| ECG-observed wrist monitoring | 46.5 min | 125 min |
| Complete AF episodes with a shared feature endpoint | 16 | 253 |
| Decision cells with features from all three routes | 513 / 705 | 5,424 / 5,463 |

The original development fitting pools contain 34 AF and 80 non-AF stable windows with shared features. Eight development participants had eligible shared feature rows on the evaluation grid. The participant and ECG coverage counts include everyone. The inventory checks the native release manifest. The waveform and digest checks cover the staged excerpts used in the benchmark.

## Future model comparisons

Use the current development folds to test architecture or training changes with the same inputs and scoring policy. Set thresholds using training-fold participants. Establish external performance on a new paired cohort after locking the model and cutoff. Record person-level support as well as the number of windows and episodes. A participant can contribute multiple episodes, so uncertainty resampling uses whole participants.

The [external wrist source review](external-wrist-validation-sources.md) documents candidate paired cohorts and which reference fields are needed to run the current episode endpoint.

For waveform reconstruction, the [paired PTT benchmark](../rddm/README.md#2-run-the-external-ptt-benchmark) scores fixed-clock waveform correlation and R-peak matching against measured ECG on participant-held-out partitions. Score waveform fidelity and AF alert utility separately. The AFPPGECG beat-based synchronization method assumes a pulse delay, so absolute R-peak timing needs a reference with independent clock synchronization.

The [MIMIC PERform AF v2.0 release](https://zenodo.org/records/15906524) supplies a separate ICU source with 35 paired ECG and PPG recordings, including 19 AF and 16 non-AF records. The two inspected MAT files matched the listed file checksums. They contain 125 Hz signals and a rhythm label for each record. The files can support a separate waveform or record-level experiment. An episode alert evaluation requires annotations for onset and offset. Check patient overlap with the RDDM training data before using a MIMIC-derived dataset for external validation of RDDM.

## Reproduce the preflight

Use the development archives from the [decision cadence study](clock-study.md#reproduce), the private protocol files and the five sealed independent archive directories. The command checks the native and feature clocks, ECG reference coverage, shared feature availability and the independent seal. It runs from staged archives and generates zero classifier predictions.

```powershell
python -m ppg_af.af_wrist_data_preflight `
  --source-manifest data/afppgecg-v6/metadata-v6.json `
  --protocol-lock runs/wrist-window-seeded-lock.json `
  --independent-lock runs/template-independent-model-lock.json `
  --clock-results experiments/af-wrist/clock-study-results.json `
  --coverage-report experiments/af-wrist/template-independent-reference-results.json `
  --development-raw data/wrist-development `
  --development-features data/wrist-development-features `
  --development-preroll runs/wrist-history `
  --development-stride-raw runs/wrist-stride10 `
  --development-stride-features runs/wrist-stride10-features `
  --independent-prepared data/wrist-independent-prepared `
  --independent-seal data/wrist-independent-prepared/sealed-input-proof.json `
  --output runs/data-preflight-results.json
```

The [preflight code](../../ppg_af/af_wrist_data_preflight.py) reports coverage separately for the original 30-second grid and the 10-second decision grid. The [aggregate output](data-preflight-results.json) records the verified counts and input fingerprints. The report hashes the exact input bytes. Use the original GitHub file bytes when comparing the two published report fingerprints across checkouts.
