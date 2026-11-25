# Trace carried alerts into earlier recorded time

In the historical miss audit, 45 carried-alert misses per policy occurred in clips that began with an active alert. The earlier origin was unavailable within the scored clips. This study traces those cases using the sealed preroll and the native ranges already cached for the same participants. [Historical miss audit](short-rescue-study.md)

## Method

1. Reproduce the original locked event and false-start counts. Find the earlier boundary of the continuous positive decision sequence that carried into each missed AF episode.
2. Trace the alert's positive tail backward on the ten-second clock. Accept an origin only after an observed negative or known unavailable reset. Low-quality decisions suppressed by the absence of an earlier saved score are ambiguous. When the sealed prefix ends in an unresolved positive tail, extend the earlier waveform context with locally cached PPG and ECG bytes. Rebuild model features on the same window clock and verify overlap with the sealed PPG.
3. At a recovered origin, read the ECG rhythm immediately before the decision endpoint. Verify finite, nonflat ECG and continuous rhythm annotations across the preceding cell. Use this extra history for attribution while preserving the original scored exposure and endpoints. Multiple later AF episodes can share one earlier alert. Report miss cases separately from the number of unique blocks. [Provenance code](../../ppg_af/af_wrist_preroll_provenance.py)

## Results at 180 seconds

| Previously unresolved carried misses | Original | Quality confirmation |
| --- | ---: | ---: |
| Total | 45 | 45 |
| Resolved with sealed history | 18 | 18 |
| Resolved with additional cached native history | 7 | 7 |
| Still unknown | 20 | 20 |
| Recovered origin classified as non-AF | 19 | 19 |
| Recovered origin classified as earlier AF | 6 | 6 |

Nineteen distinct blocks had adequate native extension data at 180 seconds. The original 155 detections and quality-confirmed 151 detections among 260 complete events reproduced exactly. The additional history contributed zero exposure to those endpoint comparisons. [Aggregate results](preroll-provenance-results.json)

The 300-second sensitivity check verified paired data in 18 of those blocks. One block with an unavailable native byte range fell back to 180 seconds. Longer history resolved two additional cases per policy, one with ECG-confirmed non-AF at the start and one with AF during the preceding interval. Across the 45 original cases per policy, the 27 resolved origins comprise 20 non-AF and seven AF starts. Eighteen remain unresolved at this lookback. The frozen endpoint counts and scoring exposure reproduced unchanged. [300-second results](preroll-provenance-longer-results.json)

## Reproduce

Use the same private archive paths as in the [miss audit reproduction](short-rescue-study.md#reproduce), the existing local range cache, PTT recordings, pinned RDDM checkout and weights. Run the original CPU or CUDA environment with a fresh `REPORT` path.

```bash
python -m ppg_af.af_wrist_preroll_provenance \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --prior-inputs "$RAW/template-independent-prepared" \
  --data-preflight experiments/af-wrist/data-preflight-results.json \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --coverage-report experiments/af-wrist/template-independent-reference-results.json \
  --historical-results experiments/af-wrist/template-independent-results.json \
  --confirmation-results experiments/af-wrist/confirmation-retrospective-results.json \
  --native-metadata "$RAW/metadata-v6.json" --native-cache "$RAW/cohort-range-cache" \
  --ptt-dir "$PTT" --upstream "$RDDM" --weights "$WEIGHTS" --device cuda --output "$REPORT"
```

For the longer sensitivity check, add `--native-lookback-seconds 300` and use a separate output path.
