# Frozen AF models on additional wrist monitoring

The first comparison used 46.5 minutes of ECG-observed wrist monitoring. We selected three additional five-minute blocks per original development participant to check whether the false-start difference persists over longer recordings. Both models were frozen before these intervals were scored. [Source AFPPGECG release](https://zenodo.org/records/23921320)

## Protocol

1. Choose from a 30-second start grid inside the recorded wrist wear segments using a deterministic participant seed. Exclude previous fit, evaluation, and supplemental training windows with a temporal buffer. This step reads wear bounds and recording duration without using AF labels, waveform quality, or model output. Preserve all 30 chosen blocks, including the blocks with gaps.
2. Reproduce the original group assignment, original held-out scores, and training-only cutoffs. For each person, use the matching held-out template models. Extract each feature from the preceding 30 seconds of PPG at the same 10-second decision cadence. Start alert-state replay using the original clock with 90 seconds of PPG history before the block.
3. Score new starts only in the planned monitoring time with valid ECG reference. Count false starts per hour of ECG-observed wrist time. Resample participants in pairs for a descriptive interval on the difference. Keep the new blocks separate from the original episode benchmark.

## Results

The 150 scheduled minutes contained 144 minutes 20 seconds of ECG reference, including 119 minutes 20 seconds of ECG-confirmed non-AF. Both routes produced scores at 810 of 900 scheduled decisions. The final 10 seconds of PPG were observed at 883 decisions. No complete AF episode was present in the chosen blocks, so this comparison does not measure episode sensitivity. [Aggregate results](monitor-extension-results.json)

| Route | Original complete AF detections | Original false starts | New false starts | New false starts per ECG-observed hour |
| --- | ---: | ---: | ---: | ---: |
| Plain template | 12/17 | 6 | 27 | 11.22 |
| Template with quality interactions | 14/17 | 4 | 35 | 14.55 |

Sixteen new false starts occurred at the same decision time for both routes. Eleven were unique to plain template and 19 to the quality variant. The quality variant added 3.33 false starts per ECG-observed wrist hour, with a paired participant-bootstrap 95% interval of +0.84 to +5.60 on these ten people. These were new times on the same development participants. The original short comparison alone gave a misleading impression of how the two routes rank on false starts for this cohort. Keep the plain template as the benchmark while seeking independent wrist data with both complete AF episodes and longer monitoring. [Original quality study](quality-study.md)

## Reproduce

Generate the private schedule with `ppg_af.af_wrist_monitor_plan` from the original release, pinned protocol, original core and preroll archives, and the 109-window training supplement. Stage planned native blocks by invoking `ppg_af.af_wrist_monitor_extension` with `--plan`, `--lock`, `--source-dir` and `--output`. The committed JSON contains only aggregate results and the plan digest. Exact recording times and staged waveforms remain in the local workspace.

```bash
python -m ppg_af.af_wrist_monitor_extension \
  --plan "$PRIVATE_PLAN" --lock "$RAW/holdout-window-seeded-lock.json" \
  --monitoring-cache "$PRIVATE_STAGE" \
  --development-raw "$RAW/cohort-seeded-development" \
  --development-features "$RAW/cohort-seeded-development-features" \
  --stride-raw "$RAW/cohort-seeded-development-stride10" \
  --stride-features "$RAW/cohort-seeded-development-stride10-features" \
  --clock-results experiments/af-wrist/clock-study-results.json \
  --quality-results experiments/af-wrist/quality-study-results.json \
  --ptt-dir "$PTT" --output runs/monitor-extension-results.json
```

[`choose_monitor_blocks`](../../ppg_af/af_wrist_monitor_plan.py) builds the schedule from wear and exclusion intervals. [`_reuse_fold_models` and `_person_runs`](../../ppg_af/af_wrist_monitor_extension.py) verify the held-out scores and replay each state from the original cutoffs.
