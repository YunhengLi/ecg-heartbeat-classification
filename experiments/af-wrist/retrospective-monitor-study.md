# Additional monitoring on 25 previously compared wrist participants

The short five-fold comparison and previous 25-person replay use limited prescheduled wrist exposure. This study locks two additional five-minute blocks per person and applies the original frozen models before comparing false starts. None of the 25 participants entered model training or calibration. They were already included in a previous evaluation of the same dataset. [Previous 25-person replay](quality-retrospective-study.md)

## Method

1. Sample two blocks for each person using their wear intervals and recording duration. Use the same 30-second start grid as the development extension. Exclude all earlier parent, pending, predecessor and stride inputs, including a one-second buffer. Use no AF labels, signal quality or model outputs to choose the blocks.
2. Stage the 50 blocks and their preceding 90 seconds of PPG from the original synchronized release. Verify each scheduled input against the fixed plan. Use trailing 30-second inputs and ten-second decisions with the frozen alert state rules. Verify the prior plain template and two quality model digests, and verify the spectral fusion model fit and cutoff frozen on the original five-plus-five development split before any supplemental scoring. [Spectral fusion freeze](direct-fusion-study.md#freeze-for-the-supplementary-monitoring-replay)
3. Count false starts per hour of ECG-observed wrist exposure. Compare model differences by resampling the 25 participants together 10,000 times. Keep model fits, positive thresholds and scoring windows fixed.

## Results

All four routes produced a score at all 1,500 scheduled decisions. Of 250 scheduled minutes, 241 minutes 30 seconds had usable ECG rhythm reference. The ECG-confirmed non-AF exposure was 191 minutes 15 seconds. There were no complete AF episodes, so the new blocks cannot supply a sensitivity comparison. [Aggregate results](retrospective-monitor-results.json)

| Frozen model | False starts | False starts per ECG-observed wrist hour | Paired difference vs plain template and 95% interval, per hour |
| --- | ---: | ---: | ---: |
| Plain template | 44 | 10.93 | — |
| Template plus quality | 76 | 18.88 | +7.95 [+3.40, +12.96] |
| Template plus quality interactions | 71 | 17.64 | +6.71 [+2.44, +11.30] |
| Quality interactions and direct PPG spectrum | 71 | 17.64 | +6.71 [+1.98, +12.13] |

The spectral fusion's lower false-start count in the original small development comparison did not recur here. The quality-interaction model again produced more false starts than the plain template, consistent with the longer monitoring on the ten development participants. These comparisons evaluate disjoint times sampled from the same source release. The rates and participant intervals describe this sampled exposure. [Development extension](monitor-extension-study.md)

## Reproduce

Run `python -m ppg_af.af_wrist_retrospective_monitor plan` with `--lock`, `--prior-inputs`, `--source-dir` and a fresh private `--plan`. `--prior-inputs` is the original sealed `template-independent-prepared` directory. Run `stage` with the same inputs and `--monitoring-cache`. Exact participant times and waveforms stay local.

```bash
python -m ppg_af.af_wrist_retrospective_monitor score \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --prior-inputs "$RAW/template-independent-prepared" --plan "$PRIVATE_PLAN" \
  --monitoring-cache "$PRIVATE_CACHE" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --variant-lock "$QUALITY_LOCK" --spectral-model-lock "$SPECTRAL_LOCK" \
  --ptt-dir "$PTT" --output runs/retrospective-monitor-results.json
```

[`verify_schedule` and `evaluate`](../../ppg_af/af_wrist_retrospective_monitor.py) check disjoint times and model provenance, and then score the same input support for all four models.
