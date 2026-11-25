# Corroborate pulse timing using red and infrared PPG

Two filters of one wrist signal share many artifacts. This study asks whether a second measured optical wavelength helps where the first channel produces an inaccurate pulse interval. PeriWaveSync-ABG provides synchronized 500 Hz finger red PPG, infrared PPG, and single-lead ECG recorded during surgery at four centers. [Dataset and CC BY 4.0 license](https://huggingface.co/datasets/DX-anes/PeriWaveSync-ABG) · [Scientific Data article](https://doi.org/10.1038/s41597-026-08836-2)

## Method

1. Pin release revision `0bd11f9dd0fa8a9329ea113b388007d9c7ae7984`. Select eight distinct participants per center by a deterministic rank on the archive membership names, choosing one segment per participant before reading waveform content. Take five nonoverlapping 30-second segments starting at 30, 90, 150, 210, and 270 seconds relative to each file. HTTP range requests retrieved 196,938,312 bytes from the four original ZIP files. The loader verifies the exact response range and reads only selected CRC-checked members.
2. Require both the project ECG detector and WFDB XQRS to agree on 90% of peaks in both directions within 100 ms and on median rate within 5 bpm. Use the original ECG sampling grid for reference intervals. Both algorithms see the same ECG lead, so this is a detector-agreement reference.
3. Resample each PPG to 100 Hz and determine its polarity from the distribution of the filtered samples within that input. Compare red alone, infrared alone, red across two filter bands, and red with infrared corroboration. The dual route requires each consecutive red peak to have an infrared peak within 80 ms. Exclude intervals touching an unverified peak without joining nonadjacent pulses.
4. Compare the same supported windows for all four routes. As a follow-up, retain red when dual support is unavailable and compare the fallback route across all red-supported windows. Resample participants together to describe uncertainty around the mean paired change in absolute error.

## Results

There were 160 staged windows, 84 with ECG detector agreement, and 61 with support from all four optical routes. The common comparison contains 21 people. [Aggregate results](periwave-results.json)

| Method | Median absolute irregularity error | Median absolute HR error |
| --- | ---: | ---: |
| Red | 0.160 | 0.319 bpm |
| Infrared | 0.187 | 0.393 bpm |
| Red across two filters | 0.160 | 0.319 bpm |
| Red with infrared corroboration | 0.153 | 0.347 bpm |

The dual route improved 12 paired windows, worsened 11 and left 38 unchanged. Its mean paired absolute irregularity error decreased by 0.056, with a participant-bootstrap interval from −0.120 to −0.007. The median paired change was zero. In the 84-window fallback comparison, the mean paired change was −0.041 [−0.086, −0.005]. The source has no AF episode annotations. The experiment measures finger pulse interval agreement under surgery, including sensor and acquisition artifacts.

## Reproduce

```bash
python -m ppg_af.af_periwave_stage --cache-dir "$PRIVATE_PERIWAVE"
python -m ppg_af.af_periwave_benchmark \
  --staged "$PRIVATE_PERIWAVE/staged" --output runs/periwave-results.json
```

Use `--zip-dir` to prepare data from the four locally downloaded release archives. Install the wrist requirements, `requests`, and `wfdb==4.3.1`. The staging runner selects native records and verifies sparse reads. `evaluate_window` and `evaluate` calculate ECG agreement, optical interval estimates and paired metrics. [Staging code](../../ppg_af/af_periwave_stage.py) · [Evaluation code](../../ppg_af/af_periwave_benchmark.py)
