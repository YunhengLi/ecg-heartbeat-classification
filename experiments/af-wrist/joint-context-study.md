# Combine short and long wrist PPG context

The 10-second wrist template previously matched the 30-second template's pooled episode sensitivity while increasing false starts. On the long measured-ECG reference, jointly fitting both input lengths decreased false starts at one sensitivity-matched operating point. We tested whether the joint representation also helps when interval measurements come from wrist PPG. [Single-context wrist comparison](template-context.md) · [Measured ECG reference](shdb-ecg-reference.md)

## Comparison

Rebuild the seven interval features from the last 10 seconds of each 30-second optical input. Verify all 881 archived long inputs against exactly reconstructed feature values and retain identical score coverage. Concatenate the seven features at each length. Compare that 14-feature logistic model against both seven-feature classifiers at C=1. The five outer and three inner participant groups, stable fit labels, original decision times, preroll and 80% calibration target are shared. Repeat at the original seed and the earlier alternate grouping. [Joint-context code](../../ppg_af/af_wrist_joint_context.py)

| Context | 2026 complete AF | 2026 false starts | 2030 complete AF | 2030 false starts |
| --- | ---: | ---: | ---: | ---: |
| 30 seconds | 12/17 | 6 | 1/17 | 2 |
| 10 seconds | 12/17 | 9 | 1/17 | 6 |
| Both input lengths | 10/17 | 5 | 1/17 | 2 |

In the primary grouping the combined model gained three detections and lost five versus the 30-second template. Its net sensitivity fell by 11.8 percentage points, beyond the ten-point tolerance. The alternate grouping reproduced the 30-second counts. Keep the long context as the template benchmark. [Aggregate results](joint-context-results.json)

## Reproduce

Run `ppg_af.af_wrist_joint_context` with the original cohort inputs detailed in the [quality confirmation study](quality-confirmation-study.md), the original `--ptt-dir`, `--context-results experiments/af-wrist/template-context-results.json`, `--confirmation-results experiments/af-wrist/quality-confirmation-results.json` and a fresh `--output` path.
