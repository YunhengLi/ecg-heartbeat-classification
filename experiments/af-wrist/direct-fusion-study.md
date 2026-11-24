# Combine direct PPG spectrum and template intervals for AF

The template extracts intervals from ECG-like pulses generated from PPG. Direct optical features also include dominant frequency and pulse-band entropy. This study tests whether adding these two features, or all nine direct features, improves an existing template and quality classifier. Use the same original wrist input, labels, group splits, quality scores and training-only cutoff selection. [Feature calculation](../../ppg_af/features.py) · [Grouped comparison](../../ppg_af/af_wrist_direct_fusion.py)

## Results

| Participant partition | Direct PPG | Original quality template | Template and PPG spectrum | Template and all direct PPG features |
| --- | --- | --- | --- | --- |
| 2026 | 7/17 detections, 4 false starts | 14/17, 4 | 12/17, 3 | 11/17, 3 |
| 2030 | 1/17, 2 | 1/17, 0 | 1/17, 0 | 1/17, 0 |

The two spectral features reproduced 12/17 detections, the aggregate count of the original plain template on the 2026 split. False starts fell from six to three over the same 46.5 ECG-observed wrist minutes. The quality template detected two additional episodes with one additional false start. The 2030 grouping remained weak, and no new development route reached the 80% sensitivity target. [Aggregate results](direct-fusion-results.json) · [Original plain template results](clock-study-results.json)

## Freeze for the supplementary monitoring replay

Fit the two spectral features with the quality template on the original five fitting participants and exactly 43 stable fitting windows. Select the cutoff on the original five separate calibration participants. The calibration detected 13/17 complete episodes with four false starts and fell below the 80% target. Freeze the candidate before scoring any new supplemental times on the 25 previously compared participants. The model remains private. Its lock digest is `fe639167e77c0c7b6951dc682992b67bf332cf0b6bf945a4f1b295fc6e9b7400`.

Run `python -m ppg_af.af_wrist_direct_fusion` with the original development archives, lock, clock and quality results as in the [motion study](motion-af-study.md#reproduce), without the acceleration argument. Add `--freeze-spectral` and a separate private `--output` to reproduce the fit used in later supplemental scoring.
