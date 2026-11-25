# Test pulse strength and its relation to beat timing

Pulse heights can vary alongside the preceding pulse interval. We tested whether local optical prominence improves held-out AF decisions beyond the quality-interaction template. Prior work has evaluated PPG amplitude and pulse deficit during AF. [Wrist PPG morphology study](https://pubmed.ncbi.nlm.nih.gov/34025448/) · [Pulse deficit study](https://pubmed.ncbi.nlm.nih.gov/41336913/)

## Method

Use the original broad-band pulse positions. Estimate each peak's prominence over at most one second on either side, inside the available 30-second optical window. For the primary comparison add the interquartile range of log prominence. For the coupling comparison also add the Spearman correlation of log prominence and the preceding optical interval. Reject long or incomplete intervals, and use zero for both added features if fewer than five paired observations remain. The seven template timing features, native quality and their interactions are the original input. The five outer participant folds, nested cutoff estimation, clock and score availability are shared. [Prominence and grouped evaluation code](../../ppg_af/af_wrist_pulse_amplitude.py)

Of 881 shared inputs, 875 supported the prominence statistics. On verified stable paired windows, median log-prominence spread was 0.645 in 74 scoreable AF windows and 0.292 in 145 scoreable non-AF windows. Median amplitude-to-interval rank correlation was 0.318 and 0.177, respectively. These descriptive samples include repeated windows per participant.

| Route | 2026 complete AF | 2026 false starts | 2030 complete AF | 2030 false starts |
| --- | ---: | ---: | ---: | ---: |
| Original quality-interaction template | 14/17 | 4 | 1/17 | 0 |
| Add prominence distribution | 11/17 | 6 | 1/17 | 0 |
| Add prominence and preceding-interval correlation | 10/17 | 2 | 0/17 | 0 |

The combined features removed two false starts in the original partition at a cost of four net detections. The held-out drop was 23.5 percentage points, outside the locked tolerance. The observed rhythm differences from stable samples did not yield a comparable-sensitivity alert improvement. [Aggregate results](pulse-amplitude-results.json)

## Reproduce

Run `ppg_af.af_wrist_pulse_amplitude` with the same locked clock, quality and waveform-pair archives as the [ECG feature regression study](ecg-feature-regression-study.md). This experiment does not need the separate `--ptt-dir` option because it reuses the original template features from the development archives. Set `--output` to a new path.
