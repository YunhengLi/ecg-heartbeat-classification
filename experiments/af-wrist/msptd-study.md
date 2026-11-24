# Test a published multiscale pulse detector on the original wrist benchmark

MSPTDfast v2 locates plausible pulse peaks using local extrema across multiple time scales. Use the NeuroKit2 0.3.0 implementation to test whether a published optical beat detector improves timing and downstream AF alerts. [MSPTDfast study](https://doi.org/10.1088/1361-6579/adb89e) · [Versioned original implementation](https://zenodo.org/records/15208943)

Use the original trailing 30-second input and 0.4–8 Hz preprocessing. Replace only the direct pulse peak detector, retaining the original present-sample mask, seven interval features, quality interactions and nested AF folds. Compare the original template, broad-band optical peaks, and MSPTDfast peaks. All 881 originally scoreable rows supplied valid intervals with MSPTDfast. [Wrist runner](../../ppg_af/af_wrist_msptd.py)

| Evaluation | Template | Broad-band optical peaks | MSPTDfast v2 |
| --- | ---: | ---: | ---: |
| Median irregularity error across 77 stable AF windows | 0.202 | 0.199 | 0.155 |
| Median irregularity error across 146 stable non-AF windows | 0.387 | 0.370 | 0.373 |
| Original AF grouping, complete detections and false starts | 14/17, 4 | 12/17, 3 | 3/17, 3 |
| Alternative AF grouping, complete detections and false starts | 1/17, 0 | 1/17, 0 | 0/17, 2 |

The detector gave lower median timing variation error in stable AF without improving the wrist alert result. These endpoints use separate signal and event references. [Aggregate results](msptd-results.json)

Run `python -m ppg_af.af_wrist_msptd` with the same original archives, `--ptt-dir`, and waveform inputs as the [pulse consensus study](pulse-consensus-study.md), with `neurokit2==0.3.0` installed.
