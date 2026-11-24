# Compare native ECG and optical intervals at false starts

The extended monitoring found more false starts for both quality variants and spectral fusion. This audit asks whether those starts occurred when earlier AF was still inside the model input or when the PPG-derived intervals showed more irregularity than the native ECG. [Longer monitoring results](retrospective-monitor-study.md)

## Method

Reproduce all frozen models and the exact false-start counts on the 50 supplemental blocks. Read the native ECG R-peak annotations from the verified recordings on the same clock. For each false start, check the entire preceding 30 seconds for AF overlap and valid ECG. Among fully observed non-AF inputs, pair each case with the nearest eligible negative decision in the same block, separated by at least 30 seconds, taking each negative control at most once per model and block.

Calculate the sum of normalized RMSSD and pNN50 separately from native ECG beat times and from the template-derived intervals. Report the median paired case-minus-control change for both, along with the change in their difference and the original PPG quality feature. The source labels cover AF versus non-AF. The ECG score describes timing variation within that reference. [Audit code](../../ppg_af/af_wrist_false_start_audit.py)

## Results

No false start included a documented AF interval in its preceding 30 seconds. Many false starts lasted one decision before returning negative. Within matched pairs, median changes in native ECG timing irregularity were close to zero while template-derived irregularity was higher at false starts. The quality model and its two variants also tended to start false alerts at lower optical quality than their matched negative decisions. These are descriptive within-block comparisons. [Aggregate results](false-start-audit-results.json)

| Route | False starts | Single-positive false starts | Matched pairs | ECG irregularity change | Template irregularity change | PPG quality change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain template | 44 | 14 | 40 | −0.0003 | +0.358 | +0.007 |
| Template plus quality | 76 | 35 | 67 | +0.0013 | +0.311 | −0.131 |
| Template plus quality interactions | 71 | 36 | 64 | −0.0003 | +0.262 | −0.129 |
| Quality interactions and PPG spectrum | 71 | 28 | 63 | −0.0015 | +0.258 | −0.180 |

The paired median increase in template-minus-ECG irregularity ranged from +0.249 to +0.446. The observations support testing optical beat reliability and independently confirmed AF before adding more aggregate irregularity features. No model or cutoff was refit using these comparisons.

## Reproduce

Run `python -m ppg_af.af_wrist_false_start_audit stage` with the original local `--source-dir`, verified `--plan`, `--lock`, `--prior-inputs` and a new private `--reference-cache`. Run `audit` with the same lock and staged reference cache. Also provide the same `--monitoring-cache`, model locks and PTT inputs as the [retrospective monitoring study](retrospective-monitor-study.md#reproduce), `--monitoring-results`, its SHA-256 and a new `--output`. Only aggregate timing statistics are included in the repository.
