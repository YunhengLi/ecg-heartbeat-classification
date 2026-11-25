# Frozen wrist motion quality on an independent exercise device

The GalaxyPPG experiment measured whether accelerometry helped predict large errors in PPG-derived interval irregularity. Shimmer exercise recordings offer a genuinely separate wrist device, lab and set of participants for checking the same timing endpoint. [GalaxyPPG release](https://zenodo.org/records/14635823) · [Shimmer exercise release](https://physionet.org/content/wrist/1.0.0/)

## Method

1. Fit separate standardized logistic models on 618 common windows from 23 GalaxyPPG participants for each of Galaxy Watch and Empatica E4. Keep the original interval-error definition and boundary of 0.20. Compare optical quality and spectral entropy with those two features plus the same two acceleration summaries.
2. Verify the official Shimmer exercise files and use its hand-marked chest ECG R peaks. Preserve the released segment clock and interpolate PPG and the three low-noise acceleration axes to the same trailing 30-second grid used by the wrist feature extractor. Keep windows where both the ECG interval reference and the optical template are valid.
3. Freeze each fitted pipeline, including feature scaling. Apply it to the same Shimmer windows as the original optical quality heuristic. Evaluate ROC AUC only when both error classes are present. After finding only one class, examine Spearman rank correlations with the continuous error as an exploratory diagnostic, both pooled and equally weighted across people.

The main implementation is in [frozen transfer](../../ppg_af/af_motion_quality_transfer.py). The two experiments share the [optical and motion feature function](../../ppg_af/af_wrist_motion_quality.py).

## Results

The reference identified 591 eligible windows of 632 scheduled across 19 recordings from eight participants. All 591 exceeded the original error boundary, with error minimum 0.236, median 1.105 and maximum 2.084. The classification AUC and average precision comparison are undefined for this endpoint on this test source.

| Frozen input | Pooled error rank correlation | Mean within-person error rank correlation |
| --- | ---: | ---: |
| Original optical quality heuristic | 0.412 | 0.021 |
| Galaxy optical quality and entropy | −0.100 | −0.079 |
| Galaxy optical features and acceleration | −0.158 | −0.114 |
| E4 optical quality and entropy | 0.056 | 0.006 |
| E4 optical features and acceleration | −0.182 | −0.102 |

The pooled heuristic correlation largely disappeared in the within-person comparison. Neither frozen acceleration model showed a positive mean within-person association with continuous error. Device, physical activity and cross-device clock precision can all affect this transfer. The exercise source supplies ECG beat locations rather than annotated AF episodes. [Aggregate results](motion-quality-transfer-results.json)
