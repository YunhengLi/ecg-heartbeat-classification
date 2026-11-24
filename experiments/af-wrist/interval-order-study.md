# Check whether the order of pulse intervals helps AF discrimination

The same set of interval lengths can occur in different orders. We compared the observed frequency of two patterns with the exact expected frequency after permuting the same observed intervals. The patterns were changes in direction above 50 ms and adjacent short–long intervals whose total returns near twice the median. Pairs and triples cannot span a signal gap. This is a local comparison inspired by published interval turning-point and ectopy work. [Turning points in AF](https://doi.org/10.1007/s10439-009-9740-z) · [PPG patterns for ectopy discrimination](https://pmc.ncbi.nlm.nih.gov/articles/PMC7582300/)

## Held-out wrist alerts

Add two features to the same quality-interaction template classifier. One arm gets the observed features and one gets only the two expectations computed from the interval distribution. This controls the number of additional classifier inputs. Both 2026 and 2030 use the original grouped fitting and nested cutoff selection. [Wrist runner](../../ppg_af/af_wrist_interval_order.py)

| Participant grouping | Original quality template | Distribution-only features | Observed interval order |
| --- | --- | --- | --- |
| 2026 | 14/17 detections, 4 false starts | 9/17, 7 | 12/17, 5 |
| 2030 | 1/17, 0 | 1/17, 1 | 1/17, 0 |

The observed sequence improved against the equal-dimension distribution control. It lost two detections and added one false start compared to the original quality template on the primary grouping. [Wrist results](interval-order-results.json)

## Independent surgical rhythm comparison

Reuse the same 30 native surgical cases, rhythm labels, train row selection, participants and folds as the original VitalDB study. Calculate both pairs of features directly from pulse timing. Hold the original participants out for each source fit. The endpoint distinguishes the published combined AF-or-flutter label from ectopy. [VitalDB data and scope](vitaldb-transfer-study.md)

| Feature set | AUC with sinus negative fitting | AUC with mixed ectopy-negative fitting |
| --- | ---: | ---: |
| Original | 0.620 | 0.736 |
| Distribution-only | 0.750 | 0.821 |
| Observed pulse order | 0.572 | 0.697 |

The distribution-only features improved the participant-weighted AUC on this predefined surgical sample. The sequence features did not. The surgical result concerns stable rhythm window ranking and does not establish a wrist alert improvement. [Surgical runner](../../ppg_af/af_vitaldb_interval_order.py) · [Surgical results](vitaldb-interval-order-results.json)
