# Limits of one global cutoff on the frozen wrist AF scores

The alert entry and release comparison decreased some false-start counts while sacrificing detections. This diagnostic checks how much one threshold can change the tradeoff with the existing scores. The analysis uses previously reviewed outcomes and estimates the best attainable operating points on this cohort. [Alert-state study](alert-state-study.md)

## Method

1. Rebuild the sealed original and quality-confirmed score streams. For each route, keep its available inputs, ten-second decision clock, known gaps, and causal prehistory fixed. Use the 260 historical complete AF episodes for detection and both additional outcome-blind wear schedules for burden. Their shared denominator is 985.5 ECG-observed minutes.
2. A definite start at a known decision requires the current score to reach the cutoff and the preceding score to fall below it. A known unavailable predecessor resets state. An unobserved predecessor can make a start ambiguous. Enumerate the exact cutoff breakpoints for those conditions. Count each complete episode once if any of its eligible decisions starts an alert. [Cutoff sweep code](../../ppg_af/af_wrist_cutoff_frontier.py)
3. Verify the archived cutoff counts for both routes. Independently replay the maximum-detection cutoff with the original causal episode scorer. For a shared burden comparison, use the published quality-confirmed count of 210 false starts as the diagnostic cap. This cap represents the existing comparator's burden.

## Results

| Score stream | Cutoffs checked | Highest complete-episode detection | False starts at that point | Highest detection with at most 210 false starts |
| --- | ---: | ---: | ---: | ---: |
| Original template | 6,222 | 163/260 (62.69%) | 238 | 145/260 (55.77%), 204 false starts |
| Quality confirmation | 5,252 | 153/260 (58.85%) | 206 | 153/260 (58.85%), 206 false starts |

The 80% target requires 208 detections. No tested global cutoff reached it, even when the threshold search used all observed outcomes and allowed any false-start count. Individually, 241 original and 239 quality-confirmed episodes could produce a definite start at some cutoff. The much lower maxima reflect the difficulty of selecting one cutoff that gives correctly timed new starts across episodes and participants. The existing threshold remains the reference. [Aggregate results](cutoff-frontier-results.json)

## Reproduce

Use the [alert-state study command](alert-state-study.md#reproduce) with module `ppg_af.af_wrist_cutoff_frontier`. Omit `--development-raw`, `--development-features`, `--stride-raw`, `--stride-features`, `--clock-results`, `--quality-results`, and `--development-confirmation-results`. Use a fresh `--output` path.
