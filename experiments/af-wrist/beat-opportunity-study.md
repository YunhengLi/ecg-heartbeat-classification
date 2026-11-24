# Optical peak opportunities against a timing rotation control

Lower prominence matched more native ECG beats in the previous comparison. A denser candidate stream also makes timing coincidences more likely. This diagnostic measures what the recorded optical extrema contribute above a density-matched control. [Weak pulse study](weak-pulse-study.md)

## Method

1. Use the same 223 stable paired wrist windows and 7,549 ECG-consensus QRS annotations as the original beat study. Remove the same one-second edge on each side. Use the documented 300 ms alignment on beat-synchronized archive rows and the fixed 190 ms external PTT-template delay on coarse rows.
2. Evaluate the original 0.18-prominence detector, half prominence, adding weak peaks only in plausible long gaps, and a denser list containing both weak peaks and troughs. Match chronological optical candidates to measured ECG beats one-to-one within 100 ms. ECG is used to count the possible matches for this diagnostic.
3. Generate 32 deterministic circular rotations of each candidate list within its same supported interval. Keep the candidate count, compare the observed recall with the mean rotated recall and summarize case differences by participant. Suppress bootstrap intervals for fewer than three participants.

The [candidate timing code](../../ppg_af/af_wrist_beat_opportunity.py) reuses the existing detector, verified clocks and beat matcher. The timing rotations preserve within-window candidate density and approximate spacing. They are a descriptive control, not a model trained to infer ECG.

## Results

| Stable AF candidate method | Observed recall | Rotated recall | Mean within-participant advantage |
| --- | ---: | ---: | ---: |
| Original optical peaks | 0.557 | 0.259 | +0.348 |
| Weak optical peaks | 0.609 | 0.297 | +0.360 |
| Weak peaks in long gaps only | 0.569 | 0.266 | +0.353 |
| Weak peaks and troughs | 0.690 | 0.526 | +0.209 |

The stable AF subset contains 77 windows from seven people. Adding both polarities raised the raw match count while increasing chance matches substantially. Across 200 beat-aligned archive windows the original candidate recall was 0.658, versus 0.226 after rotation. Across the 23 coarse-clock windows from two people, it was 0.089 versus 0.278. These results depend strongly on the documented timing convention. The coarse subgroup leaves uncertainty about delay, sensor contact and pulse measurement. [Aggregate results](beat-opportunity-results.json)
