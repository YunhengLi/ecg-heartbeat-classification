# Weak optical pulses and AF alert replay

During AF, some electrical beats can be followed by weak or absent peripheral pulses. This study tests whether detecting weaker observed pulses improves the recovery of ECG-timed beats and whether the change carries through to the same held-out AF alerts. [Pulse-deficit study](https://pubmed.ncbi.nlm.nih.gov/41336913/)

## Method

1. Run the existing 0.4–8 Hz pulse detector at the original relative prominence of 0.18 and at 0.09. Prominence is relative to the difference between the fifth and 95th filtered-signal percentiles. Keep the original minimum separation, width and usable-sample mask.
2. For the gap-only route, keep the original peaks. At most, add one observed 0.09-prominence peak within each interval that is between 1.5 and 2.5 times the median original interval. Both halves must be between 0.6 and 1.4 times that median. If more than one peak qualifies, take the one closest to the midpoint.
3. Compare beat recovery and interval errors with the same 223 ECG-consensus stable-rhythm pairs, including 77 AF and 146 non-AF windows. Use the same synchronization settings for all routes. The beat-aligned reference uses the archive's assumed delay. The coarse-clock comparison uses the independently fitted template delay and includes just 23 windows from two participants.
4. Substitute each detector's valid timing features in the same model with PPG quality interactions. Keep template features as a fallback, preserve the original participant splits, fit only on training rows and calibrate using the training-side groups. Replay the same ECG-observed clock and include the alternate frozen grouping.

The main implementation is in [pulse detection and replay](../../ppg_af/af_wrist_weak_pulses.py). The two detector settings and the gap constraint are implemented in `pulse_routes()` and `insert_gap_peaks()`. The code requires the original route to reproduce the locked direct-PPG features exactly before making the comparison.

## Results

| Comparison | Original | Half prominence | Add weak peak only in long gaps |
| --- | ---: | ---: | ---: |
| Stable AF beat recall | 0.557 | 0.609 | 0.569 |
| Stable AF beat precision | 0.692 | 0.660 | 0.689 |
| Stable AF beat F1 | 0.617 | 0.634 | 0.623 |
| Stable AF median absolute irregularity error | 0.199 | 0.192 | 0.193 |
| Stable non-AF median absolute irregularity error | 0.370 | 0.396 | 0.363 |
| Primary held-out complete AF detected | 12/17 | 12/17 | 11/17 |
| Primary held-out false starts | 3 | 4 | 3 |
| Alternate held-out complete AF detected | 1/17 | 1/17 | 1/17 |
| Alternate held-out false starts | 0 | 0 | 0 |

The gap rule added 499 observed peaks across 881 shared windows. It reduced the median irregularity error across the stable rhythm pairs from 0.274 to 0.249. The modest beat recovery increase came without a meaningful gain in held-out AF detection or false starts. The primary detection difference between original and gap-only is one event, and the alternate grouping detected one event in every detector arm. The existing template plus quality interactions reference remained at 14/17 detections and four false starts in the primary grouping. [Aggregate results](weak-pulse-results.json)
