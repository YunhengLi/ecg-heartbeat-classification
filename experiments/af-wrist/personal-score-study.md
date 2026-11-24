# Causal wearer-specific AF score centering

This experiment tests whether a wearable alert can reduce false starts by adapting the frozen template score to a wearer's recent score distribution. The baseline uses the wearer's prior PPG scores alone.

## Method

1. From the original five frozen-model fitting participants, use the three with scoreable routine wear. Take the median log odds for each of those three people, then the median across people. Their 141 training-side wear decisions yielded a reference center of 1.440 log odds.
2. Reserve the first five minutes of each sealed half-hour test block as a common warm-up. Compare a fixed median from that prefix and a rolling median of the preceding five minutes. Each estimate requires at least 20 finite decisions. At each later time, subtract half of the difference between the wearer's center and the fitting reference center from the original log odds. Use the original score when the required baseline is unavailable.
3. Preserve the classifier, cutoff and immediate alert policy. Verify the original full-block replay, then compare only the same final 25 minutes of each block across all arms. Replay the warm-up's decisions to initialize alert state without counting its exposure or starts.

The [normalization implementation](../../ppg_af/af_wrist_personal_score.py) separates past-only score estimation in `adapt()` from scoring and compares the same ECG reference across all three arms.

## Results

| Held-out final 25 minutes of each block | Original | Fixed wearer median | Rolling wearer median |
| --- | ---: | ---: | ---: |
| False starts over 619 ECG-observed minutes | 170 | 214 | 234 |
| Positive fraction of observed non-AF time | 32.0% | 46.4% | 47.3% |
| Positive fraction of observed AF time | 48.0% | 67.3% | 60.0% |

All 25 blocks could estimate a fixed center. A rolling center was available for 3,746 of 3,750 scheduled post-warm-up decisions. The median fixed offset was −1.510 log odds. Subtracting the offset raised most held-out blocks' scores. Relative to the original, the rolling rule added 6.20 false starts per observed hour, with a 95% participant-bootstrap interval from +0.98 to +11.52. The fixed change was +4.26 per hour with an interval from −0.96 to +9.45. The 1,500 observed AF seconds came from one participant and contained no complete AF episode. [Aggregate results](personal-score-results.json)
