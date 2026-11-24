# Optical entropy as a gate for AF alert confirmation

Pulse-band entropy was higher at false starts than at nearby ECG-verified controls in the half-hour audit. This follow-up tests whether spectral dispersion can replace the original optical-quality gate. It is exploratory because that audit helped motivate the challenger. [Matched false-start audit](half-hour-motion-audit.md)

## Method

1. Freeze the existing plain-template AF scores and decision cutoffs. Keep the same immediate alert and the same adjacent-score confirmation. Change only the condition that requests a second positive score.
2. On the scoreable optical wear rows of the outer training participants, set the entropy threshold to activate near the frequency of the original quality gate at 0.40. Ties at the threshold can activate one or more additional rows. The development entropy threshold for a held-out fold uses no data from that fold. The historical threshold uses only the five original model-fitting participants' wear time. Neither entropy threshold uses reference rhythm labels as an input.
3. Compare the locked development groups, the historical 260 complete episodes and the sealed half-hour wearable replay. Reproduce the original and optical quality rules before comparing the entropy variant. Keep missing confirmations unavailable and report participant-bootstrap differences on the original signal clock.

The [study runner](../../ppg_af/af_wrist_entropy_confirmation.py) uses the original PPG spectral entropy and the shared confirmation functions. The historical training support was 141 scoreable wear decisions with 106 activations for optical quality and 107 for entropy at 0.33518.

## Results

| Evaluation | Immediate | Existing optical-quality gate | Entropy gate |
| --- | --- | --- | --- |
| Primary development grouping | 12/17 and 6 false starts | 12/17 and 5 | 6/17 and 5 |
| Alternate development grouping | 1/17 and 2 false starts | 1/17 and 1 | 1/17 and 2 |
| Historical complete AF and routine wear | 155/260 and 45 false starts | 151/260 and 39 | 144/260 and 33 |
| Half-hour false starts | 209 | 169 | 147 |
| Half-hour non-AF time positive | 31.7% | 28.8% | 26.9% |

On the half-hour blocks, entropy reduced the false-start rate by 1.77 per ECG-observed hour relative to the quality gate, with a descriptive 95% participant-bootstrap interval of 0.72 to 3.14 fewer starts per hour. It also reduced positive time during ECG-confirmed AF from 40.0% to 30.6%. Those AF seconds came from one participant and the blocks had no full AF episodes.

Gate frequencies diverged outside the fitting rows. In the historical episode and wear replay, the quality rule requested confirmation on 3,422 scoreable history or evaluation rows and the entropy rule on 6,984. At the same number of primary false starts, the entropy rule lost six of the 17 complete episode detections relative to the quality rule. The existing adjacent optical-quality gate stays the comparison candidate. [Aggregate results](entropy-confirmation-results.json)
