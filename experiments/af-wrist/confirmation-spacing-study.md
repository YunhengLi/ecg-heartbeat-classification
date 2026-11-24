# Nonoverlapping confirmation and AF onset coverage

The existing low-quality rule requires a positive score from the preceding ten-second decision. Two 30-second inputs for these scores share 20 seconds. The challenger requires a positive 30 seconds earlier, giving the model two disjoint inputs at a low-quality decision.

## Method

1. Keep the original 30-second plain-template probabilities, calibration cutoffs and ten-second decision grid. Apply immediate alerting, low-quality confirmation ten seconds earlier and low-quality confirmation 30 seconds earlier. Use the same current quality boundary of 0.40. The rule checks the two specified scores. A missing required score makes the current decision unavailable.
2. Compare the two rules on both frozen development partitions, the prior 260 complete episodes and the sealed 744 ECG-observed minutes. Reproduce the full published adjacent replay before evaluating the longer separation. Use only context from the same participant and block.
3. Map candidate false starts to each uninterrupted span of original positive decisions. Count spans with at least two verified candidate false starts to measure fragmentation. Compare positive time separately from the count of new alert starts.
4. Independently of model scores, count complete AF episodes with any available ECG-valid decision endpoint, with both required endpoints inside the same episode, and with every required optical input fully within AF. The first count gives an optimistic clock and availability ceiling. It does not require that the scores exceed the threshold or the alert was previously inactive.

The [replay implementation](../../ppg_af/af_wrist_confirmation_spacing.py) calls the same confirmation and alert scoring functions with the predecessor lag changed. The [episode opportunity implementation](../../ppg_af/af_wrist_confirmation_opportunity.py) reuses the sealed ECG intervals and checks the existing coverage totals.

## Results

| Evaluation | Original | 10-second confirmation | 30-second confirmation |
| --- | --- | --- | --- |
| Primary held-out grouping | 12/17 and 6 false starts | 12/17 and 5 | 10/17 and 5 |
| Alternate held-out grouping | 1/17 and 2 false starts | 1/17 and 1 | 0/17 and 2 |
| Historical episodes and monitoring | 155/260 and 45 false starts | 151/260 and 39 | 148/260 and 38 |
| Half-hour blocks | 209 false starts | 169 | 210 |
| Half-hour blocks scoreable decisions | 4,474/4,500 | 4,469/4,500 | 4,463/4,500 |
| Half-hour non-AF time positive | 31.7% | 28.8% | 26.7% |

The half-hour blocks had no complete AF episodes. Relative to the original, the paired false-start difference was −3.23 per observed hour for the adjacent rule, with a 95% participant-bootstrap interval of −4.89 to −1.77. The 30-second rule gave +0.08 per hour, with an interval of −1.45 to +1.86. The 30-second rule generated 41 second false starts within 41 originally continuous positive spans. The adjacent rule generated none. The longer rule thus occupies less non-AF time while starting more false alerts. [Paired alert results](confirmation-spacing-results.json)

| Historical opportunity on 260 complete episodes | Original | 10-second confirmation | 30-second confirmation |
| --- | ---: | ---: | ---: |
| At least one scoreable endpoint | 253 | 253 | 253 |
| Required score endpoints all inside the same AF episode | 253 | 244 | 233 |
| Required optical windows wholly inside AF | 206 | 184 | 175 |

Reaching 80% on the historical scope requires 208 detections. Having at least one scoreable endpoint permits that target in principle. Counting only inputs fully inside AF cannot reach the target with these 30-second inputs. Among 12 episodes of up to ten seconds, seven could be scored, and only one had a viable endpoint under either confirmation rule with all required endpoints inside the episode. Detecting more short episodes requires models that tolerate a mix of rhythms within their input or accept shorter inputs. These are timing and availability counts, so they give no basis for claiming that 80% sensitivity has been reached. [Opportunity results](confirmation-opportunity-results.json)
