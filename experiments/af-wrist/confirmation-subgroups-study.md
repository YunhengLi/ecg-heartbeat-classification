# AF detection by episode duration and contributing participant

The same 260 historical episodes span a wide range of durations and contributions per person. This audit describes where the immediate template and its adjacent-confirmation variants detected or missed AF.

## Method

1. Reproduce the exact historical replay for the unmodified template, adjacent optical-quality confirmation and adjacent entropy confirmation. Keep the frozen models, policies, archived PPG inputs, ECG labels and cadence.
2. Link the first definite ECG-valid new start to each complete episode. Group complete episodes by the duration on the ECG clock. Within each range, show the number of participants and their equally weighted sensitivities as well as episode-weighted totals.
3. Calculate candidate-minus-original uncertainty from paired participant resamples within each duration range. Also report the count from the person with the most episodes and the combined count from the remaining contributors. These are descriptive subgroups of the previously scored dataset.

The [subgroup audit](../../ppg_af/af_wrist_confirmation_subgroups.py) uses the same start-state scorer and checks that attributed detections reconcile with the locked totals before summarizing any duration range.

## Results

| Complete AF duration | Episodes | Participants | Original | Adjacent optical quality | Adjacent entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up to 10 seconds | 12 | 5 | 0 | 0 | 0 |
| Over 10 to 30 seconds | 21 | 5 | 3 | 3 | 3 |
| Over 30 to 60 seconds | 83 | 13 | 42 | 40 | 36 |
| Over 60 seconds | 144 | 11 | 110 | 108 | 105 |
| All complete episodes | 260 | 16 | 155 | 151 | 144 |

The [earlier opportunity audit](confirmation-spacing-study.md#results) found that seven of the 12 shortest episodes had at least one available decision for each rule. All three variants missed those episodes. The most frequent contributor supplied 157 of the 260 events. Detections from the remaining 103 events were 48, 49 and 47. Overall participant-average sensitivity was 55.7%, 55.9% and 54.5%, respectively. The participant-bootstrap intervals for both confirmation rules' overall sensitivity changes included zero. Among shared detections of episodes over one minute, both confirmation rules delayed the first alert by a median of ten seconds. [Aggregate results](confirmation-subgroups-results.json)
