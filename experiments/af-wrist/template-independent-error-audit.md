# Independent template ECG errors and participant weighting

The [independent evaluation](template-independent-evaluation.md) recorded 155 detections among 260 complete AF episodes and 45 false starts in 125 ECG-observed wrist minutes. This post-hoc replay uses the same sealed waveforms, frozen model, cutoff, alert clock and ECG labels. It reconciles the published endpoint before attributing errors and uses the independent data only to describe failure patterns.

## Missed episodes

| Observation at the frozen cutoff | Missed episodes |
| --- | ---: |
| Alert active before onset and positive at an in-episode decision, with no new start | 66 |
| All in-episode scores below the cutoff | 32 |
| No score within the episode | 7 |
| **Total** | **105** |

Of the 66 carried alerts, 19 had a traceable start during ECG-confirmed non-AF, one started during a previous AF episode, and one started during ECG-unobserved time. The other 45 were already active when their replayed segment began. The primary endpoint credits a definite new start during the episode. Three additional scored misses had an alert active at onset and fell below cutoff at every in-episode decision, so they belong to the second row. Five of the seven unscored episodes contained no decision endpoint. In the other two, every in-episode PPG input was flat.

Only three of the 32 below-cutoff misses peaked within 0.05 of the selected cutoff, with none within 0.01. The median best in-episode score was 0.304 below cutoff. Eighteen of these 32 misses had at least one scoreable 30-second input wholly inside AF.

The largest AF contributor accounted for 107 detections in 157 episodes (68.2%). All other contributors accounted for 48 detections in 103 episodes (46.6%). Both groups contribute to the official 59.6% estimate. The two subgroup figures show the weight of the largest contributor in the pooled result.

## False starts

Ten of the 45 false starts had one positive decision followed by an observed negative reset. Thirty-three had multiple positive decisions and two were single positives at the boundary of the recorded block. Twenty-two false alerts stayed active for at least 30 seconds with continuous ECG-confirmed non-AF throughout those 30 seconds. Forty-four false starts had a fully observed non-AF trailing input. One lacked complete trailing reference. The median score at a false start was 0.330 above the cutoff.

False starts affected 15 of the 25 participants. The three largest false-start contributors produced 16 of the 45 starts. The participant with 157 complete AF episodes contributed two false starts. A new alert followed a negative reset within 30 seconds in 24 cases. This can overlap with the persistence and single-positive counts.

## Single development change

The error patterns and the difference in event sensitivity across participants motivated checking whether the classifier depends on a small number of training contributors. Two of six development participants who provide stable AF examples supply 22 of the 34 AF training rows. The test holds the rows, seven template features, unweighted standardization, C=1, 10-second decision cadence, immediate alert rule and participant folds fixed.

Class balancing already gives each rhythm half the total logistic loss weight. The candidate additionally gives the participants within a rhythm equal total weight. For each inner or outer fit, distribute each rhythm's half-weight equally among the participants with rows of that rhythm, then divide each person's share equally among their rows. Cutoffs are selected on the inner held-out development predictions using the original objective. The outer population remains the ten development participants.

| Development model | Complete episodes detected | False starts | Misses with an active alert observed during AF |
| --- | ---: | ---: | ---: |
| Original class balancing | 12/17 (70.6%) | 6 | 0 |
| Equal participant weight within each rhythm | 11/17 (64.7%) | 6 | 4 |

The two sensitivities are within the 10-percentage-point tolerance. Participant weighting gained three detections and lost four, with a zero-second median delay change across the eight episodes detected by both fits. The gain in some individual episodes did not translate to a pooled benefit or lower false burden. The original classifier remains the reference configuration for the reported independent result.

The independent audit informs this exploratory choice. All 35 paired AFPPGECG v6 participants were used in the original development or independent rosters. A fresh independent evaluation of a distinct candidate requires another paired cohort.

## Reproduce

Prepare the private lock and sealed independent inputs from [the independent evaluation](template-independent-evaluation.md#reproduce). Supply the model-free coverage receipt used for the published scoring run and its results JSON.

```powershell
python -m ppg_af.af_wrist_independent_audit --lock $privateLock --protocol-lock $sourceProtocol --independent-features runs/independent-parent-features --stride-raw runs/independent-stride10 --stride-features runs/independent-stride10-features --coverage-report experiments/af-wrist/template-independent-reference-results.json --published-results experiments/af-wrist/template-independent-results.json --output runs/template-independent-error-results.json
python -m ppg_af.af_wrist_template_weighting --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $sourceProtocol --clock-results experiments/af-wrist/clock-study-results.json --output runs/template-participant-weighting-results.json
```

[`replay_published`](../../ppg_af/af_wrist_independent_audit.py) verifies the frozen inputs and reconciles event counts. [`evaluate_weighting`](../../ppg_af/af_wrist_template_weighting.py) tests group weighting on development folds. Aggregate outputs are in the [independent error audit](template-independent-error-results.json) and the [development weighting comparison](template-participant-weighting-results.json).
