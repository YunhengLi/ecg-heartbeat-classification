# Template ECG scores around AF onset

The [failure audit](failure-audit.md) found four scorable misses for template ECG with immediate alerts at 10-second cadence. This study aligns those scores to reference ECG onset, tests one recent beat-timing feature, and follows with a small regularization search when the feature fails the development target.

## Onset analysis

Place each held-out model score on the ECG clock and subtract its own fold's cutoff. Compare scored decisions before onset with decisions during AF. For each input, measure how many seconds of its trailing 30 seconds overlap the episode. Claims about entirely non-AF inputs require continuous ECG reference for all 30 seconds. The onset-aligned output groups scores in 10-second bins and includes the number of contributing episodes in each bin.

- Three of the four scored misses had a scoreable 30-second input entirely within AF. Two had their best in-episode score in a wholly AF input. The short episode without a wholly AF input lasted 17.6 seconds.
- Two of the four misses had fully ECG-observed non-AF inputs before onset. In both, the highest pre-onset score exceeded the highest in-episode score. An alert was already active at onset in both cases, followed by entirely subthreshold in-episode decisions.
- Nine of the 12 detected episodes triggered before 30 seconds of AF had elapsed. All six false starts in prescheduled wrist monitoring had confirmed non-AF throughout their trailing inputs. The fifth miss had flat PPG and no score.

The four cases point to different timing patterns. A 30-second window can mix rhythms near onset. The persistent misses with wholly AF input and the higher pre-onset scores also motivate examining how the classifier ranks beat-interval features.

For the two misses with fully observed non-AF inputs before onset, the fold's standardized logistic model reproduced the original scores. The largest negative contribution from the pre-onset peak to the in-episode peak came from median beat interval for one episode and pNN50 for the other. pNN50 measures the fraction of successive beat intervals that differ by more than 50 milliseconds. No feature contribution fell in both comparisons. Two episodes offer too little evidence to pick a single feature as the cause.

## Controlled feature test and parameter search

The candidate extends the original seven template features with one contrast. It measures normalized root mean square of successive beat-interval differences over the last 10 seconds, then subtracts the corresponding 30-second value. When the recent segment has too few supported intervals, the contrast is zero. The new feature used supported recent intervals in 112 of 114 training rows, 507 of 513 scored-grid rows, and 171 of 175 history rows. The seven old columns and shared scoreability were verified on all 1,204 input rows.

Fit and calibrate the candidate on the original five outer participant folds with up to three inner folds. Each variant retains the 30-second feature window, 10-second decision cadence, immediate alert rule and 17 complete ECG-reference episodes. Each fold chooses the cutoff using scores from inner held-out participants only. The search favors the fewest false starts among cutoffs that reach 80% episode sensitivity and otherwise favors the highest observed sensitivity.

The recent-feature experiment used the original logistic regression parameter C=1. Its development goal was at least 80% episode sensitivity without increasing false starts. Because the feature fell short, the second experiment tuned only C on the original feature set. The grid contained 0.1, 1 and 10. Smaller C means stronger coefficient shrinkage. Each fold selected a C and cutoff from its own inner calibration results using the same sensitivity and false-start objective and preferred C=1 on an exact tie. The outer fold was always scored with its chosen setting.

## Held-out development results

| Immediate template ECG | Complete episodes detected | False starts | Median first-alert delay |
| --- | ---: | ---: | ---: |
| Seven original features, C=1 | 12/17 (70.6%) | 6 | 20.5 s |
| Added recent beat-interval feature, C=1 | 9/17 (52.9%) | 6 | 23.5 s |
| Seven original features, C selected inside each fold | 13/17 (76.5%) | 6 | 24.7 s |

False alert counts use the same 46.5 minutes of ECG-observed wrist monitoring, including 34 minutes of non-AF. The recent feature lost three of the baseline detections and gained zero. The tuned model gained three episodes and lost two relative to baseline. Across the 10 episodes both models detected, the paired median delay change was zero seconds. Three of six false starts occurred at matching timestamps. C=10 was chosen in three folds, C=1 in two, and C=0.1 in zero. None of the five selected inner calibrations reached 80%.

The modest development gain comes from changing the regularization on the original feature set. At least 14 of 17 detections are needed for 80%. These experiments reuse the development cohort to choose what to try, so the result is exploratory. The 25 independent participants remain reserved for a later frozen evaluation.

## Reproduce

Use the locked development cohort and 10-second grid from the [decision cadence study](clock-study.md#reproduce). The onset audit and classifier comparison independently reproduce the original immediate template baseline. The CLI outputs contain only aggregate results.

```powershell
python -m ppg_af.af_wrist_template_onset --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results runs/clock-study-results.json --output runs/template-onset-results.json
python -m ppg_af.af_wrist_template_study --development-raw $development --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --ptt-dir data/ptt-raw --lock $lock --clock-results runs/clock-study-results.json --output runs/template-study-results.json
```

[`analyze_template_onset`](../../ppg_af/af_wrist_template_onset.py) aligns scores and reference ECG around onset. [`append_recent_template_feature`](../../ppg_af/af_wrist_template_recent.py) adds the recent beat-timing contrast. [`select_regularization`](../../ppg_af/af_wrist_template_study.py) applies each fold's inner model selection to its held-out scores. The [onset results](template-onset-results.json) include the plotted trajectories and the [classifier results](template-study-results.json) include paired detection and false-start counts.
