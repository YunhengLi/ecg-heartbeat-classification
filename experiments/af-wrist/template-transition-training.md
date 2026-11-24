# Template ECG training examples around AF onset

The [independent error audit](template-independent-error-audit.md) found 66 missed episodes with an alert active before onset and a positive in-episode decision. Nineteen of those alerts had a traceable start in ECG-confirmed non-AF. This development experiment tests whether transition examples help the template classifier separate the two rhythms near onset.

## Training sample selection

Use the 10-second decision clock and the preceding 30 seconds of PPG. Require continuous ECG reference over that entire input window and a shared model feature vector. Use only complete ECG-confirmed episodes in the development grid.

- **Pre-onset non-AF.** Choose the latest eligible decision time in the 30 seconds up to and including onset whose trailing input is entirely non-AF.
- **Early AF.** Choose the earliest eligible decision time in the first 20 seconds of the episode whose final 10 seconds lie entirely inside AF. The earlier part of the 30-second input can contain the preceding rhythm.

Use each decision once and at most one of each class per episode. Give paired examples priority when their candidate windows overlap. Keep only new endpoints relative to the original fitting rows. The ECG reference labels and selects training examples. The classifier receives the PPG-derived features that would be available at that decision time.

The selection yielded 26 supplemental examples from three participants. Ten episodes supplied both classes, six supplied early AF only, and one supplied neither.

| Supplemental class | New windows | Contributing participants | Windows from largest contributor |
| --- | ---: | ---: | ---: |
| Pre-onset non-AF | 10 | 3 | 7 |
| Early AF | 16 | 3 | 13 |
| Combined | 26 | 3 | 20 |

All 26 windows came from previously held-out-for-fitting enriched episode rows in the development archives. The concentration permits an exploratory test of this selection and makes broader participant coverage important for learning the transition reliably.

## Controlled comparison

Keep the original seven pulse-template features, logistic regression C=1, class balance, immediate alert rule and five outer participant folds. Derive the inner and outer fold assignments with the original stable rhythm labels. Add the new labels only when their participant belongs to the relevant model fit. Preserve score availability, episode denominators and monitoring intervals. The original validation objective selects the cutoff inside each training fold.

Evaluate definite new starts during complete AF, first-alert delay for shared detections, and false starts in the same 46.5 minutes of prescheduled wrist monitoring. A separate onset diagnostic counts definite non-AF starts in the 30 seconds before completely observed episodes with continuously observed non-AF ECG in that interval. The diagnostic and the primary sensitivity retain their original scoring boundaries.

## Results

| Development model | Complete episodes detected | Wrist false starts | Pre-onset non-AF starts |
| --- | ---: | ---: | ---: |
| Original fitting rows | 12/17 (70.6%) | 6 | 3 |
| Original rows plus transition examples | 11/17 (64.7%) | 6 | 4 |

Ten complete episodes had the full 30-second non-AF reference for the pre-onset diagnostic. Three were excluded for incomplete preceding ECG coverage and four for preceding AF. The candidate gained two episode detections and lost three. For the nine episodes detected by both models, the paired median change in first-alert delay was zero seconds. The two separate medians over all detected episodes were 20.5 and 31.2 seconds.

On these development folds the transition examples produced one fewer detection, the same number of wrist false starts, and one more pre-onset start. The original classifier is the development reference. The extra training data came from three participants, with twenty of twenty-six windows from one, so this comparison gives limited evidence about the value of transition examples drawn from a broader cohort.

## Reproduce

Prepare the locked development features, 10-second grid and clock report from the [decision cadence study](clock-study.md#reproduce).

```powershell
python -m ppg_af.af_wrist_template_transition --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock --clock-results runs/clock-study-results.json --output runs/template-transition-results.json
```

[`select_transition_examples`](../../ppg_af/af_wrist_transition_examples.py) assigns the supplemental labels. [`train_oof`](../../ppg_af/af_wrist_cv.py) separates original split labels from supplemental fitting labels. [`compare_pre_onset_alert_runs`](../../ppg_af/af_wrist_transition_metrics.py) traces paired pre-onset starts. The [aggregate results](template-transition-results.json) include training support and paired episode changes.
