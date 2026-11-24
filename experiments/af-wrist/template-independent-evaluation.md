# Independent evaluation of frozen template ECG

The [fixed-split preparation](template-holdout-preparation.md) selected the same logistic model and alert threshold for the original and tuned variants. This evaluation applies that single frozen model to the prespecified independent AFPPGECG v6 roster. The development model was fitted with five participants and its threshold was selected with five different participants. The lock was written before any independent classifier scores were opened.

## Recover and verify independent recordings

The source schedule draws a five-minute wrist-monitoring block from recording times and separately identifies candidate episode blocks from reference annotations. Each eligible episode block is considered in the fixed schedule. It is eligible for sensitivity when continuous native ECG reference contains the complete episode. For false alerts, only the prescheduled wrist-monitoring blocks contribute exposure. Record missing source ranges separately from verified gaps or flat PPG. [AFPPGECG v6 release](https://zenodo.org/records/23921320)

Use validated source byte ranges and preserve the original ECG and PPG clock alignment. Check native PPG for the preceding 90 seconds to initialize each block's alert state. Check ECG-reference exposure and complete-episode membership before applying the classifier. An ECG-confirmed complete episode stays in the sensitivity denominator when no PPG decision is available. The reference and waveform audit counts both scheduled targets and distinct complete episodes, including episodes incidental to a recovered block.

All 25 prescheduled wrist blocks and 225 episode blocks decoded. Of the 225 targeted episodes, 224 had continuous native ECG reference. Across the wrist and enriched blocks, the reconciled 10-second grid contains 260 distinct complete ECG-confirmed AF episodes and 82 partial episodes. At least one 10-second decision had a usable native PPG input in 253 complete episodes. Prescheduled wrist monitoring had 125 minutes of ECG reference, including 114.53 minutes of non-AF, with raw PPG input opportunity throughout.

| Model-free coverage | Count |
| --- | ---: |
| Prespecified independent participants recovered | 25/25 |
| All evaluated ECG-reference time | 908.5 min |
| Available raw PPG decision inputs | 5,425 |
| Known unavailable raw PPG decision inputs | 38 |
| Available prehistory decision inputs | 1,723 |
| Known unavailable prehistory decision inputs | 27 |

## Apply the frozen model

Each 10-second decision uses only the previous 30 seconds of PPG. Reconstruct the pulse-template ECG and extract its original seven beat-interval features. Preserve the shared feature-availability rule used in development. The direct PPG and RDDM features are required for that gate. Neither alternate route contributes a classifier prediction. Apply the single locked standardized logistic model at C=1 and the original cutoff.

An alert starts when the decision turns positive. Previously active alerts carry through the evaluation boundary, and unavailable input interrupts the alert. The primary sensitivity counts a definite new alert during each complete ECG-reference AF episode. Report first-alert delay for detected episodes. False alert starts are counted against the same ECG clock in wrist monitoring and divided by observed wrist hours. Resample whole participants for descriptive 95% intervals so that multiple episodes from one participant stay together. Also report the mean of participant-level episode sensitivities to show how much pooled sensitivity depends on participants with more AF episodes.

## Results

| Frozen template ECG on the independent cohort | Result |
| --- | ---: |
| Participants evaluated | 25/25 |
| Complete AF episodes detected | 155/260 |
| Pooled complete-episode sensitivity | 59.6% |
| Mean episode sensitivity across the 16 participants with complete episodes | 55.7% |
| Wrist false starts | 45 |
| False starts per ECG-observed wrist hour | 21.6 |
| Median first-alert delay among detected episodes | 22.3 s |

The descriptive 95% participant-bootstrap intervals were 36.1%–66.7% for pooled sensitivity and 13.0–30.7 false starts per observed wrist hour. One participant contributed 157 of the 260 complete episodes, which affects the precision of the pooled estimate. The 25 separate wrist samples contribute 125 observed minutes to the false-alert calculation.

All 25 participants had a shared-feature decision somewhere in the evaluated recordings, and all 7,500 seconds of ECG-observed wrist time were analyzable. A shared-feature decision was available inside 253 of the 260 complete AF episodes. There were seven missed episodes without such an input and 98 missed episodes with at least one such input. The 80% episode-sensitivity target was unmet on the designated development calibration and on the independent recordings. The final evaluation used the original frozen model and cutoff.

[Model-free reference audit](template-independent-reference-results.json) · [Independent scoring results](template-independent-results.json)

## Reproduce

Prepare the prescheduled independent raw archives and their native PPG predecessor rows using the frozen roster from [test preparation](template-holdout-preparation.md). For local MAT files, run [the cohort survey](../../ppg_af/af_wrist_cohort.py) with one 150-second calendar block, a 300-second wrist block, and a 480-second episode-block limit. Use its timestamp candidate count as the episode count and set the fit count per class to zero. Retain the prescheduled wrist and enriched episode blocks for this evaluation. The calendar draw is part of the fixed schedule and its time is excluded from this evaluation.

The raw and feature archives must contain the same recorded participants. Store the two private protocol locks outside the repository and install the [feature dependencies](README.md#code-and-commands). For local native files, prepare the history with `af_wrist_preroll --independent` before these commands.

```powershell
$privateLock = 'runs/template-independent-model-lock.json'
$sourceProtocol = 'runs/wrist-window-seeded-lock.json'
$raw = 'data/wrist-independent-raw'
$history = 'data/wrist-independent-preroll'

python -m ppg_af.af_wrist_template_coverage --independent-raw $raw --preroll $history --protocol-lock $privateLock --output runs/template-independent-reference-audit.json
python -m ppg_af.af_wrist_features --cohort $raw --output runs/independent-parent-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda
python -m ppg_af.af_wrist_features --cohort $history --output runs/independent-history-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda
python -m ppg_af.af_wrist_stride --independent --cohort $raw --preroll $history --lock $privateLock --output runs/independent-stride10 --feature-output runs/independent-stride10-features --cohort-features runs/independent-parent-features --preroll-features runs/independent-history-features --ptt-dir data/ptt-raw --upstream rddm-upstream --weights rddm-weights --device cuda
python -m ppg_af.af_wrist_template_independent --lock $privateLock --protocol-lock $sourceProtocol --independent-features runs/independent-parent-features --stride-raw runs/independent-stride10 --stride-features runs/independent-stride10-features --coverage-report runs/template-independent-reference-audit.json --output runs/template-independent-results.json
```

The [coverage reporter](../../ppg_af/af_wrist_template_coverage.py) counts ECG and native PPG opportunity. The [independent evaluator](../../ppg_af/af_wrist_template_independent.py) verifies those denominators and the model lock, applies the classifier once and produces the aggregate metrics.
