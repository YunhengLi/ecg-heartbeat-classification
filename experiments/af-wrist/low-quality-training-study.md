# Lower-quality non-AF training windows on the same participant schedule

The [false-alert audit](failure-audit.md) found that wrist false starts often occurred at lower PPG quality. This comparison asks whether deliberately sampling lower-quality, ECG-confirmed non-AF training windows helps a classifier distinguish signal artifacts from AF.

## Method

Start from the published 107-window stable training supplement. Keep all 41 AF rows byte-for-byte and retain each of the 66 non-AF participant and five-minute bin assignments. Within each assignment, consider verified non-AF candidates with usable pulse-template features. Select the minimum total native PPG quality separately for each participant, subject to disjoint padded windows. The original candidate remains available as a fallback. A bin can keep its original window when that makes a lower-quality choice in the next bin possible.

The algorithm considered 347 same-bin alternatives. Of these, 148 met the gates and had strictly lower quality than the original within their slot. It replaced 44 rows from eight participants, with no changes to the recorded synchronization methods. The median quality among the 66 supplemented non-AF windows fell from 0.465 to 0.269. The 41 supplemental AF windows retained their median quality of 0.323. Comparing quality as an AF rank score in the combined training pool gave an AUC of 0.300 before replacement and 0.390 afterward. Those are descriptive composition statistics. Model outcomes never enter selection.

Use the [original training expansion comparison](training-expansion.md) to evaluate both bundles. Reproduce the same development groups, C=1, nested training-only cutoffs, original scored windows, history and ECG reference. The table compares the original 107-window supplement with the same-sized lower-quality variant. The two bundles have identical patient and rhythm support.

## Results

Each cell is detections out of 17 complete AF episodes followed by false alert starts over the same 46.5 ECG-observed wrist minutes.

| Fold seed | Plain template, original 107 | Plain template, lower-quality 107 | Quality interactions, original 107 | Quality interactions, lower-quality 107 |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12 / 6 | 13 / 6 | 11 / 1 | 6 / 1 |
| 2027 | 9 / 2 | 7 / 2 | 12 / 6 | 10 / 2 |
| 2028 | 12 / 5 | 13 / 5 | 11 / 3 | 9 / 1 |
| 2029 | 13 / 6 | 12 / 7 | 8 / 3 | 11 / 5 |
| 2030 | 2 / 2 | 2 / 3 | 12 / 2 | 4 / 3 |

For the plain template, lower-quality non-AF sampling gained one detection at the same false-start count in two splits. In the quality-interaction model, it failed to give an improvement without a trade-off in any split. The large loss in the weak original seed underscores that changing the composition of training windows within the same people can change alert behaviour. The lower-quality samples are retained as a reproducible research set. The original 107-window mix is the comparison reference.

## Reproduce

Use the original private 107-row bundle, its aggregate receipt and the original private eligibility plan. The staging command reads local source files or cached blocks and reproduces the exact selection. Retrieve the inputs as described in the [stable expansion study](training-expansion.md#reproduce).

```powershell
python -m ppg_af.af_wrist_low_quality_stage `
  --source-inventory data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --development-preroll runs/wrist-history --stride-raw runs/wrist-stride10 `
  --lock $lock --ptt-dir data/ptt-raw --original-bundle $originalBundle `
  --original-receipt $originalReceipt --eligibility-plan $privateEligibilityPlan `
  --bundle runs/private/low-quality-supplement.npz --report runs/low-quality-receipt.json
```

Run the same five-seed [training expansion command](training-expansion.md#reproduce) with the new supplement. The [aggregate results](low-quality-training-results.json) contain verified support and composition, ordinary-supplement reference outcomes, and every lower-quality trial.
