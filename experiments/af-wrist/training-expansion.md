# Additional ECG-verified stable training windows

The [split-sensitivity study](quality-stability.md) found that a fixed quality-interaction model could improve on the original grouping while performance varied widely across groupings. This experiment increases training support without changing the feature definitions, the threshold policy, or which wrist decisions are scored.

## Source and selection

The initial scan used already-cached portions of the pinned AFPPGECG v6 recordings for the ten development participants. It found 353 unused stable AF windows from six participants and 849 stable non-AF windows from nine. A candidate needs a continuous wrist PPG segment, nonflat finite PPG, measured finite ECG, and a single reference rhythm over its 30-second core and both one-second flanks. Additional windows and their flanks must be disjoint from the original fitting, monitoring, episode and predecessor windows.

For each participant and rhythm, split available five-minute source bins into up to eight chronological strata and draw at most one candidate per stratum and source bin. Use fixed seed 20261124, independent of model results. This selected 107 windows, with 41 AF and 66 non-AF. The original fit pool has 34 AF and 80 non-AF windows. All 107 selected windows generated usable pulse-template features. In the augmented AF pool, the two largest contributors supplied 37 of 75 windows, down from 22 of 34 in the original pool.

A separate, more targeted acquisition verified seven candidate AF pairs from one existing development participant who had no previous stable AF fitting rows. Two windows from different five-minute bins were selected by the same rule and added to the original 107-window bundle. This variant increases the number of AF training contributors from six to seven. Both bundle receipts contain aggregate counts and separate fingerprints.

The additional verification fetched six 1 MiB ranges from the official release. The source host, exact HTTP partial range and byte count were verified for every block. This verifies the transferred ranges and their decode; full source-file MD5 requires the complete recordings. The original and updated private eligibility snapshots record which cached native windows were available at each stage.

## Fold-safe evaluation

Fit the original template ECG logistic model and the quality-interaction model at fixed C=1 with and without the additional examples. Each extra row can enter a fit only when its participant is assigned to that training split. Original stable rows determine the grouped partitions and split support. Original prediction rows determine calibration and event scoring. The code checks that scoreable rows, outer partitions, inner fold counts and ECG denominators match.

The additional rows use pulse-template features and raw PPG quality computed from their own trailing inputs. The original evaluation retains the shared direct PPG, pulse template and RDDM availability gate. Selecting supplemental template fitting rows does not require RDDM inference. Five seeds repeat the comparison on different groupings of the same ten participants. No independent wrist predictions are generated.

## Results

Each cell is definite detections out of 17 complete AF episodes followed by wrist false starts over 46.5 ECG-observed minutes.

| Seed | Original template | Template plus 107 | Original quality interactions | Quality interactions plus 107 |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12 / 6 | 12 / 6 | 14 / 4 | 11 / 1 |
| 2027 | 3 / 4 | 9 / 2 | 12 / 2 | 12 / 6 |
| 2028 | 8 / 5 | 12 / 5 | 13 / 4 | 11 / 3 |
| 2029 | 11 / 6 | 13 / 6 | 5 / 1 | 8 / 3 |
| 2030 | 1 / 2 | 2 / 2 | 1 / 0 | 12 / 2 |

The template gained detections in four groupings at equal or lower false-start burden. The expanded quality-interaction model ranged from 8 to 12 detections across seeds, compared with 1 to 14 originally. Its original-seed sensitivity decreased by three episodes as false starts decreased by three. On the weakest seed, it gained eleven detections and two false starts. None of the expanded configurations reached the 80% target. In a follow-up on seeds 2026 and 2030, AF-only and non-AF-only augmentations each gained at most one episode on the weakest split. The larger gain came from adding both rhythms and recalibrating within the folds.

The 109-window variant included a seventh AF contributor and brought the total AF fit pool to 77 windows. Across all five seeds, both classifier families had the same reported held-out detections, false starts, paired delay and pre-onset results as with 107. Cutoff calibration status counts were also the same. The two new windows preserved all earlier training rows and demonstrated a repeatable route for growing the class donor pool.

The same four participants supplied the 17 complete events in every analysis, with 13 from the largest contributor. Improvements within these groupings require validation on a different cohort with broader episode support.

## Reproduce

The [stage command](../../ppg_af/af_wrist_expansion_stage.py) reads local full MAT files or existing sparse byte ranges and verifies the paired records and labels. An optional private eligibility snapshot reproduces the same selection even if the cache has grown. To reproduce the append-only variant, use the updated snapshot and add `--retain-prefix` with the path to the verified 107-window bundle. Use separate report and private bundle paths.

```powershell
python -m ppg_af.af_wrist_expansion_stage `
  --source-inventory data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --development-preroll runs/wrist-history --lock $lock `
  --eligibility-plan $privateEligibilityPlan `
  --bundle runs/private/training-expansion.npz --report runs/training-expansion-receipt.json

python -m ppg_af.af_wrist_training_expansion `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results runs/clock-study-results.json `
  --supplement runs/private/training-expansion.npz --ptt-dir data/ptt-raw `
  --seed 2026 2027 2028 2029 2030 --output runs/training-expansion-results.json
```

The [stage and training receipt](training-expansion-results.json) records source and bundle fingerprints, training support, paired comparisons and the append-only checks. Both evaluated bundle contents and their ordering reproduced byte-for-byte with the stage command.
