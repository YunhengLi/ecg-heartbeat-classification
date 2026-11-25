# Add one separately staged AF onset pair

The [earlier onset experiment](template-transition-training.md) used 26 examples from three development participants and obtained most of its extra rows from one person. This follow-up tests the incremental effect of an ECG-verified pair from a fourth participant. The two new windows come from native recording time outside the original training, scoring and predecessor inputs and the [109-window stable expansion](training-expansion.md).

## Source feasibility

The assessment covered only the original ten development participants in the pinned [AFPPGECG v6 release](https://zenodo.org/records/23921320). Native rhythm annotations and synchronized segment timestamps gave this sequence of counts.

| Eligibility step | AF episodes |
| --- | ---: |
| Native AF annotation | 71 |
| Continuous ECG reference across the complete episode | 57 |
| Eligible ECG labels in both 30-second inputs | 32 |
| One continuous wrist segment covering both inputs | 23 |
| Both padded inputs disjoint from staged or reserved time | 8 |
| Disjoint pairs from a participant with no earlier onset examples | 1 |

The eight candidate episodes came from three participants. One new donor supplied one feasible episode. Three verified 1 MiB range requests recovered the necessary local cache blocks, including two ECG blocks and one PPG block. The acquisition checked the pinned source inventory, Zenodo origin, HTTP status 206, `Content-Range`, and each response length. The manifest records whole-file checksums. These partial reads support verification of the target samples, while whole-file checksum verification requires the complete files.

## Controlled comparison

- The pre-onset input ends up to 30 seconds before AF onset and contains 30 seconds of ECG-verified non-AF. The early input ends in the first 20 seconds of AF with the final ten seconds confirmed as AF. Both use the 10-second decision grid. The two inputs share ten seconds of native waveform. Their overlapping ECG and PPG samples match exactly on their respective native clocks.
- Compare original stable training, original plus 26 prior onset examples, and original plus those 26 and the new pair. Fit each arm with the original seven template features and with the fixed PPG quality-interaction features, using standardization, class balancing, and C=1. The native inputs use the pinned PTT-calibrated template and only the samples available at each decision.
- Keep the original five participant folds, inner-training threshold selection, score availability, episode denominator and alert replay. The new pair enters four outer training fits and is withheld when its donor is a test participant. Nested fitting applies the same participant rule. The private 109-window stable bundle verifies reserved time. It contributes zero model fitting or calibration rows.

## Results

| Model | Stable training | Plus prior 26 | Plus prior 26 and new pair |
| --- | ---: | ---: | ---: |
| Plain template | 12/17, 6 false starts | 11/17, 6 | 11/17, 6 |
| Template with quality interactions | 14/17, 4 false starts | 12/17, 4 | 12/17, 4 |

All runs used the same 46.5 minutes of ECG-observed wrist exposure. Adding the new pair preserved exactly which episodes each family detected and its exact wrist false-alert timestamps. The paired median delay change was zero in each family. One detected episode alerted earlier in the plain model. In the quality-interaction model one alerted earlier and one later. The pre-onset diagnostic retained four starts in the plain model and three in the quality-interaction model. In the latter one pre-onset timestamp changed.

This test adds one participant and one onset. The largest contributor still supplies 20 of the 28 onset windows, and the newest two windows share ten seconds of signal. The comparison gives evidence about this single staged pair under the original grouped development protocol. No independent predictions were generated.

## Reproduce

Use the original development archives from the [decision cadence study](clock-study.md#reproduce), the original expanded stable bundle in append order, and the private two-row transition bundle and staging receipt. The private pair contains the two padded PPG vectors, the corresponding ECG voltage vectors, their decision times and the native annotation episode bounds. The aggregate receipt pins the paired bundle, source inventory, protocol and protected 109-window bundle by SHA-256.

Provide the pinned source manifest, two complete local MAT files or the corresponding local one-MiB block cache, and the original PTT files. The [study driver](../../ppg_af/af_wrist_transition_native_study.py) reopens the native data by the pinned source inventory, verifies both labels, reproduces the staged waveform samples, checks disjointness and refits both model families. It reads local inputs only.

```powershell
python -m ppg_af.af_wrist_transition_native_study `
  --source-manifest data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --development-preroll runs/wrist-history --stride-raw runs/wrist-stride10 `
  --stride-features runs/wrist-stride10-features --lock $lock `
  --pair runs/transition-pair-private.npz --receipt runs/transition-pair-receipt.json `
  --protected-supplement runs/development-expansion-109-private.npz --ptt-dir data/ptt-raw `
  --output runs/transition-native-results.json
```

The [aggregate results](transition-native-results.json) include paired outcomes, fold-calibration status and verification hashes. The numerical families match the original exploratory runs.
