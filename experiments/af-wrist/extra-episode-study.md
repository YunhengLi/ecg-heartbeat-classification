# Replay one additional short native AF episode

This diagnostic scores one ECG-confirmed AF episode lasting 5.756 seconds from a development participant who supplied zero of the 17 original complete episodes. It tests the original template and fixed quality-interaction model on the newly verified case using that participant's original held-out fit and cutoff.

## Method

The staging audit obtained seven 1 MiB byte ranges from the pinned [AFPPGECG v6 release](https://zenodo.org/records/23921320) and checked the HTTP partial response and length of each. The offline replay opens the existing local cache or checksum-verified complete files. It checks continuous ECG reference throughout the 30-second scoring block, including the native AF boundaries, and reads the aligned wrist PPG with 90 seconds of native prehistory. The entire padded interval is disjoint from the original staged data, the 109 stable expansion windows and the protected two-window transition pair. These checks bind the private plan, source inventory, protocol and protected bundles by digest.

The driver rebuilds the original participant folds and verifies both published development results. For each model it refits the selected participant's outer training roster, checks that the original held-out scores reproduce, and applies the original threshold to the extra block. The new data supply no fitting or calibration rows. Each model receives the original 30-second PPG inputs on the 10-second decision clock, including seven predecessor decisions. The diagnostic keeps the original 17-episode evaluation and adds zero wrist false-alert exposure.

## Results

Both models had usable scores on all three new evaluation decisions and all seven predecessor decisions. One evaluation decision fell inside AF.

| Model | Alert active at onset | Positive decision inside AF | New AF alert |
| --- | --- | --- | --- |
| Original template | Yes | No | No |
| Template with quality interactions | Yes | Yes | No |

The original template fell below its cutoff at the in-episode decision. The quality-interaction model remained positive while its prior alert was already active. The event audit recorded an alert already active at onset for both. The complete-episode endpoint requires a new start during AF. This single short case illustrates how the alert state can affect that endpoint.

## Reproduce offline

Use the [original development and PTT inputs](clock-study.md#reproduce) and the verified private episode plan, stable bundle and transition pair. Supply complete local MAT files or the existing source byte cache. The replay generates an aggregate only.

```powershell
python -m ppg_af.af_wrist_extra_episode_study `
  --verified-private-plan $extraEpisodePlan --source-manifest data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --development-preroll runs/wrist-history --stride-raw runs/wrist-stride10 `
  --stride-features runs/wrist-stride10-features --lock $lock --ptt-dir data/ptt-raw `
  --protected-supplement $stableBundle --protected-transition $nativeTransitionPair `
  --clock-results experiments/af-wrist/clock-study-results.json `
  --quality-results experiments/af-wrist/quality-study-results.json `
  --output runs/extra-episode-results.json
```

The [replay driver](../../ppg_af/af_wrist_extra_episode_study.py) verifies the sample clocks and applies each held-out model. [`audit_events`](../../ppg_af/af_wrist_failure_events.py) traces alert state through the predecessor decisions. The [aggregate results](extra-episode-results.json) record the outcome and verification hashes without private recording identities or timestamps.
