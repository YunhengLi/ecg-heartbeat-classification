# Replay frozen wrist alerts on disjoint half-hour blocks

The conditional confirmation rule reduced false starts from 44 to 41 on the previous two five-minute blocks per participant. We scheduled one longer block per person to better estimate alarm burden while retaining a strict separation from previously scored input times. The 25 participants had already appeared in prior evaluations and were outside all classifier fitting and threshold calibration. [Earlier confirmation replay](quality-confirmation-study.md)

## Method

1. Pick one 30-minute block per participant uniformly from eligible starts spaced 30 seconds apart. The selection uses native wrist wear bounds, ECG recording duration and a fixed random seed. Exclude all earlier parent and stride inputs, predecessor signal and both prior five-minute blocks. Reserve 90 seconds of preceding PPG and a one-second buffer at each edge. Seal the new schedule before reading its waveform and rhythm outcomes.
2. Recover all 25 blocks. Use the frozen plain-template classifier, its original cutoff, and the same trailing 30 seconds of PPG every ten seconds. The candidate asks for a preceding positive when current quality is below 0.40. Initialize alert streaks from the reserved PPG. Both routes share the full ECG reference denominator even where the confirmed score becomes unavailable. Resample participants jointly 10,000 times for paired uncertainty. [Locked planning and replay](../../ppg_af/af_wrist_long_monitor.py)
3. As a secondary outcome, treat each decision as the currently reported rhythm state for the next ten seconds. Begin each block with an unknown first state. Use exact overlap between AF annotations and the subsequent time bin. Missing scores remain unavailable. Report the AF and non-AF time reported positive and the positive state fraction on scoreable time, separately from the original alert-start metric. [State-duration code](../../ppg_af/af_state.py)

## Results

The 25 blocks total 750 scheduled minutes and 744 ECG-observed minutes. Of 4,500 scheduled decisions, 4,474 had original scores and 4,469 still had a score after confirmation. There were zero complete AF episodes. The observed AF time was 30 minutes from one participant with AF already present during the sampled block. These data support a burden comparison. Testing comparable episode sensitivity still requires a fresh complete-episode cohort.

| Metric | Original template | Low-quality confirmation |
| --- | ---: | ---: |
| False alert starts | 209 | 169 |
| False starts per ECG-observed wrist hour | 16.85 | 13.63 |
| AF time with a positive forward state | 46.7% | 40.0% |
| Non-AF time with a positive forward state | 31.7% | 28.8% |
| Positive state as a fraction of scoreable time | 32.65% | 29.58% |
| ECG AF fraction of the same scoreable time | 4.06% | 4.06% |
| Scoreable forward state coverage | 98.88% | 98.77% |

The paired false-start rate difference was −3.23 per hour, with a descriptive 95% participant interval of −4.89 to −1.77. The policies shared 80 exact-time false starts. Some of the other starts changed timing as the rule interrupted positive streaks. The drop in positive AF time reflects one person. Both state fractions substantially overestimated time in AF on this data, and both false-start rates remained high. These observations support further comparison of the confirmation rule on independent monitoring. [Aggregate results](half-hour-monitor-results.json)

## Reproduce

Use `ppg_af.af_wrist_long_monitor plan` with the original `--lock`, `--prior-inputs`, `--prior-monitor-plan`, local MAT `--source-dir` and a fresh private `--plan`. The earlier plan and archive layout are described in the [previous representative monitoring study](retrospective-monitor-study.md). Run `stage` with the same arguments and `--monitoring-cache`. The targeted local run recovered 182,452,224 source bytes from the synchronized AFPPGECG release.

```bash
python -m ppg_af.af_wrist_long_monitor score \
  --lock "$RAW/holdout-window-seeded-lock.json" \
  --prior-inputs "$RAW/template-independent-prepared" \
  --prior-monitor-plan "$OLD_PLAN" --plan "$HALF_HOUR_PLAN" \
  --monitoring-cache "$PRIVATE_CACHE" --ptt-dir "$PTT" \
  --historical-model-lock "$RAW/template-independent-model-lock.json" \
  --output runs/half-hour-monitor-results.json
```
