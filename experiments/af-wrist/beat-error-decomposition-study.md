# Separate optical beat timing and beat occurrence errors

The cutoff diagnostic found little room to improve the existing AF scores with a single threshold. This study uses measured ECG to determine whether the optical interval error is more sensitive to misplaced peaks or to missing and extra events. [Cutoff diagnostic](cutoff-frontier-study.md) · [Prior optical peak comparison](beat-opportunity-study.md)

## Method

Use the verified 223 stable wrist PPG and ECG windows from nine development participants. Match peaks one-to-one on the same 500 Hz clock, with the published archive delays and a primary 100 ms tolerance. In one replay, move only matched optical peaks to their measured ECG beat times. In a second replay, remove unmatched optical peaks and add unmatched ECG beats while leaving matched optical times as observed. The joint repair reproduces the ECG beat list. Compare the absolute error in normalized RMSSD plus pNN50, which combines two measures of interval irregularity. Summarize paired differences by giving each participant equal weight. Repeat at 65 and 150 ms and keep coarse-clock windows separate. [Repair code](../../ppg_af/af_wrist_beat_error_decomposition.py)

## Results

| Beat-aligned windows | Observed optical | Repair matched timing | Repair missing and extra events |
| --- | ---: | ---: | ---: |
| All 200 | 0.213 | 0.145 | 0.064 |
| Stable AF, 77 windows | 0.187 | 0.200 | 0.045 |
| Stable non-AF, 123 windows | 0.223 | 0.067 | 0.099 |

The table shows pooled median absolute irregularity error at 100 ms. With equal participant weighting, the reduction from repairing missing and extra events exceeded the reduction from timing repair by 0.358 across the beat-aligned windows (95% participant interval 0.119 to 0.601) and 0.660 in stable AF (0.163 to 1.391). In stable AF the same ordering held at 65 and 150 ms. Stable non-AF had the smaller pooled median under timing repair and the larger participant-weighted mean reduction under event repair. [Aggregate results](beat-error-decomposition-results.json)

On the primary clock, 4,562 of 6,918 ECG beats matched, leaving 2,356 unmatched ECG beats and 1,418 unmatched optical peaks. The 23 coarse-clock windows came from two participants and had 58 matches out of 631 ECG beats. The analysis uses ECG as an oracle and depends on the delay convention and match tolerance. It cannot determine whether an unmatched ECG beat had no peripheral pulse or whether a pulse was present but obscured or missed by the detector. The results prioritize beat recovery and artifact rejection for future PPG training and do not provide an AF alert performance estimate.

## Reproduce

Run from the repository root with `RAW` pointing to the AFPPGECG archive, `PAIRS` to the verified paired manifest and `PTT` to the local PTT recordings. Use a new `REPORT` path.

```bash
python -m ppg_af.af_wrist_beat_error_decomposition \
  --lock "$RAW/holdout-window-seeded-lock.json" --waveform-pairs "$PAIRS" \
  --waveform-results experiments/af-wrist/waveform-study-results.json \
  --ptt-dir "$PTT" --output "$REPORT"
```
