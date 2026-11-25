# Verify bedside ECG and PPG around annotated AF transitions

Physician-reviewed AF annotations are publicly released for part of the MIMIC-III matched waveforms. We checked whether those event clocks have continuously paired native waveforms. This source records intensive-care bed-side plethysmography and supports a separate cross-domain signal comparison. [AF annotation workbook](https://doi.org/10.6084/m9.figshare.12149091.v1) · [Annotation and sepsis study](https://doi.org/10.1109/JBHI.2020.2995139) · [PhysioNet waveform source](https://physionet.org/content/mimic3wdb-matched/1.0/)

## Verify source and paired coverage

1. Verify the Figshare workbook MD5 and parse the 45 actual coded participants, 23 marked AF and 22 marked non-AF. The trailing worksheet rows include notes and blanks. The workbook names 68 waveform recordings. Reject one waveform whose encoded participant differs from its worksheet row. Read all 67 other public WFDB master headers and verify their native segment frame totals and layout. One master header disagreed with the sum of its parts by 20 frames. [Header and annotation matching](../../ppg_af/af_mimic_bashar_preflight.py)
2. Forty-nine validated layouts list an ECG lead and plethysmography. Eight distinct participants have both listed channels and a complete annotated AF interval inside the waveform clock. Only 18 of the 67 reviewed workbook time lengths and master-header durations agree within one second. Require a planned window to be inside both. The eight annotated AF intervals range from about 517 to 1,815 minutes.
3. Seal two-sided 240-second windows around those eight onsets and eight offsets. Check the individual segment headers for continuous ECG and PPG before reading samples. Seven boundaries lack sufficient reviewed time, six lack continuous paired channels, and three have paired waves throughout the four minutes. All three are offsets. Re-run the staging and preserve each raw waveform and private source clock locally. [Boundary staging code](../../ppg_af/af_mimic_bashar_stage.py)

## Compare interval timing before and after the offset

For each paired record, divide the two sides into nonoverlapping 30-second windows. Require agreement between two ECG R-peak detectors. Eleven windows before and nine after the offset retain both optical and electrical interval support. Take each record's median on each side and then the median across the three records. Use the original frozen plain wrist classifier on PPG-derived template intervals as a cross-domain score check. [Offset feature comparison](../../ppg_af/af_mimic_bashar_offset.py)

| Score | Before offset | After offset | Records higher before |
| --- | ---: | ---: | ---: |
| Measured ECG irregularity | 0.950 | 0.014 | 2/3 |
| Direct PPG irregularity | 1.271 | 1.327 | 1/3 |
| PPG-derived template irregularity | 1.271 | 1.334 | 2/3 |
| Frozen wrist AF score | 0.815 | 0.771 | 2/3 |

The measured electrical interval statistic changed more clearly than the optical interval statistics around these three annotated offsets. The reported AF intervals were long ICU episodes and no onset supported both channels. Time after an annotated offset is a comparison period from the same chart. The workbook does not assign a specific alternative rhythm there. Sample size and placement limit interpretation. [Source check results](bashar-mimic-preflight-results.json) · [Offset results](bashar-mimic-offset-results.json)

## Reproduce

Download the named workbook from Figshare and install `openpyxl==3.1.5` plus the project's `wfdb==4.3.1` environment. Use fresh paths for each result and retain the exact source clocks and raw waves in private local directories.

```bash
python -m ppg_af.af_mimic_bashar_preflight --annotations "$WORKBOOK" \
  --header-cache "$HEADERS" --private-clock "$CLOCK" --output runs/bashar-preflight.json
python -m ppg_af.af_mimic_bashar_stage --private-clock "$CLOCK" \
  --private-plan "$PLAN" --signal-cache "$PAIRED_CACHE" --output runs/bashar-stage.json
python -m ppg_af.af_mimic_bashar_offset --staging-results runs/bashar-stage.json \
  --signal-cache "$PAIRED_CACHE" --protocol-lock "$PROTOCOL_LOCK" \
  --wrist-model-lock "$WRIST_MODEL_LOCK" --ptt-dir "$PTT" --output runs/bashar-offset.json
```
