# Reconstruct measured ECG on the native wrist development pairs

This study measures waveform reconstruction and beat timing on 223 verified AFPPGECG v6 pairs. It uses the 114 original stable, feature-supported fitting windows and the 109 ECG-verified stable expansion windows. The original ten-person roster determines the five participant folds. Nine participants supply paired windows. [Stable training expansion](training-expansion.md)

## Native pairing and scoring

The driver rereads the ECG at 500 Hz and the wrist PPG at 100 Hz from the pinned local source files or validated byte-range cache. It verifies that every staged padded PPG row and every original staged ECG row exactly matches its native counterpart. It confirms the stable ECG labels, the original timestamps, and the disjointness of the expansion windows. The pairs include 77 AF windows from seven participants and 146 non-AF windows from nine.

Anti-alias and downsample the measured ECG to 100 Hz, then interpolate it at the native PPG sample times using the fixed release timestamps. For 192 inputs the two sample grids start together. The remaining PPG starts are 5 ms, 7 ms, or 9 ms later, with 14, 14, and three windows respectively. The scorer retains the source QRS locations at their original ECG sample precision. The dataset used an open-source ECG detector for the QRS locations. Medical specialists reviewed and corrected the AF episode annotations. [AFPPGECG v6 dataset description](https://zenodo.org/records/23921320)

Compare a fixed pulse template fitted on sitting segments from 22 external PTT participants, a pulse template refitted on each wrist fold's training participants, and an aligned delayed-input ridge model fitted on the same wrist training participants. Each model estimates a relative ECG trace from the 30-second PPG. The training routines pool waveform statistics by person. The local models receive no measured ECG from an outer test participant. The ridge penalty is fixed at 0.01. [External PTT dataset](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/)

Trim one second at each edge and score the intersection of measured and predicted sample support. All 223 windows contributed, with 621,689 of 624,400 trimmed samples and 7,501 reference beats in the common support. Report median per-window Pearson correlation, mean per-window absolute error on the baseline-centered and amplitude-normalized ECG scale, and pooled beat F1 with a 100 ms tolerance. The reference and predictions keep their original clocks at scoring.

## Results

| Method | Median waveform correlation | Normalized MAE | Source QRS F1 |
| --- | ---: | ---: | ---: |
| Fixed external PTT template | 0.0010 | 0.2476 | 0.2110 |
| Participant-held-out wrist template | 0.0424 | 0.2225 | 0.6615 |
| Participant-held-out wrist ridge | 0.0282 | 0.2206 | 0.3681 |
| Zero waveform reference | — | 0.2153 | — |

The held-out wrist template had the highest beat F1. Its F1 was 0.6225 on AF windows and 0.6907 on non-AF windows. Waveform correlations remained low, and all three reconstructions had greater normalized absolute error than the zero waveform. This result measures beat timing and waveform fidelity on selected paired development windows. The AF alert endpoint is evaluated separately.

## Archive timing control

The source documentation says fine ECG–PPG alignment assumes a 300 ms pulse delay. The local template fit learned delays of 290–310 ms across folds, compared with 190 ms for the external PTT template. Retain the external PTT waveform shape, set a fixed 300 ms delay on beat-synchronized windows, and keep the original PTT delay on coarse windows. This control uses release metadata and a documented constant. No held-out ECG selects its delay. It tests the archive clock's influence on the beat metric. [AFPPGECG synchronization description](https://zenodo.org/records/23921320)

| Method | Overall QRS F1 | Beat-synchronized QRS F1 | Coarse-synchronized QRS F1 |
| --- | ---: | ---: | ---: |
| Fixed external PTT template | 0.211 | 0.227 | 0.073 |
| Participant-held-out wrist template | 0.661 | 0.707 | 0.279 |
| Participant-held-out wrist ridge | 0.368 | 0.380 | 0.250 |
| External PTT shape with archive beat delay | 0.640 | 0.708 | 0.073 |

The diagnostic rescored all four methods on the same support, 621,688 samples. Changing only the delay reproduced the local template's beat F1 on the 200 beat-synchronized windows, while the control's median waveform correlation there was −0.0125. The coarse group contains 23 windows from two participants. The archive-based timing comparison therefore supplies little evidence about physiological pulse delay on an independently synchronized recording.

## Reproduce

Use the original [development clock inputs](clock-study.md#reproduce), the private append-order 109-window expansion and its aggregate staging receipt. Provide the pinned release manifest, native local MAT inputs or their cache, and the local PTT WFDB files. The study writes the complete native pairs to the explicit private path and the report to a separate aggregate JSON. All reads are local.

```powershell
python -m ppg_af.af_wrist_waveform_study `
  --source-manifest data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --lock $lock `
  --supplement runs/development-expansion-109-private.npz --receipt runs/expansion-plus-new-donor-receipt.json `
  --ptt-dir data/ptt-raw --private-pair-output runs/wrist-waveform-pairs-private.npz `
  --output runs/waveform-study-results.json
```

The [`participant_template_stats` helper](../../ppg_af/af_wrist_waveform_study.py) exposes the per-person training summaries for the verified 100 Hz waveform pairs. The [aggregate report](waveform-study-results.json) records paired coverage, margins, and complete model metrics by rhythm and synchronization method.
