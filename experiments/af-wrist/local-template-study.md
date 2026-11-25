# Test a fold-local wrist waveform template for AF alerting

The [native waveform study](waveform-study.md) found higher beat F1 for the wrist-fitted template on held-out participants. This study tests the corresponding change in the downstream AF alerts using the original development clock and endpoint.

## Method

The baseline uses the original external PTT pulse template and its seven ECG interval features. The candidate learns a beat shape and pulse delay from the verified 223-pair wrist bundle separately for each outer classifier fit and for each inner calibration fit. Each fit receives paired statistics only from participants with eligible classifier fitting rows in that fit. The reconstruction for held-out or calibration participants uses their PPG waveform alone. Model inputs contain the same trailing 30 seconds and use the same reflected left padding from within that input.

Both arms use the original seed-2026 five participant folds, standardized class-balanced logistic regression at C=1, and the prespecified immediate alert policy. Select each threshold using only predictions from inner held-out participants. Preserve the original shared scoring mask and fall back to the external template features if the new reconstruction cannot support valid beat intervals. The experiment ran 20 outer and inner fits. It produced 5,766 local feature rows and used seven external fallbacks across those fits. Counts include repeated use of a window in different fits.

## Results

| Reconstruction for the AF classifier | Complete episodes detected | Wrist false starts | Median delay among detected episodes |
| --- | ---: | ---: | ---: |
| Fixed external PTT template | 12/17 (70.6%) | 6 | 20.5 s |
| Fold-local wrist template | 11/17 (64.7%) | 7 | 24.7 s |

Wrist false starts use the same 46.5 minutes of ECG-observed monitoring in both arms. The candidate gained three detections and lost four. Eight episodes were detected by both arms. Their paired median delay change was zero seconds, with two earlier and six unchanged. The pre-onset diagnostic counted three starts for the external template and five for the local template across the same ten fully supported episodes. All five inner calibrations in each arm fell short of the 80% episode sensitivity target.

Improved beat matching in the waveform benchmark did not produce better AF alerts in this development comparison. The fixed external template remains the AF reference for this route. The wrist data include beat-based synchronization that assumes a pulse delay, as described in the [waveform timing control](waveform-study.md#archive-timing-control). No independent participants were scored.

## Reproduce

Prepare the original [development clock inputs](clock-study.md#reproduce) and the verified private pair bundle from the [native waveform study](waveform-study.md#reproduce). The waveform aggregate acts as the receipt and contains the bundle hash, study roster size, protocol hash, and per-rhythm counts.

```powershell
python -m ppg_af.af_wrist_local_template_study `
  --development-raw $development --development-features $developmentFeatures `
  --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features `
  --lock $lock --clock-results experiments/af-wrist/clock-study-results.json `
  --native-pairs runs/wrist-waveform-pairs-private.npz `
  --waveform-receipt experiments/af-wrist/waveform-study-results.json `
  --output runs/local-template-study-results.json
```

The [study driver](../../ppg_af/af_wrist_local_template_study.py) verifies the native bundle and derives the reconstruction features inside each training call. The [CV callback](../../ppg_af/af_wrist_cv.py) passes each call's eligible fitting indices and scoring indices and rejects a change in feature availability. The [aggregate report](local-template-study-results.json) includes the paired alert outcomes and fallback counts.
