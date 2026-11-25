# Train a dedicated new-onset head on the original folds

The [independent error audit](template-independent-error-audit.md) found episodes with an alert already active before AF began. This experiment tests whether directly training for the early-AF state improves new alerts on the original development benchmark.

## Labels and support gate

Use the fixed 30-second PPG quality-interaction features and standardized, class-balanced logistic regression with C=1. Assign folds with the original stable AF labels. The candidate training pool contains 17 ECG-verified early-AF positives and 125 controls. The controls are 34 stable AF windows, 80 stable non-AF windows, and 11 verified pre-onset windows. Each fit selects rows from its training participants before using these labels. The [native transition study](transition-native-study.md) describes the 26 original transition rows and the separately staged pair. The native pair is available only for fitting within its donor's training groups.

Check the exact outer and inner participant assignments for the original seed 2026 and the weakest previously observed split, seed 2030. Appending the separate pair leaves both assignments unchanged. Require at least one early-AF example and a control in every fit before training either candidate. All 20 fits per seed passed this gate. Four people supply the early-AF examples, and one person supplies 13 of 17.

| Seed | Outer fits | Inner fits | Minimum early-AF windows in an outer fit | Minimum in an inner fit | Inner fits with only one early-AF donor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026 | 5 | 15 | 4 | 1 | 4 |
| 2030 | 5 | 15 | 2 | 1 | 3 |

The original shared feature mask stays in place. Each threshold uses the same episode-sensitivity objective and predictions from inner held-out participants. Replay causal immediate alerts on the same ECG reference and 2,790 seconds of wrist exposure. [Original quality split comparison](quality-stability.md)

## Results

| Seed | Standard quality model | Dedicated new-onset head |
| --- | ---: | ---: |
| 2026 | 14/17 episodes, 4 false starts | 7/17 episodes, 7 false starts |
| 2030 | 1/17 episodes, 0 false starts | 0/17 episodes, 5 false starts |

The dedicated head gained no event detections across either split. On seed 2026, pre-onset false starts dropped from three to two, while overall wrist false starts rose and seven detections were lost. The paired median delay change for the seven episodes detected by both arms was zero. On seed 2030, four of five inner calibration sets reached the sensitivity target, and the outer predictions detected zero complete episodes. The limited positive donor support and the split-dependent calibration restrict the inference to these development data. No independent predictions were generated.

## Reproduce

Use the original [development clock](clock-study.md#reproduce), the private native pair, its receipt and the protected 109-window bundle from the [native transition study](transition-native-study.md#reproduce). The protected bundle verifies disjointness and contributes zero features or fitting rows. The command checks native waveform and label timing locally, audits support, and fits only when every original split qualifies.

```powershell
python -m ppg_af.af_wrist_onset_head `
  --source-manifest data/afppgecg-v6/metadata-v6.json `
  --source-dir data/afppgecg-v6 --cache-dir data/afppgecg-v6/range-cache `
  --development-raw $development --development-features $developmentFeatures `
  --development-preroll runs/wrist-history --stride-raw runs/wrist-stride10 `
  --stride-features runs/wrist-stride10-features --lock $lock `
  --pair runs/transition-pair-private.npz --receipt runs/transition-pair-receipt.json `
  --protected-supplement runs/development-expansion-109-private.npz --ptt-dir data/ptt-raw `
  --stability-receipt experiments/af-wrist/quality-stability-results.json `
  --output runs/onset-head-results.json
```

The [study driver](../../ppg_af/af_wrist_onset_head.py) uses the existing nested group assignment and scorer. The [aggregate report](onset-head-results.json) includes support for every fit, calibration status, and paired results.
