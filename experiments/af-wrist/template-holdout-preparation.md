# Freeze template ECG before independent evaluation

The [template onset study](template-onset-study.md) found a one-episode net gain when each outer fold selected its own logistic regularization. Independent evaluation needs one model and one cutoff per candidate. This preparation fixes those values using the preassigned development split and checks the independent source coverage before opening any test predictions.

## Development freeze

The five fitting participants provide 43 original, eligible 30-second windows, including 21 AF and 22 non-AF windows. Fit the same standardized logistic classifier at C=0.1, 1 and 10 on those participants. Reuse the trailing 30-second feature window, 10-second decision cadence, common feature support and causal immediate-alert rule. The five calibration participants determine every threshold and the tuned model's C. Model weights stay fixed after calibration.

The published selection objective prefers at least 80% complete-episode sensitivity, then the fewest false starts. When the target is unattainable, choose the highest sensitivity and then the fewest false starts. Prefer C=1 if those outcomes tie.

| Candidate | Complete calibration AF episodes detected | False starts in calibration wrist monitoring |
| --- | ---: | ---: |
| C=0.1 | 11/17 (64.7%) | 4 |
| C=1 | 12/17 (70.6%) | 5 |
| C=10 | 12/17 (70.6%) | 5 |

There were 25 minutes of ECG-observed wrist monitoring in calibration, including 20 minutes of non-AF time. All three candidates fell short of the 80% target. C=1 and C=10 tied, so the tuned candidate selected C=1. Its model coefficients, scaler and cutoff are identical to the original candidate's. Applying both locked versions to the same independent inputs would produce identical scores and alerts. The earlier cross-validation gain described a fold-specific tuning procedure.

## Independent source coverage

The eligible roster contains 25 paired AFPPGECG v6 participants disjoint from development. Wrist monitoring comes from a fixed schedule based on recording times. The separate enriched episode schedule uses reference annotations to locate complete episodes and contributes no time to the false-alert denominator. Source retrieval failures and unavailable wrist signal are different outcomes. An undownloaded recording contributes no observed ECG exposure. [AFPPGECG v6 release](https://zenodo.org/records/23921320)

The cache survey found readable ECG annotations for 12 participants, paired timing for six and complete episode eligibility surveys for three. A follow-up using official source ranges verified paired timing for eight and completed the eligibility survey for seven. It found 29 timestamp candidates across six people, up from 17 across two. Timestamp candidates still require native waveform verification.

Validated source byte ranges recovered all 17 previously scheduled enriched episode blocks from two reserved participants. Sixteen of the 17 target episodes had continuous ECG reference, and 14 of those also had a usable raw PPG input at an in-episode decision on the original 30-second schedule. The complete ECG reference in the recovered blocks also contains incidental episodes, including one in wrist monitoring.

| Coverage after recovery of the original candidate list | Count |
| --- | ---: |
| Independent participants with decoded recordings | 2/25 |
| Complete distinct AF episodes | 27 |
| Partial AF episodes | 17 |
| Complete episodes with native PPG opportunity on the 10-second clock | 24/27 |
| ECG-observed time across all recovered blocks | 75.5 min |
| ECG-observed wrist monitoring | 10.0 min |
| Non-AF time in wrist monitoring | 9.53 min |

The expanded recovery decoded all 29 scheduled episode blocks and all seven prescheduled wrist blocks for the participants with completed eligibility surveys. Twenty-eight of the scheduled target episodes had continuous ECG reference. Across the combined blocks, including incidental episodes, the audit counted 42 distinct complete AF episodes and 20 partial episodes. Of the 42 complete episodes, 38 had a usable raw PPG input at a 10-second decision point. Total observed ECG exposure was 137.5 minutes. The seven wrist blocks contributed 35 minutes to future false-start evaluation, including 34.53 minutes of non-AF time.

The source audit verifies native PPG opportunity and predecessor windows without computing classifier features or generating model scores. Final range recovery resolved the input status for every evaluation and predecessor window. The current decoded set covers seven of the 25 reserved people, with the remaining 18 outside this waveform snapshot. The planned two-model comparison ends at the freeze because both models produce exactly the same results for any shared input. The recovered recordings remain available for independent validation of a distinct future candidate.

## Reproduce the freeze

Use the private source protocol and original development archives from the [decision cadence study](clock-study.md#reproduce). Preserve the private lock outside the repository. The aggregate results publish candidate counts without model parameters, thresholds or participant identifiers.

The source audit takes raw archives for the scheduled wrist and episode blocks, plus native predecessor archives prepared from the matching MAT recordings. The predecessor windows use the same [staging procedure](../../ppg_af/af_wrist_preroll.py) as the development replay and contribute no ECG exposure.

```powershell
python -m ppg_af.af_wrist_template_holdout --development-features $developmentFeatures --stride-raw runs/wrist-stride10 --stride-features runs/wrist-stride10-features --protocol-lock $lock --output-lock runs/template-independent-model-lock.json --output-report runs/template-independent-development-summary.json

python -m ppg_af.af_wrist_template_coverage --independent-raw data/wrist-independent-raw --preroll data/wrist-independent-preroll --protocol-lock runs/template-independent-model-lock.json --output runs/template-independent-coverage.json
```

[`freeze_template`](../../ppg_af/af_wrist_template_holdout.py) fits on the designated participants and applies the calibration objective. [`coverage_report`](../../ppg_af/af_wrist_template_coverage.py) reconstructs reference and 10-second raw opportunity counts from the decoded blocks and native history. The [development freeze results](template-holdout-freeze-results.json) and [source coverage results](template-holdout-coverage-results.json) contain the aggregate metrics.
