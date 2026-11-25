# Compare rhythm scores within a person

The frozen wrist models can assign different overall score levels to different people. We measured whether the AF and non-AF parts of an individual's recording receive different scores, using the previously scored 25-person cohort. [Historical replay and frozen fits](quality-retrospective-study.md)

## Method

Replay the original frozen plain-template, quality-only and quality-interaction models. Verify their published event alert totals and their shared input coverage. Use each score for the following ten seconds, then measure exact reference AF and non-AF time in that bin. Weight each class by the observed time within the bin. Calculate pooled ROC AUC across people and a person-level AUC for each participant with both rhythms. Average the person-level results equally. Bootstrap changes as paired participants where there is more than one contributor. [Time-weighted scoring code](../../ppg_af/af_wrist_state_ranking.py)

Keep routine wrist wear and the complete evaluation including AF-enriched times as two different scopes. Both share the original model fits, participants and signal coverage. The enriched scope has 51,660 forward scoreable seconds, 29,322 seconds in AF, and 16 people present in both rhythm classes. Routine monitoring has 7,250 scoreable seconds and about 608 seconds in AF, with one person present in both classes.

| Frozen route | Routine wear pooled AUC | Routine wear within-person AUC | Enriched scope pooled AUC | Enriched scope equally weighted within-person AUC |
| --- | ---: | ---: | ---: | ---: |
| Plain template | 0.603 | 0.148 | 0.589 | 0.582 |
| Template and PPG quality | 0.677 | 0.127 | 0.591 | 0.598 |
| Template and quality interactions | 0.662 | 0.089 | 0.580 | 0.576 |

The routine pooled comparison looked better for the quality-aware models while the one within-person comparison was worse. The single participant supplies a case description. In the enriched group, adding scalar quality raised the mean within-person AUC by 0.017, with a 95% paired interval from −0.062 to +0.132 and seven of 16 participants showing an improvement. The interaction model's change was −0.006 with interval −0.086 to +0.103. Neither route shows a reliable timing discrimination gain here. The 25-person roster and times were previously used in the project. [Aggregate results](wrist-state-ranking-results.json)

## Reproduce

Run `ppg_af.af_wrist_state_ranking` with the same `--retro-results`, `--protocol-lock`, `--data-preflight`, `--variant-lock`, `--historical-model-lock`, `--coverage-report` and `--sealed-inputs` as the [paired retrospective uncertainty comparison](quality-retrospective-uncertainty-study.md). Set `--output` to a fresh local path.
