# Independent validation plan for wrist AF alerts

The fixed plain template and its quality-conditioned confirmation rule are the two primary comparison routes. Both can replay identical PPG inputs with an original score cutoff. Their historical sensitivity was 59.6% and 58.1%, respectively. On separately sampled monitoring times from the same 25 people, the false-start difference was −0.75 per ECG-observed hour with a participant-bootstrap interval of −1.97 to +0.51. These observations inform planning. [Conditional confirmation study](quality-confirmation-study.md)

## Freeze the test before accessing outcomes

1. Use participants outside all development and retrospective rosters, with synchronized native wrist PPG and ECG. Record the device and recording clocks, sampling rate, wear intervals, motion and signal gaps. Verify alignment using source anchors. ECG reference readers work without PPG scores and mark AF boundaries and major alternative rhythms, including flutter and ectopy when distinguishable.
2. Keep two analyses. The episode analysis may use an enriched cohort to accrue complete AF episodes. The burden analysis uses consecutively observed wear or coverage selected independently of rhythm outcomes and predictions. Choose the target device, collection setting, monitoring duration and sampling rule before enrollment. Use a predeclared preroll when the recording allows it.
3. Hash the analysis code, reference rules, rosters and model specifications before scoring. Keep the original plain-template fit and cutoff for both routes. The candidate additionally requires a preceding positive if current PPG quality is below 0.40. Device adaptation or threshold fitting uses a separate calibration roster. The held-out test runs once under the frozen policy.
4. Report complete-episode sensitivity and its paired candidate-minus-baseline difference. Show both pooled and participant-average sensitivity, matched detection delay, false alert starts per ECG-observed wrist hour, a secondary rate per verified non-AF hour, and coverage. Report clipped episodes and ambiguous start state separately. Stratify descriptively by episode duration, device, quality and activity when support permits. Resample whole participants for intervals. Report the largest contributor's share and the sensitivity to omitting each contributor.
5. Use the existing 80% point target and ten-percentage-point tolerance in the development comparison. For an independent claim at similar sensitivity, require the paired sensitivity interval to lie inside the tolerance and the paired false-start rate interval to favor the candidate. A separate requirement on the lower sensitivity bound would need a redesigned sample size. Declare an uncertain result if either comparison remains imprecise.

## Plan the number of independent participants

The table uses the half-width of a two-sided 95% Wilson interval at the assumed sensitivity. For three eligible complete episodes per AF participant, inflate the independent episode requirement by the illustrative design effect `1 + 2 × ICC`, then divide by three and round up. ICC is the within-person correlation in episode detection. Use the higher requirements or a participant-level simulation if early observed variation differs substantially. [Wilson score interval](https://doi.org/10.1080/01621459.1927.10502953) · [Donner, Birkett and Buck on clustering](https://pubmed.ncbi.nlm.nih.gov/7315838/)

| Assumed sensitivity | 95% CI half-width | Independent episodes | AF participants, ICC .10 | AF participants, ICC .30 |
| --- | ---: | ---: | ---: | ---: |
| 80% | 10 points | 60 | 24 | 32 |
| 80% | 5 points | 245 | 98 | 131 |
| 58.1% | 10 points | 90 | 36 | 48 |
| 58.1% | 5 points | 371 | 149 | 198 |

These are marginal sensitivity precision scenarios. Paired equivalence needs the expected within-person difference, discordant episode counts and the variation in episode numbers across people. The previous roster had 260 episodes from 16 people, with 157 from one person. A power simulation based on that single dominant participant would be unstable. Estimate those inputs from an independent calibration or blinded pilot before deciding the final recruitment target.

## Plan the burden observation time

The 25-person additional replay contained 241.5 ECG-observed minutes, or an average of 9.66 minutes each. Its two-sided paired interval has a half-width of about 1.24 false starts per hour. Under an illustrative normal approximation with the same time and between-participant variation, uncertainty scales as `1.24 × sqrt(25 / n)`. A 100-person study with the same time per person would yield about 16.1 ECG-observed hours and a half-width near 0.62 per hour. A true paired reduction of 0.75 per hour would need roughly 142 people and 22.9 hours for nominal 80% power at a two-sided 5% level under the same assumptions.

The pilot times are short and were sampled from a previously compared roster. The rate variance may change substantially with longer recordings, a new device or a different activity mix. Use representative monitoring over the intended wear conditions and update the planning assumptions using blinded exposure and independently calibrated participant-level variability.
