# Motion and optical spectrum around wrist false starts

This audit tests whether false starts from the original template and the fixed low-quality confirmation rule occur during greater wrist movement or less concentrated optical pulse spectra. It uses the same 25 sealed half-hour recordings and retains the existing false-start decisions.

## Method

1. Recover 50 Hz triaxial acceleration on the 25 optical clocks. Match the first and last available PPG anchors to the sealed input and verify that they describe one consistent native acceleration sample interval. Reproduce the original and conditional replay exactly before attaching motion summaries.
2. Require a complete 30-second PPG and motion input with ECG-confirmed non-AF throughout. For each false start, find the closest same-block eligible decision within 120 seconds whose 30-second input overlaps no alert start. Use each control interval at most once within a comparison and remove control intervals that overlap it. Continued positive state is eligible when it has no overlapping new start, because this comparison concerns the timing of new starts.
3. Compare the original PPG quality score, pulse-band entropy, fraction of pulse power remaining after motion weighting, and log relative acceleration variation. Compute case-minus-control means within each participant, then average across participants and resample whole participants for descriptive 95% intervals.

The [accelerometer staging](../../ppg_af/af_wrist_half_hour_acceleration.py) verifies the frozen optical input and first and last native anchors. The [motion audit](../../ppg_af/af_wrist_half_hour_motion_audit.py) matches controls and summarizes participant-level differences. It accepts the same sealed monitoring inputs as the [half-hour replay](half-hour-monitor-study.md#reproduce), plus `--half-hour-results` and `--acceleration-cache`.

## Results

| False-start group | Matched pairs | Participants | PPG quality difference | Pulse-band entropy difference | Acceleration variation difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| All original starts | 178 | 21 | −0.060 [−0.103, −0.018] | +0.073 [+0.032, +0.121] | +0.211 [+0.004, +0.421] |
| All conditional starts | 151 | 19 | −0.013 [−0.045, +0.017] | +0.045 [+0.012, +0.083] | +0.234 [−0.025, +0.574] |
| Original-only exact timestamps | 114 | 21 | −0.142 [−0.194, −0.090] | +0.074 [+0.028, +0.128] | +0.214 [−0.036, +0.477] |
| Shared exact timestamps | 75 | 15 | +0.115 [+0.017, +0.236] | +0.064 [+0.013, +0.120] | +0.237 [+0.005, +0.523] |

All 25 blocks had native motion. Of 209 original and 169 conditional ECG-valid false starts, 204 and 166 met the stricter non-AF and signal requirements for matching. Pulse-band entropy was higher at matched false starts under both policies. The bulk motion amplitude difference for the conditional starts crossed zero, and the motion-weighted pulse-power difference crossed zero in all four comparisons. The published quality gate preferentially changed starts with low optical quality. An exact-timestamp comparison counts shifted alerts as different starts.

These comparisons describe already observed behavior. Motion features may reflect artifacts, activity or optical contact. The same release and people were previously evaluated, the 30-minute blocks had no complete AF episodes, and the intervals are unadjusted for multiple feature comparisons. [Aggregate results](half-hour-motion-audit-results.json)
