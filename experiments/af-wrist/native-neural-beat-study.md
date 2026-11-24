# Neural beat timing on the AF wrist cohort

The source-frozen PPG network improved beat detection on Shimmer. We evaluated the same network on the AFPPGECG wrist recordings, then tested whether training on paired wrist ECG improves its performance for held-out participants. [Source and independent device experiment](beat-cnn-study.md)

## Frozen beat features and AF alerts

Freeze the optical beat CNN using only the 14 ScientISST MOVE training participants. Generate peaks from each of the 881 originally scoreable wrist PPG cores. Convert the peaks to the original seven ECG timing and support features. Every row has enough predicted beats to compute those features. Retain the original PPG quality score. Compare replacing the template features with neural features and adding the neural timing features to the original classifier. All AF logits and thresholds are fitted inside each wrist training fold with the original 30-second windows and 10-second decisions.

| Partition | Original quality template | Neural timing replacement | Template plus neural timing |
| --- | --- | --- | --- |
| 2026 | 14/17 detections, 4 false starts | 3/17, 2 | 9/17, 5 |
| 2030 | 1/17, 0 | 0/17, 1 | 1/17, 0 |

The replacement and addition both exceeded the ten-percentage-point sensitivity loss tolerance on the original partition. More correctly matched beats on a separate exercise device did not translate into useful AF alert features in this test. [AF replay results](neural-beat-af-results.json)

## Native paired wrist beat timing

Use the verified 223 stable pairs from nine wrist participants. The archive includes 114 original and 109 supplemental windows, with 7,549 usable native ECG QRS marks after the one-second edge guard. Compare the release-delay template, the frozen source CNN, and two CNNs trained for 12 epochs on the same wrist training windows. Initialize one from scratch and the other from the source checkpoint. Every wrist subject stays in a single outer fold and is scored by a model that did not use their pairs. The frozen source and template see no wrist fitting targets. [Native pair verification](waveform-study.md)

| Beat route | All windows F1 | Beat-synced F1 | Coarse-synced F1 | Median absolute RR error, matched consecutive beats |
| --- | ---: | ---: | ---: | ---: |
| Template with release-delay control | 0.638 | 0.706 | 0.073 | 14 ms |
| Frozen MOVE CNN | 0.224 | 0.221 | 0.245 | 68 ms |
| Wrist-only CNN | 0.457 | 0.492 | 0.150 | 14 ms |
| MOVE initialization plus wrist fine-tuning | 0.456 | 0.491 | 0.170 | 12 ms |

The fine-tuned model recovered many more beats than the frozen network. Source initialization and training from scratch produced almost identical overall F1. The archive's fine alignment assumes a pulse delay of 300 ms, which the template comparator uses. The coarse subset has 23 windows from two participants and gives a much less certain picture of real-world clocks. In the 77 stable AF windows, F1 was 0.617 for the template and 0.340 for fine-tuning. The interval error table applies only to consecutively matched beats and should be read together with F1. [Native beat results](native-beat-transfer-results.json)

## Reproduce

Use the same wrist archives, lock, and pinned clock and quality results as the [motion study](motion-af-study.md#reproduce), and the public `--move-dir` from the [training study](beat-cnn-study.md#reproduce). Both runners accept `--device cpu` or `--device cuda` and a new local `--output`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA. The native study additionally accepts the verified `--waveform-pairs`, `--waveform-results` and external `--ptt-dir`.

- [`peak_intervals` and grouped AF comparison](../../ppg_af/af_wrist_neural_beats.py) place frozen beat predictions on the original wrist alert clock.
- [`read_pairs` and participant-held-out fitting](../../ppg_af/af_wrist_native_beat_transfer.py) verify alignment provenance and compare source initialization against training from scratch.
