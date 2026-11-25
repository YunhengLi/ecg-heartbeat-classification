# Paired-ECG guidance for a PPG-only wrist classifier

SiamAF studies whether simultaneous ECG and PPG can guide a shared representation that supports AF detection from either modality. This independent small-model experiment isolates that training idea on the public MIMIC PERform AF data and the existing wrist folds. [SiamAF paper](https://hdsr.mitpress.mit.edu/pub/vifgdvlv/release/1) · [Public ICU data](https://zenodo.org/records/15906524)

## Method

1. Start from the 1,342 PPG-eligible 30-second MIMIC windows. Require finite nonflat simultaneous ECG and at least two separate windows per person. The paired comparison retained 1,320 windows from 34 people, with 729 AF and 591 non-AF windows. One source participant lacked enough exact ECG pairs. The release labels rhythm at the record level.
2. Reuse the compact one-dimensional CNN, preprocessing, initialization, label weighting and 12-epoch training schedule across four arms. Train with PPG classification alone, with shared PPG and ECG classification, and with two versions of a cosine alignment penalty added to the joint classification loss. One uses the simultaneous ECG embedding. The other rotates the ECG assignments among disjoint windows from the same person, preserving labels, waveform distributions and pair counts while changing only the time correspondence. The ECG embedding is detached within the alignment term.
3. Freeze all four source models. Pass only wrist PPG into the model for all 881 target rows. Select thresholds from each outer fold's development-side participants using the existing target and matched-sensitivity objectives. Score complete AF detection and representative false starts on the locked original and alternate groupings.

The [paired source runner](../../ppg_af/af_wrist_paired_source.py) uses the published data checksum and defines the controlled training objectives. This benchmark uses the project's compact network and a cosine regularizer. It does not reproduce the original paper's proprietary training cohort or full model.

## Results

| Frozen wrist input route | Primary target-cutoff AF and false starts | Alternate target-cutoff AF and false starts | Pooled stable wrist AUC |
| --- | --- | --- | ---: |
| Wrist template ECG | 12/17 and 6 | 1/17 and 2 | 0.819 on the primary grouping |
| Paired-subset PPG-only training | 11/17 and 22 | 2/17 and 12 | 0.629 |
| Shared ECG and PPG classification | 9/17 and 18 | 3/17 and 10 | 0.582 |
| Synchronized ECG representation alignment | 4/17 and 20 | 2/17 and 13 | 0.558 |
| Shifted ECG representation alignment | 4/17 and 23 | 2/17 and 16 | 0.558 |

Synchronized alignment improved false-start counts versus the shifted control at the target cutoff with identical difficult event sensitivity. It detected fewer episodes than PPG-only training in the primary split. The matched calibration objective did not change the four primary detection counts. The shared source domain, small sample and whole-record rhythm labels limit the inference to this benchmark. [Aggregate results](paired-source-results.json)
