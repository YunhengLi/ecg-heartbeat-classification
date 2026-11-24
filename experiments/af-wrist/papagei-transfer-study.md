# Frozen PaPaGei-S representations for wrist AF

PaPaGei-S learned PPG representations from 57,641 hours of optical recordings in VitalDB, MIMIC-III and MESA. The released model has no wrist AF alert head. This study tests whether its frozen features improve a small, participant-held-out wrist AF model. [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7871f51d332c8661cc729539d7987b21-Paper-Conference.pdf) · [Public weights](https://zenodo.org/records/13983110)

## Method

1. Verify the official 5,785,612-parameter model code and tensor checkpoint. Freeze PaPaGei-S and an identically sized random initialization with seed 2026. Both emit 512 embedding values per ten-second PPG segment. Only the classification head is fitted on wrist data.
2. Apply the published fourth-order 0.5–12 Hz Chebyshev filter and the reference pyPPG 50 ms smoothing within each trailing 30-second core. Normalize each disjoint ten-second segment and resample to 125 Hz. Average the three embeddings to form one 512-value vector per decision. Keep the original scoreable wrist rows instead of applying the pretraining flatline exclusion. All 2,643 segments produced finite model inputs, with zero constant filtered segments.
3. Compare the original template, each embedding alone and each embedding appended to the template features. Fit the same standardized, class-balanced logistic AF head with `C=1`, and select alert cutoffs only inside the original training groups. Use the same ten participants, five outer groups, three inner groups, two grouping seeds and common replay clock. Also compare AUC on the 114 stable labeled windows.

Use the [study runner](../../ppg_af/af_wrist_papagei.py) with the original wrist inputs and `--architecture` set to the authors' `models/resnet.py`, Git blob `574d169f2dd097aa6186372b71e8f64ea34f644e`. Supply the published `papagei_s.pt` as `--weights`. Recorded inference used PyTorch 2.10.0+cu128 and deterministic CUDA. [Encoder source](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/models/resnet.py) · [pyPPG reference filter](https://pyppg.readthedocs.io/en/latest/_modules/pyPPG/preproc.html)

## Results

| Input for AF head | Primary held-out AF detections and false starts | Alternate grouping | Primary pooled stable AUC | Primary participant-average stable AUC |
| --- | --- | --- | ---: | ---: |
| Original template ECG | 12/17 and 6 | 1/17 and 2 | 0.819 | 0.952 |
| PaPaGei-S | 10/17 and 13 | 2/17 and 5 | 0.811 | 0.842 |
| Template plus PaPaGei-S | 9/17 and 13 | 2/17 and 6 | 0.813 | 0.850 |
| Frozen random encoder | 3/17 and 1 | 0/17 and 1 | 0.868 | 0.893 |
| Template plus random encoder | 3/17 and 2 | 0/17 and 2 | 0.859 | 0.893 |

The pretrained branch detected more episodes than the corresponding random branch in both groupings, with more false starts. Neither pretrained route improved on the template's primary alert trade-off. The randomly initialized branch's higher pooled stable AUC still accompanied weak episode detection. Only six participants contributed both stable rhythms for the participant-average ranking comparison. This experiment evaluates frozen, mean-pooled features with a linear AF head. [Aggregate results](papagei-transfer-results.json)
