# Reproducing RDDM checkpoint inference

**Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation**, Debaditya Shome, Pritam Sarkar, and Ali Etemad, AAAI 2024. Start with Figure 1 for the model, §4 for datasets and evaluation, and the sampling algorithm for the inference steps. [Official paper (PDF)](https://ojs.aaai.org/index.php/AAAI/article/download/29422/30683)

## Method

RDDM takes a four-second PPG segment with 512 samples at 128 Hz. It starts the ECG estimate from noise and refines it over ten reverse diffusion steps. Separate learned operations handle the QRS region and remove noise using features from the input PPG. Measured ECG is used as an evaluation reference. [Author implementation](https://github.com/DebadityaQU/RDDM/tree/7d5348843c3985c211a23ae5105a2d9497d5156a)

| Code | Role |
| --- | --- |
| [`ppg_af/rddm.py`](../../ppg_af/rddm.py) | Load the author model and run checkpoint inference and the external benchmark. |
| [`ppg_af/ptt_data.py`](../../ppg_af/ptt_data.py) | Read the paired PPG and ECG signals and verified ECG R peaks. |
| [`ppg_af/tcn_metrics.py`](../../ppg_af/tcn_metrics.py) | Score generated beats and waveform correlation on the common timeline. |
| [`ppg_af/rddm_author.py`](../../ppg_af/rddm_author.py) | Evaluate four-second waveform metrics on supplied author test arrays. |

## 1. Set up the model

Run these commands from this repository's root in a Python 3.11 or newer environment. The reference run used Python 3.14 and PyTorch 2.10.0 on CPU.

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rddm.txt
git clone https://github.com/DebadityaQU/RDDM.git rddm-upstream
git -C rddm-upstream checkout 7d5348843c3985c211a23ae5105a2d9497d5156a
```

Download these three files into `rddm-weights/`. Compare their SHA-256 digests with the reference files used in the run. [Author checkpoint folder](https://drive.google.com/drive/folders/1Z7JQ5VdTrekx4lbARJNIUiR5D-4Kz7wg?usp=sharing)

| File | SHA-256 |
| --- | --- |
| `rddm_main_network.pth` | `e90490a9039cdf147be29404be94d4b04bfcccc3c0d94ee009bbe42f2c040eb9` |
| `rddm_condition_encoder_1.pth` | `9f8ef5d758962030958674526d450b41bc188e27b1edf2c3fe5c1a4bfe067223` |
| `rddm_condition_encoder_2.pth` | `3c7cbbc1585395381faf20489bd7d9091aafc9d3cbddde67e92cc3d050f6c3bd` |

## 2. Run the external PTT benchmark

Place the WFDB `.hea`, `.dat`, and `.atr` files for the 66 sitting, walking, and running records in `data/ptt-raw/`. Keep each record's three files together at the top level. The `RECORDS` file lists the record names. [PhysioNet PTT PPG v1.1.0 files](https://physionet.org/files/pulse-transit-time-ppg/1.1.0/)

To download the required files with PowerShell:

```powershell
$pttBase = 'https://physionet.org/files/pulse-transit-time-ppg/1.1.0'
$pttDirectory = 'data/ptt-raw'
New-Item -ItemType Directory -Force -Path $pttDirectory | Out-Null
$pttRecords = ((Invoke-WebRequest -Uri "$pttBase/RECORDS" -UseBasicParsing).Content -split '\r?\n')
foreach ($pttRecord in $pttRecords) {
    if ($pttRecord -match '^s[0-9]+_(sit|walk|run)$') {
        foreach ($pttExtension in 'hea', 'dat', 'atr') {
            $pttName = "$pttRecord.$pttExtension"
            Invoke-WebRequest -Uri "$pttBase/$pttName" -UseBasicParsing -OutFile (Join-Path $pttDirectory $pttName)
        }
    }
}
```

```powershell
python -m ppg_af.rddm --upstream rddm-upstream --weights rddm-weights --output runs/ptt.json --ptt-data data/ptt-raw
```

The benchmark takes one midpoint 12-second window per participant and activity. It prepares PPG on the model's 128 Hz grid, generates three four-second segments, and returns to the 125 Hz scoring grid. It scores all models on the same samples against manually verified R peaks. Beat matching uses a 100 ms tolerance, and waveform correlation uses the recorded timing. PTT is outside the paper's listed training datasets and contains healthy participants, so this run evaluates waveform and beat timing. Mehrgardt et al. released the PhysioNet PTT PPG v1.1.0 recordings under ODbL 1.0. [PTT dataset description](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/)

The default report includes RDDM, ridge, and pulse-template results. To include the QRS-weighted CNN used in the recorded benchmark, train the fixed PTT folds and point the evaluator to those checkpoints. It checks the activity list and train, validation, and test subject IDs in each checkpoint against the evaluation split.

```powershell
python -m ppg_af.train_tcn --data-dir data/ptt-raw --output runs/tcn-results.json --folds 5 --seed 2026 --epochs 10 --batch-size 32 --threads 4 --qrs-weight 8 --checkpoint-dir runs/tcn-checkpoints
python -m ppg_af.rddm --upstream rddm-upstream --weights rddm-weights --output runs/ptt-with-cnn.json --ptt-data data/ptt-raw --tcn-checkpoints runs/tcn-checkpoints
```

The reference CPU run covers 22 participants and 66 windows. RDDM used its released weights without PTT adaptation. Each PTT baseline was fitted on other participants in the dataset. [Full results by activity](ptt-results.json)

| Model | Median waveform correlation | R-peak F1 |
| --- | ---: | ---: |
| Released RDDM | 0.011 | 0.231 |
| Ridge fitted on PTT | 0.278 | 0.864 |
| Pulse template fitted on PTT | 0.165 | 0.889 |
| QRS-weighted CNN fitted on PTT | 0.191 | 0.385 |

The follow-up fits a PPG-only timing correction while keeping the author weights fixed. [Timing experiment and results](timing-study.md)

A separate ICU comparison fits ridge and pulse-template reconstruction from scratch on patient-held-out paired ECG and PPG. It tests waveform and beat timing on recordings that include AF. [MIMIC PERform waveform benchmark](mimic-perform-waveform.md)

The [MIMIC PERform neural follow-up](mimic-perform-tcn.md) compares two small temporal CNNs on those same patient-held-out windows, including a loss that emphasizes measured ECG peak neighborhoods.

The [cross-cohort timing study](mimic-perform-ptt-delay.md) isolates the effect of a locally fitted pulse-to-ECG delay when a template pools healthy PTT and ICU data.

## 3. Evaluate the paper protocol with its test arrays

The paper describes an 80/20 subject split across WESAD, DALIA, CAPNO, BIDMC, and MIMIC-AFib. The upstream loader expects prepared arrays. The public release reviewed here leaves the prepared arrays and exact participant lists unavailable, so exact table reproduction remains pending. [Author data loader](https://github.com/DebadityaQU/RDDM/blob/7d5348843c3985c211a23ae5105a2d9497d5156a/data.py) · [Author evaluation log](https://github.com/DebadityaQU/RDDM/blob/7d5348843c3985c211a23ae5105a2d9497d5156a/eval-logs/standard_eval_RDDM.out)

With the original test arrays available locally, put each dataset's `ecg_test_4sec.npy` and `ppg_test_4sec.npy` under `data/author-arrays/<dataset>/`. The numeric arrays must contain paired 512-sample windows at 128 Hz in the same order. For BIDMC, run the command below. The report covers four-second waveform RMSE and batch-averaged Fréchet distance. Batch size and shuffle order affect the Fréchet result. The author evaluator used a batch size of 512 and a shuffled PyTorch data loader. This runner uses a seeded NumPy shuffle and reports singleton batches excluded from the Fréchet calculation.

```powershell
python -m ppg_af.rddm --upstream rddm-upstream --weights rddm-weights --output runs/author-bidmc.json --author-data data/author-arrays --dataset BIDMC
```
