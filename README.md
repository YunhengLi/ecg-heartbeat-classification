# ECG heartbeat classification and PPG-to-ECG experiments

The original scripts load MIT-BIH ECG records and extract normal and atrial premature beats. The `ppg_af` package contains experiments that reconstruct ECG from PPG and evaluate heartbeat timing.

## RDDM reproduction

The first reproduction target is **Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation** by Debaditya Shome, Pritam Sarkar, and Ali Etemad, published at AAAI 2024. [Read the paper (PDF)](https://ojs.aaai.org/index.php/AAAI/article/download/29422/30683)

The experiment runs the authors' pretrained model and measures how well its generated ECG preserves beats on an external dataset. A separate command evaluates waveform metrics on author-prepared test arrays when those arrays are available. [Setup, method, and benchmark](experiments/rddm/README.md) · [Recorded PTT results](experiments/rddm/ptt-results.json)
