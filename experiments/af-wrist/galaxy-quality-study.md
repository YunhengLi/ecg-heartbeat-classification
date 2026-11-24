# Check PPG quality on two commercial wrist sensors

The AF confirmation candidate uses an optical quality boundary of 0.40. We checked whether the same boundary transfers to two simultaneously recorded commercial devices. GalaxyPPG recruited 24 healthy adults wearing a Galaxy Watch 5 and an Empatica E4 on opposite wrists and an ECG chest strap. The dataset supports optical error and quality comparisons across activities. It contains no clinical AF annotations for our alert endpoints. [Dataset paper](https://doi.org/10.1038/s41597-025-05152-z) · [GalaxyPPG v1.0.0](https://zenodo.org/records/14635823)

## Method

1. Use the baseline, keyboard typing, mobile typing, standing, walking, jogging and running sessions. Start the first 30-second window ten seconds after each activity's entry marker and use nonoverlapping windows wholly inside that activity. The public release contains all four required waveform or event files for 23 of 24 participants.
2. Align direct Galaxy acquisition timestamps, E4 UTC microseconds and Polar phone timestamps on the same clock. Subtract the published nine-hour Polar timestamp offset. Invert raw Galaxy optical polarity as specified by the authors. Require at least 95% of nominal samples and reject gaps larger than three native sample periods. Interpolate each nominal clock from data inside its assigned window and resample both PPG signals to the project's 100 Hz input grid. [Dataset preprocessing reference](https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code/blob/main/Step02_PreProcessing/Window_Segment.py)
3. Require the project ECG detector and WFDB XQRS to agree on both timing and median heart rate. Apply the same frozen PTT pulse template to each optical window. Compare direct optical intervals as a separate control. Downsample E4 to 25 Hz with anti-aliasing before the shared input grid to test the effect of reducing its sampling rate while holding its sensor and wrist constant.
4. Apply the preset quality boundary of 0.40 to the same 618 windows supported by all routes. Measure error in normalized successive interval RMSSD plus pNN50 against ECG. Define a descriptive large error as at least 0.20 in this combined statistic. All results describe 30-second timing windows, with multiple windows per person. [Benchmark code](../../ppg_af/af_galaxy_quality.py)

## Results

Of 668 scheduled windows, all had enough ECG samples and 618 met the two-detector agreement requirement. All six optical routes scored those 618 windows from 23 people.

| Sensor and optical interval route | Median quality | Windows below 0.40 | Median absolute irregularity error | Large errors below 0.40 |
| --- | ---: | ---: | ---: | ---: |
| Galaxy Watch template | 0.388 | 325 | 1.027 | 297/569 |
| Galaxy Watch direct | 0.388 | 325 | 1.026 | 298/569 |
| Empatica E4 template | 0.994 | 0 | 0.959 | 0/533 |
| Empatica E4 direct | 0.994 | 0 | 0.969 | 0/536 |
| Empatica E4 simulated 25 Hz template | 0.994 | 0 | 0.962 | 0/532 |
| Empatica E4 simulated 25 Hz direct | 0.994 | 0 | 0.966 | 0/535 |

In the template routes, median absolute heart-rate error was 3.06 bpm on Galaxy Watch and 2.50 bpm on E4. The quality score's pooled ROC AUC for large irregularity errors was 0.490 on Galaxy Watch and 0.699 on E4. The fixed boundary did not identify any E4 windows despite errors in both device routes. The artificial sample-rate reduction preserved E4 quality and error distributions. Differences between the two devices can include optical hardware, preprocessing and wrist placement. The close direct and template errors indicate both representations share the observed limitation. Across 109 baseline windows, median absolute template irregularity error was 0.356 on Galaxy Watch and 0.261 on E4. Keyboard typing errors were 1.391 and 1.275 in 113 windows. The source tests signal reliability. Applying the AF alert policy on a new device still requires separate calibration and ECG-labeled validation. [Aggregate results](galaxy-quality-results.json)

## Reproduce

Download the original repository archive or stage these four member types by their public byte ranges. The local range-based analysis verified the 94 returned ZIP members against their published CRC values and retrieved 181,392,009 member bytes. The release's full-archive MD5 is `d9a48bfcd07928fb22b96dc40cf2e7d4`. Only aggregate results are shared here. [Source archive](https://zenodo.org/records/14635823)

```bash
python -m ppg_af.af_galaxy_quality --archive GalaxyPPG.zip \
  --ptt-dir "$PTT" --output runs/galaxy-quality-results.json
```
