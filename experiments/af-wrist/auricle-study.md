# Corroborate finger pulses with ear PPG on a separate clock calibration

PeriWave used two wavelengths on the same finger. Auricle supplies synchronized finger and ear PPG and a reference chest ECG. We examined whether aligning the two optical recordings before matching peaks improves reliable interval coverage and agreement with ECG. [Auricle dataset release](https://physionet.org/content/auricle/1.0.0/)

## Method

- Pin the PhysioNet record list and SHA-256 catalog. Select one session from ten healthy and ten heart-failure participants before seeing their waveform values. Use a 30-second optical calibration input beginning at 90 seconds and nonoverlapping evaluation inputs beginning at 330 and 930 seconds. The staging command validates byte-range geometry and WFDB header checksums. The range decoder reproduced all retained ECG and PPG samples from one source window when compared with WFDB's remote reader.
- In the optical calibration input, search a fixed ±250 ms lag using the two band-passed PPG signals. Use that lag unchanged for both later windows. Evaluate finger only, ear only, direct peak correspondence within 80 ms, and correspondence after the fixed optical lag. Only verify original adjacent finger intervals whose endpoints both match.
- Use the 1,000 Hz ECG detector for reference timing and WFDB XQRS as the agreement check. The initial raw 1,000 Hz XQRS check did not support any of the 40 windows. Normalize the ECG and run XQRS at 250 Hz while preserving the original ECG timing for interval values. This ECG-only adjustment occurred before evaluating optical errors. The same 90% peak agreement and 5 bpm rate criteria then supported 34 windows.

## Results

All 20 participants supplied finite optical calibration. The median signed ear-minus-finger lag was +5 ms. For the 34 ECG-supported windows, unshifted dual corroboration produced intervals in 26 and the calibrated route in 24. The joint accuracy comparison used 21 inputs from 14 people. Both single-channel routes produced features at all 34 ECG-supported inputs. [Aggregate results](auricle-results.json)

| Method | Median absolute interval irregularity error | Median confirmed interval fraction |
| --- | ---: | ---: |
| Finger | 0.065 | 100.0% |
| Ear | 0.268 | 100.0% |
| Dual without optical lag | 0.083 | 58.1% |
| Dual with frozen optical lag | 0.082 | 84.6% |

Among the 21 common inputs, calibration recovered more intervals than unshifted dual matching. The calibrated median irregularity error stayed above finger-only. In the 11 common heart-failure windows, calibration lowered the dual median error from 0.233 to 0.083, with finger at 0.047. The comparison concerns pulse timing in these lab sessions. Auricle provides heart-failure group membership and activity measurements, without the AF episode labels required for the wrist alert benchmark.

```bash
python -m ppg_af.af_auricle_stage --cache-dir "$PRIVATE_AURICLE"
python -m ppg_af.af_auricle_benchmark \
  --staged "$PRIVATE_AURICLE" --output runs/auricle-results.json
```

Install the wrist requirements, `requests`, and `wfdb==4.3.1`. Only the three selected waveform channels are staged. [Source loader](../../ppg_af/af_auricle_stage.py) · [Optical lag and evaluation code](../../ppg_af/af_auricle_benchmark.py)
