# Learn optical interval reliability from wrist acceleration

The original optical quality threshold treated most Galaxy Watch inputs and nearly all Empatica E4 inputs differently. We used synchronized acceleration to test whether large interval errors can be ranked more accurately within each device and after transfer between devices. [GalaxyPPG quality study](galaxy-quality-study.md)

## Method

Reuse all 618 common 30-second ECG and bilateral PPG windows from 23 people. Match Galaxy Watch and E4 acceleration to the original system timestamps and require the same finite sample coverage and gap rules as PPG. Use the frozen template on each sensor. A large error means the absolute difference in the optical and ECG irregularity statistic is at least 0.20. [Clock and targets](../../ppg_af/af_galaxy_motion_quality.py)

Compare three ways to rank interval reliability. The heuristic ranks by the original scalar quality. The optical logistic model uses quality and pulse-band entropy. The motion model adds the PPG power left after motion-coherence weighting and the median absolute variation of acceleration magnitude relative to its median. Standardize each fitting input inside the participant split. Reuse identical groups and matched windows for the two models. For transfer, train on the other device's data from fitting participants and score the target device from held-out participants. The same person stays on one side even when the sensor changes.

## Result

The motion channels supported all 618 common windows. The shared outcome class balance was 569 large errors for Galaxy Watch and 533 for E4.

| ROC AUC for large template interval error | Score on Galaxy Watch | Score on Empatica E4 |
| --- | ---: | ---: |
| Original quality heuristic | 0.490 | 0.699 |
| Optical features, train on same device | 0.756 | 0.766 |
| Optical and motion, train on same device | 0.783 | 0.848 |
| Optical features, train on other device | 0.500 | 0.825 |
| Optical and motion, train on other device | 0.511 | 0.872 |

Acceleration improved held-out error ranking within both devices in this source. Transfer was asymmetric. The E4-trained motion model was close to chance when scored on Galaxy Watch. The Galaxy-trained model performed well for the same error label on E4. The current evaluation covers ranking in healthy activity sessions. Calibration, an alert threshold and AF performance require different endpoints. Large errors are common on both devices, so use class prevalence when interpreting the average-precision results. [Aggregate results](galaxy-motion-quality-results.json)

## Reproduce

Use the public GalaxyPPG ZIP with the optical, ECG, event and accelerometer members. The run used 47 additional upstream CRC-verified acceleration members and 49,837,524 member bytes. With separate staged archives, pass the second ZIP as `--acceleration-archive`. [Original archive](https://zenodo.org/records/14635823)

```bash
python -m ppg_af.af_galaxy_motion_quality --archive GalaxyPPG.zip \
  --ptt-dir "$PTT" --output runs/galaxy-motion-quality-results.json
```
