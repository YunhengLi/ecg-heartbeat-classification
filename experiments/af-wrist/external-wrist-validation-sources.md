# Source review for a fresh wrist AF evaluation

All paired participants in the AFPPGECG v6 release have been used for development or for the published independent analysis. Testing a revised alert system on a fresh cohort requires concurrent wrist PPG and ECG with enough clock information to place each decision relative to a verified AF onset and offset. Monitor exposure is sampled without using the rhythm label. Missing PPG and gaps remain on the decision clock so they count toward coverage.

| Source | Evidence for evaluation | Access and unresolved fields |
| --- | --- | --- |
| Pulsewatch | Wrist PPG and concurrent chest-patch ECG from 106 participants in two trials. The authors describe clock-drift correction and provide an alignment implementation. The ECG record is continuous and the wrist acquisition cycles on and off. | Provider approval is required. The released 30-second labels require an AF segment to be AF throughout, so they do not yield episode boundaries. Confirm whether the provider can supply ECG AF intervals, native clocks and PPG availability across the prescribed monitoring intervals. |
| Tampere PulseOn cohort | The study analyzed 30 participants with concurrent wrist PPG and Holter ECG. It reports 1,415 PPG hours and exact arrhythmia onset and offset review. | The authors accept requests for the sponsor-owned data. The paper describes stored beat intervals, rhythm status and ECG. Raw optical waveform availability remains unverified. A beat-interval-only release would support a separate evaluation. |

Source details are described by the [Pulsewatch primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13123731/), [data project](https://www.synapse.org/Synapse:syn64402376), [clock alignment implementation](https://github.com/Cassey2016/PulsewatchAlignment) and [Tampere PulseOn study](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2023.1100127/full).

For Pulsewatch, a small permitted fixture would establish the participant and session key, signal clocks, PPG gap flags and ECG interval format. The trial used patch AF detection with cardiologist review of suspected events, so the fixture should verify what episode annotations are available. Keep evaluated time and uncertainty on the original clock, including the five-minute wrist off periods. [Pulsewatch trial](https://cardio.jmir.org/2023/1/e45137)

MIMIC PERform AF supplies paired ICU waveforms with whole-record rhythm labels. It supports the [separate reconstruction benchmark](../rddm/mimic-perform-waveform.md). The source review identified candidate data but obtained no new wrist waveforms.
