# RDDM window stability and expanded AF evaluation

An earlier control found that processing other windows before the same wrist PPG could change its RDDM AF score. This experiment verifies window-level repeatability, refits the classifiers with the repaired features, and checks source waveform availability for an independent test. [Earlier episode and batch audit](episode-follow-up.md)

## Window-level repeatability

`window_seed()` derives a seed from the PPG samples and sampling rate. Each window uses a private Torch generator for the initial noise and every reverse diffusion step. The RDDM call processes all eight four-second chunks of the padded wrist window in one model call. Setting `--batch-size` changes how many windows are prepared together. The learned model, diffusion schedule, and timing correction use the existing checkpoint and parameters. [Window sampler](../../ppg_af/rddm_window.py)

On five real CUDA-processed wrist windows with the same AF classifier, the largest score changes with the original sampler were 0.241 after reordering windows, 0.184 after changing the batch size, and 0.288 after adding windows. All three changes were zero with the window-seeded sampler. Reordering with a different preparation batch and adding another window produced bitwise-identical waveforms on two actual inputs. The reverse diffusion loop also matched the pinned author's sampler bitwise on one window given the same initial random state.

Cached features carry a sampler version. The trainer checks that all participants in a feature directory use the same version, and the independent evaluator checks that the version matches the frozen models. The older replay accepts the archived feature version used by its recorded report. [Feature cache](../../ppg_af/af_wrist_features.py) · [Independent evaluator](../../ppg_af/af_wrist_holdout.py)

## Split and endpoint

The original fitting group contains five participants. The calibration group uses the other three original participants plus the two participants scored in the earlier feasibility run. The remaining 25 paired participants form the eligible independent roster. All three models share the same 30-second windows and feature-availability rule. Fix all thresholds before evaluating the independent roster.

The waveform review begins with the 29 timestamp candidates identified previously. Twelve are from the two previously scored participants and can support calibration. Seventeen are from two participants on the independent roster. The waveform check verifies native ECG reference and whether each episode offers at least one finite, nonflat PPG decision window. Scoring eligibility is checked again after feature extraction. Complete ECG-confirmed episodes with flat or missing PPG remain in the sensitivity denominator. False alerts use separate wrist blocks drawn from recording timestamps without consulting rhythm labels.

The comparison keeps the 80% calibration target and the ten-percentage-point test tolerance. The frozen split checks every distinct calibration score and the threshold that disables alerts. An alert is a new positive streak that starts during the reference episode. Positive windows from a streak that began before onset are tracked as a separate diagnostic. [Alert scoring](../../ppg_af/af_wrist_operating.py)

## Recorded development results

The previously scored pair supplied 11 decoded candidate runs out of 12. Some runs included another complete reference episode in their context. The expanded calibration set contained 17 complete episodes, of which 16 had an ECG-valid shared score endpoint. This presents a coverage ceiling of 94.1%. Achieving the 80% target would require a new alert in at least 14 of the 17 episodes.

| Route | New alert during the episode | Alert already active at onset | Scores below threshold | No usable score |
| --- | ---: | ---: | ---: | ---: |
| Direct PPG | 9 | 5 | 2 | 1 |
| Template ECG | 11 | 3 | 2 | 1 |
| RDDM ECG | 10 | 1 | 5 | 1 |

The earlier alert streaks began 11 to 27 seconds before the ECG-confirmed onset. All three routes fell short of 80% in the expanded calibration, even after checking all 137 candidate thresholds. The independent participants were kept unscored so that a revised alert approach can be evaluated on fresh participants. The separate five-fold development results are included in the aggregate report. [Repeatability, calibration, and development results](window-seeded-results.json)

The independent waveform review decoded both prescheduled five-minute monitoring blocks. Their ECG reference covers 600 seconds, including 572 seconds of non-AF rhythm, and all 20 monitoring windows had finite, nonflat PPG. Source range timeouts left the 17 independent candidate episode recordings incomplete. These retrieval failures are tracked separately from physiological coverage gaps.

## Reproduce

Run the staging and feature commands from the [grouped benchmark](grouped-evaluation.md#reproduce) in this revision on separately staged development and test directories. For candidate waveform review, set `--episode-count 12` on the four surveyed candidate files. This includes every timestamp candidate under the recorded schedule and duration cap when all paired waveforms are available. Use the five calibration prefixes below in the development cohort. Create a local test roster array from the paired v6 inventory after excluding the ten previously scored participants. Place the lock in a private directory.

```powershell
python -m unittest discover -s tests -p test_rddm_window.py
python -m ppg_af.af_wrist_holdout freeze --development-features data/wrist-development-features --calibration-subject 001 --calibration-subject 005 --calibration-subject 008 --calibration-subject 012 --calibration-subject 039 --test-roster data/wrist-test-roster.json --output runs/wrist-window-seeded-lock.json
python -m ppg_af.af_wrist_holdout evaluate --lock runs/wrist-window-seeded-lock.json --test-features data/wrist-test-features --output runs/wrist-window-seeded-holdout.json
```
