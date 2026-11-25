# VitalDB paired rhythm source preflight

VitalDB is an openly available surgical monitoring source for testing rhythm features and training on a broader set of ECG reference rhythms. The 1.0.0 annotation release and the parent waveform dataset are CC BY 4.0. The PPG is a monitor plethysmogram, and the file encodes atrial fibrillation and flutter together as `AFIB/AFL`. The annotations focus on candidate arrhythmia regions and mark rhythm on ECG beats. These properties support labeled-window research on the source device. The wrist endpoint needs separately adjudicated AF boundaries and rhythm-blind wrist exposure. [Eun et al., annotation release](https://doi.org/10.13026/axd6-wm13) · [Lee et al., VitalDB](https://doi.org/10.1038/s41597-022-01411-5) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Source support

The analysis verified the published SHA-256 for each annotation file. It joined case keys to the public track list and grouped repeat cases by the published participant key. It built rhythm intervals only between clean adjacent beats of the same label, breaking at quality markers, duplicate timestamps, uncertain rhythms, or a gap over two seconds. A candidate 30-second window must fit wholly inside one such interval on the 10-second decision grid. The counts below describe annotation and catalog support, with waveform coverage checked in the six-case pilot.

| Source measure | Count |
| --- | ---: |
| Annotated cases / participants | 482 / 473 |
| Catalog cases / participants with ECG II and plethysmogram | 429 / 422 |
| Paired catalog cases / participants with an AF or flutter label | 105 / 102 |
| Candidate AF or flutter windows / cases | 8,721 / 102 |
| Candidate sinus rhythm windows / cases | 17,912 / 312 |
| Candidate patterned atrial / ventricular ectopy windows | 522 / 626 |

## Verify the actual waveform clock

The pilot chose two distinct participants in each of three strata before fetching waveforms. The strata were AF or flutter with clean sinus windows, AF or flutter without clean sinus windows, and a combination of clean ectopy and sinus windows without a source AF or flutter label. The code chooses participants based on a fixed ordering of case keys, verifies both channels at 500 Hz, and preserves gaps. It tests the same annotated windows with the track CSV and the original `.vital` file loaded with `EVENT` anchoring the case clock. No fitted shift from ECG labels is applied. The R-peak check measures the presence of a detected waveform peak within 150 ms of an annotation in disjoint trailing windows.

| Representation | Finite ECG and PPG windows | Annotated R peaks with a nearby waveform detection |
| --- | ---: | ---: |
| Legacy per-track CSV | 371 / 371 | 1,447 / 4,129 (35.0%) |
| Original `.vital` with `EVENT` origin | 371 / 371 | 4,128 / 4,129 (99.98%) |

The native ECG track starts 0.130 to 0.870 seconds later than the first populated CSV sample in these cases. Median difference is 0.499 seconds. The waveform availability test alone would have accepted both versions. The R-peak comparison distinguishes them and verifies the original case clock for this pilot. The 371 native windows comprise 146 AF or flutter, 176 sinus rhythm, 39 patterned ventricular ectopy and 10 sinus node dysfunction windows. [Aggregate preflight results](vitaldb-preflight-results.json) · [VitalDB timestamp audit](https://vitaldb.metrology-atlas.org/studies/time-zeros.html)

## Reproduce

Prepare `ANNOTATIONS` from the published annotation archive, `TRACKS` from the public `trks` endpoint, and separate local folders `LEGACY` and `NATIVE`. Use `vitaldb==1.7.2` alongside the wrist research dependencies. The plan contains case and track keys and stays local.

```bash
python -m ppg_af.af_vitaldb_preflight --annotations-dir "$ANNOTATIONS" \
  --track-list "$TRACKS" --private-plan --output runs/vitaldb-private-plan.json

python -m ppg_af.af_vitaldb_preflight --annotations-dir "$ANNOTATIONS" \
  --track-list "$TRACKS" --waveform-cache "$LEGACY" --native-cache "$NATIVE" \
  --output runs/vitaldb-preflight-results.json
```

Use the API to save `https://api.vitaldb.net/{tid}` as `{case}-ECG_II.csv.gz` or `{case}-PLETH.csv.gz` in `LEGACY`, and `https://api.vitaldb.net/{case}.vital` as `{case}.vital` in `NATIVE`. [`stable_reference`, `load_native_case`, and `verify_case`](../../ppg_af/af_vitaldb_preflight.py) define eligibility and the clock comparison. [Legacy track API](https://vitaldb.net/dataset/?query=api) · [Native waveform API](https://api.vitaldb.net/)
