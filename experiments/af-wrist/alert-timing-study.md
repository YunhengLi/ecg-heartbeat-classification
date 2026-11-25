# AF alert timing on development participants

This experiment traces alerts that were already active at a reference AF onset and tests a causal two-window confirmation rule. It uses the ten participants from the window-seeded development cohort. The independent cohort remains reserved. [Window-seeded calibration](window-seeded-evaluation.md)

## What the earlier signal shows

The frozen scores produced nine route-and-episode cases in which an alert started 11–27 seconds before the target onset and stayed positive into AF. They represent six distinct episodes from two development participants. Five triggering windows contained some or all of an earlier ECG-confirmed AF episode. In four cases, the alert itself began while that earlier episode was still active.

Seven starts were the first available decision in a staged block. Three earlier 30-second windows of paired ECG and PPG were recovered from validated local source ranges for all seven cases. RDDM used the window-seeded pipeline, and all route classifiers and thresholds stayed frozen. Five cases already had a positive decision immediately before the staged block. This accounted for all four cases whose apparent triggering window was AF-free. The two observed negative-to-positive transitions occurred in windows containing prior AF. In two cases the recovered 90 seconds began with an active positive streak, so its exact start remains uncertain.

This audit explains why resetting the alert state at a recording boundary can make an ongoing alert appear new. The nine cases came from AF-enriched blocks. False-alert burden is measured separately using the scheduled wrist-monitoring blocks. [Aggregate onset audit and study results](alert-timing-results.json)

## One versus two positive windows

The original rule can emit an alert at the end of the first positive 30-second window. The confirmation rule waits until the second consecutive positive window ends. A negative or unavailable decision, or a gap between runs, resets the streak. Confirmation uses the route's decisions and their timing. Reference ECG supplies episode labels for scoring. The study retains the strict endpoint of a new alert inside the reference episode. [Confirmation code](../../ppg_af/af_events.py) · [Experiment runner](../../ppg_af/af_wrist_alert_study.py)

First, hold the original score thresholds fixed on the five calibration participants. Then search all 137 candidate thresholds separately for each rule with the 80% sensitivity target. Each cell shows complete episodes detected out of 17, followed by false alerts over the same 25 minutes of ECG-observed wrist monitoring. Twenty of those minutes were non-AF.

| Route | One window at original threshold | Two windows at original threshold | Two windows after recalibration |
| --- | --- | --- | --- |
| Direct PPG | 9/17, 4 false alerts | 9/17, 3 false alerts | 13/17, 3 false alerts |
| Template ECG | 11/17, 4 false alerts | 8/17, 3 false alerts | 13/17, 3 false alerts |
| RDDM ECG | 10/17, 9 false alerts | 2/17, 2 false alerts | 13/17, 4 false alerts |

At the original threshold, direct PPG detected five additional episodes and missed five that its original rule detected. The total of nine stayed the same. All three recalibrated confirmation rules stopped one detection short of the 80% target.

## Five-fold development check

Each participant stays in one outer fold. Each outer split fits the classifier on the other participants and uses predictions from up to three inner folds to choose its threshold. The same outer participants and cutoff search are used for both confirmation rules. This check pools 17 complete reference episodes and 46.5 minutes of ECG-observed wrist monitoring, including 34 minutes of non-AF.

| Route | One window | Two windows |
| --- | --- | --- |
| Direct PPG | 9/17, 6 false alerts | 12/17, 3 false alerts |
| Template ECG | 10/17, 7 false alerts | 10/17, 2 false alerts |
| RDDM ECG | 3/17, 9 false alerts | 13/17, 7 false alerts |

Each two-window route reached the 80% inner calibration target in one of the five folds. These development comparisons leave independent false-alert performance at matched sensitivity unresolved.

## Next development experiment

Recover preceding ECG and PPG for every scheduled run where the source permits it. Use the earlier PPG decisions to initialize alert state, and count exposure and outcomes within the prescheduled evaluation interval. Check the two confirmation rules on those extended blocks first. Then keep the 30-second feature window and test a 10-second update interval separately. A shorter update interval can expose brief inter-episode gaps while preserving the input horizon. Use the same monitoring time and participant splits.

## Reproduce

Use the development feature archives and original frozen lock from the [window-seeded protocol](window-seeded-evaluation.md#reproduce).

```powershell
python -m unittest discover -s tests
python -m ppg_af.af_wrist_alert_study --development-features data/wrist-development-features --lock runs/wrist-window-seeded-lock.json --output runs/alert-confirmation.json
```
