# Predictive Maintenance and Anomaly Detection

When a machine breaks down unexpectedly, the line stops. I wanted to see how far sensor data can get you in two questions: which runs are about to fail, and how early you can see a problem coming, so that maintenance can be planned instead of being a reaction.

The project has two parts:

1. **Failure prediction** on 10,000 milling-machine runs (AI4I 2020) with five labelled failure types. I chose the alarm threshold with a simple maintenance cost model instead of using the default.
2. **Early warning** on a temperature sensor from a real industrial machine (NAB). I compared three detectors that don't use the failure labels at all and measured how many hours before a failure each one raised an alarm.

The whole analysis, with explanations, is in [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).

## Main results

| | Result |
|---|---|
| Failures caught (test set, 2,000 unseen runs) | 58 of 68 (85%), with 8 false alarms |
| With a planned tool change at 205 min | 64 of 68 (94%), with 113 extra inspections |
| Maintenance cost vs. running to failure | 68% lower (illustrative costs; a simple tool-change rule gives 20%) |
| Heat, power and overstrain failures | all caught |
| Early warning on the real machine | 4 of 4 labelled events, median 17 hours ahead, 0.5 false alarms per week |

## Part 1: Failure prediction (AI4I 2020)

![Operating range](assets/eda_operating_range.png)

Failures happen at the edges of the operating range: very high torque at low speed (overstrain, too much power) and very low torque at high speed (too little power). Low-quality (L) parts fail almost twice as often as high-quality ones (3.9% vs 2.1%). Based on that I added three features from the physics of the process: spindle power (torque × angular speed), strain (tool wear × torque) and the difference between process and air temperature, which points to heat-dissipation problems.

### Models (5-fold cross-validation on 8,000 training runs)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Rule of thumb: change the tool at 200 min | – | recall 0.36, precision 0.16 |
| Logistic regression | 0.455 | 0.922 |
| Gradient boosting, raw sensors only | 0.792 | 0.967 |
| Gradient boosting with the extra features | 0.875 | 0.968 |
| **Random forest** (the one I used) | **0.893** | **0.977** |

<sub>I used PR-AUC as the main metric because only 3.4% of runs fail. A model that never raises an alarm would still be 96.6% "accurate".</sub>

![Precision vs recall](assets/pr_curves.png)

### Picking the alarm threshold

A missed failure, a planned repair and a false alarm cost very different amounts. With the illustrative costs in [`configs/costs.yaml`](configs/costs.yaml) (unplanned breakdown €10,000, planned intervention €2,000, false alarm €500), the cheapest threshold on the training data is 0.34 rather than the usual 0.50.

| Strategy (test set, 2,000 runs) | Caught | Missed | False alarms | Cost per 1,000 runs | Saving |
|---|---|---|---|---|---|
| Run to failure (no prediction) | 0 | 68 | 0 | €340,000 | – |
| Rule of thumb: change the tool at 200 min | 26 | 42 | 149 | €273,250 | 20% |
| Model, default threshold 0.50 | 55 | 13 | 3 | €120,750 | 64% |
| **Model, cost-based threshold 0.34** | **58** | **10** | **8** | **€110,000** | **68%** |
| Model + planned tool change at 205 min | 64 | 4 | 113 | €112,250 | 67% |

![Cost comparison](assets/cost_comparison.png)

### What I noticed

The extra features mattered more than the choice of model. They took the same gradient boosting model from 0.79 to 0.88 PR-AUC. Rotational speed, temperature difference, strain and power came out as the strongest risk drivers.

Failures with a physical cause were completely predictable: all heat-dissipation (29/29), power (13/13) and overstrain (16/16) failures in the test set were caught. Tool-wear failures were not. They happen at a random point somewhere between about 200 and 240 minutes of use, so there's nothing in the sensors to see beforehand (only 2 of 10 caught). Adding a planned tool change at 205 minutes (limit chosen on the training data) raises that to 7 of 10 and overall recall to 94%.

Whether that tool change is worth it is really a business decision. On the test set both options cost about the same. The tool change means 105 extra inspections in exchange for 6 fewer breakdowns, which pays off once an unplanned breakdown costs more than about €10,750:

| Cost of one unplanned breakdown | Model only | Model + tool change | Cheaper |
|---|---|---|---|
| €5,000 | €170,000 | €204,500 | Model only |
| €10,000 | €220,000 | €224,500 | Model only |
| €20,000 | €320,000 | €264,500 | Model + tool change |
| €50,000 | €620,000 | €384,500 | Model + tool change |

| What drives the risk? | Where do failures happen? |
|---|---|
| ![Feature importance](assets/feature_importance.png) | ![Failure map](assets/failure_map.png) |

<sub>Right: predicted risk for low-quality (L) parts across torque and tool wear. The diagonal edge is the overstrain limit (wear × torque), and the band on the left is power failure at low torque.</sub>

## Part 2: Early warning from a temperature sensor (NAB)

The data is a temperature sensor on an internal component of a real industrial machine: 11 weeks of readings every 5 minutes (22,683 after cleaning) with four labelled anomalies. According to Numenta, this machine had a planned shutdown and a hard-to-spot anomaly that led straight to a catastrophic failure.

I built the detectors without the labels, as if it were a new machine with no failure history. They only see past readings, and I used standard settings (1st percentile, 3 standard deviations, 30 minutes persistence) instead of tuning them to the four events. The labels are only used for scoring.

| Detector | Events detected | Median warning before event | False alarms |
|---|---|---|---|
| **Fixed low-temperature limit** (1st percentile of a normal week) | **4 / 4** | **17.4 h** | **0.5 per week** (5) |
| Rolling z-score (> 3σ vs. the last 24 h) | 3 / 4 ¹ | 5.4 h | 3.0 per week (31) |
| Isolation Forest (6 rolling features) | 3 / 4 ¹ | 7.1 h | 1.3 per week (13) |

<sub>¹ Includes the 8 Feb event, which both only flagged about 22 hours after it happened. Both missed the 28 Jan event.</sub>

![Anomaly timeline](assets/anomaly_timeline.png)

![Anomaly zoom](assets/anomaly_zoom.png)

### What I noticed

The simplest detector won. On this machine, problems show up as long temperature drops, and a fixed limit set at the right level catches those early.

The two adaptive detectors kind of get used to a slow decline. Before the 8 Feb event they kept updating what "normal" looks like and only fired when the machine restarted, about 22 hours after the labelled anomaly. They were quicker with sudden changes, though: on 16 Dec they warned 18–21 hours ahead, compared to 12 hours for the fixed limit. So the methods complement each other, and combining them is what I'd try next. The main lesson for me was to always compare an ML model against a simple rule.

The data also needed some cleaning. On 7 Jan the sensor clock jumps back 55 minutes and repeats 12 timestamps, so the loader sorts the readings and averages duplicates.

<sub>Warning times are measured from the first alarm inside NAB's labelled window, which opens about 24 hours before each event, so about 24 hours is the most warning that can be measured.</sub>

## Daily maintenance worklist

`predict.py` scores new runs and suggests an action. The output is a CSV, so it could feed an Excel or Power BI dashboard.

```
$ python src/predict.py --input samples/new_runs.csv

product_id  type  speed_rpm  torque_nm  tool_wear_min  failure_risk  recommended_action
    L54191     L       2633       12.2             17         0.994  STOP & INSPECT - high failure risk
    L51841     L       1350       48.1             32         0.987  STOP & INSPECT - high failure risk
    L53258     L       1339       59.1            203         0.976  STOP & INSPECT - high failure risk
    M22885     M       1422       50.7            219         0.193  PLAN TOOL CHANGE - tool wear >= 205 min
    L52571     L       1572       34.4            228         0.132  PLAN TOOL CHANGE - tool wear >= 205 min
    L53382     L       1296       54.9             69         0.009  OK
    ...
5 of 12 runs need attention.
```

<sub>These sample rows come from the AI4I dataset the model was trained on, so this only shows how the tool works. The accuracy numbers are from the held-out test set above.</sub>

## Project structure

```
predictive-maintenance/
├── configs/costs.yaml              # cost assumptions (change them to your own numbers)
├── src/
│   ├── data.py                     # downloads both datasets, cleaning, extra features
│   ├── failure_prediction.py       # Part 1: models, cost threshold, test evaluation, charts
│   ├── anomaly_detection.py        # Part 2: three detectors, warning times, false alarms, charts
│   └── predict.py                  # score new runs -> maintenance worklist CSV
├── notebooks/walkthrough.ipynb     # the full analysis with explanations
├── models/                         # trained model + model card (threshold, costs, test scores)
├── results/                        # every table as CSV
├── assets/                         # charts used in this README
└── samples/new_runs.csv            # example input for predict.py
```

## Running it

```bash
pip install -r requirements.txt
python src/failure_prediction.py     # Part 1 (downloads the data on the first run, ~1 minute)
python src/anomaly_detection.py      # Part 2 (~10 seconds)
python src/predict.py                # score new runs with the trained model
```

If you want to try other cost assumptions, change `configs/costs.yaml` and rerun `failure_prediction.py`. The threshold, savings and break-even point are recalculated.

## Limitations and what I'd do next

- The costs are made up for illustration. Real numbers (line stop per minute, scrap, technician time) would change the threshold and the tool-change decision. The break-even table shows how sensitive that choice is.
- AI4I is synthetic. It's generated to resemble a real milling process, which is probably why the physics-based failures are so easy to predict. Real plant data will be noisier.
- One sensor with four labelled events is not much for comparing detectors. I'd want more machines and more failures before trusting the ranking.
- Next steps: combine the fixed limit with an adaptive detector, add remaining-useful-life prediction (for example on NASA's C-MAPSS turbofan data) and build a Power BI dashboard on top of the worklist CSV.

## Tech stack

Python, scikit-learn (random forest, gradient boosting, Isolation Forest), pandas, NumPy, matplotlib, Jupyter

## Credits and licences

- AI4I 2020 Predictive Maintenance Dataset: S. Matzka (2020), UCI Machine Learning Repository, CC BY 4.0. Downloaded from a GitHub mirror and checked against the published statistics (10,000 runs, 339 failures)
- Numenta Anomaly Benchmark (NAB): machine_temperature_system_failure, A. Lavin and S. Ahmad (2015), [github.com/numenta/NAB](https://github.com/numenta/NAB), MIT licence
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning)

Code under the MIT licence. The raw data is downloaded when you run the scripts and isn't stored in this repository.

Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck. [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/)
