# Smart Energy Fraud Detection System using Machine Learning

**Samagra Gune | 25BCE10462 | B.Tech CSE — 1st Year | VIT Bhopal**

---

## Background

Electricity theft is a bigger problem than it sounds. In India alone, distribution companies lose thousands of crores every year to meter tampering, bypassed connections, and manipulated readings. The tricky part is catching it — manual inspection of every account isn't practical at scale.

The idea here is to use machine learning to flag suspicious accounts automatically based on their consumption patterns. Theft accounts tend to look different from normal ones: lower average usage, more erratic readings, and sometimes negative values when a meter has been reversed or bypassed entirely.

This project builds that detection system from scratch — data generation, feature extraction, model training, and evaluation.

---

## How it works

Since no real smart-meter dataset was available, synthetic data is generated for 300 user accounts over 30 days each.

**Normal users (210 accounts)**
Stable consumption drawn from a normal distribution centred around 15 units/day. Low variance — the kind of pattern you'd expect from a regular household.

**Theft users (90 accounts)**
Low average consumption (~4 units/day) with high variance. About 30% of these accounts also have at least one negative reading in the 30-day window — this simulates a bypassed or reversed meter, which is a classic physical tampering method.

Four features are extracted from each account's 30-day window:

| Feature | What it captures |
|---|---|
| `mean_use` | Average daily consumption |
| `std_use` | How erratic the usage is day-to-day |
| `min_use` | Lowest single-day reading (negative = strong theft signal) |
| `low_days` | Count of days where usage was below 0.5 units |

Two classifiers are trained on these features and compared — Logistic Regression as a simple baseline, and Random Forest as the main model.

---

## Project structure

```
electricity_theft_detection.py   main script
requirements.txt                 dependencies
README.md                        this file
```

The script does everything in one file: generates data, trains both models, prints evaluation metrics, and shows a scatter plot.

---

## Setup and running

Make sure you have Python 3.8 or above. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python electricity_theft_detection.py
```

### What you'll see

The script prints dataset info first, then accuracy and confusion matrices for both models:

```
(300, 5)
0    210
1     90

LR acc: 0.9333
RF acc: 0.9667

[[57  6]
 [ 0 27]]    <- confusion matrix for RF (rows: actual, cols: predicted)
```

Then feature importances from Random Forest — `mean_use` ends up being the most useful, followed by `min_use`. After that a scatter plot opens showing how the two groups separate in mean vs std space.

---

## Results

Random Forest consistently outperforms Logistic Regression on this data. LR struggles a bit because the boundary between normal and theft accounts isn't cleanly linear — RF handles that better with its ensemble of decision trees.

The most informative feature turned out to be `mean_use`, which makes sense since theft accounts systematically have lower consumption. `min_use` adds signal through the negative readings. `low_days` contributes the least but still helps at the margin.

---

## Limitations

- All data is synthetic. The model would need retraining on real smart-meter data before it could be used in practice.
- Only 4 features are used. Real deployments would likely use many more (time-of-day patterns, seasonal variation, billing history, etc.).
- Class imbalance isn't handled — in real datasets, theft accounts are a much smaller fraction than 30%.

---

## Possible extensions

- Use a real dataset (some state electricity boards have published anonymised billing data)
- Add time-series features like week-over-week change or peak/off-peak ratios
- Handle class imbalance with SMOTE or class weights
- Try XGBoost or a simple neural net for comparison
- Build a small CLI tool that takes a CSV of readings and outputs flagged accounts
