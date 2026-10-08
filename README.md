# Data Mining Techniques: Assignments 1 & 2 (Group 62)

Coursework for the Data Mining Techniques course at Vrije Universiteit Amsterdam. This repository contains the code, experiments and write-ups for two applied data mining projects, both following the CRISP-DM process model:

| | Assignment | Task | Data | Models |
|---|---|---|---|---|
| 1 | Data Mining Practice and Theory | Predict next-day mood (classification and regression) | Smartphone sensor + self-report data, 27 participants | XGBoost, LSTM |
| 2 | A Real Life Competition: Hotel Ranking | Rank hotels per search by booking likelihood | Expedia ICDM 2013 Kaggle dataset | KNN (baseline), LightGBM LambdaMART |

**Authors (Group 62):** Anna De Martin, Alicja Dorobis, David Zantovský

---

## Assignment 1: Mood Prediction from Smartphone Data

### Goal
Predict a participant's average mood on the following day from longitudinal smartphone data collected to support research on mood changes and depression monitoring (Asselbergs et al., 2016). The problem is framed both as a **3-class classification** task and as a **regression** task.

### Data
- 376,912 rows in long format, 19 variables, 27 participants, February to June 2014.
- Self-reports: mood (1-10), circumplex arousal and valence (-2 to +2).
- Passive sensors: screen time, app usage in 12 categories, activity score, call and SMS events.
- The raw data is **not included** in this repository (see [Data](#data-not-included)).

### Pipeline
1. **Exploratory analysis**: variable distributions, sampling frequencies, outliers, missingness patterns, correlation matrix.
2. **Cleaning**: removal of physically impossible negative durations; aggregation to one row per participant per day (1,973 rows, 40 columns); removal of data preceding long recording gaps, leaving 1,268 participant-days.
3. **Imputation**: zero-fill for app, screen, call and SMS variables; linear interpolation (gaps up to 3 days) for mood, arousal, valence and activity, chosen over LOCF to avoid flat plateaus.
4. **Feature engineering**: 5-day sliding window per participant, with mean/std/min/max, OLS slope and exponential moving average for self-reports, absolute and normalised (proportion) app usage, and day-of-week and month dummies. This yields 81 features over 1,154 training instances.
5. **Targets**: next-day mood. For classification, quantile-based thresholds (q33 = 6.8, q66 = 7.25) give three roughly balanced classes (low, neutral, high).
6. **Evaluation**: chronological per-participant split (80/20 train/test; 70/15/15 for the LSTM), `TimeSeriesSplit` cross-validation, randomized hyperparameter search for XGBoost, grid search (32 configurations) for the LSTM.

### Results

**Classification** (macro-F1 is the primary metric)

| Model | Accuracy | Macro F1 | AUC | Cohen's κ |
|---|---|---|---|---|
| XGBoost | 0.531 | 0.508 | 0.734 | 0.292 |
| LSTM (normalised app usage) | 0.554 | 0.556 | 0.729 | 0.333 |
| LSTM (absolute app usage) | 0.491 | 0.488 | 0.700 | 0.236 |

**Regression**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 0.432 | 0.597 | 0.383 |
| LSTM (normalised app usage) | 0.622 | 0.853 | -0.114 |
| LSTM (absolute app usage) | 0.638 | 0.786 | -0.054 |

**Key findings**
- Past mood (the 5-day EMA of mood) is by far the strongest predictor in both tasks.
- The neutral class is the hardest to classify for both models.
- XGBoost clearly wins on regression; the LSTM underperforms the mean baseline (negative R²), consistent with the difficulty of training recurrent networks on only 27 participants.
- The report also covers a case study of a winning Kaggle solution (Child Mind Institute, Problematic Internet Use), a discussion of association rule mining (Apriori and hierarchical negative rules), and a comparison of MSE and MAE as evaluation metrics.

---

## Assignment 2: Expedia Hotel Ranking

### Goal
Given a user's hotel search on Expedia, rank the returned properties in descending order of booking likelihood. This is the setting of the 2013 ICDM / Kaggle "Personalize Expedia Hotel Searches" competition. Performance is measured with **NDCG@5**, with relevance defined as 5 for a booking, 1 for a click only, and 0 otherwise.

### Data
- 4,958,347 training rows (199,795 searches, 129,113 properties) and 4,959,183 test rows.
- Very sparse user-history and competitor columns; extremely low positive rates (4.47% clicks, 2.79% bookings).
- The raw data is **not included** in this repository (see [Data](#data-not-included)).

### Pipeline
1. **Memory-aware loading**: dtype downcasting for the ~5M-row dataset.
2. **Preprocessing**: missing values kept as `NaN` with explicit missingness flags for LightGBM; zero or destination-median imputation for KNN; the 24 competitor columns collapsed into two flags (`is_cheapest_any`, `is_expensive_any`); `gross_bookings_usd` dropped to prevent leakage.
3. **Feature engineering**:
   - *Within-search* features (grouped by `srch_id`): price and quality ranks, ratios to the search mean, user-history vs. hotel match, quality interactions and a star-rating vs. review-score mismatch signal.
   - *Hotel performance* features (grouped by `prop_id`): price statistics, historical click and booking rates, popularity, and an estimated-position feature.
4. **Validation**: 80/20 split at the **search level**, so no search appears in both train and validation.
5. **Models**:
   - **KNN baseline**: scikit-learn pipeline with RobustScaler, Manhattan distance and distance weighting; score = P(booked) + 0.1 × P(clicked); k chosen from a dedicated sweep (k = 400).
   - **LightGBM LambdaMART** (`LGBMRanker`, `lambdarank` objective): learning rate 0.05, 255 leaves, `min_child_samples` 50, `label_gain=[0,1,5]`, L1/L2 regularisation, column subsampling 0.7, early stopping.

### Results

| Model | CV NDCG@5 | Validation NDCG@5 | Kaggle NDCG@5 |
|---|---|---|---|
| KNN (baseline) | 0.3479 | 0.3366 | n/a |
| LightGBM LambdaMART | n/a | 0.4106 | 0.41058 |

### Bias analysis: position and promotion bias
- Booking rates fall sharply with display position, and promoted hotels are displayed higher on average (mean position 14.8 vs. 17.4).
- A **conditional logistic regression** grouped by search shows the promotion odds ratio dropping from 2.151 (raw) to 1.951 (quality controls) to 1.119 once log-position is controlled for. A random-sort robustness check gives 1.729, so a modest genuine promotion effect remains.
- KNN ranked promoted hotels about 3.5 positions higher and could not be fixed by post-processing, due to its zero-inflated scores.
- For LightGBM, feeding display position into training (inverse-propensity-style correction) reduced the mean score gap between top and low position buckets from +1.193 to +0.739 and the promoted-hotel rank advantage from 4.948 to 4.015 positions, at a small cost in NDCG@5 (0.4108 to 0.4026).

---

## Setup

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Library versions are pinned in `requirements.txt` and random seeds are fixed to make the experiments reproducible.

## Data (not included)

The datasets are not redistributed here.

- **Assignment 1:** the mood and smartphone dataset provided by the course (Asselbergs et al., 2016). Place the file in `assignment1_mood_prediction/data/`.
- **Assignment 2:** the Expedia dataset from the Kaggle competition [Personalize Expedia Hotel Searches, ICDM 2013](https://www.kaggle.com/expedia-personalized-sort). Place `training_set_VU_DM.csv` and `test_set_VU_DM.csv` in `assignment2_hotel_ranking/data/`.

## Running the Code

> Replace the placeholders below with the actual scripts or notebooks.

```bash
# Assignment 1
python assignment1_mood_prediction/src/<preprocess_script>.py
python assignment1_mood_prediction/src/<train_xgboost_script>.py
python assignment1_mood_prediction/src/<train_lstm_script>.py

# Assignment 2
python assignment2_hotel_ranking/src/<feature_engineering_script>.py
python assignment2_hotel_ranking/src/<train_knn_script>.py
python assignment2_hotel_ranking/src/<train_lightgbm_script>.py
```

Assignment 2 is computationally heavy (~5M rows); a machine with substantial RAM or access to a compute cluster is recommended.


## Acknowledgements and Key References

- Asselbergs et al. (2016). Mobile phone-based unobtrusive ecological momentary assessment of day-to-day mood. *J. Med. Internet Res.*
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD.*
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation.*
- Liu et al. (2013). Combination of Diverse Ranking Models for Personalized Expedia Hotel Searches.
- Burges (2010). From RankNet to LambdaRank to LambdaMART: An Overview.
- Craswell et al. (2008). An experimental comparison of click position-bias models. *WSDM.*

The full reference lists are in each assignment's report.

## License

It is a shared for portfolio for academic purposes only.
