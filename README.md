# RETINA AI — Student Dropout Risk Prediction

Predicting student dropout risk (**Low / Medium / High**) using academic records, attendance history, and counsellor notes. Built for the RETINA AI Hackathon.

## Overview

Early identification of at-risk students allows institutions to intervene before a student disengages or drops out. This project builds a machine learning pipeline that classifies each student into one of three risk categories based on their academic, behavioral, and qualitative (counsellor note) history.

- **Final Macro F1 (10-fold OOF):** 0.7161
- **Overall Accuracy:** 76%
- **Classes:** Low Risk, Medium Risk, High Risk

## Problem Understanding

Student dropout is rarely caused by a single factor — it is typically the result of a combination of declining attendance, weakening academic performance, and behavioral or emotional warning signs noted by counsellors. The challenge is therefore inherently **multi-modal** (numeric, time-series, and text data) and **imbalanced** (most students are Low risk, with progressively fewer Medium and High risk students).

| Class | Description | Count (train) |
|---|---|---|
| Low Risk | 0 | 7,200 |
| Medium Risk | 1 | 3,000 |
| High Risk | 2 | 1,800 |

## Datasets

| File | Description | Shape |
|---|---|---|
| `train.csv` | Student records with target label | (12,000, 18) |
| `test.csv` | Student records without target label | (3,000, 17) |
| `Attendance_series.csv` | Weekly attendance log per student/subject/semester | (1,048,575, 5) |
| `Counsellor_notes.csv` | Free-text counsellor notes per student | (15,000, 2) |

## Approach

### 1. Data Preprocessing

- Missing values handled via **median imputation** (`SimpleImputer`, with `keep_empty_features=True` to preserve sparse/empty columns)
- Categorical features encoded with `LabelEncoder`
- Numerical features scaled with `RobustScaler` to reduce the influence of outliers
- Train/test alignment to ensure consistent feature columns across both sets

### 2. Feature Engineering (144 final features)

**Attendance features**
- Global statistics: mean, std, min, max, median, quantiles, skew, kurtosis
- Threshold-based features: percentage of weeks below 75% / 60% / 50% / 40% attendance
- Attendance trend (linear slope over time) and semester-over-semester decline
- Consecutive absence streaks (longest streak, streak count, most recent streak)
- Per-subject attendance breakdown (best/worst subject, subject spread)
- Volatility measures (rolling drops, semester-level standard deviation)

**NLP features (from counsellor notes)**
- Sentiment analysis via VADER (compound, positive, negative, neutral scores)
- Custom risk-word and positive-word lexicon counts and ratios
- Text statistics: word count, lexical diversity, exclamation/question usage
- TF-IDF (1-3 grams) + TruncatedSVD (60 components), **fit only on training-set students** to prevent data leakage
- Note frequency and length statistics

**Tabular / academic features**
- Grade statistics: mean, std, min, max, range, trend, failing/warning counts
- Cross-feature interactions: grade × attendance, grade × sentiment, attendance × risk-word frequency, combined decline indicators

### 3. Model Architecture

A **stacked ensemble** of 7 base models:

| Model | Role |
|---|---|
| LightGBM (×2 configs) | Gradient boosting, tuned via Optuna |
| XGBoost (×2 configs) | Gradient boosting, alternate regularization |
| ExtraTrees | Variance reduction via extreme randomization |
| RandomForest | Bagged decision trees |
| CatBoost | Gradient boosting with explicit class weighting |

**Level-2 meta-learners:** LightGBM + Logistic Regression, trained on out-of-fold predictions (probabilities + hard predictions) from the base models.

**Class imbalance handling:** SMOTE oversampling applied per-fold (re-fitting the scaler on augmented data to avoid leakage), plus class-weighted loss functions across all base models.

**Pseudo-labeling:** High-confidence test predictions (probability ≥ 0.85) are added back into training data to refine the final model.

**Hyperparameter tuning:** LightGBM hyperparameters tuned via Optuna (40 trials, 5-fold CV, optimizing macro F1).

### 4. Evaluation Methodology

- **Stratified 10-fold cross-validation** throughout, optimizing macro F1 to fairly weight all three risk classes (since the dataset is imbalanced)
- Final predictions are a weighted blend of direct ensemble averaging and stacked meta-learner outputs
- Decision thresholds for Medium/High risk probabilities are tuned post-hoc to maximize macro F1 without affecting the base evaluation

## Results

```
              precision    recall  f1-score   support

    Low Risk       0.90      0.84      0.87      7200
 Medium Risk       0.53      0.56      0.54      3000
   High Risk       0.68      0.80      0.74      1800

    accuracy                           0.76     12000
   macro avg       0.70      0.73      0.72     12000
weighted avg       0.77      0.76      0.77     12000
```

**Observations:**
- The model performs strongly on **Low Risk** and **High Risk** classification, with High Risk recall reaching **0.80** — important for a dropout-prevention system, since failing to flag a genuinely at-risk student is the costliest type of error.
- **Medium Risk** remains the hardest class to separate (F1 = 0.54), which is expected given it sits between the other two categories on a continuous risk spectrum.
- Pseudo-labeling and threshold tuning provided incremental improvements to the final macro F1.

## Repository Structure

```
.
├── model.py              # Final model pipeline (feature engineering + ensemble + stacking)
├── submission_v3.csv      # Final predictions on test set
├── README.md              # This file
```

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna imbalanced-learn nltk scipy
   ```

2. Place the dataset files (`train.csv`, `test.csv`, `Attendance_series.csv`, `Counsellor_notes.csv`) in the project directory.

3. Run the pipeline:
   ```bash
   python model.py
   ```

4. The final predictions will be saved to `submission_v3.csv`.

## Technologies Used

- **Python** (pandas, numpy, scipy)
- **Machine Learning:** scikit-learn, LightGBM, XGBoost, CatBoost
- **NLP:** NLTK (VADER sentiment), TF-IDF, TruncatedSVD
- **Tuning:** Optuna
- **Imbalance handling:** imbalanced-learn (SMOTE)

## Conclusion

This project demonstrates that combining structured academic/attendance data with unstructured counsellor notes — through careful feature engineering and a diverse stacked ensemble — produces a robust dropout risk classifier. The model's strength in identifying High Risk students makes it particularly suited for early-intervention use cases, while the remaining challenge in Medium Risk classification highlights an area for future work, such as ordinal-aware modeling approaches.

## Author

- **Name:** Shivam Kumar
- **Kaggle Username:** shivamkumarcse
- **Branch & Year:** CSE-3rd year
