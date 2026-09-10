# Predicting Parkinson's Disease Progression from Voice Recordings

## Overview

Parkinson's Disease (PD) is a complex neurological disorder affecting populations worldwide. Anticipating how the disease will progress is critical for planning effective, timely treatment. Clinicians commonly use the **Unified Parkinson's Disease Rating Scale (UPDRS)** to assess both motor and non-motor symptoms.

This project investigates whether **sustained-vowel voice recordings** can serve as a simple, non-invasive proxy for a patient's **total UPDRS score**. Because voice-based assessment requires no physical examination, it opens the door to low-cost, remote, at-home monitoring — potentially enabling earlier detection and more frequent tracking of disease progression than periodic in-clinic visits allow.

Two complementary modeling angles are explored:
- **Regression** — predict the continuous `total_UPDRS` score directly.
- **Classification** — bucket `total_UPDRS` into discrete severity classes and predict the class.

Feature engineering, target discretization, and feature scaling are used throughout to improve model performance, and models are compared using a **70/30 train-test split** (with 80/20 also used in some sections for cross-checking).

## Dataset

- **File:** `parkinsons_updrs.csv`
- **Size:** 5,875 voice recordings collected from PD patients (multiple recordings per subject over time via `test_time`).
- **Identifier:** `subject#`
- **Targets:** `motor_UPDRS` and `total_UPDRS` (clinician-assessed UPDRS scores).
- **Demographic columns:** `age`, `sex`
- **Voice/acoustic features (16):**
  - Frequency variation: `Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`
  - Amplitude variation: `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA`
  - Noise/harmonicity: `NHR`, `HNR`
  - Nonlinear dynamics: `RPDE`, `DFA`, `PPE`
- **Modeling feature set (`X`):** `age`, `sex`, `test_time`, + the 16 voice features above = **19 features**. `subject#`, `motor_UPDRS`, and `total_UPDRS` are dropped from `X` (the latter two because they leak the target).

### Derived classification targets
| Target | Definition | Class balance |
|---|---|---|
| `mild_park` (binary) | `1` if `total_UPDRS > 32`, else `0` | 3,800 healthy/mild (0) vs. 2,075 more advanced (1) |
| Severity class (3-way) | `total_UPDRS` split into **Low / Medium / High** via `KBinsDiscretizer` (quantile strategy, 3 bins) | roughly balanced (~1,900–1,960 per class) |

## Repository Contents

| File | Description |
|---|---|
| `Regression_ML_models.ipynb` | Predicts the continuous `total_UPDRS` score directly from voice/demographic features. |
| `Classification_ML_models.ipynb` | Predicts PD severity category (`mild_park` binary flag, or Low/Medium/High class) from the same feature set. |

## Methodology

1. **Data loading & exploration** — load `parkinsons_updrs.csv`, inspect head/columns/distribution.
2. **Target definition:**
   - *Regression:* `total_UPDRS` used as-is (continuous).
   - *Classification:* `total_UPDRS` converted into `mild_park` (binary, threshold 32) or a 3-class Low/Medium/High label via quantile binning.
3. **Train/test split:** `train_test_split` with `random_state=42`; **test_size = 0.3 (70/30 split)** used as the primary validation approach, with `test_size = 0.2` also used in several sections for comparison.
4. **Feature scaling:** `StandardScaler` applied ahead of scale-sensitive models (ANN, KNN, LightGBM run).
5. **Target discretization for regression-via-NB:** Gaussian Naive Bayes can't regress directly, so `total_UPDRS` is binned into 15 uniform bins with `KBinsDiscretizer`; predictions are reconstructed as a probability-weighted average of bin centers.
6. **Model training:** each algorithm below is trained and evaluated independently per notebook section.
7. **Evaluation metrics:**
   - *Regression:* R² score, Mean Squared Error (MSE), actual-vs-predicted scatter plots.
   - *Classification:* Accuracy, Precision, Recall, F1-score, ROC-AUC, Gini coefficient, confusion matrix / classification report.

## Models Used

| Model | Regression notebook | Classification notebook | Notes |
|---|---|---|---|
| Random Forest | ✅ `RandomForestRegressor` / `RandomForestClassifier` | ✅ `RandomForestClassifier` | Also tuned with `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features='sqrt'` in one variant |
| Naive Bayes | ✅ `GaussianNB` on binned target | ✅ `GaussianNB` on `mild_park` | Regression version needs target discretization (15 bins) |
| XGBoost | ✅ `XGBRegressor` | ✅ `XGBClassifier` | Best-performing model in both tasks |
| KNN | ✅ `KNeighborsRegressor` (k=5) | ✅ `KNeighborsClassifier` (3-class, scaled features) | |
| ANN (TensorFlow/Keras) | ✅ Dense(64)–Dense(32)–Dense(1), linear output, Adam/MSE, 100 epochs | ✅ Dense(32)–Dense(16)–Dense(1), sigmoid output | Features standardized before training |
| LightGBM | — | ✅ `LGBMClassifier` | Extra comparison model added only in the classification notebook |

## Results

### Regression — predicting `total_UPDRS` directly (test_size = 0.2, unless noted)

| Model | R² | MSE |
|---|---|---|
| Naive Bayes (binned) | 0.363 | 70.60 |
| KNN (k=5) | 0.513 | 53.97 |
| ANN | 0.770 | 25.07 |
| XGBoost | *(R² not separately captured — see MSE)* | **1.85** |

XGBoost produced by far the lowest MSE of the regression models tested, followed by the ANN. Naive Bayes performed weakest, consistent with its target-discretization workaround losing precision.

### Classification — predicting `mild_park` (binary, test_size = 0.2 / 0.3 as noted)

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|
| **XGBoost** | **99.23%** | 99.50% | 98.28% | 98.89% | 0.9996 |
| LightGBM | 99.40% | ~99% | — | — | — |
| Random Forest | 92.26% | 95.93% | 81.08% | 87.88% | 0.983 |
| ANN | 89.53% | 86.22% | 83.05% | 84.61% | 0.955 |
| Naive Bayes | 41.28% | 35.84% | 87.96% | 50.92% | 0.574 |

### Classification — 3-class severity (Low / Medium / High, via `KBinsDiscretizer`)

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|
| KNN | 81.11% | 81.11% | 81.11% | 81.10% |

**Observations:**
- **XGBoost and LightGBM (gradient-boosted trees) were the clear top performers** on both the regression and classification tasks — XGBoost hit up to 99.23% accuracy / AUC ≈ 1.0 on the binary classification task and the lowest regression MSE (1.85).
- **Random Forest and the ANN** were strong second-tier performers (~89–92% accuracy on binary classification).
- **Naive Bayes performed poorly** on both tasks (41.3% classification accuracy, R² of 0.36 on regression) — its independence assumption doesn't hold well for these correlated acoustic features, and the target-binning workaround for regression adds further error.
- **KNN** gave moderate results on both tasks (~81% for 3-class severity, R² of 0.51 for regression).
- Metrics shifted somewhat between the 70/30 and 80/20 splits and between binary vs. 3-class targets, showing that **reported performance is sensitive to validation strategy and how the target is framed**, not just the choice of algorithm.
- Despite this variability, the results support the overall premise: **voice-based, non-invasive tools show strong potential for remote PD monitoring**, with gradient-boosted tree models in particular offering high predictive accuracy for symptom severity from vocal features alone.

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
tensorflow
matplotlib
seaborn
```

## How to Run

1. Open either notebook in Google Colab or Jupyter.
2. Ensure `parkinsons_updrs.csv` is available in the working directory (or upload it to your Colab session).
3. Run all cells sequentially — each model section is largely self-contained (imports, data prep, training, evaluation) but several sections reuse `df`/`X`/`X_train` etc. defined in earlier cells, so running top-to-bottom is recommended.

## Limitations & Future Work

- **Standardize preprocessing across models:** feature scaling, split ratio, and random seed vary somewhat between sections; a single shared pipeline would make model comparisons more rigorous.
- **Add cross-validation** (e.g., stratified k-fold) alongside the single 70/30 / 80/20 holdout splits to check the stability of reported metrics.
- **Systematic hyperparameter tuning** (grid/random/Bayesian search), especially for the weaker models (Naive Bayes, KNN).
- **Data leakage check:** confirm that `total_UPDRS`/`motor_UPDRS` derived labels and voice features from the same subject don't leak across the train/test split (recordings are repeated per subject over time), e.g. via subject-level (grouped) splitting.
- **Expand feature engineering and data augmentation** — the project description notes these were used to improve prediction; documenting the specific techniques applied would strengthen reproducibility.
- **Explainability:** build on the Random Forest `feature_importances_` output already computed to identify which acoustic features (e.g., jitter, shimmer, PPE) are most predictive of PD severity.
