# Predicting Parkinson's Disease Progression from Voice Recordings

## Overview

Parkinson's Disease (PD) is a complex neurological disorder affecting populations worldwide. Anticipating how the disease will progress is critical for planning effective treatment. Clinicians commonly use the **Unified Parkinson's Disease Rating Scale (UPDRS)** to assess both motor and non-motor symptoms.

This project investigates whether **voice recordings** can serve as a simple, non-invasive tool to predict a patient's **total UPDRS score**. Because voice-based assessment requires no physical examination, it opens the door to low-cost, remote, at-home monitoring — potentially enabling earlier detection and more frequent tracking of disease progression.

Feature engineering and data preparation techniques were applied to improve model performance, and both **regression** (predicting the continuous UPDRS score) and **classification** (predicting a UPDRS-derived category) approaches were explored.

## Dataset

- **File:** `parkinsons_updrs.csv`
- Contains biomedical voice measurements from PD patients, along with `motor_UPDRS` and `total_UPDRS` scores and a `subject#` identifier.
- **Target variable:** `total_UPDRS` (used directly for regression; binarized/binned for classification).
- Non-feature columns (`subject#`, `motor_UPDRS`, `total_UPDRS`) are dropped from the feature set `X` before modeling.

## Repository Contents

| File | Description |
|---|---|
| `Regression_ML_models.ipynb` | Trains and evaluates regression models to predict the continuous `total_UPDRS` score directly from voice features. |
| `Classification_ML_models.ipynb` | Converts `total_UPDRS` into discrete classes (e.g., healthy/mild PD, or Low/Medium/High severity via quantile binning) and trains classifiers to predict the category. |

## Methodology

1. **Data loading & exploration** – load `parkinsons_updrs.csv`, inspect features and target distribution.
2. **Target definition:**
   - *Regression:* `total_UPDRS` predicted as a continuous value.
   - *Classification:* `total_UPDRS` converted into classes — either a binary `mild_park` flag (threshold at UPDRS > 32) or multi-class Low/Medium/High bins via `KBinsDiscretizer`.
3. **Train/test split** – models are validated using a **70/30 train-test split** (test size 0.3; some experiments also use an 80/20 split for comparison).
4. **Feature scaling** – `StandardScaler` is applied where required (ANN, KNN).
5. **Model training** – each algorithm below is trained and evaluated independently.
6. **Evaluation:**
   - *Regression:* R² score, Mean Squared Error (MSE), and actual-vs-predicted scatter plots.
   - *Classification:* Accuracy, Precision, Recall, F1-score, AUC/Gini, and confusion matrices.

## Models Used

Both notebooks explore the same family of algorithms, applied to regression and classification versions of the problem respectively:

- **Random Forest** (Regressor / Classifier)
- **Naive Bayes** (Gaussian NB, with target discretization for the regression case)
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Network (ANN)** — built with TensorFlow/Keras (Dense layers; sigmoid output for classification, linear output for regression)
- **LightGBM** (classification notebook, additional comparison model)

## Results Summary

Using the **70/30 train-test split**, classification accuracy varied notably across models:

| Model | Accuracy |
|---|---|
| XGBoost | up to **99.23%** |
| Random Forest | ~92.25% |
| Naive Bayes | ~89.53% |
| KNN | ~81.1% |
| (lowest-performing model/setting) | ~41.27% |

**Key takeaways:**
- **XGBoost** consistently delivered the strongest performance and lowest prediction error across settings.
- Model performance (accuracy, precision, recall, F1-score) varied noticeably depending on **which model** was used and **how the data was split/validated**, underscoring the importance of robust validation strategy.
- Despite this variability, results support the overall potential of **voice-based, non-invasive tools** for PD monitoring — enabling more frequent, remote check-ins that could help detect symptom progression earlier than traditional in-clinic assessment.

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
3. Run all cells sequentially — each model section is self-contained (imports, data prep, training, evaluation).

## Future Work

- Standardize the train/test split and preprocessing steps across all models for a fairer comparison.
- Add cross-validation (e.g., k-fold) alongside the 70/30 split to check robustness of results.
- Perform systematic hyperparameter tuning (especially for the lower-performing models).
- Expand feature engineering and explore data augmentation to further improve generalization.
