# Heart Disease ML Classifier

Binary heart disease prediction using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). Built as part of MSc Cybersecurity coursework at Wrexham University.

## Overview

Compares three classifiers on 13 clinical features (age, cholesterol, chest pain type, etc.) to predict heart disease presence. Evaluation prioritises recall to minimise false negatives, as missed diagnoses carry higher clinical risk than false alarms.

**Models evaluated:**
- Logistic Regression (baseline, interpretable)
- Random Forest (100 trees)
- XGBoost (gradient boosted)

## Results

| Model | Accuracy (μ±σ) | F1 (μ±σ) | Recall (μ±σ) |
|---|---|---|---|
| Logistic Regression | 0.869 ±0.012 | 0.877 ±0.009 | 0.905 ±0.029 |
| Random Forest | 0.990 ±0.009 | 0.990 ±0.009 | 0.981 ±0.017 |
| XGBoost | **0.992 ±0.007** | **0.992 ±0.007** | **0.985 ±0.014** |

All metrics from stratified 5-fold cross-validation. While XGBoost achieves the highest accuracy, logistic regression shows superior probability calibration (Brier score: 0.092 vs 0.011 for XGBoost), making it more suitable where interpretability matters. McNemar's test confirms no statistically significant difference between Random Forest and XGBoost (p=0.479).

## Setup

Run in Google Colab or locally:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

Open `Bill_New_Task_2.ipynb` and run all cells. The UCI dataset is loaded automatically via `sklearn.datasets` or UCI's API.

## Key Findings

- XGBoost and Random Forest achieve near-perfect cross-validated accuracy but show overconfidence in calibration curves
- Logistic regression's lower accuracy (86.9%) is offset by well-calibrated probability outputs, relevant in clinical settings
- All three models significantly outperform each other pairwise except Random Forest vs XGBoost (p=0.479)
- Dataset: 1,025 patients, 51% positive class, no missing values
