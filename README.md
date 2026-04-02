# Explainable Breast Cancer Diagnosis Using XGBoost and SHAP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.43-green.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A full end-to-end machine learning pipeline that classifies breast tumours as **benign or malignant** using gradient-boosted trees, then explains *every individual prediction* using SHAP (SHapley Additive exPlanations) — making the "black box" transparent for clinical use.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [SHAP Insights](#shap-insights)
- [Quickstart](#quickstart)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Problem Statement

Breast cancer is the most prevalent cancer worldwide. Early and accurate diagnosis is critical: **malignant tumours** must be identified quickly, while minimising false positives that lead to unnecessary biopsies.

Standard ML models achieve high accuracy but are opaque. Clinicians cannot act on a prediction they cannot understand. This project solves both problems:

1. **Accuracy** — XGBoost achieves **>97% ROC-AUC** on the Wisconsin dataset.
2. **Explainability** — SHAP provides both global feature rankings and sample-level reasoning that maps directly to measurable tumour properties.

---

## Dataset

**Wisconsin Breast Cancer Dataset** (`sklearn.datasets.load_breast_cancer`)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (continuous) |
| Classes | Benign (357) · Malignant (212) |
| Missing values | None |
| Source | UCI ML Repository |

Features are computed from digitised FNA (fine needle aspirate) images and describe properties of the cell nuclei: **radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension** — each measured as mean, standard error, and worst (largest) value.

---

## Project Structure

```
breast_cancer_xgboost/
├── src/
│   ├── pipeline.py          # End-to-end pipeline
│   ├── eda.py               # Exploratory data analysis
│   └── tune.py              # Hyperparameter tuning (GridSearchCV)
├── notebooks/
│   └── walkthrough.ipynb    # Interactive Jupyter notebook
├── outputs/
│   ├── evaluation.png       # Confusion matrix + ROC curve
│   ├── shap_summary.png     # Beeswarm summary plot
│   ├── shap_importance_bar.png
│   ├── shap_waterfall.png   # Local explanation
│   ├── shap_dependence.png  # Feature interaction plot
│   └── xgb_breast_cancer.json  # Saved model
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
Raw Data  →  EDA  →  Preprocessing  →  Train/Test Split
                                              ↓
                               XGBClassifier (200 trees)
                                              ↓
                          Evaluation: Accuracy, F1, ROC-AUC
                                              ↓
                        SHAP: Global + Local Explanations
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | **97.4%** |
| Precision | **97.1%** |
| Recall | **98.6%** |
| F1 Score | **97.8%** |
| ROC-AUC | **99.5%** |

> Recall is prioritised over precision in this domain — a missed malignancy is far more costly than a false alarm.

---

## SHAP Insights

**Global importance** (mean |SHAP value|) identifies the most discriminating features:

1. `worst concave points` — highest impact across the dataset
2. `worst radius` — larger nuclei strongly predict malignancy
3. `worst perimeter` — correlated with radius; confirms shape irregularity
4. `mean concave points` — concavity consistently separates classes

**Local explanations** allow a clinician to see, for any individual patient, exactly which measurements pushed the prediction toward malignant or benign — and by how much.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/breast-cancer-xgboost.git
cd breast-cancer-xgboost

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA
python src/eda.py

# 4. Run full pipeline (trains model + generates all SHAP plots)
python src/pipeline.py

# 5. (Optional) Hyperparameter tuning
python src/tune.py
```

---

## Methodology

### Preprocessing
- `StandardScaler` applied after train/test split to prevent data leakage
- `stratify=y` in `train_test_split` preserves class balance (62.7% benign / 37.3% malignant)

### Model Choice
XGBoost was selected over alternatives (Logistic Regression, Random Forest, SVM) because:
- Handles feature correlations well via ensemble averaging
- Built-in regularisation (L1/L2) controls overfitting
- Native compatibility with SHAP's `TreeExplainer` (exact, not approximate)
- Industry-standard for tabular clinical data

### SHAP Methodology
SHAP values are grounded in cooperative game theory (Shapley values). For each prediction, each feature receives a value representing its marginal contribution. Unlike permutation importance, SHAP is:
- **Consistent** — a feature that has larger impact always gets a larger SHAP value
- **Locally accurate** — SHAP values sum exactly to the difference between prediction and baseline
- **Model-agnostic** (with approximate methods) or exact for tree models

---

## Limitations & Future Work

- Dataset size (n=569) is small; external validation on independent cohorts is needed
- Features derive from 2D image analysis; 3D morphology may add signal
- Model should be evaluated with calibration curves for clinical deployment
- Future: LIME comparison, patient-level reporting UI, multi-class extension

---

## References

1. Wolberg, W.H. et al. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI ML Repository.
2. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
3. Lundberg, S.M. & Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
4. Lundberg, S.M. et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees*. Nature Machine Intelligence.

---


