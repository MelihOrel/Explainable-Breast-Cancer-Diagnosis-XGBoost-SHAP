"""
Hyperparameter Tuning with GridSearchCV + Stratified K-Fold
===========================================================
Run this after pipeline.py to find optimal XGBoost hyperparameters.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ── Load data ─────────────────────────────────────────────────────────────────
raw   = load_breast_cancer()
X, y  = raw.data, raw.target

# ── Define pipeline (scaler + model) ─────────────────────────────────────────
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb",    xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# ── Hyperparameter grid ────────────────────────────────────────────────────────
param_grid = {
    "xgb__n_estimators" : [100, 200, 300],
    "xgb__max_depth"    : [3, 4, 5],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample"    : [0.8, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8]
}

# ── 5-fold stratified CV ──────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    pipe, param_grid,
    cv      = cv,
    scoring = "roc_auc",     # optimise for AUC
    n_jobs  = -1,
    verbose = 1,
    refit   = True
)

print("Starting grid search (this may take a few minutes)...")
grid_search.fit(X, y)

print(f"\nBest AUC   : {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")

# ── Baseline cross-validation with best params ────────────────────────────────
best_model = grid_search.best_estimator_
cv_scores  = cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc")
print(f"\nCV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
