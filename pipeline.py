"""
Explainable Breast Cancer Diagnosis Using XGBoost and SHAP
==========================================================
Full end-to-end pipeline: load → preprocess → train → evaluate → explain.
"""

# ─── 1. IMPORTS ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

import xgboost as xgb
import shap

# ─── 2. CONFIGURATION ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.20
OUTPUT_DIR    = "outputs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 3. DATASET LOADING & EXPLORATION ─────────────────────────────────────────
def load_and_explore():
    """Load Wisconsin Breast Cancer dataset and print key statistics."""
    raw = load_breast_cancer()

    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"] = raw.target                        # 0 = malignant, 1 = benign
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

    print("=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Samples   : {df.shape[0]}")
    print(f"  Features  : {df.shape[1] - 2}")
    print(f"  Benign    : {(df.target == 1).sum()} ({(df.target==1).mean()*100:.1f}%)")
    print(f"  Malignant : {(df.target == 0).sum()} ({(df.target==0).mean()*100:.1f}%)")
    print(f"\n  Missing values: {df.isnull().sum().sum()}")
    print(f"\n  Feature value ranges (mean ± std):")
    for col in raw.feature_names[:5]:
        print(f"    {col:<40} {df[col].mean():.3f} ± {df[col].std():.3f}")
    print("    ... (30 features total)")

    return df, raw


# ─── 4. PREPROCESSING ─────────────────────────────────────────────────────────
def preprocess(df):
    """Split into X/y, apply stratified train-test split, scale features."""
    X = df.drop(columns=["target", "diagnosis"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = TEST_SIZE,
        random_state= RANDOM_STATE,
        stratify    = y          # preserve class balance in both sets
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit only on train
    X_test_s  = scaler.transform(X_test)        # apply same transform

    print(f"\n  Train set : {X_train.shape[0]} samples")
    print(f"  Test  set : {X_test.shape[0]}  samples")

    return X_train_s, X_test_s, y_train, y_test, X.columns.tolist()


# ─── 5. MODEL TRAINING ────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """
    Train XGBClassifier.

    Hyperparameter rationale:
    - n_estimators=200   : enough trees to capture complex patterns without
                           overfitting on this ~455-sample training set
    - max_depth=4        : shallow trees reduce variance; breast cancer features
                           are already informative so depth >5 rarely helps
    - learning_rate=0.05 : low LR + more trees = better generalisation (shrinkage)
    - subsample=0.8      : stochastic gradient boosting reduces overfitting
    - colsample_bytree=0.8: random feature subsets, akin to Random Forest trick
    - scale_pos_weight   : handles mild class imbalance (benign > malignant)
    - eval_metric='logloss': standard for binary classification
    - use_label_encoder=False: suppress deprecation warning
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    model = xgb.XGBClassifier(
        n_estimators        = 200,
        max_depth           = 4,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        scale_pos_weight    = n_neg / n_pos,
        eval_metric         = "logloss",
        use_label_encoder   = False,
        random_state        = RANDOM_STATE,
        n_jobs              = -1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)

    print("\n  Model training complete.")
    return model


# ─── 6. EVALUATION ────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, feature_names):
    """Print full classification report and save ROC + Confusion Matrix plots."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUATION METRICS")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Malignant (0)", "Benign (1)"]))

    # --- Plot: Confusion Matrix + ROC Curve side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # Confusion Matrix
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Malignant", "Benign"])
    axes[0].set_yticklabels(["Malignant", "Benign"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm[i, j] > cm.max()/2 else "black")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve"); axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/evaluation.png")

    return {"accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


# ─── 7. SHAP EXPLAINABILITY ───────────────────────────────────────────────────
def shap_analysis(model, X_train, X_test, feature_names):
    """
    Full SHAP analysis:
      a) TreeExplainer for XGBoost (fast, exact)
      b) Global importance (mean |SHAP|)
      c) Summary beeswarm plot
      d) Local explanation for one high-risk sample
      e) Waterfall / force plot
      f) Dependence plot for top feature
    """
    print("\n" + "=" * 60)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)   # (n_samples, n_features)
    expected_val = explainer.expected_value

    # ── a) Global Feature Importance ──────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "mean_|shap|": mean_abs_shap
    }).sort_values("mean_|shap|", ascending=False)

    print("\n  Top 10 most important features (mean |SHAP|):")
    print(importance_df.head(10).to_string(index=False))

    # ── b) Summary Plot ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Summary Plot — Feature Impact on Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/shap_summary.png")

    # ── c) Bar Plot: Global Importance ────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/shap_importance_bar.png")

    # ── d) Local Explanation: one malignant sample ─────────────────────────────
    # Find first test sample predicted as malignant (high risk)
    preds      = model.predict(X_test)
    mal_idx    = np.where(preds == 0)[0][0]   # index of first malignant prediction
    sample     = X_test[mal_idx]
    sv_sample  = shap_values[mal_idx]

    print(f"\n  Local explanation for test sample #{mal_idx} (predicted: Malignant)")

    # ── e) Waterfall Plot ─────────────────────────────────────────────────────
    shap_exp = shap.Explanation(
        values         = sv_sample,
        base_values    = expected_val,
        data           = sample,
        feature_names  = feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, max_display=12, show=False)
    plt.title(f"Waterfall Plot — Sample #{mal_idx} (Malignant)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/shap_waterfall.png")

    # ── f) Dependence Plot: top feature ───────────────────────────────────────
    top_feature  = importance_df.iloc[0]["feature"]
    top_idx      = feature_names.index(top_feature)
    interact_idx = importance_df.iloc[1]["feature"]   # auto-colour by 2nd feature

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        top_idx, shap_values, X_test,
        feature_names     = feature_names,
        interaction_index = feature_names.index(interact_idx),
        show              = False,
        alpha             = 0.7
    )
    plt.title(f"SHAP Dependence: '{top_feature}'\n(coloured by '{interact_idx}')", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/shap_dependence.png")

    return importance_df


# ─── 8. MAIN ──────────────────────────────────────────────────────────────────
def main():
    df, raw          = load_and_explore()
    X_tr, X_te, y_tr, y_te, feat_names = preprocess(df)
    model            = train_model(X_tr, y_tr)
    metrics          = evaluate(model, X_te, y_te, feat_names)
    importance_df    = shap_analysis(model, X_tr, X_te, feat_names)

    # Save model
    model.save_model(f"{OUTPUT_DIR}/xgb_breast_cancer.json")
    importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model  → {OUTPUT_DIR}/xgb_breast_cancer.json")
    print(f"  Plots  → {OUTPUT_DIR}/")
    print(f"\n  Final ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Final F1      : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
