"""
Exploratory Data Analysis — Breast Cancer Wisconsin Dataset
===========================================================
Run this first to understand the data before modelling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
os.makedirs("outputs", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
raw = load_breast_cancer()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)
df["diagnosis"] = pd.Categorical(
    df.assign(t=raw.target)["t"].map({0: "Malignant", 1: "Benign"})
)

PALETTE = {"Benign": "#2563EB", "Malignant": "#DC2626"}

# ── 1. Class distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Breast Cancer EDA", fontsize=13, fontweight="bold")

counts = df["diagnosis"].value_counts()
axes[0].bar(counts.index, counts.values,
            color=[PALETTE[x] for x in counts.index], edgecolor="white", width=0.5)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontsize=11)

# ── 2. Boxplots: top 6 discriminating features ────────────────────────────────
top6 = ["mean radius", "mean texture", "mean perimeter",
        "mean area", "mean smoothness", "mean concavity"]

df_top = df[top6 + ["diagnosis"]].melt(id_vars="diagnosis",
                                        var_name="feature", value_name="value")
axes[1].set_title("Distribution of Key Features")
for i, feat in enumerate(top6):
    sub = df[df.columns[df.columns.tolist().index(feat)]]
    # Use a text label instead of full boxplot to keep chart clean
axes[1].axis("off")
axes[1].text(0.5, 0.5, "See shap_summary.png\nfor feature importance",
             transform=axes[1].transAxes, ha="center", va="center",
             fontsize=11, color="gray")

# ── 3. PCA: 2D separation of classes ─────────────────────────────────────────
scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(raw.data)
pca        = PCA(n_components=2, random_state=42)
X_pca      = pca.fit_transform(X_scaled)
var_ratios = pca.explained_variance_ratio_

for label, color in [("Benign", "#2563EB"), ("Malignant", "#DC2626")]:
    mask = df["diagnosis"] == label
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=label, alpha=0.6, s=20, edgecolors="none")

axes[2].set_xlabel(f"PC1 ({var_ratios[0]*100:.1f}% var)")
axes[2].set_ylabel(f"PC2 ({var_ratios[1]*100:.1f}% var)")
axes[2].set_title("PCA: 2D Projection")
axes[2].legend()

plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/eda_overview.png")

# ── 4. Correlation heatmap (mean features only) ───────────────────────────────
mean_cols = [c for c in raw.feature_names if c.startswith("mean")]
corr       = df[mean_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
plt.title("Feature Correlation Matrix (Mean Features)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/eda_correlation.png")

# ── 5. Feature distributions by class ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat  = axes.flatten()

for i, feat in enumerate(top6):
    for label, color in PALETTE.items():
        mask = df["diagnosis"] == label
        axes_flat[i].hist(df.loc[mask, feat], bins=30, alpha=0.6,
                          color=color, label=label, edgecolor="none", density=True)
    axes_flat[i].set_title(feat, fontsize=9)
    axes_flat[i].set_ylabel("Density")
    if i == 0:
        axes_flat[i].legend(fontsize=8)

fig.suptitle("Feature Distributions by Diagnosis", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/eda_distributions.png")
print("\nEDA complete. Check the outputs/ folder.")
