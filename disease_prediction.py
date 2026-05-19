# ============================================================
#   Disease Prediction from Medical Data
#   CodeAlpha Machine Learning Internship — Task 4
#   Author: Amr Hany
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ── Styling ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccff",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "text.color":       "#ffffff",
    "grid.color":       "#2a2a4a",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
})

PALETTE = ["#7b68ee", "#00d4aa", "#ff6b9d", "#ffd166"]
ACCENT  = "#7b68ee"

# ══════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATA
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  DISEASE PREDICTION — Breast Cancer Dataset")
print("=" * 60)

raw   = load_breast_cancer()
df    = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target          # 0 = Malignant, 1 = Benign

print(f"\n📦 Dataset shape  : {df.shape}")
print(f"🎯 Target classes : {dict(zip(raw.target_names, np.bincount(raw.target)))}")
print(f"\n📊 Sample preview :")
print(df.head(3).to_string())
print(f"\n🔍 Missing values : {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════
# 2. EDA — Figure 1
# ══════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.patch.set_facecolor("#0f0f1a")
fig1.suptitle("Exploratory Data Analysis", fontsize=16, color="white", fontweight="bold", y=1.02)

# 2a. Class distribution
ax = axes[0]
counts = df["target"].value_counts()
bars = ax.bar(["Malignant", "Benign"], counts.values, color=[PALETTE[2], PALETTE[1]], width=0.5, edgecolor="none")
ax.set_title("Class Distribution", color="white")
ax.set_ylabel("Count")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
            ha="center", color="white", fontsize=12, fontweight="bold")

# 2b. Feature correlation heatmap (top 10 features)
ax = axes[1]
top10 = df.corr()["target"].abs().nlargest(11).index.tolist()
top10.remove("target")
corr_matrix = df[top10 + ["target"]].corr()
sns.heatmap(corr_matrix, ax=ax, cmap="RdPu", linewidths=0.3,
            linecolor="#0f0f1a", annot=False, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap\n(Top 10 Features vs Target)", color="white")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)

# 2c. Feature distribution for top feature
top_feat = df.corr()["target"].abs().nlargest(2).index[1]
ax = axes[2]
for label, color, name in zip([0, 1], [PALETTE[2], PALETTE[1]], ["Malignant", "Benign"]):
    ax.hist(df[df["target"] == label][top_feat], bins=30, alpha=0.7,
            color=color, label=name, edgecolor="none")
ax.set_title(f"Distribution: {top_feat[:30]}", color="white", fontsize=9)
ax.set_xlabel(top_feat[:30], fontsize=8)
ax.legend()

plt.tight_layout()
plt.savefig("/home/claude/plot_eda.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("\n✅ EDA plot saved.")

# ══════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n✂️  Train size : {X_train.shape[0]} | Test size : {X_test.shape[0]}")

# ══════════════════════════════════════════════════════════
# 4. MODELS
# ══════════════════════════════════════════════════════════
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, random_state=42,
                                         eval_metric="logloss", verbosity=0),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "─" * 60)
print(f"{'Model':<25} {'CV Acc':>8} {'Test Acc':>10} {'AUC':>8}")
print("─" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    acc       = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_proba)

    results[name] = {
        "model":    model,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "accuracy": acc,
        "auc":      auc,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "report":   classification_report(y_test, y_pred,
                        target_names=["Malignant", "Benign"], output_dict=True),
    }
    print(f"{name:<25} {cv_scores.mean():.4f}   {acc:.4f}     {auc:.4f}")

print("─" * 60)

best_name  = max(results, key=lambda k: results[k]["auc"])
best       = results[best_name]
print(f"\n🏆 Best Model : {best_name}  (AUC = {best['auc']:.4f})")

# ══════════════════════════════════════════════════════════
# 5. RESULTS — Figure 2 (4 subplots)
# ══════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 16))
fig2.patch.set_facecolor("#0f0f1a")
fig2.suptitle("Model Evaluation Results", fontsize=18, color="white",
              fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.35)

model_names = list(results.keys())
colors_bar  = PALETTE

# 5a. Accuracy comparison
ax1 = fig2.add_subplot(gs[0, 0])
accs = [results[m]["accuracy"] for m in model_names]
bars = ax1.barh(model_names, accs, color=colors_bar, edgecolor="none", height=0.5)
ax1.set_xlim(0.85, 1.01)
ax1.set_title("Test Accuracy", color="white", fontsize=13)
ax1.set_xlabel("Accuracy")
for bar, val in zip(bars, accs):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", color="white", fontsize=10)

# 5b. ROC Curves
ax2 = fig2.add_subplot(gs[0, 1])
for (name, res), color in zip(results.items(), colors_bar):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax2.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, lw=2)
ax2.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
ax2.set_title("ROC Curves — All Models", color="white", fontsize=13)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=9, loc="lower right")

# 5c. Confusion Matrix — best model
ax3 = fig2.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Malignant", "Benign"])
disp.plot(ax=ax3, colorbar=False, cmap="RdPu")
ax3.set_title(f"Confusion Matrix — {best_name}", color="white", fontsize=13)
ax3.tick_params(colors="white")
ax3.xaxis.label.set_color("white")
ax3.yaxis.label.set_color("white")

# 5d. Feature Importance — Random Forest
ax4 = fig2.add_subplot(gs[1, 1])
rf_model  = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=raw.feature_names)
top15 = importances.nlargest(15)
ax4.barh(top15.index[::-1], top15.values[::-1], color=ACCENT, edgecolor="none", height=0.6)
ax4.set_title("Top 15 Feature Importances\n(Random Forest)", color="white", fontsize=13)
ax4.set_xlabel("Importance Score")
ax4.tick_params(labelsize=8)

plt.savefig("/home/claude/plot_results.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("✅ Results plot saved.")

# ══════════════════════════════════════════════════════════
# 6. DETAILED REPORT — Best Model
# ══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  DETAILED REPORT — {best_name}")
print(f"{'='*60}")
print(classification_report(y_test, best["y_pred"],
      target_names=["Malignant", "Benign"]))

# CV Summary
print("Cross-Validation (5-fold):")
for name, res in results.items():
    print(f"  {name:<25}: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

print("\n✅ All done! Check plot_eda.png and plot_results.png")
