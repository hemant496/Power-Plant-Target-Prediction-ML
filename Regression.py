# Machine learning models - Monthly Power Generation October 2023
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

os.makedirs("model_plots", exist_ok=True)
df = pd.read_csv("C:/Users/hk389/Downloads/power_cleaned.csv")

# make the target here if the file doesn't have it yet
if "met_target" not in df.columns:
    df = df[df["oct_program"] > 0].copy()
    df["met_target"] = (df["oct_actual"] >= df["oct_program"]).astype(int)

# actual October values are removed because the target is made from them
features = ["capacity_mw", "target_annual", "oct_program", "oct_actual_last_year",
            "ytd_program", "ytd_actual_last_year", "plf_oct_program", "plf_oct_last_year",
            "plf_ytd_program", "plf_ytd_last_year", "plf_reported",
            "region", "sector", "fuel_type"]
X = pd.get_dummies(df[features], columns=["region", "sector", "fuel_type"], drop_first=True)
y = df["met_target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Train:", X_train.shape, "Test:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results, matrices = [], {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    matrices[name] = confusion_matrix(y_test, pred)
    results.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                    recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, prob)])

res = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]).round(4)
print(res)
res.to_csv("model_results.csv", index=False)

# confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (name, cm) in zip(axes.flat, matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Missed", "Met"], yticklabels=["Missed", "Met"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("model_plots/confusion_matrices.png", dpi=200)
plt.show()

# model comparison
res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", figsize=(9, 5))
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("model_plots/model_comparison.png", dpi=200)
plt.show()

# top features from random forest
imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).nlargest(10).sort_values()
imp.plot(kind="barh", color="#4c82b8")
plt.title("Top 10 Features (Random Forest)")
plt.tight_layout()
plt.savefig("model_plots/feature_importance.png", dpi=200)
plt.show()
print(imp.sort_values(ascending=False).round(3))
for n, cm in matrices.items():
    print(n, cm.tolist())