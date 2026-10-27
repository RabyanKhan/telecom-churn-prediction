
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the same held-out test data used during training
X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
y_test = pd.read_csv(REPORTS_DIR / "y_test.csv").iloc[:, 0]

# Normalize target labels to 0 and 1 if needed
if y_test.dtype == "object":
    y_test = y_test.astype(str).str.strip().str.lower().map(
        {"no": 0, "yes": 1, "0": 0, "1": 1}
    )

if y_test.isna().any():
    raise ValueError("Unknown target labels found in y_test.csv.")

y_test = y_test.astype(int)

model_files = {
    "Logistic Regression (Baseline)": "logistic_regression_baseline.joblib",
    "Decision Tree (Baseline)": "decision_tree_baseline.joblib",
    "Random Forest (Baseline)": "random_forest_baseline.joblib",
    "XGBoost (Baseline)": "xgboost_baseline.joblib",
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
    "XGBoost (Tuned)": "xgboost_tuned.joblib",
}

results = []

for model_name, filename in model_files.items():
    model_path = MODELS_DIR / filename

    if not model_path.exists():
        print(f"Skipping missing model: {filename}")
        continue

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    row = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        row["ROC-AUC"] = roc_auc_score(y_test, probabilities)
    else:
        row["ROC-AUC"] = float("nan")

    results.append(row)

comparison = pd.DataFrame(results)

if comparison.empty:
    raise RuntimeError("No models were evaluated.")

comparison = comparison.sort_values(
    by="F1 Score", ascending=False
).reset_index(drop=True)

comparison.to_csv(
    REPORTS_DIR / "final_model_comparison.csv",
    index=False,
)

print("\nFINAL MODEL COMPARISON")
print("=" * 90)
print(comparison.round(4).to_string(index=False))
print("\nSaved: reports/final_model_comparison.csv")

best_model = comparison.iloc[0]

recommendation = f"""
FINAL MODEL RECOMMENDATION

Top model by test F1 score: {best_model['Model']}

Test metrics:
- Accuracy:  {best_model['Accuracy']:.4f}
- Precision: {best_model['Precision']:.4f}
- Recall:    {best_model['Recall']:.4f}
- F1 Score:  {best_model['F1 Score']:.4f}
- ROC-AUC:   {best_model['ROC-AUC']:.4f}

Reason:
F1 score balances precision and recall, making it useful when evaluating
customer churn models. Recall is also important because a false negative
means a customer who may churn is missed. The final business choice should
consider the cost of missed churners versus unnecessary retention offers.

This model is selected by the test F1 score, not by accuracy alone.
"""

(REPORTS_DIR / "final_model_recommendation.txt").write_text(
    recommendation.strip() + "\n",
    encoding="utf-8",
)

print("\n" + recommendation)
print("Saved: reports/final_model_recommendation.txt")
