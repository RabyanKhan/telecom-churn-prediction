
# tune_models.py
# Tune Random Forest and XGBoost for Telecom Customer Churn

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.base import clone

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "telecom_churn_modeling.csv"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"

for folder in [REPORT_DIR, PLOT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

print("=" * 65)
print("RANDOM FOREST AND XGBOOST HYPERPARAMETER TUNING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=[TARGET]).copy()

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    })
)

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# Recreate the same engineered features used in baseline training.
if {"support_calls", "late_payments"}.issubset(df.columns):
    df["customer_risk_signals"] = (
        pd.to_numeric(df["support_calls"], errors="coerce")
        + pd.to_numeric(df["late_payments"], errors="coerce")
    )

if {"total_charges", "tenure_months"}.issubset(df.columns):
    total = pd.to_numeric(df["total_charges"], errors="coerce")
    tenure = pd.to_numeric(df["tenure_months"], errors="coerce")
    df["calculated_monthly_charge"] = total / tenure.where(tenure > 0)

drop_columns = [TARGET]

for column in ["customer_id", "signup_date"]:
    if column in df.columns:
        drop_columns.append(column)

X = df.drop(columns=drop_columns)
y = df[TARGET]

# Convert numeric-looking text columns only when all present values
# can be converted successfully.
for column in X.columns:
    if (
        pd.api.types.is_object_dtype(X[column])
        or pd.api.types.is_string_dtype(X[column])
    ):
        converted = pd.to_numeric(X[column], errors="coerce")
        if converted.notna().sum() == X[column].notna().sum():
            X[column] = converted

empty_columns = X.columns[X.isna().all()].tolist()
if empty_columns:
    print("Dropping entirely empty columns:", empty_columns)
    X = X.drop(columns=empty_columns)

numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

if numeric_columns:
    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

# Same deterministic stratified split settings as train_model.py.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Dataset:", df.shape)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# --------------------------------------------------
# 4. DEFINE RANDOM FOREST AND XGBOOST
# --------------------------------------------------

random_forest = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1,
)

from xgboost import XGBClassifier

xgboost = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

model_configs = {
    "Random Forest": {
        "estimator": random_forest,
        "parameters": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
    },
    "XGBoost": {
        "estimator": xgboost,
        "parameters": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5],
        },
    },
}

# --------------------------------------------------
# 5. HYPERPARAMETER TUNING AND EVALUATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

all_results = []
best_pipelines = {}

for model_name, config in model_configs.items():
    print("\n" + "-" * 65)
    print(f"Tuning {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", config["estimator"]),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["parameters"],
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_pipelines[model_name] = best_pipeline

    predictions = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "CV Best F1": search.best_score_,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    all_results.append(metrics)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nHeld-out test metrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    # Save confusion matrix as a separate image for each tuned model.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )

    display.plot(values_format="d")
    plt.title(f"{model_name} - Tuned Confusion Matrix")
    plt.tight_layout()

    image_name = (
        model_name.lower().replace(" ", "_")
        + "_tuned_confusion_matrix.png"
    )
    image_path = PLOT_DIR / image_name
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved:", image_path)

    # Save tuned model, including fitted preprocessing.
    model_name_safe = model_name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"{model_name_safe}_tuned.joblib"
    joblib.dump(best_pipeline, model_path)

    print("Tuned pipeline saved:", model_path)

    # Save selected parameters.
    params_path = REPORT_DIR / f"{model_name_safe}_best_parameters.csv"
    pd.DataFrame(
        list(search.best_params_.items()),
        columns=["Parameter", "Best Value"],
    ).to_csv(params_path, index=False)

# --------------------------------------------------
# 6. SAVE FINAL TUNED MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(all_results).sort_values(
    by="F1-Score",
    ascending=False,
)

comparison_path = REPORT_DIR / "tuned_model_comparison.csv"
results_df.to_csv(comparison_path, index=False)

print("\n" + "=" * 65)
print("TUNED MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}",
))

print("\nComparison saved:", comparison_path)
print("\nTuning complete.")
print("Next: compare tuned models with baselines and justify the final model.")
