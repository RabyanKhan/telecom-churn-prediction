
# train_model.py
# Telecom Customer Churn Prediction
# Baseline models with leakage-safe preprocessing

from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "telecom_churn_modeling.csv"
OUTPUT_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "churn"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("TELECOM CUSTOMER CHURN PREDICTION")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please check the file path and filename."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded: {DATA_PATH}")
print(f"Dataset shape: {df.shape}")

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Unknown target labels cannot be used for supervised training.
df = df.dropna(subset=[TARGET]).copy()

# Normalize target labels only; feature missing values remain untouched.
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

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The target must contain both churn classes: Yes and No."
    )

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

# Create features only when their required source columns exist.
if {"support_calls", "late_payments"}.issubset(df.columns):
    df["customer_risk_signals"] = (
        pd.to_numeric(df["support_calls"], errors="coerce")
        + pd.to_numeric(df["late_payments"], errors="coerce")
    )

if {"total_charges", "tenure_months"}.issubset(df.columns):
    total = pd.to_numeric(df["total_charges"], errors="coerce")
    tenure = pd.to_numeric(df["tenure_months"], errors="coerce")

    # Avoid dividing by zero or creating invalid values.
    df["calculated_monthly_charge"] = (
        total / tenure.where(tenure > 0)
    )

print("\nFeature engineering completed where source columns were available.")

# --------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# --------------------------------------------------

# Customer IDs and raw signup dates are excluded:
# ID is not a meaningful predictive feature, and raw date strings
# need a separate, deliberate date-feature strategy.
drop_columns = [TARGET]

if "customer_id" in df.columns:
    drop_columns.append("customer_id")

if "signup_date" in df.columns:
    drop_columns.append("signup_date")

X = df.drop(columns=drop_columns)
y = df[TARGET]

# Convert numeric-looking columns to numeric values.
# Columns containing genuine text remain categorical.
for column in X.columns:
    if (
        pd.api.types.is_object_dtype(X[column])
        or pd.api.types.is_string_dtype(X[column])
    ):
        converted = pd.to_numeric(X[column], errors="coerce")

        # Convert only when every existing non-missing value is numeric.
        original_non_missing = X[column].notna().sum()
        converted_non_missing = converted.notna().sum()

        if original_non_missing == converted_non_missing:
            X[column] = converted

# Remove columns that contain no usable values at all.
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    print("\nDropping entirely empty feature columns:", empty_columns)
    X = X.drop(columns=empty_columns)

# Replace infinite numeric values with missing values so the
# training-fitted imputer can handle them.
numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

if numeric_columns:
    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nFeatures: {X.shape[1]}")
print(f"Numeric features: {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")
print(f"Training records available: {len(X)}")

print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "No churn", 1: "Churn"}))

# --------------------------------------------------
# 5. STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTraining churn proportions:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTest churn proportions:")
print(y_test.value_counts(normalize=True).round(3))

# --------------------------------------------------
# 6. PREPROCESSING PIPELINES
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
)

# --------------------------------------------------
# 7. FOUR BASELINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

# XGBoost is an explicit requirement for this project.
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

except ImportError as exc:
    raise ImportError(
        "XGBoost is required for requirement 12. Install it using:\n"
        "python -m pip install xgboost"
    ) from exc

# --------------------------------------------------
# 8. TRAIN AND EVALUATE BASELINE MODELS
# --------------------------------------------------

results = []
trained_models = {}

for model_name, estimator in models.items():
    print("\n" + "-" * 65)
    print(f"Training: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
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

    results.append(metrics)
    trained_models[model_name] = pipeline

    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")

# --------------------------------------------------
# 9. SAVE BASELINE COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    by="F1-Score",
    ascending=False,
)

results_path = REPORT_DIR / "baseline_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("BASELINE MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nComparison saved to: {results_path}")

# --------------------------------------------------
# 10. SAVE BASELINE PIPELINES
# --------------------------------------------------

for model_name, pipeline in trained_models.items():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = OUTPUT_DIR / f"{safe_name}_baseline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved {model_name}: {model_path}")

# Save the test split so later tuning and comparisons can use
# the same held-out test records.
X_test.to_csv(REPORT_DIR / "X_test.csv", index=False)
y_test.to_csv(REPORT_DIR / "y_test.csv", index=False)

print("\nBaseline training complete.")
print("Next: tune Random Forest and XGBoost, then evaluate tuned models.")