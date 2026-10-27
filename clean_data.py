
"""Clean and audit the Telecom Customer Churn dataset.

The raw CSV is never overwritten. Cleaned datasets and audit reports
are saved for reproducibility.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "telecom_churn_raw.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

# Required deliverable and compatibility copies
CLEAN_OUTPUTS = [
    BASE_DIR / "telecom_churn_clean.csv",
    PROCESSED_DIR / "telecom_churn_clean.csv",
    PROCESSED_DIR / "telecom_churn_cleaned.csv",
]

MODELING_DATA_PATH = (
    PROCESSED_DIR / "telecom_churn_modeling.csv"
)

DUPLICATE_REPORT_PATH = (
    REPORT_DIR / "conflicting_customer_ids.csv"
)

AUDIT_REPORT_PATH = (
    REPORT_DIR / "data_audit_report.txt"
)

CLEANING_SUMMARY_PATH = (
    REPORT_DIR / "cleaning_summary.csv"
)

MISSING_PATTERN = (
    r"(?i)^\s*(?:\?|--|not available|n/a|na|null|none)?\s*$"
)


# ============================================================
# 2. LOAD RAW DATASET
# ============================================================

def load_raw(path: Path) -> pd.DataFrame:
    """Load the original CSV without modifying it."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    try:
        return pd.read_csv(path, dtype="string")

    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            dtype="string",
            encoding="latin-1",
        )


# ============================================================
# 3. STANDARDIZE WHITESPACE AND MISSING VALUES
# ============================================================

def standardize_missing_and_whitespace(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Trim whitespace and normalize missing-value markers."""

    df = df.copy()

    for column in df.columns:
        df[column] = df[column].astype("string").str.strip()

    df = df.replace(
        MISSING_PATTERN,
        pd.NA,
        regex=True,
    )

    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].str.upper()

    return df


# ============================================================
# 4. NORMALIZE BOOLEAN-LIKE CATEGORIES
# ============================================================

def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize common yes/no representations."""

    mapping = {
        "yes": "Yes",
        "y": "Yes",
        "1": "Yes",
        "true": "Yes",
        "no": "No",
        "n": "No",
        "0": "No",
        "false": "No",
    }

    normalized = series.str.casefold().map(mapping)

    return normalized.where(
        normalized.notna(),
        series,
    )


# ============================================================
# 5. STANDARDIZE CATEGORICAL COLUMNS
# ============================================================

def standardize_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize inconsistent categorical labels."""

    df = df.copy()

    # Standardize binary yes/no columns.
    binary_columns = [
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
        "multiple_lines",
        "online_security",
        "tech_support",
        "streaming_tv",
    ]

    for column in binary_columns:
        if column in df.columns:
            df[column] = normalize_boolean(df[column])

    # Standardize senior-citizen values.
    if "senior_citizen" in df.columns:
        senior_mapping = {
            "0": 0,
            "no": 0,
            "false": 0,
            "1": 1,
            "yes": 1,
            "true": 1,
        }

        df["senior_citizen"] = (
            df["senior_citizen"]
            .str.casefold()
            .map(senior_mapping)
            .astype("Int64")
        )

    # Standardize gender while preserving unknown labels.
    if "gender" in df.columns:
        gender_mapping = {
            "female": "Female",
            "f": "Female",
            "fem": "Female",
            "male": "Male",
            "m": "Male",
        }

        normalized = df["gender"].str.casefold().map(
            gender_mapping
        )

        df["gender"] = normalized.where(
            normalized.notna(),
            df["gender"],
        )

    # Standardize internet-service labels.
    if "internet_service" in df.columns:
        internet_mapping = {
            "dsl": "DSL",
            "fiber": "Fiber optic",
            "fibre": "Fiber optic",
            "fiber optic": "Fiber optic",
            "fibre optic": "Fiber optic",
            "no": "No",
        }

        normalized = (
            df["internet_service"]
            .str.casefold()
            .map(internet_mapping)
        )

        df["internet_service"] = normalized.where(
            normalized.notna(),
            df["internet_service"],
        )

    # Standardize contract labels.
    if "contract_type" in df.columns:
        contract_mapping = {
            "m2m": "Month-to-month",
            "monthly": "Month-to-month",
            "month to month": "Month-to-month",
            "month-to-month": "Month-to-month",
            "one year": "One year",
            "1 year": "One year",
            "two year": "Two year",
            "two years": "Two year",
            "2 year": "Two year",
            "2 years": "Two year",
        }

        normalized = (
            df["contract_type"]
            .str.casefold()
            .map(contract_mapping)
        )

        df["contract_type"] = normalized.where(
            normalized.notna(),
            df["contract_type"],
        )

    # Standardize payment methods.
    if "payment_method" in df.columns:
        payment_mapping = {
            "electronic check": "Electronic check",
            "e-check": "Electronic check",
            "e check": "Electronic check",
            "mailed check": "Mailed check",
            "bank transfer": "Bank transfer",
            "bank transfer (auto)": "Bank transfer",
            "credit card": "Credit card",
            "credit card (auto)": "Credit card",
            "cc": "Credit card",
        }

        normalized = (
            df["payment_method"]
            .str.casefold()
            .map(payment_mapping)
        )

        df["payment_method"] = normalized.where(
            normalized.notna(),
            df["payment_method"],
        )

    # Standardize customer-satisfaction labels.
    if "customer_satisfaction" in df.columns:
        satisfaction_mapping = {
            "excellent": "Excellent",
            "excelent": "Excellent",
            "good": "Good",
            "average": "Average",
            "avg": "Average",
            "avrage": "Average",
            "poor": "Poor",
            "pooor": "Poor",
            "poorr": "Poor",
            "v.good": "Very good",
            "very good": "Very good",
        }

        normalized = (
            df["customer_satisfaction"]
            .str.casefold()
            .map(satisfaction_mapping)
        )

        df["customer_satisfaction"] = normalized.where(
            normalized.notna(),
            df["customer_satisfaction"],
        )

    # Standardize churn target labels.
    if "churn" in df.columns:
        churn_mapping = {
            "yes": "Yes",
            "y": "Yes",
            "1": "Yes",
            "churned": "Yes",
            "true": "Yes",
            "no": "No",
            "n": "No",
            "0": "No",
            "stayed": "No",
            "false": "No",
        }

        normalized = (
            df["churn"]
            .str.casefold()
            .map(churn_mapping)
        )

        df["churn"] = normalized.where(
            normalized.notna(),
            df["churn"],
        )

    # Normalize capitalization in remaining string columns.
    excluded_columns = {
        "customer_id",
        "gender",
        "internet_service",
        "contract_type",
        "payment_method",
        "customer_satisfaction",
        "churn",
    }

    for column in df.select_dtypes(
        include="string"
    ).columns:
        if column not in excluded_columns:
            df[column] = df[column].map(
                lambda value: (
                    value.title()
                    if pd.notna(value)
                    else value
                )
            )

    # Restore canonical service-label capitalization.
    for column in [
        "multiple_lines",
        "online_security",
        "tech_support",
        "streaming_tv",
    ]:
        if column in df.columns:
            df[column] = df[column].replace({
                "No Phone Service": "No phone service",
                "No Internet Service": "No internet service",
            })

    return df


# ============================================================
# 6. CLEAN AND VALIDATE NUMERIC COLUMNS
# ============================================================

def clean_numeric_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert numeric columns and flag invalid domain values."""

    df = df.copy()
    invalid_counts = {}

    numeric_columns = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_monthly_gb",
        "support_calls",
        "late_payments",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        original = df[column]

        values = original.str.replace(
            r"(?i)\bUSD\b",
            "",
            regex=True,
        )

        values = values.str.replace(
            r"[$,]",
            "",
            regex=True,
        ).str.strip()

        parsed = pd.to_numeric(
            values,
            errors="coerce",
        )

        invalid_counts[column] = int(
            (original.notna() & parsed.isna()).sum()
        )

        df[column] = parsed

    # Conservative domain checks.
    # Invalid values become missing instead of being clipped.
    bounds = {
        "tenure_months": (0, 120),
        "monthly_charges": (0, 500),
        "total_charges": (0, 100000),
        "avg_monthly_gb": (0, 1000),
        "support_calls": (0, 30),
        "late_payments": (0, 30),
    }

    for column, (lower, upper) in bounds.items():
        if column not in df.columns:
            continue

        invalid = (
            df[column].notna()
            & (
                (df[column] < lower)
                | (df[column] > upper)
            )
        )

        invalid_counts[column] = (
            invalid_counts.get(column, 0)
            + int(invalid.sum())
        )

        df.loc[invalid, column] = pd.NA

    # Convert integer-like columns to nullable integers.
    for column in [
        "tenure_months",
        "support_calls",
        "late_payments",
    ]:
        if column in df.columns:
            df[column] = (
                df[column].round().astype("Int64")
            )

    return df, invalid_counts


# ============================================================
# 7. MAIN CLEANING AND AUDIT PIPELINE
# ============================================================

def main() -> None:
    """Run the complete data-cleaning workflow."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load raw data.
    raw = load_raw(RAW_DATA_PATH)
    original_shape = raw.shape

    print("=" * 60)
    print("TELECOM CUSTOMER CHURN - DATA CLEANING")
    print("=" * 60)
    print(f"Raw dataset shape: {original_shape}")

    # Standardize missing values and whitespace.
    standardized = standardize_missing_and_whitespace(
        raw
    )

    missing_after_standardization = (
        standardized.isna().sum()
    )

    # Normalize categories and numeric values.
    df = standardize_categories(standardized)

    df, invalid_numeric_counts = clean_numeric_columns(
        df
    )

    # Parse dates safely.
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(
            df["signup_date"],
            errors="coerce",
            format="mixed",
        )

    # Remove fully identical rows after normalization.
    before_deduplication = len(df)

    df = df.drop_duplicates().copy()

    duplicates_removed = (
        before_deduplication - len(df)
    )

    # Report repeated customer IDs for manual review.
    if "customer_id" in df.columns:
        repeated_ids = df[
            df["customer_id"].notna()
            & df["customer_id"].duplicated(keep=False)
        ].copy()

        repeated_ids = repeated_ids.sort_values(
            "customer_id"
        )

        repeated_ids.to_csv(
            DUPLICATE_REPORT_PATH,
            index=False,
        )
    else:
        repeated_ids = pd.DataFrame()

        repeated_ids.to_csv(
            DUPLICATE_REPORT_PATH,
            index=False,
        )

    # Save all required cleaned-dataset copies.
    for output_path in CLEAN_OUTPUTS:
        df.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d",
        )

    # Keep rows with known churn labels for supervised learning.
    if "churn" not in df.columns:
        raise ValueError(
            "Required target column 'churn' is missing."
        )

    modeling_df = df.dropna(
        subset=["churn"]
    ).copy()

    # Convert target labels into numeric values.
    modeling_df["churn"] = modeling_df["churn"].map({
        "No": 0,
        "Yes": 1,
    })

    # Remove any unrecognized target labels rather than
    # silently converting them into incorrect target values.
    modeling_df = modeling_df.dropna(
        subset=["churn"]
    ).copy()

    modeling_df["churn"] = (
        modeling_df["churn"].astype("int64")
    )

    modeling_df.to_csv(
        MODELING_DATA_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    # Create a column-level cleaning summary.
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count_after_cleaning": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "missing_percentage_after_cleaning": [
            round(
                float(df[column].isna().mean() * 100),
                2,
            )
            for column in df.columns
        ],
        "data_type_after_cleaning": [
            str(df[column].dtype)
            for column in df.columns
        ],
        "unique_non_missing_values": [
            int(df[column].nunique(dropna=True))
            for column in df.columns
        ],
    })

    summary.to_csv(
        CLEANING_SUMMARY_PATH,
        index=False,
    )

    # Write a human-readable audit report.
    with AUDIT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "TELECOM CUSTOMER CHURN - DATA AUDIT REPORT\n"
        )
        report.write("=" * 55 + "\n\n")

        report.write(
            f"Raw dataset shape: "
            f"{original_shape[0]} rows x "
            f"{original_shape[1]} columns\n"
        )

        report.write(
            f"Cleaned dataset shape: "
            f"{df.shape[0]} rows x "
            f"{df.shape[1]} columns\n"
        )

        report.write(
            f"Exact duplicate rows removed: "
            f"{duplicates_removed}\n"
        )

        report.write(
            f"Rows with known churn labels for modeling: "
            f"{len(modeling_df)}\n"
        )

        report.write(
            f"Rows with missing churn labels excluded: "
            f"{int(df['churn'].isna().sum())}\n\n"
        )

        report.write(
            "Missing values detected after normalizing "
            "raw missing-value markers:\n"
        )

        missing_raw = (
            missing_after_standardization[
                missing_after_standardization.gt(0)
            ].sort_values(ascending=False)
        )

        report.write(
            missing_raw.to_string()
            if not missing_raw.empty
            else "No missing values detected at this stage."
        )

        report.write("\n\nInvalid numeric values by column:\n")

        report.write(
            pd.Series(invalid_numeric_counts).to_string()
        )

        report.write(
            "\n\nRepeated customer IDs are retained for review, "
            "not automatically deleted.\n"
        )

        report.write(
            f"Repeated-ID report: "
            f"{DUPLICATE_REPORT_PATH.name}\n"
        )

        report.write(
            "\nNumeric validation ranges:\n"
            "tenure_months: 0-120\n"
            "monthly_charges: 0-500\n"
            "total_charges: 0-100000\n"
            "avg_monthly_gb: 0-1000\n"
            "support_calls: 0-30\n"
            "late_payments: 0-30\n"
        )

        report.write(
            "\nValues outside these ranges are marked missing, "
            "not clipped.\n"
        )

    # Display final results.
    print(f"\nRows removed as duplicates: {duplicates_removed}")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Modeling dataset shape: {modeling_df.shape}")

    print("\nMissing values after cleaning:")

    missing_after_cleaning = (
        df.isna().sum()
    )

    missing_after_cleaning = (
        missing_after_cleaning[
            missing_after_cleaning.gt(0)
        ].sort_values(ascending=False)
    )

    if missing_after_cleaning.empty:
        print("No missing values.")
    else:
        print(missing_after_cleaning.to_string())

    print("\nChurn distribution:")
    print(df["churn"].value_counts(dropna=False).to_string())

    print("\nSaved output files:")

    for output_path in CLEAN_OUTPUTS:
        print(f"- {output_path}")

    print(f"- {MODELING_DATA_PATH}")
    print(f"- {DUPLICATE_REPORT_PATH}")
    print(f"- {CLEANING_SUMMARY_PATH}")
    print(f"- {AUDIT_REPORT_PATH}")

    print("\nDATA CLEANING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
