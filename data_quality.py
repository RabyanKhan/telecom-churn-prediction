
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "telecom_churn_raw.csv"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

MISSING_MARKERS = {
    "", "?", "--", "not available", "n/a", "na", "null", "none"
}

CHURN_LABELS = {
    "yes": "Yes",
    "y": "Yes",
    "1": "Yes",
    "churned": "Yes",
    "no": "No",
    "n": "No",
    "0": "No",
    "stayed": "No",
}


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, dtype="string")

    # Standardize whitespace and common missing-value markers.
    df = df.apply(lambda column: column.str.strip())
    df = df.replace(
        r"(?i)^\s*(?:\?|--|not available|n/a|na|null|none)?\s*$",
        pd.NA,
        regex=True,
    )

    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n--- Duplicate Customer IDs ---")
    print(f"Missing IDs: {df['customer_id'].isna().sum()}")
    print(
        "Repeated ID rows:",
        df["customer_id"].duplicated(keep=False).sum(),
    )

    print("\n--- Missing Values After Standardization ---")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing.gt(0)].to_string())

    # Normalize labels for analysis only.
    churn = df["churn"].str.casefold().map(CHURN_LABELS)

    print("\n--- Churn Target Audit ---")
    print("Recognized labels:")
    print(churn.value_counts(dropna=False).to_string())

    unrecognized = df.loc[
        df["churn"].notna() & churn.isna(), "churn"
    ].value_counts()

    print("\nUnrecognized churn labels:")
    print(
        unrecognized.to_string()
        if not unrecognized.empty
        else "None"
    )

    # Check numeric columns without modifying the original CSV.
    numeric_columns = [
        "senior_citizen",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_monthly_gb",
        "support_calls",
        "late_payments",
    ]

    print("\n--- Numeric Conversion Audit ---")
    for column in numeric_columns:
        cleaned = df[column].str.replace(
            r"[$,]", "", regex=True
        )
        converted = pd.to_numeric(cleaned, errors="coerce")

        invalid = df[column].notna() & converted.isna()

        print(
            f"{column}: "
            f"{invalid.sum()} non-numeric values; "
            f"{converted.isna().sum()} missing/unparseable values")
        

    print("\n--- Senior Citizen: Raw Categories ---")
    print(
        df["senior_citizen"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n--- Repeated Customer ID Examples ---")
    repeated_ids = df.loc[
        df["customer_id"].duplicated(keep=False),
        "customer_id",
    ]

    example_ids = repeated_ids.drop_duplicates().head(5)

    for customer_id in example_ids:
        records = df.loc[
            df["customer_id"].eq(customer_id)
        ]

        print(f"\nCustomer ID: {customer_id}")
        print(
            records[
                [
                    "customer_id",
                    "tenure_months",
                    "monthly_charges",
                    "contract_type",
                    "churn",
                ]
            ].to_string(index=False)
        )

    print("\n--- Exact Duplicate Rows ---")
    print(f"Exact duplicates: {df.duplicated().sum()}")

    # Save a report, not a modified training dataset.
    missing.rename("missing_count").to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
