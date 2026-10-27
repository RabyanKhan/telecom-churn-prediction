
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "telecom_churn_raw.csv"


def inspect_dataset(file_path: Path) -> None:
    """Load the raw dataset and generate an initial quality report."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place the CSV file in the project root folder."
        )

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    print("\n" + "=" * 60)
    print("TELECOM CUSTOMER CHURN — DATA AUDIT")
    print("=" * 60)

    print(f"\nDataset file : {file_path.name}")
    print(f"Rows         : {df.shape[0]:,}")
    print(f"Columns      : {df.shape[1]:,}")
    print(f"Memory usage : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n--- Column Names ---")
    for column in df.columns:
        print(f"  {column}")

    print("\n--- First Five Records ---")
    print(df.head().to_string(index=False))

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values ---")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print("\n--- Duplicate Rows ---")
    print(f"Duplicate records: {df.duplicated().sum():,}")

    print("\n--- Unique Values Per Column ---")
    print(df.nunique(dropna=False).to_string())

    target_candidates = [
        column for column in df.columns
        if any(term in column.lower() for term in ("churn", "exited", "attrition"))
    ]

    print("\n--- Potential Target Columns ---")
    if target_candidates:
        for column in target_candidates:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).to_string())
    else:
        print("No obvious churn target column identified.")

    print("\n--- Numeric Summary ---")
    print(df.describe().to_string())

    print("\n" + "=" * 60)
    print("INITIAL DATA AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset(DATA_PATH)
