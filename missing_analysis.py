
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telecom_churn_cleaned.csv"
)

REPORT_DIR = BASE_DIR / "reports"

REPORT_PATH = REPORT_DIR / "missing_values_analysis.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    total_rows = len(df)

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })

    report["missing_percentage"] = (
        report["missing_count"] / total_rows * 100
    ).round(2)

    report["data_type"] = [
        str(df[column].dtype)
        for column in df.columns
    ]

    report = report.sort_values(
        "missing_percentage",
        ascending=False,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\nMISSING VALUE ANALYSIS")
    print("=" * 65)
    print(f"Dataset rows: {total_rows}")
    print(
        report[
            [
                "column",
                "missing_count",
                "missing_percentage",
                "data_type",
            ]
        ].to_string(index=False)
    )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
