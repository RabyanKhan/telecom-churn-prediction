
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telecom_churn_modeling.csv"
)

REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Modeling dataset not found: {DATA_PATH}\n"
        "Run clean_data.py first."
    )

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("\nTELECOM CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 65)
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# =========================================================
# 2. VALIDATE AND PREPARE REQUIRED COLUMNS
# =========================================================

required_columns = [
    "churn",
    "contract_type",
    "monthly_charges",
    "tenure_months",
    "internet_service",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Standardize churn labels to numeric 0/1.
# This supports common variations in the source dataset.
if not pd.api.types.is_numeric_dtype(df["churn"]):
    churn_mapping = {
        "no": 0,
        "n": 0,
        "stayed": 0,
        "0": 0,
        "yes": 1,
        "y": 1,
        "churned": 1,
        "1": 1,
    }

    df["churn"] = (
        df["churn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(churn_mapping)
    )
else:
    df["churn"] = pd.to_numeric(
        df["churn"], errors="coerce"
    )

# Convert numeric analysis columns safely.
for column in ["monthly_charges", "tenure_months"]:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Normalize categories for consistent charts and grouping.
for column in ["contract_type", "internet_service"]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

# Use only valid binary churn labels for churn analysis.
analysis_df = df[df["churn"].isin([0, 1])].copy()

if analysis_df.empty:
    raise ValueError(
        "No valid churn labels (0/1) were found in the dataset."
    )

analysis_df["churn"] = analysis_df["churn"].astype(int)

analysis_df["churn_label"] = analysis_df["churn"].map({
    0: "Retained",
    1: "Churned",
})

print("\nCHURN DISTRIBUTION")
print("-" * 40)
print(analysis_df["churn_label"].value_counts())
print("\nChurn percentages:")
print(
    (
        analysis_df["churn_label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def save_plot(filename):
    """Save the current chart as a high-resolution PNG."""
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved chart: {filename}")


def calculate_churn_rates(column):
    """Calculate churn rates by category, excluding missing labels."""
    temp = analysis_df[[column, "churn"]].dropna()

    if temp.empty:
        return pd.Series(dtype=float)

    rates = (
        temp.groupby(column, observed=True)["churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    return rates


# Store one-sentence insights for all five charts.
insights = []


# =========================================================
# 4. VISUALIZATION 1 - OVERALL CHURN DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=analysis_df,
    x="churn_label",
    order=["Retained", "Churned"],
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

churn_rate = analysis_df["churn"].mean() * 100
churn_count = int(analysis_df["churn"].sum())
total_customers = len(analysis_df)

insights.append(
    "1. Churn distribution: "
    f"{churn_count:,} of {total_customers:,} customers "
    f"({churn_rate:.1f}%) are labeled as churned."
)

save_plot("churn_distribution.png")


# =========================================================
# 5. VISUALIZATION 2 - CHURN BY CONTRACT TYPE
# =========================================================

contract_df = analysis_df.dropna(
    subset=["contract_type", "churn"]
).copy()

plt.figure(figsize=(9, 5))

sns.countplot(
    data=contract_df,
    x="contract_type",
    hue="churn_label",
    hue_order=["Retained", "Churned"],
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.legend(title="Customer Status")

contract_rates = calculate_churn_rates("contract_type")

if not contract_rates.empty:
    top_contract = contract_rates.index[0]
    top_contract_rate = contract_rates.iloc[0]

    insights.append(
        "2. Churn by contract: "
        f"{top_contract} has the highest observed churn rate "
        f"({top_contract_rate:.1f}%) among contract types."
    )
else:
    insights.append(
        "2. Churn by contract: Insufficient valid contract data "
        "to calculate churn rates."
    )

save_plot("churn_by_contract.png")


# =========================================================
# 6. VISUALIZATION 3 - MONTHLY CHARGES VS CHURN
# =========================================================

charges_df = analysis_df.dropna(
    subset=["monthly_charges", "churn_label"]
).copy()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=charges_df,
    x="churn_label",
    y="monthly_charges",
    order=["Retained", "Churned"],
)

plt.title("Monthly Charges by Customer Status")
plt.xlabel("Customer Status")
plt.ylabel("Monthly Charges")

charge_medians = (
    charges_df.groupby("churn_label")["monthly_charges"]
    .median()
)

if {"Retained", "Churned"}.issubset(charge_medians.index):
    retained_charge = charge_medians["Retained"]
    churned_charge = charge_medians["Churned"]

    if churned_charge > retained_charge:
        comparison = "higher"
    elif churned_charge < retained_charge:
        comparison = "lower"
    else:
        comparison = "the same as"

    insights.append(
        "3. Monthly charges: The median monthly charge for churned "
        f"customers ({churned_charge:.2f}) is {comparison} the median "
        f"for retained customers ({retained_charge:.2f})."
    )
else:
    insights.append(
        "3. Monthly charges: Insufficient valid data to compare "
        "median charges for both customer groups."
    )

save_plot("monthly_charges_vs_churn.png")


# =========================================================
# 7. VISUALIZATION 4 - TENURE VS CHURN
# =========================================================

tenure_df = analysis_df.dropna(
    subset=["tenure_months", "churn_label"]
).copy()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=tenure_df,
    x="churn_label",
    y="tenure_months",
    order=["Retained", "Churned"],
)

plt.title("Customer Tenure by Customer Status")
plt.xlabel("Customer Status")
plt.ylabel("Tenure (Months)")

tenure_medians = (
    tenure_df.groupby("churn_label")["tenure_months"]
    .median()
)

if {"Retained", "Churned"}.issubset(tenure_medians.index):
    retained_tenure = tenure_medians["Retained"]
    churned_tenure = tenure_medians["Churned"]

    if churned_tenure < retained_tenure:
        comparison = "shorter"
    elif churned_tenure > retained_tenure:
        comparison = "longer"
    else:
        comparison = "the same as"

    insights.append(
        "4. Tenure: The median tenure of churned customers "
        f"({churned_tenure:.1f} months) is {comparison} the median "
        f"tenure of retained customers ({retained_tenure:.1f} months)."
    )
else:
    insights.append(
        "4. Tenure: Insufficient valid data to compare median tenure "
        "for both customer groups."
    )

save_plot("tenure_vs_churn.png")


# =========================================================
# 8. VISUALIZATION 5 - CHURN BY INTERNET SERVICE
# =========================================================

internet_df = analysis_df.dropna(
    subset=["internet_service", "churn"]
).copy()

plt.figure(figsize=(9, 5))

sns.barplot(
    data=internet_df,
    x="internet_service",
    y="churn",
    errorbar=None,
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate")
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"{value:.0%}")
)

internet_rates = calculate_churn_rates("internet_service")

if not internet_rates.empty:
    top_service = internet_rates.index[0]
    top_service_rate = internet_rates.iloc[0]

    insights.append(
        "5. Churn by internet service: "
        f"{top_service} has the highest observed churn rate "
        f"({top_service_rate:.1f}%) among internet-service groups."
    )
else:
    insights.append(
        "5. Churn by internet service: Insufficient valid data "
        "to calculate service-wise churn rates."
    )

save_plot("churn_by_internet_service.png")


# =========================================================
# 9. MISSING VALUES REPORT
# =========================================================

missing_values = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_values = missing_values[missing_values > 0]

print("\nMISSING VALUES IN MODELING DATA")
print("-" * 40)

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values)

missing_report_path = REPORT_DIR / "missing_values_eda.csv"

missing_values.rename("missing_count").to_csv(
    missing_report_path,
    header=True,
)

print(f"Missing-value report saved: {missing_report_path}")


# =========================================================
# 10. SAVE ONE-SENTENCE INSIGHTS
# =========================================================

insights_path = REPORT_DIR / "eda_insights.txt"

with insights_path.open("w", encoding="utf-8") as file:
    file.write(
        "TELECOM CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS INSIGHTS\n"
    )
    file.write("=" * 65 + "\n\n")

    for insight in insights:
        file.write(insight + "\n\n")

print("\nEDA INSIGHTS")
print("=" * 65)

for insight in insights:
    print(f"- {insight}")

print(f"\nInsights saved to: {insights_path}")


# =========================================================
# 11. COMPLETION SUMMARY
# =========================================================

print("\nEDA COMPLETE")
print("=" * 65)
print(f"Rows analyzed: {len(analysis_df)}")
print(f"Columns available: {len(analysis_df.columns)}")
print(f"Charts generated: 5")
print(f"Chart directory: {PLOT_DIR}")
print(f"Insights report: {insights_path}")
print("\nAll five EDA charts and their insights have been generated.")
