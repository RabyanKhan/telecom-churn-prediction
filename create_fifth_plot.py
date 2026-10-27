
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "telecom_churn_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "plots" / "churn_by_internet_service.png"

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Keep records with known churn labels
df = df.dropna(subset=["churn", "internet_service"])

# Calculate churn rate for each internet service type
churn_rate = (
    df.assign(churn_flag=df["churn"].astype(str).str.strip().str.lower())
    .assign(churn_flag=lambda x: x["churn_flag"].map(
        {"yes": 1, "no": 0}
    ))
    .dropna(subset=["churn_flag"])
    .groupby("internet_service")["churn_flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Create visualization
plt.figure(figsize=(9, 6))
bars = plt.bar(churn_rate.index.astype(str), churn_rate.values)

plt.title("Customer Churn Rate by Internet Service", fontsize=14)
plt.xlabel("Internet Service Type")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar, rate in zip(bars, churn_rate.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{rate:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("Fifth visualization saved:", OUTPUT_PATH)
print("\nChurn rate by internet service:")
print(churn_rate.round(2).to_string())

if len(churn_rate) > 0:
    highest = churn_rate.idxmax()
    lowest = churn_rate.idxmin()
    print(
        f"\nInsight: {highest} customers have the highest observed churn "
        f"rate ({churn_rate[highest]:.2f}%), while {lowest} has the "
        f"lowest ({churn_rate[lowest]:.2f}%)."
    )
