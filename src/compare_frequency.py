import pandas as pd

# Read the frequency analysis results
df = pd.read_csv("analysis/frequency_results.csv")

# Frequency columns to compare
columns = [
    "low_0_500",
    "low_mid_500_1000",
    "mid_1000_2000",
    "high_2000_4000",
    "very_high_4000_8000",
    "upper_8000_11025",
    "dominant_frequency"
]

# Calculate statistics for each class
for label in ["scream", "non_scream"]:

    data = df[df["label"] == label]

    print("\n================================")
    print("CLASS:", label.upper())
    print("================================")

    for column in columns:

        values = data[column]

        print(f"\n{column}")
        print("Minimum :", round(values.min(), 2))
        print("25%     :", round(values.quantile(0.25), 2))
        print("Median  :", round(values.median(), 2))
        print("75%     :", round(values.quantile(0.75), 2))
        print("Maximum :", round(values.max(), 2))