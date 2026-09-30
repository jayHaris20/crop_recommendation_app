import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("Crop_recommendation.csv")

print("--- Dataset Shape ---")
print(df.shape)  # Should show (2200, 8)

print("\n--- Class Distribution ---")
print(df["label"].value_counts())  # 100 per crop type

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 3. Feature Outlier Inspection (Boxplots)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop("label", axis=1))
plt.title("Feature Value Distributions & Outliers")
plt.tight_layout()
plt.savefig("feature_boxplots.png")
plt.show()

# 4. Average Soil pH per Crop
print("\n--- Average Soil pH per Crop ---")
print(df.groupby("label")["ph"].mean())

