import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("amazon_delivery_cleaned.csv")

# Derive On-Time proxy: delayed if delivery time > median delivery time for that distance decile
df["Distance_Bin"] = pd.qcut(df["Distance_Km"], 10, duplicates="drop")
median_by_bin = df.groupby("Distance_Bin", observed=True)["Delivery_Time"].transform("median")
df["On_Time"] = (df["Delivery_Time"] <= median_by_bin).astype(int)

# Delivery efficiency: km covered per minute of delivery time
df["Efficiency_KmPerMin"] = df["Distance_Km"] / df["Delivery_Time"]

print("Rows:", len(df))
print("Avg Delivery Time (min):", round(df["Delivery_Time"].mean(), 1))
print("Median Delivery Time (min):", df["Delivery_Time"].median())
print("On-Time Rate (proxy):", round(df["On_Time"].mean() * 100, 1))
print("Avg Efficiency (km/min):", round(df["Efficiency_KmPerMin"].mean(), 3))
print(df.groupby("Traffic")["Delivery_Time"].mean().sort_values())
print(df.groupby("Weather")["Delivery_Time"].mean().sort_values())
print(df.groupby("Vehicle")["Delivery_Time"].mean().sort_values())
print(df[["Distance_Km", "Delivery_Time", "Agent_Rating", "Agent_Age"]].corr())

# --- Chart 1: Distribution of Delivery Time ---
plt.figure(figsize=(7, 4.2))
sns.histplot(df["Delivery_Time"], bins=30, kde=True, color="#4C72B0")
plt.title("Distribution of Delivery Time (Real Data, n=%d)" % len(df))
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("chart1_delivery_time_dist.png", dpi=150)
plt.close()

# --- Chart 2: Delivery Time by Traffic ---
plt.figure(figsize=(7, 4.2))
order = ["Low", "Medium", "High", "Jam"]
sns.boxplot(data=df, x="Traffic", y="Delivery_Time", order=order, hue="Traffic", palette="Blues", legend=False)
plt.title("Delivery Time by Traffic Condition")
plt.xlabel("Traffic Condition")
plt.ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("chart2_delivery_time_by_traffic.png", dpi=150)
plt.close()

# --- Chart 3: Distance vs Delivery Time by Vehicle ---
plt.figure(figsize=(7, 4.2))
sample = df.sample(min(4000, len(df)), random_state=1)
sns.scatterplot(data=sample, x="Distance_Km", y="Delivery_Time", hue="Vehicle", alpha=0.4, palette="Set2", s=18)
plt.title("Delivery Time vs. Distance by Vehicle Type")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("chart3_distance_vs_time.png", dpi=150)
plt.close()

# --- Chart 4: Avg Delivery Time by Weather ---
plt.figure(figsize=(7, 4.2))
order_w = df.groupby("Weather")["Delivery_Time"].mean().sort_values().index
avg_by_weather = df.groupby("Weather")["Delivery_Time"].mean().reindex(order_w)
sns.barplot(x=avg_by_weather.index, y=avg_by_weather.values, hue=avg_by_weather.index, palette="crest", legend=False)
plt.title("Average Delivery Time by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("chart4_time_by_weather.png", dpi=150)
plt.close()

# --- Chart 5: Correlation heatmap ---
plt.figure(figsize=(6, 5))
num_cols = ["Distance_Km", "Delivery_Time", "Agent_Rating", "Agent_Age", "On_Time"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Real Data)")
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", dpi=150)
plt.close()

print("charts saved")
