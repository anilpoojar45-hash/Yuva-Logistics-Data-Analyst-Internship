import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
df = pd.read_csv("amazon_delivery_cleaned.csv")

# --- Feature set ---
num_features = ["Distance_Km", "Agent_Age", "Agent_Rating"]
cat_features = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
target = "Delivery_Time"

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

results = {}

# --- Model 1: Linear Regression ---
lr_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
results["Linear Regression"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
    "MAE": mean_absolute_error(y_test, lr_pred),
    "R2": r2_score(y_test, lr_pred),
}

# --- Model 2: Random Forest ---
rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=200, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)
results["Random Forest"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
    "MAE": mean_absolute_error(y_test, rf_pred),
    "R2": r2_score(y_test, rf_pred),
}

for name, m in results.items():
    print(name, {k: round(v, 3) for k, v in m.items()})

# --- Chart 1: Actual vs Predicted (Random Forest, best model) ---
plt.figure(figsize=(6.5, 5.5))
sample_idx = np.random.RandomState(0).choice(len(y_test), size=min(3000, len(y_test)), replace=False)
y_test_arr = y_test.values
plt.scatter(y_test_arr[sample_idx], rf_pred[sample_idx], alpha=0.3, s=14, color="#4C72B0")
lims = [min(y_test_arr.min(), rf_pred.min()), max(y_test_arr.max(), rf_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Delivery Time (min)")
plt.ylabel("Predicted Delivery Time (min)")
plt.title("Random Forest: Actual vs. Predicted Delivery Time")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_actual_vs_predicted.png", dpi=150)
plt.close()

# --- Chart 2: Model comparison bar chart ---
plt.figure(figsize=(6.5, 4.2))
model_names = list(results.keys())
rmse_vals = [results[m]["RMSE"] for m in model_names]
mae_vals = [results[m]["MAE"] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, rmse_vals, width, label="RMSE", color="#4C72B0")
plt.bar(x + width/2, mae_vals, width, label="MAE", color="#DD8452")
plt.xticks(x, model_names)
plt.ylabel("Error (minutes)")
plt.title("Model Comparison: RMSE and MAE")
plt.legend()
plt.tight_layout()
plt.savefig("chart2_model_comparison.png", dpi=150)
plt.close()

# --- Chart 3: Feature importance (Random Forest) ---
ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_features)
all_feature_names = num_features + list(cat_names)
importances = rf_pipe.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
# Aggregate one-hot importances back to their parent categorical feature
imp_df["group"] = imp_df["feature"].apply(
    lambda f: f if f in num_features else next((c for c in cat_features if f.startswith(c + "_")), f)
)
grouped_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=True)

plt.figure(figsize=(7, 4.5))
plt.barh(grouped_imp.index, grouped_imp.values, color="#55A868")
plt.xlabel("Aggregated Feature Importance")
plt.title("Random Forest — Feature Importance (Grouped)")
plt.tight_layout()
plt.savefig("chart3_feature_importance.png", dpi=150)
plt.close()

print("\nGrouped feature importance:")
print(grouped_imp.sort_values(ascending=False))

import json
with open("model_results.json", "w") as f:
    json.dump({
        "results": results,
        "feature_importance": grouped_imp.sort_values(ascending=False).to_dict(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, f, indent=2, default=str)

print("\ndone")
