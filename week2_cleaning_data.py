import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

df = pd.read_csv("amazon_delivery.csv")

report = {}
report["raw_shape"] = df.shape
report["missing_before"] = df.isnull().sum().to_dict()
report["duplicates_before"] = int(df.duplicated().sum())
report["raw_traffic_values"] = df["Traffic"].unique().tolist()
report["raw_vehicle_values"] = df["Vehicle"].unique().tolist()
report["raw_area_values"] = df["Area"].unique().tolist()
report["raw_weather_values"] = df["Weather"].dropna().unique().tolist()

# --- 1. Standardize categorical text (strip whitespace) ---
cat_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", np.nan)

# --- 2. Handle missing values ---
report["missing_after_strip"] = df.isnull().sum().to_dict()
df["Weather"] = df["Weather"].fillna("Unknown")
df["Traffic"] = df["Traffic"].fillna(df["Traffic"].mode()[0])
df["Agent_Rating"] = df["Agent_Rating"].fillna(df["Agent_Rating"].median())

# --- 3. Remove duplicate orders ---
before = len(df)
df = df.drop_duplicates(subset="Order_ID", keep="first")
report["duplicates_removed"] = before - len(df)

# --- 4. Parse datetimes & derive cycle-time-related fields ---
df["Order_DateTime"] = pd.to_datetime(df["Order_Date"] + " " + df["Order_Time"], errors="coerce")
df["Pickup_DateTime"] = pd.to_datetime(df["Order_Date"] + " " + df["Pickup_Time"], errors="coerce")
# Pickup sometimes rolls past midnight relative to order time in raw data; fix negative gaps
mask = df["Pickup_DateTime"] < df["Order_DateTime"]
df.loc[mask, "Pickup_DateTime"] += pd.Timedelta(days=1)
df["Pickup_Delay_Min"] = (df["Pickup_DateTime"] - df["Order_DateTime"]).dt.total_seconds() / 60

# --- 5. Derive distance via haversine ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

df["Distance_Km"] = df.apply(
    lambda r: haversine(r.Store_Latitude, r.Store_Longitude, r.Drop_Latitude, r.Drop_Longitude), axis=1
)

report["distance_desc_before_clean"] = df["Distance_Km"].describe().to_dict()
report["pickup_delay_desc_before_clean"] = df["Pickup_Delay_Min"].describe().to_dict()

# --- 6. Handle outliers: some rows have near-zero or absurd distance due to bad coordinates ---
# A number of rows in this raw dataset have store/drop coordinates near (0,0) -> huge/invalid distance
invalid_coords = ((df["Store_Latitude"].abs() < 1) & (df["Store_Longitude"].abs() < 1)) | \
                  ((df["Drop_Latitude"].abs() < 1) & (df["Drop_Longitude"].abs() < 1))
report["invalid_coord_rows"] = int(invalid_coords.sum())
df = df[~invalid_coords]

# Winsorize remaining distance outliers at 1st/99th percentile
lower, upper = df["Distance_Km"].quantile([0.01, 0.99])
df["Distance_Km"] = df["Distance_Km"].clip(lower, upper)

# Winsorize pickup delay outliers
lower_p, upper_p = df["Pickup_Delay_Min"].quantile([0.01, 0.99])
df["Pickup_Delay_Min"] = df["Pickup_Delay_Min"].clip(lower_p, upper_p)

report["final_shape"] = df.shape
report["distance_desc_after_clean"] = df["Distance_Km"].describe().to_dict()
report["missing_after_clean"] = df.isnull().sum().to_dict()

df.to_csv("amazon_delivery_cleaned.csv", index=False)

import json
with open("cleaning_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

for k, v in report.items():
    print(k, ":", v)
