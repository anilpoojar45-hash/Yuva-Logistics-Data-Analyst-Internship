import pandas as pd import numpy as np

df = pd.read_csv('delivery_data.csv')

#1. Handle missing values

df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())

df['Weather'] = df['Weather'].fillna('Unknown')

#-2. Standardize categorical text

for col in ['Weather', "Traffic', 'Vehicle', 'Area', 'Category']:

df[col] = df[col].str.strip().str.title()

#3. Remove duplicate orders -

df = df.drop_duplicates(subset='Order_ID', keep='first')

#4. Derive KPI-related fields

df['Order_Datetime'] = pd.to_datetime(

df['Order_Date']+''+ df['Order_Time'])

df['Delivery_Datetime'] = pd.to_datetime(

df['Order_Date'] + '' + df['Delivery_Time'])

df['Cycle_Time_Hrs'] = (

df['Delivery_Datetime'] - df['Order_Datetime']

).dt.total_seconds()/3600
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):

R6371 #km

dlat, dlon radians(lat2-lat1), radians(lon2 lon1)

a = sin(dlat/2)**2 cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2

return 2*R* atan2(sqrt(a), sqrt(1-a))

df['Distance_Km'] = df.apply(

lambda r: haversine(r.Store Latitude, r.Store_Longitude,

r.Drop Latitude, r. Drop Longitude), axis=1)

#-5. Handle outliers (winsorize at 1st/99th percentile)

for col in ['Cycle_Time_Hrs', 'Distance_Km']:

lower, upper df[col].quantile([0.01, 0.99])

df[col] = df[col].clip(lower, upper)

#-6. Save cleaned dataset for Week 3 EDA -

df.to_csv('delivery_data cleaned.csv', index=False)

print(df.info())
