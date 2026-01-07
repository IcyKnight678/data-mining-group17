from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

energy_path = r"dataset/energy_dataset.csv"
weather_path = r"dataset/weather_features.csv"

energy = pd.read_csv(energy_path)
weather = pd.read_csv(weather_path)

# merging
energy['time'] = pd.to_datetime(energy['time'], utc=True)
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True)

weather = weather.rename(columns={'dt_iso': 'time'})

weather_numeric = weather.select_dtypes(include=[np.number])
weather_numeric['time'] = weather['time']

weather_agg = weather_numeric.groupby('time').mean().reset_index()
merged_df = pd.merge(energy, weather_agg, on='time', how='inner')

# data cleaning
# handle NaN datas
gen_cols = [col for col in merged_df.columns if 'generation' in col.lower()]
merged_df[gen_cols] = merged_df[gen_cols].fillna(0)

# use forward fill (standard for time series data)
merged_df = merged_df.fillna(method='ffill')

# drop columns where all the rows are 0 
merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0)]

# extract time features before droppoing time column for modelling
merged_df['hour'] = merged_df['time'].dt.hour
merged_df['day_of_week'] = merged_df['time'].dt.dayofweek
merged_df['month'] = merged_df['time'].dt.month

print(f"Original Energy Rows: {len(energy)}")
print(f"Final Merged Rows: {len(merged_df)}")
print(f"Total Columns: {len(merged_df.columns)}")

# export to new csv file 
merged_df.to_csv('cleaned_national_energy_weather.csv', index=False)

# 1. Define target and features
target = 'total load actual'
features = ['temp', 'humidity', 'wind_speed', 'clouds_all', 'hour', 'day_of_week', 'month']

# 2. Test different partitions
partitions = [0.6, 0.7, 0.8, 0.9]
results = []

print("--- Testing Train/Test Partitions ---")

for p in partitions:
    # Calculate split index
    split_idx = int(len(merged_df) * p)
    
    # Split chronologically (No Shuffling!)
    train_df = merged_df.iloc[:split_idx]
    test_df = merged_df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Initialize and train Random Forest (Baseline)
    # Using small n_estimators for speed during testing
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Record result
    results.append({'Partition': f"{int(p*100)}/{int((1-p)*100)}", 'MAE': mae, 'RMSE': rmse})
    print(f"Split {int(p*100)}/{int((1-p)*100)} -> MAE: {mae:.2f}")

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)
print("\nSummary of results:")
print(results_df)

    