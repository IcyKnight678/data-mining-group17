import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb

pd.set_option('display.max_columns', None)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
energy = pd.read_csv('dataset/energy_dataset.csv')
weather = pd.read_csv('dataset/weather_features.csv')

# =============================================================================
# 2. PREPROCESSING & MERGING
# =============================================================================
# Datetime handling
energy['time'] = pd.to_datetime(energy['time'], utc=True).dt.tz_localize(None)
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True).dt.tz_localize(None)
weather = weather.rename(columns={'dt_iso': 'time'})

# Aggregate weather
weather_numeric = weather.select_dtypes(include=[np.number])
weather_numeric['time'] = weather['time']
weather_agg = weather_numeric.groupby('time').mean().reset_index()

# Merge
merged_df = pd.merge(energy, weather_agg, on='time', how='inner')
print("Merged shape:", merged_df.shape)

# =============================================================================
# 3. DATA CLEANING
# =============================================================================
# Fill generation NaNs with 0
generation_cols = [c for c in merged_df.columns if 'generation' in c.lower()]
merged_df[generation_cols] = merged_df[generation_cols].fillna(0)

# Forward fill remaining values
merged_df = merged_df.ffill().bfill()

print(f"NaN count: {merged_df.isna().sum().sum()}")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
# Time features
merged_df['hour'] = merged_df['time'].dt.hour
merged_df['day_of_week'] = merged_df['time'].dt.dayofweek
merged_df['month'] = merged_df['time'].dt.month

# 24-hour lag for load
merged_df['load_lag_24'] = merged_df['total load actual'].shift(24)
merged_df = merged_df.ffill().bfill()

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
plt.figure(figsize=(10,4))
sns.histplot(merged_df['total load actual'], bins=50)
plt.title('Distribution of Electricity Load')
plt.show()

plt.figure(figsize=(12,4))
plt.plot(merged_df['time'], merged_df['total load actual'])
plt.title('Electricity Load Over Time')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=merged_df['temp'],
    y=merged_df['total load actual'],
    alpha=0.3
)
plt.title('Temperature vs Load')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=merged_df['generation solar'],
    y=merged_df['total load actual'],
    alpha=0.3
)
plt.title('Solar Generation vs Load')
plt.show()

key_features = [
    'total load actual',
    'temp',
    'wind_speed',
    'clouds_all',
    'generation solar',
    'generation wind onshore',
    'hour',
    'load_lag_24'
]

plt.figure(figsize=(10,8))
sns.heatmap(
    merged_df[key_features].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Feature Correlation Heatmap')
plt.show()

# =============================================================================
# 6. MODEL PREPARATION
# =============================================================================
features = [
    'temp',
    'wind_speed',
    'clouds_all',
    'generation solar',
    'generation wind onshore',
    'hour',
    'day_of_week',
    'month',
    'load_lag_24'
]

X = merged_df[features]
y = merged_df['total load actual']

split = int(len(merged_df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

def evaluate(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy (%)': 100 - mape
    }

# =============================================================================
# 7. MODEL TRAINING
# =============================================================================
results = []

# RANDOM FOREST
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_p = rf.predict(X_test)
results.append(evaluate('Random Forest', y_test, rf_p))

# XGBOOST
print("Training XGBoost...")
xgb_m = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_m.fit(X_train, y_train)
xgb_p = xgb_m.predict(X_test)
results.append(evaluate('XGBoost', y_test, xgb_p))

# STACKED LSTM
print("Training Stacked LSTM...")
scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
xt_s = scaler_X.fit_transform(X_train)
xv_s = scaler_X.transform(X_test)
yt_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Reshape to [samples, time_steps, features]
xt_l = xt_s.reshape((xt_s.shape[0], 1, xt_s.shape[1]))
xv_l = xv_s.reshape((xv_s.shape[0], 1, xv_s.shape[1]))

lstm_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(1, xt_s.shape[1])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_model.fit(xt_l, yt_s, epochs=15, batch_size=64, validation_split=0.1, callbacks=[early_stop], verbose=0)

lp_s = lstm_model.predict(xv_l)
lp = scaler_y.inverse_transform(lp_s).flatten()
results.append(evaluate('Stacked LSTM', y_test, lp))

# =============================================================================
# 8. FINAL COMPARISON
# =============================================================================
comparison_df = pd.DataFrame(results)
print("\n--- Final Model Comparison Table ---")
pd.options.display.float_format = '{:.2f}'.format
print(comparison_df)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual', color='black', linewidth=1.5)
plt.plot(rf_p[:100], label='Random Forest', linestyle='--')
plt.plot(xgb_p[:100], label='XGBoost', linestyle='--')
plt.plot(lp[:100], label='LSTM', linestyle='--')
plt.title('24-Hour Ahead Forecast: Actual vs Predicted Load')
plt.ylabel('Load (MW)')
plt.legend()
plt.show()
