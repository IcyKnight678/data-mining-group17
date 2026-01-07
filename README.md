# Energy Load Forecasting using Machine Learning

A data mining project that predicts electricity load demand using weather and energy generation data from Spain's national grid.

## 📊 Project Overview

This project applies machine learning techniques to forecast 24-hour ahead electricity load demand. We compare three different models:
- **Random Forest** - Ensemble tree-based model
- **XGBoost** - Gradient boosting model
- **Stacked LSTM** - Deep learning recurrent neural network

## 👥 Team Members (Group 17)

| Name | Student ID |
|------|------------|
| Vicky Leow Ming Fong | 22009591 |
| Hong Jing Jay | 22008338 |
| Geoi Ree Gyn | 22001952 |
| Wong Wei Ting | 24001869 |

## 📁 Project Structure

```
├── dataset/
│   ├── energy_dataset.csv              # Energy generation and load data
│   ├── weather_features.csv            # Weather data (temp, humidity, wind, etc.)
│   └── cleaned_national_energy_weather.csv  # Merged and cleaned dataset
├── graphs/                             # EDA and model comparison visualizations
├── Final_Energy_Forecast.py            # Main script with EDA + Model Training
├── partitiontest.py                    # Train/Test partition analysis
└── README.md
```

## 🔧 Features Used

- **Weather Features:** Temperature, Wind Speed, Cloud Cover
- **Generation Features:** Solar Generation, Wind Generation
- **Time Features:** Hour, Day of Week, Month
- **Lag Features:** 24-hour load lag (for day-ahead forecasting)

## 📈 Model Performance

| Model | MAE | RMSE | R² | Accuracy (%) |
|-------|-----|------|----|--------------| 
| Random Forest | 1628.33 | 2433.11 | 0.71 | 94.39 |
| XGBoost | 1634.90 | 2402.25 | 0.72 | 94.36 |
| Stacked LSTM | 1890.33 | 2533.41 | 0.69 | 93.33 |

## 🚀 How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
```

2. Run the main forecasting script:
```bash
python Final_Energy_Forecast.py
```

3. (Optional) Run partition testing:
```bash
python partitiontest.py
```

## 📊 Visualizations

The `graphs/` folder contains:
- Distribution of Electricity Load
- Load Over Time Series
- Temperature vs Load Scatter Plot
- Correlation Heatmap
- Model Predictions Comparison

## 📝 Dataset Source

The dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain (2015-2018).
