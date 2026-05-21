# 🔵 Smart Warehouse Demand Prediction — LPG Cylinders
Mitra Bharatgas Agency · Murshidabad, West Bengal
Dataset: **Warehouse_Demand_Realistic_v2.xlsx**

## Setup & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files needed in same folder
```
app.py
requirements.txt
Warehouse_Demand_Realistic_v2.xlsx
```

## App Pages
| Page | Description |
|------|-------------|
| 📊 Overview & Data | Dataset preview, KPIs, download |
| 🤖 Train Models | 4 regression + 4 classification models with SMOTE |
| 📈 Visualizations | 8 charts — performance, trends, zones, feature importance |
| 🔮 Predict Demand | Single scenario: calendar-synced date, gas forecast + stockout risk |
| 📋 Batch Report | All zones × months → heatmaps + Excel export |

## ML Details
| | Model | Score |
|---|---|---|
| Regression | Gradient Boosting | R²=0.9451, MAE=0.72 kg |
| Classification | Logistic Regression | F1=1.00 (SMOTE balanced) |

- **Regression target**: `Gas_Consumption_kg` (5.8 – 25.9 kg)
- **Classification target**: `Stockout_Occurred` (real 0/1, 9.2% rate)
- **Imbalance fix**: SMOTE oversampling (k=5) on training set only
- **Separate scalers**: regression and classification use independent StandardScalers
- **Calendar sync**: Year/Month inputs locked to current date onwards
