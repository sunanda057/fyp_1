# Smart Warehouse — LPG Cylinder Demand Prediction
**Mitra Bharatgas Agency · Murshidabad, West Bengal**

A full Streamlit ML application to forecast LPG cylinder demand and predict
stockout risk across 10 warehouse zones.

---

## Project Structure

```
smart_warehouse/
├── app.py                  # Streamlit entry point
├── config.py               # All settings, paths, constants
├── pipeline.py             # Data loading, preprocessing, training, prediction
├── requirements.txt
├── data/
│   └── Warehouse_Demand_Realistic_v2.xlsx
├── models/                 # Saved .pkl files (auto-created after training)
├── outputs/                # CSV artefacts (auto-created)
└── pages/
    ├── 1_overview.py       # Home + dataset summary
    ├── 2_train.py          # Train all models
    ├── 3_performance.py    # Evaluation charts
    ├── 4_predict.py        # Month-locked prediction form
    └── 5_analytics.py      # EDA dashboards
```

---

## Quick Start

```bash
# 1. Clone / copy the project folder
cd smart_warehouse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage Flow

1. **Overview** — check dataset stats and model status
2. **Train Models** — click "Start Training" (takes ~60 seconds)
3. **Model Performance** — review R², confusion matrix, feature importance
4. **Predict Demand** — pick a future month (calendar-locked to current year),
   fill in zone/household details, get demand forecast + stockout risk
5. **Analytics** — explore monthly trends, zone heatmaps, year-on-year charts

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set **Main file path** to `app.py`
4. Add the Excel file to `data/` and commit it
5. Click **Deploy**

---

## Key Improvements vs Original Scripts

| Area | Change |
|---|---|
| Year/month in prediction | Dynamically locked to current year; future months only |
| Season/festival flags | Auto-derived from month number, never hardcoded |
| Class imbalance | `class_weight='balanced'` + sample weights for Gradient Boosting |
| Cross-validation | 5-fold CV added to all models |
| Config | Single `config.py` — no magic numbers scattered across files |
| Validation | Negative stock/demand checks on load |
| Duplicate columns | Only Unicode ₹ column used; Rs alias removed |
| Model versioning | Timestamped filenames in future releases |
