"""
pages/1_overview.py — Project overview and dataset summary.
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from pipeline import load_data, load_trained_models

st.title("Smart Warehouse — LPG Cylinder Demand Prediction")
st.caption("Mitra Bharatgas Agency · Murshidabad, West Bengal")
st.divider()

# ── Status banner ──────────────────────────────────────────────────────────────
artefacts = load_trained_models()
if artefacts:
    mi = artefacts["model_info"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Demand Model",   mi["best_regression_model"].split()[0])
    c2.metric("R² Score",            f"{mi['regression_r2']:.4f}")
    c3.metric("Best Stockout Model", mi["best_classification_model"].split()[0])
    c4.metric("Stockout F1",         f"{mi['classification_f1']:.4f}")
    st.success("✅ Models are trained and ready. Go to **🔮 Predict Demand** to forecast.")
else:
    st.warning("⚠️ Models not trained yet. Go to **⚙️ Train Models** first.")

st.divider()

# ── Dataset overview ───────────────────────────────────────────────────────────
st.subheader("Dataset Overview")

@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    return load_data()

try:
    train_raw, test_raw, zone_summary, warnings_list = get_data()

    if warnings_list:
        for w in warnings_list:
            st.warning(f"⚠️ Data quality: {w}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records",     f"{len(train_raw)+len(test_raw):,}")
    col2.metric("Training Records",  f"{len(train_raw):,}")
    col3.metric("Test Records",      f"{len(test_raw):,}")
    col4.metric("Warehouse Zones",   "10")
    col5.metric("Feature Columns",   "18")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Zone Summary")
        if 'Warehouse_Zone' in zone_summary.columns:
            display_cols = [c for c in zone_summary.columns if c in [
                'Warehouse_Zone','Total_Records','Avg_Monthly_Demand',
                'Stockout_Records','Avg_Lead_Time'
            ]]
            st.dataframe(
                zone_summary[display_cols].set_index('Warehouse_Zone'),
                use_container_width=True,
            )

    with col_b:
        st.subheader("Training Data Sample")
        st.dataframe(train_raw.head(8), use_container_width=True)

    st.divider()
    st.subheader("How the Pipeline Works")

    steps = [
        ("⚙️ Step 1 — Preprocess",
         "Load Excel → drop unused columns → encode zones & subsidy → StandardScaler"),
        ("🤖 Step 2 — Train Models",
         "4 regression models (demand) + 4 classification models (stockout) with 5-fold CV"),
        ("📊 Step 3 — Evaluate",
         "Compare R², MAE, RMSE for regression; Accuracy, F1, Precision, Recall for classification"),
        ("🔮 Step 4 — Predict",
         "Select zone + future month → auto-derive season/festival flags → get demand + risk"),
        ("📈 Step 5 — Analytics",
         "Explore monthly trends, zone comparisons, feature importance, stockout patterns"),
    ]
    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.write(desc)

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.info(f"Expected file at: `{config.DATA_PATH}`")
