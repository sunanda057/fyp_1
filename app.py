"""
app.py — Streamlit entry point.
Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Smart Warehouse — LPG Demand Prediction",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔵 Smart Warehouse")
st.sidebar.caption("LPG Cylinder Demand Prediction\nMitra Bharatgas Agency · Murshidabad")
st.sidebar.divider()

PAGES = {
    "🏠 Overview":          "pages/1_overview.py",
    "⚙️ Train Models":      "pages/2_train.py",
    "📊 Model Performance": "pages/3_performance.py",
    "🔮 Predict Demand":    "pages/4_predict.py",
    "📈 Analytics":         "pages/5_analytics.py",
}

page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit + scikit-learn")

# ── Route to page ──────────────────────────────────────────────────────────────
import importlib.util, os, sys

page_file = PAGES[page]
spec = importlib.util.spec_from_file_location("page_module", page_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
