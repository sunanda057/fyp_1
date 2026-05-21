"""
Smart Warehouse Demand Prediction — LPG Cylinders
Streamlit App | Mitra Bharatgas Agency, Murshidabad
Dataset: Warehouse_Demand_Realistic_v2.xlsx
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, os, joblib, calendar
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LPG Demand Predictor", page_icon="🔵",
                   layout="wide", initial_sidebar_state="expanded")

# ── Calendar anchor ────────────────────────────────────────────────────────────
NOW           = datetime.now()
CURRENT_YEAR  = NOW.year
CURRENT_MONTH = NOW.month

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,[class*="css"]{font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]{background:#0a1628 !important;border-right:1px solid #1a2e4a;}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1a2e4a 60%,#0d1b2a 100%);
  border:1px solid #1e3a5f;border-radius:14px;padding:2rem 2.5rem;margin-bottom:1.5rem;}
.hero-title{font-size:1.9rem;font-weight:800;color:#e8f4fd;margin:0;letter-spacing:-.5px;}
.hero-sub{font-size:.9rem;color:#6b8faf;margin-top:.4rem;font-family:'IBM Plex Mono',monospace;}
.hero-badge{display:inline-block;background:rgba(30,90,160,.25);border:1px solid #1e5aa0;
  color:#5ba3d9;padding:.15rem .7rem;border-radius:20px;font-size:.72rem;
  font-family:'IBM Plex Mono',monospace;margin-top:.8rem;}
.kpi{background:#0f1e30;border:1px solid #1a2e4a;border-radius:10px;padding:1rem 1.2rem;margin:.3rem 0;}
.kpi-label{font-size:.7rem;color:#5b7a99;text-transform:uppercase;letter-spacing:1px;}
.kpi-value{font-size:1.7rem;font-weight:700;color:#e8f4fd;margin:.15rem 0;}
.kpi-sub{font-size:.72rem;color:#3d7ab5;}
.sec{background:#0f1e30;border-left:4px solid #1e5aa0;border-radius:0 8px 8px 0;
  padding:.7rem 1rem;margin:1.2rem 0 .8rem 0;}
.sec h3{margin:0;color:#c8dcf0;font-size:1rem;font-weight:700;}
.sec p{margin:.15rem 0 0 0;color:#5b7a99;font-size:.78rem;}
.stTabs [data-baseweb="tab-list"]{background:#0f1e30;border-radius:8px;gap:3px;padding:3px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#5b7a99;border-radius:6px;font-weight:600;font-size:.82rem;}
.stTabs [aria-selected="true"]{background:#1e3a5f !important;color:#5ba3d9 !important;}
div[data-testid="stSelectbox"] label,div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label{color:#8aacc8 !important;font-size:.83rem;}
.cal-info{background:#0f1e30;border:1px solid #1e3a5f;border-radius:8px;
  padding:.6rem 1rem;margin-bottom:.8rem;font-size:.8rem;color:#5ba3d9;}
.cal-info span{color:#c8dcf0;font-weight:700;}
.warn-box{background:#1a1200;border:1px solid #b7950b;border-radius:8px;
  padding:.6rem 1rem;margin-bottom:.8rem;font-size:.8rem;color:#b7950b;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Warehouse_Demand_Realistic_v2.xlsx")

NUMERIC_COLS = ['Year','Month_Number','Is_Winter','Is_Festival_Month','Days_in_Month',
    'No_of_Family_Members','No_of_Adults','No_of_Children','Monthly_Income (₹)',
    'LPG_Price_per_Cylinder (₹)','Zone_Opening_Stock','Zone_Cylinders_Ordered',
    'Lead_Time_Days','Zone_Cylinders_Delivered','Zone_Safety_Stock','Zone_Reorder_Point',
    'Gas_Consumption_kg','Actual_Demand (cylinders)','Fulfilled_Demand (cylinders)',
    'Units_Short','Stockout_Occurred','Damaged_Cylinders','Zone_Closing_Stock','Avg_Daily_Demand']
DROP_COLS = ['Record_ID','Family_ID','Month','Season','Festival']

# Separate exclude lists for regression vs classification
REG_EXCLUDE = ['Gas_Consumption_kg','Stockout_Occurred',
               'Actual_Demand (cylinders)','Fulfilled_Demand (cylinders)',
               'Avg_Daily_Demand','Units_Short']
CLF_EXCLUDE = ['Stockout_Occurred','Gas_Consumption_kg',
               'Actual_Demand (cylinders)','Fulfilled_Demand (cylinders)',
               'Avg_Daily_Demand']

ZONES       = ['Raghunathganj','Kandi','Samserganj','Berhampore','Bali',
               'Domkal','Farakka','Suti','Lalbagh','Jangipur']
ZONE_ENC    = {z: i for i, z in enumerate(sorted(ZONES))}
SUBSIDY_ENC = {'Non-Subsidized': 0, 'PMUY': 1}
COLORS      = ['#1a5276','#1e8449','#b7950b','#c0392b','#7d3c98','#117a65']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_FULL  = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.dpi':120,
    'figure.facecolor':'#0f1e30','axes.facecolor':'#0f1e30',
    'axes.labelcolor':'#8aacc8','xtick.color':'#5b7a99','ytick.color':'#5b7a99',
    'text.color':'#c8dcf0','grid.color':'#1a2e4a','grid.alpha':.4,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
})

# ── Calendar helpers ───────────────────────────────────────────────────────────
def get_allowed_years():
    return [CURRENT_YEAR, CURRENT_YEAR + 1, CURRENT_YEAR + 2]

def get_allowed_months(selected_year):
    if selected_year == CURRENT_YEAR:
        return list(range(CURRENT_MONTH, 13))
    return list(range(1, 13))

# ── Data & model loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_excel(DATA_FILE, sheet_name='Warehouse_5000_Records', header=1)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def preprocess(_df):
    # Use the pre-split sheets from the Excel file
    train_raw = pd.read_excel(DATA_FILE, sheet_name='Train_80pct', header=1)
    test_raw  = pd.read_excel(DATA_FILE, sheet_name='Test_20pct',  header=1)
    for df in [train_raw, test_raw]:
        for c in NUMERIC_COLS:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    tr = train_raw.drop(columns=DROP_COLS, errors='ignore').copy()
    te = test_raw.drop(columns=DROP_COLS,  errors='ignore').copy()

    for col in tr.select_dtypes(include=[np.number]).columns:
        med = tr[col].median()
        tr[col] = tr[col].fillna(med)
        te[col] = te[col].fillna(med)

    enc = {}
    for col in ['Warehouse_Zone','Subsidy_Type']:
        if col in tr.columns:
            le = LabelEncoder()
            tr[col] = le.fit_transform(tr[col].astype(str))
            te[col] = le.transform(te[col].astype(str))
            enc[col] = le

    reg_feat = [c for c in tr.columns if c not in REG_EXCLUDE]
    clf_feat = [c for c in tr.columns if c not in CLF_EXCLUDE]

    Xtr_r, Xte_r = tr[reg_feat], te[reg_feat]
    Xtr_c, Xte_c = tr[clf_feat], te[clf_feat]
    ytr_d, yte_d  = tr['Gas_Consumption_kg'], te['Gas_Consumption_kg']
    ytr_s, yte_s  = tr['Stockout_Occurred'].astype(int), te['Stockout_Occurred'].astype(int)

    sc_r = StandardScaler()
    Xtr_rs = pd.DataFrame(sc_r.fit_transform(Xtr_r), columns=reg_feat)
    Xte_rs = pd.DataFrame(sc_r.transform(Xte_r),     columns=reg_feat)

    sc_c = StandardScaler()
    Xtr_cs = pd.DataFrame(sc_c.fit_transform(Xtr_c), columns=clf_feat)
    Xte_cs = pd.DataFrame(sc_c.transform(Xte_c),     columns=clf_feat)

    # SMOTE balancing for classification
    sm = SMOTE(random_state=42, k_neighbors=5)
    Xtr_sm, ytr_sm = sm.fit_resample(Xtr_cs, ytr_s)

    return (Xtr_rs, Xte_rs, Xtr_sm, ytr_sm, Xte_cs,
            ytr_d, yte_d, ytr_s, yte_s,
            sc_r, sc_c, enc, reg_feat, clf_feat, train_raw)

@st.cache_resource(show_spinner=False)
def train_all(_Xtr_r, _ytr_d, _Xtr_sm, _ytr_sm):
    regs = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }
    clfs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }
    for m in regs.values(): m.fit(_Xtr_r,  _ytr_d)
    for m in clfs.values(): m.fit(_Xtr_sm, _ytr_sm)
    return regs, clfs

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔵 LPG Predictor")
    st.caption("Mitra Bharatgas · Murshidabad")
    page = st.radio("Navigation", [
        "📊 Overview & Data",
        "🤖 Train Models",
        "📈 Visualizations",
        "🔮 Predict Demand",
        "📋 Batch Report",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Dataset: 5,000 records · 10 zones\nJan 2022 – Dec 2024\n(Train: 4,000 | Test: 1,000)")
    st.caption("Targets:\n• Gas Consumption (kg) — Regression\n• Stockout Occurred — Classification\n• SMOTE used for class balancing")
    st.divider()
    st.markdown(
        f"<div style='font-size:.75rem;color:#5b7a99;'>🗓️ Today: "
        f"<b style='color:#5ba3d9'>{NOW.strftime('%d %b %Y')}</b><br>"
        f"Predictions from <b style='color:#5ba3d9'>"
        f"{MONTH_FULL[CURRENT_MONTH-1]} {CURRENT_YEAR}</b> onwards.</div>",
        unsafe_allow_html=True
    )

# ── Load raw data ──────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    raw_df = load_raw()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Data":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔵 Smart Warehouse Demand Prediction</div>
      <div class="hero-sub">LPG Cylinder Supply Chain · Mitra Bharatgas Agency · Murshidabad, West Bengal</div>
      <div class="hero-badge">Gradient Boosting Regression · R²=0.9451 · Logistic Regression · F1=1.00 · SMOTE Balanced</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Total Records</div><div class="kpi-value">{len(raw_df):,}</div><div class="kpi-sub">Household transactions</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Warehouse Zones</div><div class="kpi-value">{raw_df["Warehouse_Zone"].nunique()}</div><div class="kpi-sub">Murshidabad district</div></div>', unsafe_allow_html=True)
    with c3:
        avg_gas = raw_df['Gas_Consumption_kg'].mean()
        st.markdown(f'<div class="kpi"><div class="kpi-label">Avg Gas Consumption</div><div class="kpi-value">{avg_gas:.1f} kg</div><div class="kpi-sub">Per household / month</div></div>', unsafe_allow_html=True)
    with c4:
        sr = raw_df['Stockout_Occurred'].mean() * 100
        st.markdown(f'<div class="kpi"><div class="kpi-label">Actual Stockout Rate</div><div class="kpi-value">{sr:.1f}%</div><div class="kpi-sub">Real stockout events in data</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec"><h3>📂 Dataset Preview</h3><p>First 300 rows — Warehouse_Demand_Realistic_v2.xlsx</p></div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(300), use_container_width=True, height=360)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec"><h3>📍 Records by Zone</h3></div>', unsafe_allow_html=True)
        zc = raw_df['Warehouse_Zone'].value_counts().reset_index()
        zc.columns = ['Zone', 'Count']
        st.dataframe(zc, use_container_width=True, hide_index=True)
    with col2:
        st.markdown('<div class="sec"><h3>📊 Key Statistics</h3></div>', unsafe_allow_html=True)
        st.dataframe(raw_df[['Gas_Consumption_kg','Monthly_Income (₹)',
                               'LPG_Price_per_Cylinder (₹)','Zone_Opening_Stock',
                               'No_of_Family_Members','Stockout_Occurred']].describe().round(2),
                     use_container_width=True)

    buf = io.BytesIO()
    raw_df.to_excel(buf, index=False, engine='openpyxl')
    st.download_button("⬇ Download Full Dataset (.xlsx)", buf.getvalue(),
                       "warehouse_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🤖 Model Training</div>
      <div class="hero-sub">Pre-split sheets (Train_80pct / Test_20pct) · SMOTE balancing · 4 Regression + 4 Classification models</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Preprocessing — loading pre-split sheets & applying SMOTE…"):
        (Xtr_r, Xte_r, Xtr_sm, ytr_sm, Xte_c,
         ytr_d, yte_d, ytr_s, yte_s,
         sc_r, sc_c, enc, reg_feat, clf_feat, tr_raw) = preprocess(raw_df)

    c1, c2, c3 = st.columns(3)
    with c1: st.success(f"Train: **{len(Xtr_r):,}** rows")
    with c2: st.success(f"Test:  **{len(Xte_r):,}** rows")
    with c3: st.success(f"After SMOTE: **{len(Xtr_sm):,}** rows (balanced)")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 Regression feature columns"):
            st.code(str(reg_feat))
    with col2:
        with st.expander("🔍 Classification feature columns"):
            st.code(str(clf_feat))

    with st.spinner("Training 8 models… (~30 seconds first run)"):
        regs, clfs = train_all(Xtr_r, ytr_d, Xtr_sm, ytr_sm)

    # ── Regression table ──────────────────────────────────────────────────────
    st.markdown('<div class="sec"><h3>📉 Task A — Gas Consumption Regression (kg)</h3><p>Predict monthly LPG gas usage per household · Target: Gas_Consumption_kg (5.8 – 25.9 kg)</p></div>', unsafe_allow_html=True)
    reg_rows = []
    for name, m in regs.items():
        p = m.predict(Xte_r)
        reg_rows.append({"Model": name,
                          "R² Score": round(r2_score(yte_d, p), 4),
                          "MAE (kg)": round(mean_absolute_error(yte_d, p), 4),
                          "RMSE (kg)": round(np.sqrt(mean_squared_error(yte_d, p)), 4)})
    rdf = pd.DataFrame(reg_rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    best_reg_name = rdf.iloc[0]["Model"]

    def hi(row):
        return ['background-color:#1e3a5f;color:#5ba3d9;font-weight:bold' if row.name == 0 else '' for _ in row]
    st.dataframe(rdf.style.apply(hi, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 Best Regression: **{best_reg_name}** · R²={rdf.iloc[0]['R² Score']}")

    # ── Classification table ──────────────────────────────────────────────────
    st.markdown('<div class="sec"><h3>⚠️ Task B — Stockout Classification (SMOTE balanced)</h3><p>Predict Stockout_Occurred (real 0/1) · 9.2% actual stockout rate · SMOTE applied to training set</p></div>', unsafe_allow_html=True)
    clf_rows = []
    for name, m in clfs.items():
        p = m.predict(Xte_c)
        clf_rows.append({"Model": name,
                          "Accuracy": round(accuracy_score(yte_s, p), 4),
                          "Precision": round(precision_score(yte_s, p, zero_division=0), 4),
                          "Recall": round(recall_score(yte_s, p, zero_division=0), 4),
                          "F1 Score": round(f1_score(yte_s, p, zero_division=0), 4)})
    cdf = pd.DataFrame(clf_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_clf_name = cdf.iloc[0]["Model"]
    st.dataframe(cdf.style.apply(hi, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 Best Classification: **{best_clf_name}** · F1={cdf.iloc[0]['F1 Score']}")

    st.session_state.update({
        "trained": True, "regs": regs, "clfs": clfs,
        "rdf": rdf, "cdf": cdf,
        "best_reg": best_reg_name, "best_clf": best_clf_name,
        "Xte_r": Xte_r, "Xte_c": Xte_c,
        "yte_d": yte_d, "yte_s": yte_s,
        "sc_r": sc_r, "sc_c": sc_c,
        "enc": enc, "reg_feat": reg_feat, "clf_feat": clf_feat,
        "tr_raw": tr_raw,
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">📈 Visualizations</div>
      <div class="hero-sub">8 charts · Model performance · Demand trends · Zone analysis · Feature importance</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    regs     = st.session_state["regs"]
    clfs     = st.session_state["clfs"]
    rdf      = st.session_state["rdf"]
    cdf      = st.session_state["cdf"]
    br_name  = st.session_state["best_reg"]
    bc_name  = st.session_state["best_clf"]
    Xte_r    = st.session_state["Xte_r"]
    Xte_c    = st.session_state["Xte_c"]
    yte_d    = st.session_state["yte_d"]
    yte_s    = st.session_state["yte_s"]
    reg_feat = st.session_state["reg_feat"]
    tr_raw   = st.session_state["tr_raw"]
    br = regs[br_name]; bc = clfs[bc_name]
    br_pred = br.predict(Xte_r)
    bc_pred = bc.predict(Xte_c)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", "📉 Demand Analysis",
        "🏭 Zone & Stockout",  "⭐ Feature Importance"])

    # Tab 1 — Model Comparison
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(rdf['Model'], rdf['R² Score'], color=COLORS[:4], height=0.5)
            ax.set_xlim(0, 1.1); ax.set_xlabel('R² Score')
            ax.set_title('Regression Model Comparison — R²', fontweight='bold', fontsize=11)
            ax.axvline(0.9, color='#e74c3c', linestyle='--', alpha=.6, lw=1.2, label='Target 0.90')
            ax.legend(fontsize=8)
            for bar, v in zip(bars, rdf['R² Score']):
                ax.text(bar.get_width()+.01, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(cdf['Model'], cdf['F1 Score'], color=COLORS[:4], height=0.5)
            ax.set_xlim(0, 1.1); ax.set_xlabel('F1 Score')
            ax.set_title('Classification F1 Comparison (SMOTE)', fontweight='bold', fontsize=11)
            for bar, v in zip(bars, cdf['F1 Score']):
                ax.text(bar.get_width()+.01, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            cm = confusion_matrix(yte_s, bc_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Stockout', 'Stockout'],
                        yticklabels=['No Stockout', 'Stockout'],
                        annot_kws={'size': 13, 'weight': 'bold'}, cbar=False)
            ax.set_title(f'Confusion Matrix\n{bc_name}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col4:
            errors = yte_d.values - br_pred
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.hist(errors, bins=35, color='#1a5276', alpha=.85, edgecolor='#0d1b2a')
            ax.axvline(0, color='#e74c3c', linestyle='--', lw=1.5, label='Zero Error')
            ax.axvline(errors.mean(), color='#f39c12', linestyle='--', lw=1.5,
                       label=f'Mean: {errors.mean():.3f}')
            ax.set_xlabel('Error (Actual − Predicted)'); ax.set_ylabel('Frequency')
            ax.set_title('Prediction Error Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 2 — Demand Analysis
    with tab2:
        r2 = r2_score(yte_d, br_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(yte_d, br_pred, alpha=.3, color='#1a5276', s=12)
        lims = [yte_d.min() - .5, yte_d.max() + .5]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect Prediction')
        ax.set_xlabel('Actual Gas Consumption (kg)'); ax.set_ylabel('Predicted (kg)')
        ax.set_title(f'Actual vs Predicted Gas Consumption\n{br_name} | R²={r2:.4f}',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=10); plt.tight_layout(); st.pyplot(fig); plt.close()

        tr_raw['Gas_Consumption_kg'] = pd.to_numeric(tr_raw['Gas_Consumption_kg'], errors='coerce')
        tr_raw['Month_Number'] = pd.to_numeric(tr_raw['Month_Number'], errors='coerce')
        monthly = tr_raw.groupby('Month_Number')['Gas_Consumption_kg'].mean().reset_index()
        s_colors = {1:'#5dade2',2:'#5dade2',3:'#f39c12',4:'#f39c12',5:'#f39c12',
                    6:'#27ae60',7:'#27ae60',8:'#27ae60',9:'#27ae60',
                    10:'#e67e22',11:'#e67e22',12:'#5dade2'}
        fig, ax = plt.subplots(figsize=(11, 5))
        bc2 = [s_colors.get(int(m), '#1a5276') for m in monthly['Month_Number']]
        bars = ax.bar([MONTH_NAMES[int(m)-1] for m in monthly['Month_Number']],
                      monthly['Gas_Consumption_kg'], color=bc2, edgecolor='#0d1b2a', width=0.65)
        ax.set_title('Average Monthly Gas Consumption (kg)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Month'); ax.set_ylabel('Avg Gas (kg)')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        ax.legend(handles=[mpatches.Patch(color='#5dade2', label='Winter'),
                            mpatches.Patch(color='#f39c12', label='Summer'),
                            mpatches.Patch(color='#27ae60', label='Monsoon'),
                            mpatches.Patch(color='#e67e22', label='Autumn')], fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 3 — Zone & Stockout
    with tab3:
        tr_raw['Stockout_Occurred'] = pd.to_numeric(tr_raw['Stockout_Occurred'], errors='coerce')
        zs = tr_raw.groupby('Warehouse_Zone')['Stockout_Occurred'].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        bc3 = ['#c0392b' if v > zs.mean() else '#1e8449' for v in zs.values]
        bars = ax.bar(zs.index, zs.values, color=bc3, edgecolor='#0d1b2a', width=0.65)
        ax.axhline(zs.mean(), color='#f39c12', linestyle='--', lw=1.5, label=f'Avg: {zs.mean():.1f}%')
        ax.set_title('Actual Stockout Rate by Warehouse Zone', fontweight='bold', fontsize=13)
        ax.set_ylabel('Stockout Rate (%)'); ax.set_xlabel('Zone')
        ax.set_xticklabels(zs.index, rotation=30, ha='right'); ax.legend(fontsize=10)
        for bar, v in zip(bars, zs.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.1,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        tr_raw['Gas_Consumption_kg'] = pd.to_numeric(tr_raw['Gas_Consumption_kg'], errors='coerce')
        zd = tr_raw.groupby('Warehouse_Zone')['Gas_Consumption_kg'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(zd.index, zd.values, color='#1a5276', height=0.6)
        ax.set_title('Avg Gas Consumption per Household by Zone', fontweight='bold', fontsize=12)
        ax.set_xlabel('Avg Gas Consumption (kg)')
        for i, (idx, v) in enumerate(zd.items()):
            ax.text(v+.05, i, f'{v:.2f} kg', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 4 — Feature Importance
    with tab4:
        if hasattr(br, 'feature_importances_'):
            fi = pd.Series(br.feature_importances_, index=reg_feat).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(fi.index, fi.values,
                    color=[COLORS[i % len(COLORS)] for i in range(len(fi))], height=0.6)
            ax.set_title(f'Feature Importance — {br_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Importance Score')
            for i, (idx, v) in enumerate(fi.items()):
                ax.text(v+.001, i, f'{v:.4f}', va='center', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Feature importances not available for Linear Regression.")

        tr_raw['Monthly_Income (₹)'] = pd.to_numeric(tr_raw['Monthly_Income (₹)'], errors='coerce')
        sample = tr_raw.sample(min(1000, len(tr_raw)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 5))
        sc2 = ax.scatter(sample['Monthly_Income (₹)'], sample['Gas_Consumption_kg'],
                         alpha=.35, c=sample['Gas_Consumption_kg'],
                         cmap='YlOrRd', s=15)
        plt.colorbar(sc2, ax=ax, label='Gas (kg)')
        ax.set_xlabel('Monthly Income (₹)'); ax.set_ylabel('Gas Consumption (kg)')
        ax.set_title('Income vs Gas Consumption', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT DEMAND (calendar-synced + improved income)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Demand":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔮 Predict Future Demand</div>
      <div class="hero-sub">Enter household & zone parameters → get gas forecast + real stockout risk prediction</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    br       = st.session_state["regs"][st.session_state["best_reg"]]
    bc       = st.session_state["clfs"][st.session_state["best_clf"]]
    sc_r     = st.session_state["sc_r"]
    sc_c     = st.session_state["sc_c"]
    reg_feat = st.session_state["reg_feat"]
    clf_feat = st.session_state["clf_feat"]

    # Calendar sync banner
    st.markdown(
        f'<div class="cal-info">🗓️ Calendar synced to today — '
        f'<span>{NOW.strftime("%d %B %Y")}</span>. '
        f'Predictions available from <span>{MONTH_FULL[CURRENT_MONTH-1]} {CURRENT_YEAR}</span> onwards only.</div>',
        unsafe_allow_html=True
    )

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)

        # Column 1 — Location & Time
        with col1:
            st.markdown("**📍 Location & Time**")
            zone  = st.selectbox("Warehouse Zone", sorted(ZONES), key="p_zone")
            year  = st.selectbox("Year", get_allowed_years(), index=0, key="p_year",
                                 help=f"Only {CURRENT_YEAR} and future years available.")
            allowed_months = get_allowed_months(year)
            month = st.selectbox("Month", allowed_months,
                                 format_func=lambda m: MONTH_FULL[m-1],
                                 index=0, key="p_month",
                                 help=("Only months from today onwards shown."
                                       if year == CURRENT_YEAR else "All months available."))
            if year == CURRENT_YEAR:
                st.caption(f"📅 {len(allowed_months)} month(s) available in {year} "
                           f"({MONTH_FULL[allowed_months[0]-1]} → December)")
            is_win  = st.checkbox("Winter Month?", value=(month in [12,1,2]), key="p_win",
                                  help="Dec, Jan, Feb — higher LPG usage.")
            is_fest = st.checkbox("Festival Month?", value=(month in [10,1]), key="p_fest",
                                  help="Oct (Durga Puja), Jan (Makar Sankranti), etc.")

        # Column 2 — Household Profile
        with col2:
            st.markdown("**👨‍👩‍👧 Household Profile**")
            subsidy  = st.selectbox("Subsidy Type", ['PMUY','Non-Subsidized'], key="p_sub")
            members  = st.slider("Family Members", 1, 10, 4, key="p_mem")
            adults   = st.slider("Adults", 1, min(8, members), min(2, members), key="p_adu")
            children = st.slider("Children", 0, max(0, members - adults), key="p_chi")

            st.markdown("**Monthly Income (₹)**")
            inc_c1, inc_c2, inc_c3 = st.columns([3, 1, 1])
            with inc_c1:
                income = st.number_input(
                    "Income", min_value=1000, max_value=500000,
                    value=st.session_state.get("income_val", 22000),
                    step=500, key="p_income",
                    label_visibility="collapsed",
                    help="Type any amount. Step = ₹500. Use ▲▼ for ₹1,000 jumps."
                )
            with inc_c2:
                if st.form_submit_button("▲ +1k"):
                    st.session_state["income_val"] = income + 1000
                    st.rerun()
            with inc_c3:
                if st.form_submit_button("▼ −1k"):
                    st.session_state["income_val"] = max(1000, income - 1000)
                    st.rerun()
            st.caption(f"💰 ₹{income:,} per month")

        # Column 3 — Zone Operations
        with col3:
            st.markdown("**🏭 Zone Operations**")
            lpg_price  = st.number_input("LPG Price/Cylinder (₹)", 700, 1200, 903, step=10, key="p_lpg")
            open_stock = st.number_input("Zone Opening Stock", 100, 2000, 800, step=50, key="p_ops")
            ordered    = st.number_input("Cylinders Ordered", 100, 1500, 435, step=25, key="p_ord")
            lead_time  = st.slider("Lead Time (days)", 1, 15, 5, key="p_lt")
            delivered  = st.number_input("Cylinders Delivered", 0, 1500, 410, step=25, key="p_del")
            safety_s   = st.number_input("Zone Safety Stock", 40, 200, 75, step=5, key="p_saf")
            reorder_p  = st.number_input("Zone Reorder Point", 60, 300, 125, step=5, key="p_rop")
            closing_s  = st.number_input("Zone Closing Stock", 0, 2000, 780, step=50, key="p_cls")
            units_short= st.number_input("Units Short", 0, 20, 0, step=1, key="p_ush",
                                         help="0 = no shortage. >0 = demand exceeded supply.")
            damaged    = st.number_input("Damaged Cylinders", 0, 50, 2, key="p_dmg")

        st.divider()
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: st.caption(f"📍 **{zone}**")
        with sc2: st.caption(f"🗓️ **{MONTH_FULL[month-1]} {year}**")
        with sc3: st.caption(f"👪 **{members}** members")
        with sc4: st.caption(f"💰 **₹{income:,}**")

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    if submitted:
        days = calendar.monthrange(year, month)[1]
        base = {
            'Warehouse_Zone': ZONE_ENC[zone], 'Year': year, 'Month_Number': month,
            'Is_Winter': int(is_win), 'Is_Festival_Month': int(is_fest),
            'Days_in_Month': days, 'No_of_Family_Members': members,
            'No_of_Adults': adults, 'No_of_Children': children,
            'Monthly_Income (₹)': income, 'Subsidy_Type': SUBSIDY_ENC[subsidy],
            'LPG_Price_per_Cylinder (₹)': lpg_price,
            'Zone_Opening_Stock': open_stock, 'Zone_Cylinders_Ordered': ordered,
            'Lead_Time_Days': lead_time, 'Zone_Cylinders_Delivered': delivered,
            'Zone_Safety_Stock': safety_s, 'Zone_Reorder_Point': reorder_p,
            'Zone_Closing_Stock': closing_s, 'Units_Short': units_short,
            'Damaged_Cylinders': damaged,
        }
        # Regression prediction
        inp_r  = pd.DataFrame([{c: base.get(c, 0) for c in reg_feat}])
        inp_rs = sc_r.transform(inp_r)
        gas_pred = max(0.1, round(br.predict(inp_rs)[0], 2))

        # Classification prediction
        inp_c  = pd.DataFrame([{c: base.get(c, 0) for c in clf_feat}])
        inp_cs = sc_c.transform(inp_c)
        risk_prob = bc.predict_proba(inp_cs)[0][1] * 100

        cyl_needed  = round(gas_pred / 14.2, 2)
        recommended = max(1, round(cyl_needed * 1.15))
        safety_stk  = max(1, round(cyl_needed * 0.15))

        risk_label = "🔴 HIGH" if risk_prob > 60 else "🟡 MEDIUM" if risk_prob > 30 else "🟢 LOW"
        risk_color = "#e74c3c" if risk_prob > 60 else "#f39c12" if risk_prob > 30 else "#27ae60"

        st.markdown('<div class="sec"><h3>📊 Prediction Results</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Gas Consumption</div><div class="kpi-value" style="color:#5ba3d9">{gas_pred} kg</div><div class="kpi-sub">{MONTH_FULL[month-1]} {year}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Cylinders Needed</div><div class="kpi-value" style="color:#27ae60">{cyl_needed}</div><div class="kpi-sub">@ 14.2 kg per cylinder</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Stockout Probability</div><div class="kpi-value" style="color:{risk_color}">{risk_prob:.1f}%</div><div class="kpi-sub">{risk_label} RISK</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Recommended Order</div><div class="kpi-value" style="color:#f39c12">{recommended}</div><div class="kpi-sub">+15% buffer · Safety: {safety_stk} cyl</div></div>', unsafe_allow_html=True)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(8, 1.6))
        ax.barh(['Risk'], [risk_prob], color=risk_color, height=0.4)
        ax.barh(['Risk'], [100 - risk_prob], left=risk_prob, color='#1a2e4a', height=0.4)
        ax.set_xlim(0, 100); ax.set_xlabel('Probability (%)')
        ax.set_title(f'Zone Stockout Risk: {risk_prob:.1f}%  {risk_label}', fontweight='bold')
        ax.text(min(risk_prob / 2, 90), 0, f'{risk_prob:.1f}%',
                ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.axvline(30, color='#f39c12', linestyle=':', lw=1, alpha=.7)
        ax.axvline(60, color='#e74c3c', linestyle=':', lw=1, alpha=.7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            f"💡 **Interpretation**: Household in **{zone}** expected to consume **{gas_pred} kg** "
            f"(≈ {cyl_needed} cylinders) in {MONTH_FULL[month-1]} {year}. "
            f"Zone stockout probability: **{risk_prob:.1f}%** — "
            f"recommended zone order: **{recommended} cylinders** (15% safety buffer included)."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH REPORT (calendar-synced)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Report":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">📋 Batch Prediction Report</div>
      <div class="hero-sub">All 10 zones × selected months → heatmaps + Excel export</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    br       = st.session_state["regs"][st.session_state["best_reg"]]
    bc       = st.session_state["clfs"][st.session_state["best_clf"]]
    sc_r     = st.session_state["sc_r"]
    sc_c     = st.session_state["sc_c"]
    reg_feat = st.session_state["reg_feat"]
    clf_feat = st.session_state["clf_feat"]
    rdf      = st.session_state["rdf"]
    cdf      = st.session_state["cdf"]

    st.markdown(
        f'<div class="cal-info">🗓️ Calendar synced — predictions from '
        f'<span>{MONTH_FULL[CURRENT_MONTH-1]} {CURRENT_YEAR}</span> onwards only.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        year_sel = st.selectbox("Year", get_allowed_years(), index=0, key="b_year")
    with col2:
        allowed_months = get_allowed_months(year_sel)
        if year_sel == CURRENT_YEAR and CURRENT_MONTH > 1:
            st.markdown(
                f'<div class="warn-box">⚠️ For {year_sel}: only '
                f'<b>{MONTH_FULL[CURRENT_MONTH-1]}</b> → <b>December</b> available '
                f'({len(allowed_months)} months).</div>',
                unsafe_allow_html=True
            )
        default_sel = allowed_months[:min(4, len(allowed_months))]
        month_sel = st.multiselect("Months", options=allowed_months,
                                   default=default_sel,
                                   format_func=lambda m: MONTH_FULL[m-1],
                                   key="b_months")

    if not month_sel:
        st.warning("Please select at least one month.")
        st.stop()

    defaults = {
        'No_of_Family_Members': 4, 'No_of_Adults': 2, 'No_of_Children': 2,
        'Monthly_Income (₹)': 22000, 'Subsidy_Type': 1,
        'LPG_Price_per_Cylinder (₹)': 903,
        'Zone_Opening_Stock': 794, 'Zone_Cylinders_Ordered': 435,
        'Lead_Time_Days': 5, 'Zone_Cylinders_Delivered': 409,
        'Zone_Safety_Stock': 75, 'Zone_Reorder_Point': 125,
        'Zone_Closing_Stock': 780, 'Units_Short': 0, 'Damaged_Cylinders': 2,
    }

    if st.button("🚀 Generate Batch Predictions", use_container_width=True):
        results = []
        prog  = st.progress(0)
        total = len(sorted(ZONES)) * len(month_sel)
        n = 0

        for zone in sorted(ZONES):
            for month in sorted(month_sel):
                days = calendar.monthrange(year_sel, month)[1]
                base = {'Warehouse_Zone': ZONE_ENC[zone], 'Year': year_sel,
                        'Month_Number': month,
                        'Is_Winter': int(month in [12,1,2]),
                        'Is_Festival_Month': int(month in [10,1,3]),
                        'Days_in_Month': days, **defaults}

                inp_r  = pd.DataFrame([{c: base.get(c, 0) for c in reg_feat}])
                inp_rs = sc_r.transform(inp_r)
                gas_pred = max(0.1, round(br.predict(inp_rs)[0], 2))

                inp_c  = pd.DataFrame([{c: base.get(c, 0) for c in clf_feat}])
                inp_cs = sc_c.transform(inp_c)
                risk_prob = bc.predict_proba(inp_cs)[0][1] * 100

                cyl_need = round(gas_pred / 14.2, 2)
                rec      = max(1, round(cyl_need * 1.15))
                safety   = max(1, round(cyl_need * 0.15))
                risk     = "HIGH" if risk_prob > 60 else "MEDIUM" if risk_prob > 30 else "LOW"

                results.append({
                    'Zone': zone, 'Year': year_sel, 'Month': MONTH_FULL[month-1],
                    'Gas_Consumption_kg': gas_pred,
                    'Cylinders_Needed': cyl_need,
                    'Stockout_Prob_%': round(risk_prob, 1),
                    'Risk_Level': risk,
                    'Recommended_Order': rec,
                    'Safety_Stock': safety,
                })
                n += 1; prog.progress(n / total)

        res_df = pd.DataFrame(results)

        def cr(v):
            if v == "HIGH":   return 'background-color:#3d0000;color:#e74c3c;font-weight:bold'
            if v == "MEDIUM": return 'background-color:#2d1d00;color:#f39c12;font-weight:bold'
            return 'background-color:#001a0d;color:#27ae60;font-weight:bold'

        st.markdown('<div class="sec"><h3>📊 Batch Results</h3></div>', unsafe_allow_html=True)
        st.dataframe(res_df.style.applymap(cr, subset=['Risk_Level']),
                     use_container_width=True, height=420)

        ordered_cols = [MONTH_FULL[m-1] for m in sorted(month_sel)]

        st.markdown('<div class="sec"><h3>🌡️ Gas Consumption Heatmap (kg)</h3></div>', unsafe_allow_html=True)
        pivot = res_df.pivot_table(values='Gas_Consumption_kg', index='Zone', columns='Month')
        pivot = pivot[[c for c in ordered_cols if c in pivot.columns]]
        fig, ax = plt.subplots(figsize=(max(6, len(month_sel)*1.8), max(5, len(ZONES)*0.7)))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    annot_kws={'size':9,'weight':'bold'}, cbar_kws={'label':'Gas (kg)'})
        ax.set_title('Predicted Gas Consumption — Zone × Month', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="sec"><h3>⚠️ Stockout Risk Heatmap (%)</h3></div>', unsafe_allow_html=True)
        pivot_r = res_df.pivot_table(values='Stockout_Prob_%', index='Zone', columns='Month')
        pivot_r = pivot_r[[c for c in ordered_cols if c in pivot_r.columns]]
        fig, ax = plt.subplots(figsize=(max(6, len(month_sel)*1.8), max(5, len(ZONES)*0.7)))
        sns.heatmap(pivot_r, annot=True, fmt='.0f', cmap='Reds', ax=ax,
                    annot_kws={'size':9,'weight':'bold'}, cbar_kws={'label':'Stockout Risk (%)'})
        ax.set_title('Stockout Risk — Zone × Month (%)', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            res_df.to_excel(w, sheet_name='Batch_Predictions', index=False)
            pd.DataFrame({
                'Metric': ['Dataset','Best Demand Model','R² Score','MAE (kg)','RMSE (kg)',
                            'Best Stockout Model','F1 Score','Accuracy',
                            'Balancing Method','Report Generated','Prediction Period'],
                'Value':  ['Warehouse_Demand_Realistic_v2.xlsx',
                           st.session_state["best_reg"],
                           f"{rdf.iloc[0]['R² Score']:.4f}",
                           f"{rdf.iloc[0]['MAE (kg)']:.4f}",
                           f"{rdf.iloc[0]['RMSE (kg)']:.4f}",
                           st.session_state["best_clf"],
                           f"{cdf.iloc[0]['F1 Score']:.4f}",
                           f"{cdf.iloc[0]['Accuracy']:.4f}",
                           'SMOTE oversampling (k=5)',
                           NOW.strftime('%d %B %Y %H:%M'),
                           f"{MONTH_FULL[min(month_sel)-1]} – {MONTH_FULL[max(month_sel)-1]} {year_sel}"]
            }).to_excel(w, sheet_name='Model_Summary', index=False)

        st.download_button(
            "⬇ Download Batch Report (.xlsx)", buf.getvalue(),
            f"lpg_predictions_{year_sel}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
