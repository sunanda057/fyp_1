"""
Smart Warehouse Demand Prediction — LPG Cylinders
FINAL OPTIMIZED VERSION
✔ Income arrows removed
✔ Model trains only once
✔ Saved model loading enabled
✔ Faster prediction
✔ Streamlit cloud optimized
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import os
import joblib
import calendar

from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from imblearn.over_sampling import SMOTE


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="LPG Demand Predictor",
    page_icon="🔵",
    layout="wide"
)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "Warehouse_Demand_Realistic_v2.xlsx"
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

REG_MODEL_FILE = os.path.join(MODEL_DIR, "best_reg_model.pkl")
CLF_MODEL_FILE = os.path.join(MODEL_DIR, "best_clf_model.pkl")

SCALER_R_FILE = os.path.join(MODEL_DIR, "scaler_reg.pkl")
SCALER_C_FILE = os.path.join(MODEL_DIR, "scaler_clf.pkl")

META_FILE = os.path.join(MODEL_DIR, "meta.pkl")

NOW = datetime.now()

CURRENT_YEAR = NOW.year
CURRENT_MONTH = NOW.month

ZONES = [
    'Raghunathganj',
    'Kandi',
    'Samserganj',
    'Berhampore',
    'Bali',
    'Domkal',
    'Farakka',
    'Suti',
    'Lalbagh',
    'Jangipur'
]

ZONE_ENC = {
    z: i for i, z in enumerate(sorted(ZONES))
}

SUBSIDY_ENC = {
    'Non-Subsidized': 0,
    'PMUY': 1
}

MONTH_FULL = [
    'January', 'February', 'March',
    'April', 'May', 'June',
    'July', 'August', 'September',
    'October', 'November', 'December'
]

NUMERIC_COLS = [
    'Year',
    'Month_Number',
    'Is_Winter',
    'Is_Festival_Month',
    'Days_in_Month',
    'No_of_Family_Members',
    'No_of_Adults',
    'No_of_Children',
    'Monthly_Income (₹)',
    'LPG_Price_per_Cylinder (₹)',
    'Zone_Opening_Stock',
    'Zone_Cylinders_Ordered',
    'Lead_Time_Days',
    'Zone_Cylinders_Delivered',
    'Zone_Safety_Stock',
    'Zone_Reorder_Point',
    'Gas_Consumption_kg',
    'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)',
    'Units_Short',
    'Stockout_Occurred',
    'Damaged_Cylinders',
    'Zone_Closing_Stock',
    'Avg_Daily_Demand'
]

DROP_COLS = [
    'Record_ID',
    'Family_ID',
    'Month',
    'Season',
    'Festival'
]

REG_EXCLUDE = [
    'Gas_Consumption_kg',
    'Stockout_Occurred',
    'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)',
    'Avg_Daily_Demand',
    'Units_Short'
]

CLF_EXCLUDE = [
    'Stockout_Occurred',
    'Gas_Consumption_kg',
    'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)',
    'Avg_Daily_Demand'
]


# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>

.hero{
background:linear-gradient(135deg,#0d1b2a,#1a2e4a);
padding:2rem;
border-radius:15px;
margin-bottom:1rem;
}

.hero h1{
color:white;
}

.kpi{
background:#0f1e30;
padding:1rem;
border-radius:12px;
text-align:center;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def get_allowed_years():
    return [CURRENT_YEAR, CURRENT_YEAR + 1, CURRENT_YEAR + 2]


def get_allowed_months(selected_year):

    if selected_year == CURRENT_YEAR:
        return list(range(CURRENT_MONTH, 13))

    return list(range(1, 13))


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_raw():

    df = pd.read_excel(
        DATA_FILE,
        sheet_name='Warehouse_5000_Records',
        header=1
    )

    for c in NUMERIC_COLS:

        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# =============================================================================
# PREPROCESS
# =============================================================================

@st.cache_data(show_spinner=False)
def preprocess():

    train_raw = pd.read_excel(
        DATA_FILE,
        sheet_name='Train_80pct',
        header=1
    )

    test_raw = pd.read_excel(
        DATA_FILE,
        sheet_name='Test_20pct',
        header=1
    )

    for df in [train_raw, test_raw]:

        for c in NUMERIC_COLS:

            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    tr = train_raw.drop(columns=DROP_COLS, errors='ignore').copy()

    te = test_raw.drop(columns=DROP_COLS, errors='ignore').copy()

    for col in tr.select_dtypes(include=[np.number]).columns:

        med = tr[col].median()

        tr[col] = tr[col].fillna(med)

        te[col] = te[col].fillna(med)

    encoders = {}

    for col in ['Warehouse_Zone', 'Subsidy_Type']:

        le = LabelEncoder()

        tr[col] = le.fit_transform(tr[col].astype(str))

        te[col] = le.transform(te[col].astype(str))

        encoders[col] = le

    reg_feat = [c for c in tr.columns if c not in REG_EXCLUDE]

    clf_feat = [c for c in tr.columns if c not in CLF_EXCLUDE]

    Xtr_r = tr[reg_feat]
    Xte_r = te[reg_feat]

    Xtr_c = tr[clf_feat]
    Xte_c = te[clf_feat]

    ytr_d = tr['Gas_Consumption_kg']
    yte_d = te['Gas_Consumption_kg']

    ytr_s = tr['Stockout_Occurred'].astype(int)
    yte_s = te['Stockout_Occurred'].astype(int)

    sc_r = StandardScaler()

    Xtr_rs = pd.DataFrame(
        sc_r.fit_transform(Xtr_r),
        columns=reg_feat
    )

    Xte_rs = pd.DataFrame(
        sc_r.transform(Xte_r),
        columns=reg_feat
    )

    sc_c = StandardScaler()

    Xtr_cs = pd.DataFrame(
        sc_c.fit_transform(Xtr_c),
        columns=clf_feat
    )

    Xte_cs = pd.DataFrame(
        sc_c.transform(Xte_c),
        columns=clf_feat
    )

    sm = SMOTE(random_state=42)

    Xtr_sm, ytr_sm = sm.fit_resample(Xtr_cs, ytr_s)

    return (
        Xtr_rs,
        Xte_rs,
        Xtr_sm,
        ytr_sm,
        Xte_cs,
        ytr_d,
        yte_d,
        ytr_s,
        yte_s,
        sc_r,
        sc_c,
        reg_feat,
        clf_feat
    )


# =============================================================================
# TRAIN / LOAD MODELS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_or_train_models():

    (
        Xtr_r,
        Xte_r,
        Xtr_sm,
        ytr_sm,
        Xte_c,
        ytr_d,
        yte_d,
        ytr_s,
        yte_s,
        sc_r,
        sc_c,
        reg_feat,
        clf_feat
    ) = preprocess()

    # LOAD SAVED MODELS
    if os.path.exists(REG_MODEL_FILE):

        reg_model = joblib.load(REG_MODEL_FILE)

        clf_model = joblib.load(CLF_MODEL_FILE)

        sc_r = joblib.load(SCALER_R_FILE)

        sc_c = joblib.load(SCALER_C_FILE)

        meta = joblib.load(META_FILE)

        reg_feat = meta["reg_feat"]

        clf_feat = meta["clf_feat"]

    else:

        # TRAIN FIRST TIME

        reg_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        clf_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        reg_model.fit(Xtr_r, ytr_d)

        clf_model.fit(Xtr_sm, ytr_sm)

        # SAVE

        joblib.dump(reg_model, REG_MODEL_FILE)

        joblib.dump(clf_model, CLF_MODEL_FILE)

        joblib.dump(sc_r, SCALER_R_FILE)

        joblib.dump(sc_c, SCALER_C_FILE)

        joblib.dump(
            {
                "reg_feat": reg_feat,
                "clf_feat": clf_feat
            },
            META_FILE
        )

    return (
        reg_model,
        clf_model,
        sc_r,
        sc_c,
        reg_feat,
        clf_feat,
        Xte_r,
        Xte_c,
        yte_d,
        yte_s
    )


# =============================================================================
# LOAD EVERYTHING
# =============================================================================

raw_df = load_raw()

(
    reg_model,
    clf_model,
    sc_r,
    sc_c,
    reg_feat,
    clf_feat,
    Xte_r,
    Xte_c,
    yte_d,
    yte_s
) = load_or_train_models()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🔵 LPG Predictor")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Prediction"
        ]
    )


# =============================================================================
# OVERVIEW PAGE
# =============================================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
    <h1>🔵 Smart Warehouse Demand Prediction</h1>
    <p style='color:white'>
    LPG Cylinder Supply Chain Forecasting
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class='kpi'>
            <h2>{len(raw_df)}</h2>
            <p>Total Records</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        r2 = r2_score(
            yte_d,
            reg_model.predict(Xte_r)
        )

        st.markdown(
            f"""
            <div class='kpi'>
            <h2>{r2:.4f}</h2>
            <p>Regression R²</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        pred = clf_model.predict(Xte_c)

        f1 = f1_score(yte_s, pred)

        st.markdown(
            f"""
            <div class='kpi'>
            <h2>{f1:.4f}</h2>
            <p>Classification F1</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Dataset Preview")

    st.dataframe(raw_df.head(100))


# =============================================================================
# PREDICTION PAGE
# =============================================================================

elif page == "Prediction":

    st.markdown("""
    <div class="hero">
    <h1>🔮 Predict LPG Demand</h1>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:

            zone = st.selectbox(
                "Warehouse Zone",
                sorted(ZONES)
            )

            year = st.selectbox(
                "Year",
                get_allowed_years()
            )

            month = st.selectbox(
                "Month",
                get_allowed_months(year),
                format_func=lambda x: MONTH_FULL[x - 1]
            )

            is_win = st.checkbox(
                "Winter Month",
                value=(month in [12, 1, 2])
            )

            is_fest = st.checkbox(
                "Festival Month",
                value=(month in [10, 1])
            )

        with c2:

            subsidy = st.selectbox(
                "Subsidy Type",
                ['PMUY', 'Non-Subsidized']
            )

            members = st.slider(
                "Family Members",
                1,
                10,
                4
            )

            adults = st.slider(
                "Adults",
                1,
                members,
                2
            )

            children = st.slider(
                "Children",
                0,
                members - adults,
                2
            )

            # NO ARROWS HERE
            income_text = st.text_input(
                "Monthly Income (₹)",
                value="22000"
            )

            income = int(income_text) if income_text.isdigit() else 22000

        with c3:

            lpg_price = st.number_input(
                "LPG Price",
                700,
                1200,
                903
            )

            opening_stock = st.number_input(
                "Opening Stock",
                100,
                5000,
                800
            )

            ordered = st.number_input(
                "Ordered Cylinders",
                0,
                5000,
                435
            )

            delivered = st.number_input(
                "Delivered Cylinders",
                0,
                5000,
                410
            )

            safety_stock = st.number_input(
                "Safety Stock",
                0,
                5000,
                75
            )

            reorder = st.number_input(
                "Reorder Point",
                0,
                5000,
                125
            )

            closing = st.number_input(
                "Closing Stock",
                0,
                5000,
                780
            )

            units_short = st.number_input(
                "Units Short",
                0,
                100,
                0
            )

            damaged = st.number_input(
                "Damaged Cylinders",
                0,
                100,
                2
            )

            lead_time = st.slider(
                "Lead Time",
                1,
                15,
                5
            )

        submit = st.form_submit_button(
            "🔮 Predict",
            use_container_width=True
        )

    if submit:

        days = calendar.monthrange(year, month)[1]

        base = {
            'Warehouse_Zone': ZONE_ENC[zone],
            'Year': year,
            'Month_Number': month,
            'Is_Winter': int(is_win),
            'Is_Festival_Month': int(is_fest),
            'Days_in_Month': days,
            'No_of_Family_Members': members,
            'No_of_Adults': adults,
            'No_of_Children': children,
            'Monthly_Income (₹)': income,
            'Subsidy_Type': SUBSIDY_ENC[subsidy],
            'LPG_Price_per_Cylinder (₹)': lpg_price,
            'Zone_Opening_Stock': opening_stock,
            'Zone_Cylinders_Ordered': ordered,
            'Lead_Time_Days': lead_time,
            'Zone_Cylinders_Delivered': delivered,
            'Zone_Safety_Stock': safety_stock,
            'Zone_Reorder_Point': reorder,
            'Zone_Closing_Stock': closing,
            'Units_Short': units_short,
            'Damaged_Cylinders': damaged,
        }

        inp_r = pd.DataFrame([
            {
                c: base.get(c, 0)
                for c in reg_feat
            }
        ])

        inp_rs = sc_r.transform(inp_r)

        gas_pred = round(
            reg_model.predict(inp_rs)[0],
            2
        )

        inp_c = pd.DataFrame([
            {
                c: base.get(c, 0)
                for c in clf_feat
            }
        ])

        inp_cs = sc_c.transform(inp_c)

        risk = clf_model.predict_proba(inp_cs)[0][1] * 100

        cyl = round(gas_pred / 14.2, 2)

        st.success("Prediction Generated Successfully")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Gas Consumption",
                f"{gas_pred} kg"
            )

        with c2:
            st.metric(
                "Cylinders Needed",
                cyl
            )

        with c3:
            st.metric(
                "Stockout Risk",
                f"{risk:.1f}%"
            )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ['Predicted Gas'],
            [gas_pred]
        )

        ax.set_ylabel("Gas Consumption (kg)")

        st.pyplot(fig)
