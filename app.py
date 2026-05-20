"""
Smart Warehouse Demand Prediction — LPG Cylinders
Streamlit App | Mitra Bharatgas Agency, Murshidabad

RUN:
streamlit run app.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import os
import joblib
import calendar

from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LPG Demand Predictor",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html,[class*="css"]{
    font-family:'Syne',sans-serif;
}

[data-testid="stSidebar"]{
    background:#0a1628 !important;
    border-right:1px solid #1a2e4a;
}

.hero{
    background:linear-gradient(
        135deg,
        #0d1b2a 0%,
        #1a2e4a 60%,
        #0d1b2a 100%
    );

    border:1px solid #1e3a5f;
    border-radius:14px;
    padding:2rem 2.5rem;
    margin-bottom:1.5rem;
}

.hero-title{
    font-size:1.9rem;
    font-weight:800;
    color:#e8f4fd;
    margin:0;
}

.hero-sub{
    font-size:.9rem;
    color:#6b8faf;
    margin-top:.4rem;
}

.kpi{
    background:#0f1e30;
    border:1px solid #1a2e4a;
    border-radius:10px;
    padding:1rem;
    margin:.3rem 0;
}

.kpi-label{
    font-size:.72rem;
    color:#5b7a99;
    text-transform:uppercase;
}

.kpi-value{
    font-size:1.7rem;
    font-weight:700;
    color:#e8f4fd;
}

.sec{
    background:#0f1e30;
    border-left:4px solid #1e5aa0;
    border-radius:0 8px 8px 0;
    padding:.7rem 1rem;
    margin:1rem 0;
}

.sec h3{
    margin:0;
    color:#c8dcf0;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOUR DATASET FILE NAME
DATA_FILE = os.path.join(
    BASE_DIR,
    "Warehouse_Demand_Realistic_v2(2).xlsx"
)

MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

COLORS = [
    '#1a5276',
    '#1e8449',
    '#b7950b',
    '#c0392b',
    '#7d3c98',
    '#117a65'
]

DROP_COLS = [
    'Record_ID',
    'Family_ID',
    'Month',
    'Season',
    'Festival',
    'Stockout_Occurred',
    'Units_Short'
]

EXCLUDE_TGT = [
    'Gas_Consumption_kg',
    'Stockout_Risk',
    'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)',
    'Avg_Daily_Demand',
    'Zone_Closing_Stock',
    'Zone_Safety_Stock',
    'Zone_Reorder_Point'
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_raw():

    df = pd.read_excel(
        DATA_FILE,
        sheet_name=0
    )

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # CREATE STOCKOUT RISK
    df['Stockout_Risk'] = (
        (
            df['Zone_Closing_Stock']
            <
            df['Zone_Safety_Stock'] * 1.5
        )
        |
        (
            df['Zone_Cylinders_Delivered']
            <
            df['Zone_Cylinders_Ordered'] * 0.9
        )
    ).astype(int)

    return df

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def preprocess(df):

    train = df.sample(
        frac=0.8,
        random_state=42
    )

    test = df.drop(train.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    tr_raw_copy = train.copy()

    tr = train.drop(
        columns=DROP_COLS,
        errors='ignore'
    )

    te = test.drop(
        columns=DROP_COLS,
        errors='ignore'
    )

    # FILL MISSING VALUES
    for col in tr.select_dtypes(include=[np.number]).columns:

        med = tr[col].median()

        tr[col] = tr[col].fillna(med)
        te[col] = te[col].fillna(med)

    # ENCODE CATEGORICAL
    encoders = {}

    cat_cols = tr.select_dtypes(
        include=['object']
    ).columns

    for col in cat_cols:

        le = LabelEncoder()

        tr[col] = le.fit_transform(
            tr[col].astype(str)
        )

        te[col] = le.transform(
            te[col].astype(str)
        )

        encoders[col] = le

    # FEATURES
    feat = [
        c for c in tr.columns
        if c not in EXCLUDE_TGT
    ]

    Xtr = tr[feat]
    Xte = te[feat]

    ytr_d = tr['Gas_Consumption_kg']
    yte_d = te['Gas_Consumption_kg']

    ytr_s = tr['Stockout_Risk']
    yte_s = te['Stockout_Risk']

    scaler = StandardScaler()

    Xtr_s = pd.DataFrame(
        scaler.fit_transform(Xtr),
        columns=feat
    )

    Xte_s = pd.DataFrame(
        scaler.transform(Xte),
        columns=feat
    )

    return (
        Xtr_s,
        Xte_s,
        ytr_d,
        yte_d,
        ytr_s,
        yte_s,
        scaler,
        encoders,
        feat,
        tr_raw_copy
    )

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_all(Xtr, ytr_d, ytr_s):

    regs = {

        'Linear Regression':
            LinearRegression(),

        'Decision Tree':
            DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),

        'Random Forest':
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        'Gradient Boosting':
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
    }

    clfs = {

        'Logistic Regression':
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        'Decision Tree':
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),

        'Random Forest':
            RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        'Gradient Boosting':
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                random_state=42
            )
    }

    # TRAIN REGRESSION
    for model in regs.values():
        model.fit(Xtr, ytr_d)

    # TRAIN CLASSIFICATION
    for model in clfs.values():
        model.fit(Xtr, ytr_s)

    return regs, clfs

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading dataset..."):

    raw_df = load_raw()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## 🔵 LPG Predictor")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🤖 Train Models",
            "🔮 Predict Demand"
        ]
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🔵 Smart Warehouse Demand Prediction
        </div>

        <div class="hero-sub">
            Mitra Bharatgas Agency · Murshidabad
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{len(raw_df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Zones</div>
            <div class="kpi-value">
                {raw_df['Warehouse_Zone'].nunique()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        avg_gas = raw_df['Gas_Consumption_kg'].mean()

        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Avg Gas</div>
            <div class="kpi-value">
                {avg_gas:.2f} kg
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:

        sr = raw_df['Stockout_Risk'].mean() * 100

        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Stockout Risk</div>
            <div class="kpi-value">
                {sr:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec">
        <h3>📂 Dataset Preview</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        raw_df.head(300),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Train Models":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🤖 Train Machine Learning Models
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Preprocessing dataset..."):

        (
            Xtr,
            Xte,
            ytr_d,
            yte_d,
            ytr_s,
            yte_s,
            scaler,
            encoders,
            feat,
            tr_raw
        ) = preprocess(raw_df)

    st.success(
        f"""
        ✅ Train: {len(Xtr):,}
        | Test: {len(Xte):,}
        | Features: {len(feat)}
        """
    )

    with st.spinner("Training models..."):

        regs, clfs = train_all(
            Xtr,
            ytr_d,
            ytr_s
        )

    # REGRESSION RESULTS
    reg_rows = []

    for name, model in regs.items():

        pred = model.predict(Xte)

        reg_rows.append({

            "Model": name,

            "R² Score":
                round(r2_score(yte_d, pred), 4),

            "MAE":
                round(mean_absolute_error(yte_d, pred), 4),

            "RMSE":
                round(
                    np.sqrt(
                        mean_squared_error(yte_d, pred)
                    ),
                    4
                )
        })

    rdf = pd.DataFrame(reg_rows)

    st.markdown("""
    <div class="sec">
        <h3>📉 Regression Results</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        rdf.sort_values(
            "R² Score",
            ascending=False
        ),
        use_container_width=True
    )

    # CLASSIFICATION RESULTS
    clf_rows = []

    for name, model in clfs.items():

        pred = model.predict(Xte)

        clf_rows.append({

            "Model": name,

            "Accuracy":
                round(accuracy_score(yte_s, pred), 4),

            "Precision":
                round(precision_score(yte_s, pred), 4),

            "Recall":
                round(recall_score(yte_s, pred), 4),

            "F1":
                round(f1_score(yte_s, pred), 4)
        })

    cdf = pd.DataFrame(clf_rows)

    st.markdown("""
    <div class="sec">
        <h3>⚠️ Classification Results</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        cdf.sort_values(
            "F1",
            ascending=False
        ),
        use_container_width=True
    )

    # BEST MODELS
    best_reg = rdf.sort_values(
        "R² Score",
        ascending=False
    ).iloc[0]["Model"]

    best_clf = cdf.sort_values(
        "F1",
        ascending=False
    ).iloc[0]["Model"]

    st.success(
        f"""
        🏆 Best Regression:
        {best_reg}

        🏆 Best Classification:
        {best_clf}
        """
    )

    # SAVE MODELS
    joblib.dump(
        regs[best_reg],
        "best_regressor.pkl"
    )

    joblib.dump(
        clfs[best_clf],
        "best_classifier.pkl"
    )

    joblib.dump(
        scaler,
        "scaler.pkl"
    )

    # SESSION STATE
    st.session_state["trained"] = True
    st.session_state["regs"] = regs
    st.session_state["clfs"] = clfs
    st.session_state["best_reg"] = best_reg
    st.session_state["best_clf"] = best_clf
    st.session_state["scaler"] = scaler
    st.session_state["features"] = feat

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT DEMAND
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict Demand":

    if "trained" not in st.session_state:

        st.warning(
            "⚠️ Please train models first."
        )

        st.stop()

    regs = st.session_state["regs"]
    clfs = st.session_state["clfs"]

    best_reg = st.session_state["best_reg"]
    best_clf = st.session_state["best_clf"]

    scaler = st.session_state["scaler"]
    feat = st.session_state["features"]

    br = regs[best_reg]
    bc = clfs[best_clf]

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🔮 Predict Future LPG Demand
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CURRENT DATE LOGIC
    today = datetime.today()

    current_year = today.year
    current_month = today.month

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        # ──────────────────────────────
        # COLUMN 1
        # ──────────────────────────────

        with col1:

            st.markdown("### 📍 Date & Zone")

            zone = st.selectbox(
                "Warehouse Zone",
                sorted(
                    raw_df['Warehouse_Zone']
                    .unique()
                    .tolist()
                )
            )

            # YEAR
            allowed_years = [
                current_year,
                current_year + 1
            ]

            year = st.selectbox(
                "Year",
                allowed_years
            )

            # MONTH FILTER
            if year == current_year:

                allowed_months = list(
                    range(current_month, 13)
                )

            else:

                allowed_months = list(
                    range(1, 13)
                )

            month = st.selectbox(
                "Month",
                allowed_months,
                format_func=lambda x:
                    MONTH_NAMES[x - 1]
            )

            is_winter = st.checkbox(
                "Winter Month?",
                value=(month in [12,1,2])
            )

            is_festival = st.checkbox(
                "Festival Month?",
                value=(month in [10,1])
            )

        # ──────────────────────────────
        # COLUMN 2
        # ──────────────────────────────

        with col2:

            st.markdown("### 👨‍👩‍👧 Household")

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
                8,
                2
            )

            children = st.slider(
                "Children",
                0,
                6,
                2
            )

            # TYPEABLE INCOME
            income = st.number_input(
                "Monthly Income (₹)",
                min_value=0,
                max_value=1000000,
                value=22000,
                step=500,
                format="%d"
            )

        # ──────────────────────────────
        # COLUMN 3
        # ──────────────────────────────

        with col3:

            st.markdown("### 🏭 Warehouse Ops")

            lpg_price = st.number_input(
                "LPG Price/Cylinder (₹)",
                700,
                1500,
                903,
                step=10
            )

            opening_stock = st.number_input(
                "Opening Stock",
                100,
                5000,
                800,
                step=50
            )

            ordered = st.number_input(
                "Cylinders Ordered",
                100,
                5000,
                450,
                step=25
            )

            lead_time = st.slider(
                "Lead Time (days)",
                1,
                15,
                5
            )

            delivered = st.number_input(
                "Cylinders Delivered",
                0,
                5000,
                430,
                step=25
            )

            damaged = st.number_input(
                "Damaged Cylinders",
                0,
                50,
                2
            )

        submitted = st.form_submit_button(
            "🔮 Predict Now",
            use_container_width=True
        )

    # ════════════════════════════════
    # PREDICTION
    # ════════════════════════════════

    if submitted:

        selected_date = datetime(
            year,
            month,
            1
        )

        current_date = datetime(
            current_year,
            current_month,
            1
        )

        # PREVENT PAST PREDICTION
        if selected_date < current_date:

            st.error(
                "❌ Cannot predict past months."
            )

            st.stop()

        days = calendar.monthrange(
            year,
            month
        )[1]

        # ENCODE
        zone_encoded = (
            raw_df['Warehouse_Zone']
            .astype('category')
            .cat
            .categories
            .tolist()
            .index(zone)
        )

        subsidy_encoded = (
            1 if subsidy == "PMUY"
            else 0
        )

        scen = {

            'Warehouse_Zone':
                zone_encoded,

            'Year':
                year,

            'Month_Number':
                month,

            'Is_Winter':
                int(is_winter),

            'Is_Festival_Month':
                int(is_festival),

            'Days_in_Month':
                days,

            'No_of_Family_Members':
                members,

            'No_of_Adults':
                adults,

            'No_of_Children':
                children,

            'Monthly_Income (₹)':
                income,

            'Subsidy_Type':
                subsidy_encoded,

            'LPG_Price_per_Cylinder (₹)':
                lpg_price,

            'Zone_Opening_Stock':
                opening_stock,

            'Zone_Cylinders_Ordered':
                ordered,

            'Lead_Time_Days':
                lead_time,

            'Zone_Cylinders_Delivered':
                delivered,

            'Damaged_Cylinders':
                damaged
        }

        inp = pd.DataFrame([
            {
                c: scen.get(c, 0)
                for c in feat
            }
        ])

        inp_sc = pd.DataFrame(
            scaler.transform(inp),
            columns=feat
        )

        # PREDICTIONS
        gas_pred = round(
            max(
                0.1,
                br.predict(inp_sc)[0]
            ),
            2
        )

        risk_prob = round(
            bc.predict_proba(inp_sc)[0][1] * 100,
            2
        )

        cylinders = round(
            gas_pred / 14.2,
            2
        )

        # WINTER BUFFER
        season_factor = (
            1.20
            if month in [11,12,1,2]
            else 1.10
        )

        recommended = max(
            1,
            round(cylinders * season_factor)
        )

        # RISK LABEL
        if risk_prob > 60:

            risk_label = "🔴 HIGH"
            risk_color = "#e74c3c"

        elif risk_prob > 30:

            risk_label = "🟡 MEDIUM"
            risk_color = "#f39c12"

        else:

            risk_label = "🟢 LOW"
            risk_color = "#27ae60"

        # RESULTS
        st.markdown("""
        <div class="sec">
            <h3>📊 Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">
                    Gas Consumption
                </div>

                <div class="kpi-value">
                    {gas_pred} kg
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">
                    Cylinders Needed
                </div>

                <div class="kpi-value">
                    {cylinders}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">
                    Stockout Risk
                </div>

                <div class="kpi-value"
                     style="color:{risk_color}">
                    {risk_prob}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">
                    Recommended Order
                </div>

                <div class="kpi-value">
                    {recommended}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # RISK BAR
        fig, ax = plt.subplots(
            figsize=(8,1.5)
        )

        ax.barh(
            ['Risk'],
            [risk_prob],
            color=risk_color
        )

        ax.barh(
            ['Risk'],
            [100-risk_prob],
            left=risk_prob,
            color='#1a2e4a'
        )

        ax.set_xlim(0,100)

        ax.set_title(
            f"Stockout Risk: {risk_prob}%"
        )

        st.pyplot(fig)

        # INTERPRETATION
        st.info(f"""
        💡 Household in **{zone}**
        is predicted to consume
        **{gas_pred} kg LPG**
        in **{MONTH_NAMES[month-1]} {year}**.

        Approx cylinders needed:
        **{cylinders}**

        Recommended order:
        **{recommended} cylinders**

        Predicted stockout risk:
        **{risk_label}**
        """)
