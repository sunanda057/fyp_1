"""
config.py — single source of truth for all project settings.
Edit this file to change paths, feature lists, or business rules.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data" / "Warehouse_Demand_Realistic_v2.xlsx"
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"

for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Excel sheet config ─────────────────────────────────────────────────────────
TRAIN_SHEET = "Train_80pct"
TEST_SHEET  = "Test_20pct"
ZONE_SHEET  = "Zone_Summary"
HEADER_ROW  = 1          # header=1 → row index 1 (0-based) is the column header

# ── Columns to drop before training ───────────────────────────────────────────
DROP_COLS = ['Record_ID', 'Family_ID', 'Month', 'Season', 'Festival']

# ── Numeric columns (for type coercion) ───────────────────────────────────────
NUMERIC_COLS = [
    'Year', 'Month_Number', 'Is_Winter', 'Is_Festival_Month', 'Days_in_Month',
    'No_of_Family_Members', 'No_of_Adults', 'No_of_Children',
    'Monthly_Income (₹)', 'LPG_Price_per_Cylinder (₹)',
    'Zone_Opening_Stock', 'Zone_Cylinders_Ordered', 'Lead_Time_Days',
    'Zone_Cylinders_Delivered', 'Zone_Safety_Stock', 'Zone_Reorder_Point',
    'Gas_Consumption_kg', 'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)', 'Units_Short',
    'Stockout_Occurred', 'Damaged_Cylinders', 'Zone_Closing_Stock',
    'Avg_Daily_Demand',
]

# ── Columns excluded from features (leakage / derived) ────────────────────────
EXCLUDE_FROM_FEATURES = [
    'Actual_Demand (cylinders)',     # regression target
    'Stockout_Occurred',             # classification target
    'Fulfilled_Demand (cylinders)',  # derived from demand
    'Units_Short',                   # derived from demand
    'Avg_Daily_Demand',              # derived from demand
    'Gas_Consumption_kg',            # too correlated with target
    'Zone_Safety_Stock',             # computed from demand
    'Zone_Reorder_Point',            # computed from demand
]

# ── Categorical columns ────────────────────────────────────────────────────────
CAT_COLS = ['Warehouse_Zone', 'Subsidy_Type']

# ── Zone definitions ───────────────────────────────────────────────────────────
ZONES = [
    'Bali', 'Berhampore', 'Domkal', 'Farakka', 'Jangipur',
    'Kandi', 'Lalbagh', 'Raghunathganj', 'Samserganj', 'Suti',
]
ZONE_ENCODING = {z: i for i, z in enumerate(ZONES)}  # alphabetical

# ── Subsidy types ──────────────────────────────────────────────────────────────
SUBSIDY_TYPES    = ['PMUY', 'Non-Subsidized']
SUBSIDY_ENCODING = {'Non-Subsidized': 0, 'PMUY': 1}

# ── Festival months (Murshidabad-specific) ─────────────────────────────────────
FESTIVAL_MONTHS = {1, 4, 8, 9, 10, 11}   # Makar Sankranti, Eid, etc.

# ── Winter months ──────────────────────────────────────────────────────────────
WINTER_MONTHS = {12, 1, 2}

# ── Business rules (configurable) ─────────────────────────────────────────────
SAFETY_STOCK_PCT  = 0.15   # 15% of predicted demand
REORDER_POINT_PCT = 0.25   # 25% of predicted demand
ORDER_BUFFER_PCT  = 1.10   # order 10% more than predicted

# ── LPG price range (for UI slider) ──────────────────────────────────────────
LPG_PRICE_MIN = 850
LPG_PRICE_MAX = 970
LPG_PRICE_DEFAULT = 915

# ── Income range (for UI slider) ─────────────────────────────────────────────
INCOME_MIN = 10000
INCOME_MAX = 50000
INCOME_DEFAULT = 23000

# ── Model artifact filenames ──────────────────────────────────────────────────
DEMAND_MODEL_FILE   = MODEL_DIR / "best_demand_model.pkl"
STOCKOUT_MODEL_FILE = MODEL_DIR / "best_stockout_model.pkl"
SCALER_FILE         = MODEL_DIR / "scaler.pkl"
ENCODERS_FILE       = MODEL_DIR / "label_encoders.pkl"
FEATURE_COLS_FILE   = MODEL_DIR / "feature_cols.pkl"
MODEL_INFO_FILE     = MODEL_DIR / "model_info.pkl"
TRAIN_DATA_FILE     = OUTPUT_DIR / "X_train.csv"
