"""
pipeline.py — data loading, preprocessing, and model training.
Called from Streamlit pages. Returns cached artefacts.
"""
import calendar
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight
import config

warnings.filterwarnings("ignore")


# ── Data validation ────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, name: str) -> list[str]:
    """Return a list of data-quality warnings."""
    issues = []
    if 'Actual_Demand (cylinders)' in df.columns:
        neg = (df['Actual_Demand (cylinders)'] < 0).sum()
        if neg:
            issues.append(f"{name}: {neg} negative demand values found.")
    if 'Zone_Opening_Stock' in df.columns:
        neg = (df['Zone_Opening_Stock'] < 0).sum()
        if neg:
            issues.append(f"{name}: {neg} negative opening-stock values.")
    return issues


# ── Loading ────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns (train_raw, test_raw, zone_summary, warnings_list).
    Reads straight from the Excel file every time — Streamlit caches the result.
    """
    train_raw = pd.read_excel(
        config.DATA_PATH, sheet_name=config.TRAIN_SHEET, header=config.HEADER_ROW
    )
    test_raw = pd.read_excel(
        config.DATA_PATH, sheet_name=config.TEST_SHEET, header=config.HEADER_ROW
    )
    zone_summary = pd.read_excel(
        config.DATA_PATH, sheet_name=config.ZONE_SHEET, header=0
    )
    warnings_list = validate_dataframe(train_raw, "Train") + \
                    validate_dataframe(test_raw, "Test")
    return train_raw, test_raw, zone_summary, warnings_list


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> dict:
    """
    Full preprocessing pipeline.
    Returns a dict with X_train, X_test, y_* arrays, scaler, encoders, feature_cols.
    """
    train = train_raw.drop(columns=config.DROP_COLS, errors="ignore").copy()
    test  = test_raw.drop(columns=config.DROP_COLS, errors="ignore").copy()

    # Type coercion
    for col in config.NUMERIC_COLS:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col]  = pd.to_numeric(test[col],  errors="coerce")

    # Fill missing with training median (fit on train, apply to both)
    for col in train.select_dtypes(include=[np.number]).columns:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col]  = test[col].fillna(median_val)

    # Encode categoricals
    encoders = {}
    for col in config.CAT_COLS:
        if col in train.columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            test[col]  = le.transform(test[col].astype(str))
            encoders[col] = le

    # Feature / target split
    feature_cols = [c for c in train.columns if c not in config.EXCLUDE_FROM_FEATURES]

    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train_demand = train["Actual_Demand (cylinders)"]
    y_test_demand  = test["Actual_Demand (cylinders)"]
    y_train_stock  = train["Stockout_Occurred"].astype(int)
    y_test_stock   = test["Stockout_Occurred"].astype(int)

    # Scale
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

    # Persist artefacts
    joblib.dump(encoders,     config.ENCODERS_FILE)
    joblib.dump(scaler,       config.SCALER_FILE)
    joblib.dump(feature_cols, config.FEATURE_COLS_FILE)
    X_train_sc.to_csv(config.TRAIN_DATA_FILE, index=False)

    return dict(
        X_train=X_train_sc, X_test=X_test_sc,
        y_train_demand=y_train_demand, y_test_demand=y_test_demand,
        y_train_stock=y_train_stock,   y_test_stock=y_test_stock,
        scaler=scaler, encoders=encoders, feature_cols=feature_cols,
        train_raw=train_raw,
    )


# ── Training ───────────────────────────────────────────────────────────────────
def train_models(prep: dict) -> dict:
    """
    Train all regression + classification models.
    Returns full results dict including best models and comparison tables.
    """
    X_tr  = prep["X_train"]
    X_te  = prep["X_test"]
    y_d_tr = prep["y_train_demand"]
    y_d_te = prep["y_test_demand"]
    y_s_tr = prep["y_train_stock"]
    y_s_te = prep["y_test_stock"]

    # ── Regression ────────────────────────────────────────────────────────────
    reg_models = {
        "Linear Regression":  LinearRegression(),
        "Decision Tree":      DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=12,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
    }
    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_tr, y_d_tr)
        preds = model.predict(X_te)
        cv    = cross_val_score(model, X_tr, y_d_tr, cv=5, scoring="r2")
        reg_results[name] = dict(
            model=model, preds=preds,
            r2=r2_score(y_d_te, preds),
            mae=mean_absolute_error(y_d_te, preds),
            rmse=np.sqrt(mean_squared_error(y_d_te, preds)),
            cv_r2_mean=cv.mean(), cv_r2_std=cv.std(),
        )

    best_reg_name  = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_reg_model = reg_results[best_reg_name]["model"]
    best_reg_preds = reg_results[best_reg_name]["preds"]

    # ── Classification ────────────────────────────────────────────────────────
    # Handle class imbalance with sample weights
    sample_weights = compute_sample_weight("balanced", y_s_tr)

    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42,
                                                   class_weight="balanced"),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42,
                                                      class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=12,
                                                      random_state=42, n_jobs=-1,
                                                      class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                          learning_rate=0.1, random_state=42),
    }
    clf_results = {}
    for name, model in clf_models.items():
        if name == "Gradient Boosting":
            model.fit(X_tr, y_s_tr, sample_weight=sample_weights)
        else:
            model.fit(X_tr, y_s_tr)
        preds = model.predict(X_te)
        cv    = cross_val_score(model, X_tr, y_s_tr, cv=5, scoring="f1")
        clf_results[name] = dict(
            model=model, preds=preds,
            acc=accuracy_score(y_s_te, preds),
            prec=precision_score(y_s_te, preds, zero_division=0),
            rec=recall_score(y_s_te, preds, zero_division=0),
            f1=f1_score(y_s_te, preds, zero_division=0),
            cv_f1_mean=cv.mean(), cv_f1_std=cv.std(),
        )

    best_clf_name  = max(clf_results, key=lambda k: clf_results[k]["f1"])
    best_clf_model = clf_results[best_clf_name]["model"]
    best_clf_preds = clf_results[best_clf_name]["preds"]

    # ── Persist best models ───────────────────────────────────────────────────
    joblib.dump(best_reg_model, config.DEMAND_MODEL_FILE)
    joblib.dump(best_clf_model, config.STOCKOUT_MODEL_FILE)

    model_info = dict(
        best_regression_model=best_reg_name,
        regression_r2=reg_results[best_reg_name]["r2"],
        regression_mae=reg_results[best_reg_name]["mae"],
        regression_rmse=reg_results[best_reg_name]["rmse"],
        best_classification_model=best_clf_name,
        classification_accuracy=clf_results[best_clf_name]["acc"],
        classification_f1=clf_results[best_clf_name]["f1"],
    )
    joblib.dump(model_info, config.MODEL_INFO_FILE)

    # ── Comparison tables ─────────────────────────────────────────────────────
    reg_df = pd.DataFrame([
        {"Model": k, "R² Score": v["r2"], "MAE": v["mae"], "RMSE": v["rmse"],
         "CV R² (mean)": v["cv_r2_mean"], "CV R² (std)": v["cv_r2_std"]}
        for k, v in reg_results.items()
    ]).sort_values("R² Score", ascending=False).reset_index(drop=True)

    clf_df = pd.DataFrame([
        {"Model": k, "Accuracy": v["acc"], "Precision": v["prec"],
         "Recall": v["rec"], "F1 Score": v["f1"],
         "CV F1 (mean)": v["cv_f1_mean"], "CV F1 (std)": v["cv_f1_std"]}
        for k, v in clf_results.items()
    ]).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    pred_df = pd.DataFrame({
        "Actual_Demand":      y_d_te.values,
        "Predicted_Demand":   best_reg_preds,
        "Actual_Stockout":    y_s_te.values,
        "Predicted_Stockout": best_clf_preds,
    })

    return dict(
        reg_results=reg_results,   clf_results=clf_results,
        best_reg_model=best_reg_model, best_clf_model=best_clf_model,
        best_reg_name=best_reg_name,   best_clf_name=best_clf_name,
        best_reg_preds=best_reg_preds, best_clf_preds=best_clf_preds,
        reg_df=reg_df, clf_df=clf_df, pred_df=pred_df,
        model_info=model_info,
        y_test_demand=y_d_te, y_test_stock=y_s_te,
    )


# ── Prediction helpers ─────────────────────────────────────────────────────────
def load_trained_models():
    """Load persisted models + artefacts. Returns None if not trained yet."""
    try:
        return dict(
            demand_model  = joblib.load(config.DEMAND_MODEL_FILE),
            stockout_model= joblib.load(config.STOCKOUT_MODEL_FILE),
            scaler        = joblib.load(config.SCALER_FILE),
            encoders      = joblib.load(config.ENCODERS_FILE),
            feature_cols  = joblib.load(config.FEATURE_COLS_FILE),
            model_info    = joblib.load(config.MODEL_INFO_FILE),
        )
    except FileNotFoundError:
        return None


def build_input_vector(
    zone: str, year: int, month: int,
    family_members: int, adults: int, children: int,
    income: float, subsidy: str, lpg_price: float,
    opening_stock: int, cylinders_ordered: int,
    lead_time: int, cylinders_delivered: int,
    damaged: int, closing_stock: int,
    feature_cols: list,
) -> pd.DataFrame:
    """Build a single-row input DataFrame for prediction."""
    is_winter   = 1 if month in config.WINTER_MONTHS   else 0
    is_festival = 1 if month in config.FESTIVAL_MONTHS else 0
    days        = calendar.monthrange(year, month)[1]

    row = {
        "Warehouse_Zone":              config.ZONE_ENCODING[zone],
        "Year":                        year,
        "Month_Number":                month,
        "Is_Winter":                   is_winter,
        "Is_Festival_Month":           is_festival,
        "Days_in_Month":               days,
        "No_of_Family_Members":        family_members,
        "No_of_Adults":                adults,
        "No_of_Children":              children,
        "Monthly_Income (₹)":          income,
        "Subsidy_Type":                config.SUBSIDY_ENCODING[subsidy],
        "LPG_Price_per_Cylinder (₹)":  lpg_price,
        "Zone_Opening_Stock":          opening_stock,
        "Zone_Cylinders_Ordered":      cylinders_ordered,
        "Lead_Time_Days":              lead_time,
        "Zone_Cylinders_Delivered":    cylinders_delivered,
        "Damaged_Cylinders":           damaged,
        "Zone_Closing_Stock":          closing_stock,
    }
    return pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])


def predict(input_df: pd.DataFrame, artefacts: dict) -> dict:
    """Run demand + stockout prediction and compute business metrics."""
    scaled = artefacts["scaler"].transform(input_df)
    demand = float(artefacts["demand_model"].predict(scaled)[0])
    demand = max(1.0, round(demand, 2))

    stockout       = int(artefacts["stockout_model"].predict(scaled)[0])
    stockout_prob  = float(artefacts["stockout_model"].predict_proba(scaled)[0][1]) * 100

    safety_stock   = max(1, round(demand * config.SAFETY_STOCK_PCT))
    reorder_point  = max(1, round(demand * config.REORDER_POINT_PCT))
    recommended_order = round(demand * config.ORDER_BUFFER_PCT + safety_stock)

    risk = "High" if stockout_prob > 60 else "Medium" if stockout_prob > 30 else "Low"

    return dict(
        demand=demand,
        stockout=stockout,
        stockout_prob=round(stockout_prob, 1),
        risk=risk,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        recommended_order=recommended_order,
    )
