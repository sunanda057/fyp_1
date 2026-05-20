"""
pages/3_performance.py — Model evaluation charts.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import load_trained_models, load_data, preprocess
import config

st.title("📊 Model Performance")
st.caption("Evaluation metrics, charts, and confusion matrix.")
st.divider()

artefacts = load_trained_models()
if not artefacts:
    st.warning("⚠️ No trained models found. Please go to **⚙️ Train Models** first.")
    st.stop()

mi = artefacts["model_info"]

# ── KPI row ────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Demand Model",  mi["best_regression_model"].split()[0])
c2.metric("R² Score",      f"{mi['regression_r2']:.4f}")
c3.metric("MAE",           f"{mi['regression_mae']:.4f}")
c4.metric("Stockout Model",mi["best_classification_model"].split()[0])
c5.metric("Accuracy",      f"{mi['classification_accuracy']*100:.2f}%")
c6.metric("F1 Score",      f"{mi['classification_f1']:.4f}")
st.divider()

# ── Reload predictions ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing predictions…")
def get_predictions():
    train_raw, test_raw, _, _ = load_data()
    prep = preprocess(train_raw, test_raw)
    X_te = prep["X_test"]
    y_d  = prep["y_test_demand"]
    y_s  = prep["y_test_stock"]
    artes = load_trained_models()
    pred_demand   = artes["demand_model"].predict(X_te)
    pred_stockout = artes["stockout_model"].predict(X_te)
    return y_d.values, pred_demand, y_s.values, pred_stockout

y_d_actual, y_d_pred, y_s_actual, y_s_pred = get_predictions()

COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B']

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Regression", "🎯 Classification", "🗺️ Error Analysis", "📋 Feature Importance"
])

# ─── TAB 1: Regression ────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Actual vs Predicted Demand**")
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(y_d_actual, y_d_pred, alpha=0.35, color='#1A5276', s=15)
        lims = [min(y_d_actual.min(), y_d_pred.min())-0.2,
                max(y_d_actual.max(), y_d_pred.max())+0.2]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
        ax.set_xlabel('Actual demand (cylinders)')
        ax.set_ylabel('Predicted demand (cylinders)')
        ax.set_title(f'{mi["best_regression_model"]}  |  R²={mi["regression_r2"]:.4f}')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Prediction Error Distribution**")
        errors = y_d_actual - y_d_pred
        fig, ax = plt.subplots(figsize=(5,5))
        ax.hist(errors, bins=40, color='#1A5276', alpha=0.75, edgecolor='white')
        ax.axvline(0,            color='red',    linestyle='--', linewidth=1.5, label='Zero error')
        ax.axvline(errors.mean(),color='orange', linestyle='--', linewidth=1.5,
                   label=f'Mean: {errors.mean():.3f}')
        ax.set_xlabel('Error (Actual − Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error distribution (centred ≈ unbiased)')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Residuals vs Predicted (heteroscedasticity check)**")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(y_d_pred, errors, alpha=0.3, s=12, color='#1A5276')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted demand')
    ax.set_ylabel('Residual')
    ax.set_title('Residual plot — random scatter around 0 is ideal')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── TAB 2: Classification ────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_s_actual, y_s_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Stockout','Stockout'],
                    yticklabels=['No Stockout','Stockout'],
                    annot_kws={'size':14, 'weight':'bold'}, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(mi["best_classification_model"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Classification Metrics**")
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "True Negatives (correct no-stockout)": int(tn),
            "True Positives (correct stockout)":    int(tp),
            "False Positives (false alarm)":        int(fp),
            "False Negatives (missed stockout)":    int(fn),
        }
        for k,v in metrics.items():
            color = "normal"
            if "False Negative" in k and v > 0: color = "inverse"
            st.metric(k, v)

        st.info(
            "False Negatives are the most costly — missed stockouts mean "
            "real shortages in the field."
        )

# ─── TAB 3: Error Analysis ─────────────────────────────────────────────────────
with tab3:
    st.markdown("**Prediction accuracy buckets**")
    errors = y_d_actual - y_d_pred
    buckets = {
        "Within ±0.5 cyl":  ((np.abs(errors) <= 0.5).sum()),
        "Within ±1 cyl":    ((np.abs(errors) <= 1.0).sum()),
        "Within ±2 cyl":    ((np.abs(errors) <= 2.0).sum()),
        "Error > 2 cyl":    ((np.abs(errors)  > 2.0).sum()),
    }
    total = len(errors)
    cols = st.columns(4)
    for i, (label, count) in enumerate(buckets.items()):
        cols[i].metric(label, f"{count}", f"{count/total*100:.1f}%")

    st.markdown("**Error by demand range**")
    df_err = pd.DataFrame({"actual": y_d_actual, "error": np.abs(errors)})
    df_err["demand_bin"] = pd.cut(df_err["actual"],
                                   bins=[0,1,2,3,4,5,100],
                                   labels=["0–1","1–2","2–3","3–4","4–5","5+"])
    grouped = df_err.groupby("demand_bin", observed=True)["error"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(grouped["demand_bin"].astype(str), grouped["error"], color='#1A5276', edgecolor='white')
    ax.set_xlabel("Actual demand range (cylinders)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("MAE by demand bucket")
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── TAB 4: Feature Importance ────────────────────────────────────────────────
with tab4:
    artes = load_trained_models()
    dm = artes["demand_model"]
    feature_cols = artes["feature_cols"]

    if hasattr(dm, "feature_importances_"):
        imp = pd.Series(dm.feature_importances_, index=feature_cols) \
                .sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(imp.index, imp.values, color='#1A5276', height=0.6)
        ax.set_xlabel("Importance score")
        ax.set_title(f"Top feature importances — {mi['best_regression_model']}")
        for bar, val in zip(bars, imp.values):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Feature importance not available for linear models.")
