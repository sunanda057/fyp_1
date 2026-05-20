"""
pages/2_train.py — Model training page with live progress.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from pipeline import load_data, preprocess, train_models, load_trained_models

st.title("⚙️ Train Models")
st.caption("Preprocess the dataset and train all ML models.")
st.divider()

# ── Check if already trained ───────────────────────────────────────────────────
existing = load_trained_models()
if existing:
    mi = existing["model_info"]
    st.success(
        f"✅ Models already trained.  "
        f"Best demand model: **{mi['best_regression_model']}** (R²={mi['regression_r2']:.4f}) · "
        f"Best stockout model: **{mi['best_classification_model']}** (F1={mi['classification_f1']:.4f})"
    )
    st.info("You can retrain below to refresh with the latest dataset.")

st.subheader("Training Configuration")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
| Setting | Value |
|---|---|
| Train sheet | `{config.TRAIN_SHEET}` |
| Test sheet | `{config.TEST_SHEET}` |
| Safety stock | {int(config.SAFETY_STOCK_PCT*100)}% of demand |
| Reorder point | {int(config.REORDER_POINT_PCT*100)}% of demand |
""")
with c2:
    st.markdown(f"""
| Setting | Value |
|---|---|
| Regression models | 4 (Linear, DT, RF, GB) |
| Classification models | 4 (Logistic, DT, RF, GB) |
| Cross-validation | 5-fold |
| Class imbalance | Balanced weights |
""")

st.divider()

if st.button("🚀 Start Training", type="primary", use_container_width=True):
    try:
        progress = st.progress(0, text="Loading data…")
        status   = st.empty()

        # Step 1: Load
        status.info("📂 Loading dataset from Excel…")
        train_raw, test_raw, zone_summary, warnings_list = load_data()
        progress.progress(15, text="Data loaded.")

        if warnings_list:
            for w in warnings_list:
                st.warning(w)

        st.success(f"Data loaded — {len(train_raw):,} train rows, {len(test_raw):,} test rows.")

        # Step 2: Preprocess
        status.info("🔧 Preprocessing — encoding, scaling…")
        prep = preprocess(train_raw, test_raw)
        progress.progress(35, text="Preprocessing complete.")
        st.success(
            f"Preprocessing done — {prep['X_train'].shape[1]} features, "
            f"stockout rate: {prep['y_train_stock'].mean()*100:.1f}%"
        )

        # Step 3: Train regression
        status.info("📉 Training regression models (demand)…")
        progress.progress(45, text="Training regression…")

        # Step 4: Train all
        status.info("🤖 Training all models with 5-fold cross-validation…")
        results = train_models(prep)
        progress.progress(90, text="Models trained.")

        # Step 5: Done
        progress.progress(100, text="Complete!")
        status.empty()

        st.divider()
        st.subheader("Training Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Regression models (demand)**")
            reg_display = results["reg_df"].copy()
            reg_display["R² Score"]     = reg_display["R² Score"].map("{:.4f}".format)
            reg_display["MAE"]          = reg_display["MAE"].map("{:.4f}".format)
            reg_display["RMSE"]         = reg_display["RMSE"].map("{:.4f}".format)
            reg_display["CV R² (mean)"] = reg_display["CV R² (mean)"].map("{:.4f}".format)
            reg_display["CV R² (std)"]  = reg_display["CV R² (std)"].map("{:.4f}".format)
            st.dataframe(reg_display.set_index("Model"), use_container_width=True)
            st.success(f"✅ Best: **{results['best_reg_name']}**")

        with col2:
            st.markdown("**Classification models (stockout)**")
            clf_display = results["clf_df"].copy()
            for col in ["Accuracy","Precision","Recall","F1 Score","CV F1 (mean)","CV F1 (std)"]:
                clf_display[col] = clf_display[col].map("{:.4f}".format)
            st.dataframe(clf_display.set_index("Model"), use_container_width=True)
            st.success(f"✅ Best: **{results['best_clf_name']}**")

        st.divider()
        mi = results["model_info"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Best Demand Model",   mi["best_regression_model"].split()[0])
        m2.metric("R² Score",            f"{mi['regression_r2']:.4f}")
        m3.metric("MAE",                 f"{mi['regression_mae']:.4f}")
        m4.metric("Best Stockout Model", mi["best_classification_model"].split()[0])
        m5.metric("Accuracy",            f"{mi['classification_accuracy']*100:.2f}%")
        m6.metric("F1 Score",            f"{mi['classification_f1']:.4f}")

        st.balloons()
        st.info("✅ Models saved. Go to **📊 Model Performance** or **🔮 Predict Demand**.")

    except Exception as e:
        st.error(f"Training failed: {e}")
        raise
