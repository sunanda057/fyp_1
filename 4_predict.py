"""
pages/4_predict.py — Demand prediction with calendar-locked month picker.
"""
import streamlit as st
import pandas as pd
import calendar
from datetime import date
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from pipeline import load_trained_models, build_input_vector, predict

st.title("🔮 Predict Demand")
st.caption("Forecast LPG cylinder demand for a future month.")
st.divider()

# ── Guard: models must be trained ─────────────────────────────────────────────
artefacts = load_trained_models()
if not artefacts:
    st.warning("⚠️ No trained models found. Please go to **⚙️ Train Models** first.")
    st.stop()

# ── Calendar lock logic ────────────────────────────────────────────────────────
today        = date.today()
current_year = today.year
current_month = today.month   # 1-indexed

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

st.subheader("1️⃣  Select Prediction Month")

# Year badge — locked
col_yr, col_info = st.columns([1, 4])
with col_yr:
    st.markdown(
        f"<div style='background:#1A5276;color:white;padding:8px 18px;"
        f"border-radius:8px;font-weight:600;font-size:16px;display:inline-block'>"
        f"📅 {current_year}</div>",
        unsafe_allow_html=True,
    )
with col_info:
    st.caption(
        f"Year is locked to **{current_year}**. "
        f"Only months after **{MONTH_NAMES[current_month-1]} {current_year}** are selectable."
    )

st.markdown("**Select a future month:**")

# Render month buttons in a 4-column grid
btn_cols = st.columns(12)
selected_month = st.session_state.get("selected_month", None)

for i, name in enumerate(MONTH_NAMES):
    mn = i + 1
    is_past    = mn <= current_month
    is_current = mn == current_month
    is_sel     = (selected_month == mn)

    label = f"~~{name}~~" if is_past else name
    disabled = is_past

    with btn_cols[i]:
        if is_sel:
            st.markdown(
                f"<div style='background:#1A5276;color:white;text-align:center;"
                f"padding:8px 4px;border-radius:8px;font-weight:600;cursor:pointer'>"
                f"{name}</div>",
                unsafe_allow_html=True,
            )
            if st.button("✓", key=f"m_{mn}", help=f"Selected: {name}", use_container_width=True):
                pass
        elif disabled:
            st.button(name, key=f"m_{mn}", disabled=True, use_container_width=True,
                      help="Past month — not selectable")
        else:
            if st.button(name, key=f"m_{mn}", use_container_width=True,
                         help=f"Predict for {name} {current_year}"):
                st.session_state["selected_month"] = mn
                st.rerun()

if selected_month:
    days_in_sel = calendar.monthrange(current_year, selected_month)[1]
    is_winter   = selected_month in config.WINTER_MONTHS
    is_festival = selected_month in config.FESTIVAL_MONTHS

    st.success(
        f"📅 Predicting for: **{MONTH_NAMES[selected_month-1]} {current_year}**  |  "
        f"Days: {days_in_sel}  |  "
        f"Winter: {'Yes' if is_winter else 'No'}  |  "
        f"Festival month: {'Yes' if is_festival else 'No'}"
    )
else:
    st.info("👆 Select a future month above to continue.")
    st.stop()

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("2️⃣  Zone & Household Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Zone & Subsidy**")
        zone    = st.selectbox("Warehouse Zone", config.ZONES)
        subsidy = st.selectbox("Subsidy Type",   config.SUBSIDY_TYPES)
        lpg_price = st.slider(
            "LPG Price / Cylinder (₹)",
            min_value=config.LPG_PRICE_MIN,
            max_value=config.LPG_PRICE_MAX,
            value=config.LPG_PRICE_DEFAULT, step=5,
        )

    with col2:
        st.markdown("**Household**")
        family_members = st.slider("Family Members",  2, 10, 4)
        adults         = st.slider("Adults",           1,  8, 2)
        children       = st.slider("Children",         0,  6, 2)
        income         = st.slider(
            "Monthly Income (₹)",
            min_value=config.INCOME_MIN,
            max_value=config.INCOME_MAX,
            value=config.INCOME_DEFAULT, step=500,
        )

    with col3:
        st.markdown("**Stock & Logistics**")
        opening_stock      = st.number_input("Opening Stock (cyl)",    50,  2000, 600, step=10)
        cylinders_ordered  = st.number_input("Cylinders Ordered",       0,  1500, 400, step=10)
        lead_time          = st.slider("Lead Time (days)",  1, 15, 5)
        cylinders_delivered= st.number_input("Cylinders Delivered",     0,  1500, 380, step=10)
        damaged            = st.number_input("Damaged Cylinders",       0,    50,   2)
        closing_stock      = st.number_input("Closing Stock (cyl)",     0,  2000, 500, step=10)

    submitted = st.form_submit_button("🔮 Run Prediction", type="primary", use_container_width=True)

# ── Run prediction ─────────────────────────────────────────────────────────────
if submitted:
    try:
        input_df = build_input_vector(
            zone=zone, year=current_year, month=selected_month,
            family_members=family_members, adults=adults, children=children,
            income=income, subsidy=subsidy, lpg_price=lpg_price,
            opening_stock=int(opening_stock),
            cylinders_ordered=int(cylinders_ordered),
            lead_time=lead_time,
            cylinders_delivered=int(cylinders_delivered),
            damaged=int(damaged),
            closing_stock=int(closing_stock),
            feature_cols=artefacts["feature_cols"],
        )

        result = predict(input_df, artefacts)

        st.divider()
        st.subheader("3️⃣  Prediction Results")

        # Risk colour
        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        risk_icon  = risk_color[result["risk"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Demand",    f"{result['demand']:.1f} cylinders")
        c2.metric("Recommended Order",   f"{result['recommended_order']} cylinders")
        c3.metric("Stockout Probability",f"{result['stockout_prob']}%",
                  delta=f"{risk_icon} {result['risk']} risk",
                  delta_color="off")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Business Recommendations**")
            details = {
                "Safety Stock":             f"{result['safety_stock']} cylinders",
                "Reorder Point":            f"{result['reorder_point']} cylinders",
                "Recommended Order Qty":    f"{result['recommended_order']} cylinders",
                "Stockout Risk Level":      f"{risk_icon} {result['risk']}",
                "Stockout Probability":     f"{result['stockout_prob']}%",
            }
            for k, v in details.items():
                st.markdown(f"- **{k}:** {v}")

        with col_b:
            st.markdown("**Input Summary (auto-derived)**")
            auto = {
                "Month":            f"{MONTH_NAMES[selected_month-1]} {current_year}",
                "Year":             current_year,
                "Is Winter":        "Yes" if selected_month in config.WINTER_MONTHS else "No",
                "Is Festival Month":"Yes" if selected_month in config.FESTIVAL_MONTHS else "No",
                "Days in Month":    calendar.monthrange(current_year, selected_month)[1],
                "Zone":             zone,
                "Subsidy Type":     subsidy,
            }
            for k, v in auto.items():
                st.markdown(f"- **{k}:** {v}")

        if result["risk"] == "High":
            st.error(
                f"🚨 **High stockout risk ({result['stockout_prob']}%)** for "
                f"{zone} in {MONTH_NAMES[selected_month-1]}. "
                f"Consider ordering **{result['recommended_order']}** cylinders proactively."
            )
        elif result["risk"] == "Medium":
            st.warning(
                f"⚠️ Medium stockout risk. Monitor stock levels closely for {zone}."
            )
        else:
            st.success(
                f"✅ Low stockout risk for {zone}. Normal ordering should be sufficient."
            )

        # Store in session for export
        st.session_state["last_prediction"] = {
            "Zone": zone,
            "Month": MONTH_NAMES[selected_month-1],
            "Year": current_year,
            **result,
        }

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

# ── Batch mode ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("4️⃣  Batch Prediction — All Zones")
st.caption("Run prediction for all 10 zones for the selected month at once.")

if st.button("Run All Zones", use_container_width=True):
    if not selected_month:
        st.warning("Select a month first.")
    else:
        rows = []
        for z in config.ZONES:
            try:
                idf = build_input_vector(
                    zone=z, year=current_year, month=selected_month,
                    family_members=4, adults=2, children=2,
                    income=config.INCOME_DEFAULT, subsidy="PMUY",
                    lpg_price=config.LPG_PRICE_DEFAULT,
                    opening_stock=600, cylinders_ordered=400, lead_time=5,
                    cylinders_delivered=380, damaged=2, closing_stock=500,
                    feature_cols=artefacts["feature_cols"],
                )
                r = predict(idf, artefacts)
                rows.append({
                    "Zone":             z,
                    "Predicted Demand": r["demand"],
                    "Recommended Order":r["recommended_order"],
                    "Stockout %":       r["stockout_prob"],
                    "Risk":             r["risk"],
                    "Safety Stock":     r["safety_stock"],
                    "Reorder Point":    r["reorder_point"],
                })
            except Exception as ex:
                st.warning(f"Skipped {z}: {ex}")

        if rows:
            df_batch = pd.DataFrame(rows)
            st.dataframe(
                df_batch.style.applymap(
                    lambda v: "background-color:#fde8e8" if v=="High"
                              else "background-color:#fef9e7" if v=="Medium"
                              else "background-color:#e9f7ef",
                    subset=["Risk"]
                ),
                use_container_width=True,
            )

            csv = df_batch.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Batch Predictions (CSV)",
                csv,
                file_name=f"lpg_predictions_{MONTH_NAMES[selected_month-1]}_{current_year}.csv",
                mime="text/csv",
                use_container_width=True,
            )
