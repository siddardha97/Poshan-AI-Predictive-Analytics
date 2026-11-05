import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from app.data_sources import fetch_world_bank_nutrition, load_demo_dataset
from app.ml import train_demo_model, predict_demo_risk

st.set_page_config(page_title="Poshan AI Analytics", page_icon="🥗", layout="wide")
st.title("🥗 Poshan AI | Nutrition Predictive Analytics")
st.caption("Student prototype • Public indicators + clearly labelled demonstration data")
st.info("This prototype is for education and population-level analytics, not medical diagnosis. Demo records are synthetic; do not use predictions for real children or clinical decisions.")

with st.sidebar:
    st.header("Data controls")
    data_mode = st.radio("Dashboard data", ["Demo district dataset", "World Bank public indicators"], index=0)
    st.caption("World Bank indicators update periodically, not in real time. Poshan Tracker operational data requires authorized access.")
    refresh = st.button("Refresh public data")

if data_mode == "World Bank public indicators":
    try:
        wb = fetch_world_bank_nutrition()
        if wb.empty:
            st.warning("No values returned by the public API. Check your internet connection or use the demo dataset.")
        else:
            latest = wb.sort_values("year").groupby("indicator", as_index=False).tail(1)
            c1, c2, c3 = st.columns(3)
            for col, name in zip([c1,c2,c3], ["Stunting","Wasting","Underweight"]):
                row = latest[latest["indicator"].str.contains(name, case=False)]
                with col:
                    st.metric(name, f"{row.iloc[0]['value']:.1f}%" if not row.empty else "Not available")
            st.subheader("India: historical public nutrition indicators")
            fig = px.line(wb, x="year", y="value", color="indicator", markers=True,
                          labels={"year":"Year","value":"Estimate (%)","indicator":"Indicator"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(wb, use_container_width=True)
            st.download_button("Download public indicators CSV", wb.to_csv(index=False), "world_bank_nutrition_indicators.csv", "text/csv")
            st.markdown("Source: World Bank Indicators API, drawing on the Joint Child Malnutrition Estimates (UNICEF/WHO/World Bank).")
    except Exception as e:
        st.error(f"Could not retrieve public API data: {e}")
        st.caption("Choose Demo district dataset to continue.")
else:
    df = load_demo_dataset()
    state_options = ["All"] + sorted(df["state"].dropna().unique().tolist())
    selected_state = st.selectbox("Select state", state_options)
    view = df if selected_state == "All" else df[df["state"] == selected_state]
    a,b,c,d = st.columns(4)
    a.metric("District records", f"{len(view):,}")
    b.metric("Avg. stunting estimate", f"{view['stunting_pct'].mean():.1f}%")
    c.metric("Avg. wasting estimate", f"{view['wasting_pct'].mean():.1f}%")
    d.metric("Avg. underweight estimate", f"{view['underweight_pct'].mean():.1f}%")
    left, right = st.columns(2)
    with left:
        st.subheader("Nutrition indicators by district")
        fig = px.bar(view.sort_values("stunting_pct", ascending=False), x="district", y=["stunting_pct","wasting_pct","underweight_pct"],
                     barmode="group", labels={"value":"Estimate (%)","district":"District","variable":"Indicator"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Indicator relationship")
        fig = px.scatter(view, x="stunting_pct", y="underweight_pct", size="population_under5_est",
                         color="state", hover_name="district",
                         labels={"stunting_pct":"Stunting estimate (%)","underweight_pct":"Underweight estimate (%)"})
        st.plotly_chart(fig, use_container_width=True)
    st.caption("These rows are synthetic demo data created for testing the dashboard. They are NOT official district estimates.")
    st.dataframe(view, use_container_width=True)
    st.download_button("Download demo data CSV", view.to_csv(index=False), "demo_nutrition_data.csv", "text/csv")

st.divider()
st.header("AI risk demonstration")
st.write("This section demonstrates the user flow for a future model. It uses synthetic training data and is not validated for healthcare use.")
model, score = train_demo_model()
st.metric("Demo model test accuracy", f"{score:.0%}", help="Measured only on generated synthetic data; not a real-world performance claim.")
with st.form("risk_form"):
    x1,x2,x3 = st.columns(3)
    age_months = x1.slider("Child age (months)", 0, 59, 24)
    height_cm = x2.number_input("Height (cm)", min_value=30.0, max_value=130.0, value=82.0)
    weight_kg = x3.number_input("Weight (kg)", min_value=2.0, max_value=35.0, value=10.5)
    x4,x5,x6 = st.columns(3)
    dietary_diversity = x4.slider("Dietary diversity score (demo)", 0, 8, 4)
    recent_illness = x5.selectbox("Recent illness reported?", ["No","Yes"])
    food_security = x6.selectbox("Household food security (demo)", ["Secure","Moderate concern","High concern"])
    submitted = st.form_submit_button("Estimate demo risk")
if submitted:
    result = predict_demo_risk(model, age_months, height_cm, weight_kg, dietary_diversity, recent_illness == "Yes", food_security)
    st.subheader(f"Demonstration output: {result}")
    st.warning("Not a medical assessment. Real deployment requires ethically sourced, representative labelled data, WHO-standard growth calculations, expert validation, privacy safeguards, and approval.")
st.divider()
st.subheader("Upload a dataset")
uploaded = st.file_uploader("Upload a CSV file to inspect (no automatic clinical prediction)", type=["csv"])
if uploaded:
    try:
        up = pd.read_csv(uploaded)
        st.success(f"Loaded {len(up):,} rows and {len(up.columns)} columns.")
        st.write("Columns:", ", ".join(map(str, up.columns)))
        st.dataframe(up.head(100), use_container_width=True)
        st.download_button("Download uploaded data copy", up.to_csv(index=False), "uploaded_data_preview.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
