import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from app.data_sources import (
    fetch_world_bank_nutrition,
    load_demo_dataset,
    load_nfhs_dataset,
)
from app.real_ml import train_real_model, compare_models
st.set_page_config(
    page_title="Poshan AI Analytics",
    page_icon="🥗",
    layout="wide",
)
st.title("🥗 Poshan AI | Nutrition Predictive Analytics")
st.caption(
    "Student prototype | NFHS-5 survey data, public indicators "
    "and demonstration data"
)
st.info(
    "For education and population-level analysis only, not medical diagnosis. "
    "NFHS-5 contains historical survey estimates from 2019–2021. "
    "The Linear Regression model uses historical district-level survey data "
    "and its predictions are exploratory, not clinically validated."
)


with st.sidebar:
    st.header("Data controls")

    data_mode = st.radio(
        "Dashboard data",
        [
            "NFHS-5 District Survey Data",
            "Demo district dataset",
            "World Bank public indicators",
        ],
        index=0,
    )

    st.caption(
        "NFHS-5: historical district survey data. "
        "World Bank: historical national indicators. "
        "Neither is a live measurement feed."
    )

    refresh = st.button("Refresh public data")


# --------------------------------------------------
# OPTION 1: WORLD BANK PUBLIC INDICATORS
# --------------------------------------------------

if data_mode == "World Bank public indicators":
    try:
        with st.spinner("Loading World Bank indicators..."):
            wb = fetch_world_bank_nutrition()

        if wb.empty:
            st.warning(
                "No indicator values were returned. "
                "Check your internet connection and try again."
            )
        else:
            latest = (
                wb.sort_values("year")
                .groupby("indicator", as_index=False)
                .tail(1)
            )

            c1, c2, c3 = st.columns(3)

            for col, name in zip(
                [c1, c2, c3],
                ["Stunting", "Wasting", "Underweight"],
            ):
                row = latest[
                    latest["indicator"].str.contains(
                        name, case=False, na=False
                    )
                ]

                with col:
                    if not row.empty:
                        st.metric(
                            name,
                            f"{row.iloc[0]['value']:.1f}%",
                        )
                        st.caption(
                            f"Latest available year: "
                            f"{int(row.iloc[0]['year'])}"
                        )
                    else:
                        st.metric(name, "Not available")

            st.subheader("India: historical nutrition indicators")

            fig = px.line(
                wb,
                x="year",
                y="value",
                color="indicator",
                markers=True,
                labels={
                    "year": "Year",
                    "value": "Estimate (%)",
                    "indicator": "Indicator",
                },
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(wb, use_container_width=True)

            st.download_button(
                "Download public indicators CSV",
                wb.to_csv(index=False),
                "world_bank_nutrition_indicators.csv",
                "text/csv",
            )

            st.caption(
                "Source: World Bank Indicators API. "
                "These are periodic historical estimates, not real-time data."
            )

    except Exception as e:
        st.error(f"Could not retrieve public API data: {e}")


# --------------------------------------------------
# OPTION 2: DISTRICT DATA
# --------------------------------------------------

elif data_mode in [
    "NFHS-5 District Survey Data",
    "Demo district dataset",
]:
    is_nfhs = data_mode == "NFHS-5 District Survey Data"

    try:
        if is_nfhs:
            df = load_nfhs_dataset()
            source_label = "NFHS-5 district factsheets, 2019–2021"
        else:
            df = load_demo_dataset()
            source_label = "Synthetic demonstration dataset"

        st.subheader(source_label)

        if is_nfhs:
            st.warning(
                "These are historical survey estimates, not live readings. "
                "District comparisons describe the survey data and should "
                "not be interpreted as current conditions."
            )
        else:
            st.warning(
                "This dataset is synthetic and is used only to test "
                "the dashboard. It is not official government data."
            )

        # State filter
        state_options = ["All"] + sorted(
            df["state"].dropna().unique().tolist()
        )

        selected_state = st.selectbox("Select state / UT", state_options)

        view = (
            df.copy()
            if selected_state == "All"
            else df[df["state"] == selected_state].copy()
        )

        if view.empty:
            st.warning("No district records found for this selection.")
        else:
            # Summary metrics
            a, b, c, d = st.columns(4)

            a.metric("District records", f"{len(view):,}")

            b.metric(
                "Average stunting",
                (
                    f"{view['stunting_pct'].mean():.1f}%"
                    if view["stunting_pct"].notna().any()
                    else "Not available"
                ),
            )

            c.metric(
                "Average wasting",
                (
                    f"{view['wasting_pct'].mean():.1f}%"
                    if view["wasting_pct"].notna().any()
                    else "Not available"
                ),
            )

            d.metric(
                "Average underweight",
                (
                    f"{view['underweight_pct'].mean():.1f}%"
                    if view["underweight_pct"].notna().any()
                    else "Not available"
                ),
            )

            # District comparison chart
            st.subheader("Nutrition indicators by district")

            indicator_columns = [
                "stunting_pct",
                "wasting_pct",
                "underweight_pct",
            ]

            chart_data = view.dropna(
                subset=indicator_columns, how="all"
            )

            if not chart_data.empty:
                chart_data = chart_data.sort_values(
                    "stunting_pct",
                    ascending=False,
                    na_position="last",
                )

                fig = px.bar(
                    chart_data,
                    x="district",
                    y=indicator_columns,
                    barmode="group",
                    labels={
                        "district": "District",
                        "value": "Estimate (%)",
                        "variable": "Nutrition indicator",
                        "stunting_pct": "Stunting",
                        "wasting_pct": "Wasting",
                        "underweight_pct": "Underweight",
                    },
                )

                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid nutrition indicator values to plot.")

            # Relationship chart
            st.subheader("Stunting and underweight relationship")

            scatter_data = view.dropna(
                subset=["stunting_pct", "underweight_pct"]
            )

            if not scatter_data.empty:
                fig = px.scatter(
                    scatter_data,
                    x="stunting_pct",
                    y="underweight_pct",
                    color="state",
                    hover_name="district",
                    labels={
                        "stunting_pct": "Stunting estimate (%)",
                        "underweight_pct": "Underweight estimate (%)",
                        "state": "State / UT",
                    },
                    title="District-level indicator comparison",
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Not enough valid stunting and underweight values "
                    "to create the relationship chart."
                )

            # Additional indicators available in the NFHS dataset
            if is_nfhs:
                st.subheader("Other available district indicators")

                extra_columns = [
                    col for col in [
                        "improved_drinking_water_pct",
                        "improved_sanitation_pct",
                        "clean_cooking_fuel_pct",
                        "child_anaemia_pct",
                        "overweight_under5_pct",
                        "severe_wasting_under5_pct",
                    ]
                    if col in view.columns
                ]

                if extra_columns:
                    extra_chart = view.melt(
                        id_vars=["district"],
                        value_vars=extra_columns,
                        var_name="Indicator",
                        value_name="Estimate (%)",
                    )

                    fig = px.bar(
                        extra_chart,
                        x="district",
                        y="Estimate (%)",
                        color="Indicator",
                        barmode="group",
                    )

                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("District data table")
            st.dataframe(view, use_container_width=True)

            download_name = (
                "nfhs5_district_nutrition.csv"
                if is_nfhs
                else "demo_nutrition_data.csv"
            )

            st.download_button(
                "Download displayed district data",
                view.to_csv(index=False),
                download_name,
                "text/csv",
            )

    except Exception as e:
        st.error(f"Could not load district data: {e}")
        st.caption(
            "Check the CSV filename, column names and data_sources.py."
        )



# --------------------------------------------------
# CSV UPLOAD PREVIEW
# --------------------------------------------------

st.divider()
st.subheader("Upload a dataset")

uploaded = st.file_uploader(
    "Upload a CSV file to inspect",
    type=["csv"],
)

if uploaded is not None:
    try:
        up = pd.read_csv(uploaded)

        st.success(
            f"Loaded {len(up):,} rows and {len(up.columns)} columns."
        )

        st.write("Columns:", ", ".join(map(str, up.columns)))
        st.dataframe(up.head(100), use_container_width=True)

        st.download_button(
            "Download uploaded data copy",
            up.to_csv(index=False),
            "uploaded_data_preview.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"Could not read CSV: {e}")


# --------------------------------------------------
# REAL NFHS-5 MACHINE LEARNING MODEL
# --------------------------------------------------

st.divider()
st.header("Machine Learning Model Comparison")

comparison_df = compare_models()

st.write("Comparison of four machine learning algorithms:")
st.dataframe(comparison_df, use_container_width=True)

st.subheader("Algorithm Performance Comparison")

error_data = comparison_df.melt(
    id_vars="Algorithm",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error"
)

fig1 = px.bar(
    error_data,
    x="Algorithm",
    y="Error",
    color="Metric",
    barmode="group",
    title="MAE and RMSE Comparison"
)

st.plotly_chart(fig1, use_container_width=True)
st.subheader("R² Score Comparison")

r2_data = comparison_df.melt(
    id_vars="Algorithm",
    value_vars=["Test R2", "CV Mean R2"],
    var_name="Metric",
    value_name="R2 Score"
)

fig2 = px.bar(
    r2_data,
    x="Algorithm",
    y="R2 Score",
    color="Metric",
    barmode="group",
    title="Test R² and Cross-Validation R²"
)

st.plotly_chart(fig2, use_container_width=True)
st.header("AI-Based Stunting Prediction")

st.write(
    "This model uses actual NFHS-5 district survey data "
    "to estimate the percentage of stunting among children "
    "under five years of age."
)

# Train the real model
model, results, validation_df = train_real_model()

st.subheader("Model Validation: Actual vs Predicted")

st.write(
    "Comparison of actual NFHS-5 stunting values "
    "with model-predicted values on the test dataset."
)

# Display validation table
st.dataframe(
    validation_df.head(20).round(2),
    use_container_width=True
)

# Actual vs Predicted Scatter Plot
st.subheader("Actual vs Predicted Scatter Plot")

import plotly.express as px

fig = px.scatter(
    validation_df,
    x="Actual Stunting (%)",
    y="Predicted Stunting (%)",
    title="Actual vs Predicted Stunting Percentage",
    labels={
        "Actual Stunting (%)": "Actual Stunting (%)",
        "Predicted Stunting (%)": "Predicted Stunting (%)"
    }
)

# Add perfect prediction reference line
min_value = min(
    validation_df["Actual Stunting (%)"].min(),
    validation_df["Predicted Stunting (%)"].min()
)

max_value = max(
    validation_df["Actual Stunting (%)"].max(),
    validation_df["Predicted Stunting (%)"].max()
)
fig.add_shape(
    type="line",
    x0=min_value,
    y0=min_value,
    x1=max_value,
    y1=max_value,
    line=dict(dash="dash")
)

st.plotly_chart(fig, use_container_width=True)
# --------------------------------------------------
# LINEAR REGRESSION COEFFICIENT ANALYSIS
# --------------------------------------------------

st.subheader("Linear Regression Coefficient Analysis")

feature_names = [
    "Wasting (%)",
    "Severe Wasting (%)",
    "Underweight (%)",
    "Improved Drinking Water (%)",
    "Improved Sanitation (%)",
    "Clean Cooking Fuel (%)",
    "Households Surveyed"
]

coefficients = model.named_steps["regressor"].coef_

coefficient_df = pd.DataFrame({
    "Indicator": feature_names,
    "Coefficient": coefficients
})

coefficient_df["Relationship"] = coefficient_df[
    "Coefficient"
].apply(
    lambda x: "Positive" if x > 0
    else "Negative" if x < 0
    else "Zero"
)

coefficient_df = coefficient_df.sort_values(
    by="Coefficient",
    ascending=False
)

st.write(
    "The coefficients show how each indicator is associated "
    "with the model's estimated stunting percentage."
)

st.dataframe(coefficient_df, use_container_width=True)

fig = px.bar(
    coefficient_df,
    x="Coefficient",
    y="Indicator",
    color="Relationship",
    orientation="h",
    title="Linear Regression Coefficient Analysis"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Positive coefficients indicate a positive model relationship; "
    "negative coefficients indicate a negative model relationship. "
    "These are associations, not proof of causation."
)

# Display model evaluation metrics
col1, col2, col3 = st.columns(3)

col1.metric("MAE", results["MAE"])
col2.metric("RMSE", results["RMSE"])
col3.metric("Test R² Score", results["R2 Score"])

col4, col5 = st.columns(2)

col4.metric("5-Fold CV Mean R²", results["CV R2 Mean"])
col5.metric("CV Standard Deviation", results["CV R2 Std"])

st.caption(
    f"Training records: {results['Training Records']} | "
    f"Testing records: {results['Testing Records']}"
)

st.info(
    "The model estimates district-level stunting from other "
    "NFHS-5 indicators. It does not predict an individual "
    "child's health or future malnutrition."
)

st.subheader("Estimate District Stunting Percentage")

with st.form("stunting_prediction_form"):

    col1, col2 = st.columns(2)

    wasted = col1.number_input(
        "Wasted children (%)",
        min_value=0.0,
        max_value=100.0,
        value=18.5
    )

    severely_wasted = col2.number_input(
        "Severely wasted children (%)",
        min_value=0.0,
        max_value=100.0,
        value=7.5
    )

    col3, col4 = st.columns(2)

    underweight = col3.number_input(
        "Underweight children (%)",
        min_value=0.0,
        max_value=100.0,
        value=29.5
    )

    drinking_water = col4.number_input(
        "Improved drinking water (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

    col5, col6 = st.columns(2)

    sanitation = col5.number_input(
        "Improved sanitation (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    cooking_fuel = col6.number_input(
        "Clean cooking fuel (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    households = st.number_input(
        "Households surveyed",
        min_value=1,
        value=1000
    )

    predict_button = st.form_submit_button(
        "Predict Stunting Percentage"
    )

if predict_button:

    try:
        input_data = pd.DataFrame([{
            "wasted_under5_pct": wasted,
            "severely_wasted_under5_pct": severely_wasted,
            "underweight_under5_pct": underweight,
            "improved_drinking_water_pct": drinking_water,
            "improved_sanitation_pct": sanitation,
            "clean_cooking_fuel_pct": cooking_fuel,
            "households_surveyed": households
        }])

        prediction = float(model.predict(input_data)[0])

        if not np.isfinite(prediction):
            st.error("Invalid prediction. Please check your inputs.")

        elif prediction < 0 or prediction > 100:
            st.warning(
                f"Model produced an out-of-range estimate: "
                f"{prediction:.2f}%. Please interpret cautiously."
            )

        else:
            st.success(
                f"Estimated district stunting percentage: "
                f"{prediction:.2f}%"
            )

        st.caption(
            "This is an exploratory model estimate, not an official "
            "district statistic or medical assessment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    