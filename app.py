"""
Climate Risk & Insurance Affordability — Interactive Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# === PATHS ===
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data" / "processed"

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Climate Risk & Insurance Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === STATE FIPS LOOKUP ===
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR", "78": "VI",
}

FEMA_REGIONS = {
    1: ["CT", "ME", "MA", "NH", "RI", "VT"],
    2: ["NJ", "NY", "PR", "VI"],
    3: ["DE", "DC", "MD", "PA", "VA", "WV"],
    4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
    5: ["IL", "IN", "MI", "MN", "OH", "WI"],
    6: ["AR", "LA", "NM", "OK", "TX"],
    7: ["IA", "KS", "MO", "NE"],
    8: ["CO", "MT", "ND", "SD", "UT", "WY"],
    9: ["AZ", "CA", "HI", "NV"],
    10: ["AK", "ID", "OR", "WA"],
}

STATE_TO_FEMA = {}
for region, states in FEMA_REGIONS.items():
    for s in states:
        STATE_TO_FEMA[s] = region


# === DATA LOADING (cached) ===
@st.cache_data
def load_data():
    data = {}

    # Risk scores
    df = pd.read_csv(MODELS / "classifier_risk_scores.csv")
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    df["state_abbr"] = df["state_fips"].map(STATE_FIPS)
    df["fema_region"] = df["state_abbr"].map(STATE_TO_FEMA)
    data["risk_scores"] = df

    # Predictions
    df = pd.read_csv(MODELS / "classifier_predictions.csv")
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    data["predictions"] = df

    # SHAP importance
    data["shap"] = pd.read_csv(MODELS / "classifier_shap_importance.csv")

    # Gamma GLM coefficients
    data["gamma_coef"] = pd.read_csv(MODELS / "gamma_glm_coefficients.csv")

    # Disasters
    data["disasters"] = pd.read_csv(DATA / "disasters_cleaned.csv")

    # County-year disasters
    df = pd.read_csv(DATA / "disasters_county_year.csv")
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    data["disasters_cy"] = df

    # Cross-module summary
    data["cross_module"] = pd.read_csv(MODELS / "validation_cross_module_summary.csv")

    # Validation files
    data["geo_cv"] = pd.read_csv(MODELS / "validation_geographic_cv.csv")
    data["temporal"] = pd.read_csv(MODELS / "validation_temporal_stability.csv")
    data["sensitivity"] = pd.read_csv(MODELS / "validation_sensitivity_thresholds.csv")
    data["ablation"] = pd.read_csv(MODELS / "validation_feature_ablation.csv")

    # ROC curves
    data["roc"] = pd.read_csv(MODELS / "classifier_roc_curves.csv")

    # Model comparison
    data["comparison"] = pd.read_csv(MODELS / "classifier_comparison_metrics.csv")

    return data


data = load_data()

# === SIDEBAR ===
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Disaster Explorer",
        "Risk Rankings",
        "Model Performance",
        "Validation & Robustness",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Climate Risk & Insurance**  \n"
    "Priya More  \n"
    "[GitHub Repo](https://github.com/Priya8975/climate-risk-insurance)"
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Climate Risk & Property Insurance Affordability Crisis")
    st.markdown(
        "Predicting which U.S. counties are at risk of becoming **uninsurable** "
        "due to escalating climate disasters."
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_disasters = len(data["disasters"])
        st.metric("Disaster Records", f"{n_disasters:,}")
    with col2:
        n_counties = data["risk_scores"]["county_fips"].nunique()
        st.metric("Counties Analyzed", f"{n_counties:,}")
    with col3:
        gb_auc = data["comparison"][
            data["comparison"]["model"] == "Gradient Boosting"
        ]["auc_roc"].values[0]
        st.metric("Best Model AUC", f"{gb_auc:.3f}")
    with col4:
        high_risk = (data["risk_scores"]["mean_risk"] > 0.5).sum()
        st.metric("High-Risk Counties", f"{high_risk:,}")

    st.markdown("---")

    # Cross-module summary
    st.subheader("Cross-Module Performance Summary")
    summary = data["cross_module"].copy()
    display_cols = [c for c in summary.columns if c != "Unnamed: 0"]
    summary = summary[display_cols]

    # Clean up: deduplicate SARIMA rows, round notes, rename columns
    summary = summary.drop_duplicates(
        subset=["module", "model_name", "primary_metric"], keep="first"
    )
    summary["notes"] = summary["notes"].apply(
        lambda x: x if pd.isna(x)
        else "AIC = {:,.0f}".format(float(x.split("=")[1])) if "AIC=" in str(x)
        else "Brier = {:.3f}".format(float(x.split("=")[1])) if "Brier=" in str(x)
        else x
    )
    summary = summary.rename(columns={
        "module": "Module",
        "model_name": "Model",
        "primary_metric": "Metric",
        "metric_value": "Test Score",
        "cv_metric": "CV Score",
        "cv_std": "CV Std",
        "notes": "Notes",
    })
    summary["Test Score"] = summary["Test Score"].apply(
        lambda x: f"{x:,.1f}" if pd.notna(x) and x > 10 else
        (f"{x:.3f}" if pd.notna(x) else "—")
    )
    summary["CV Score"] = summary["CV Score"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    summary["CV Std"] = summary["CV Std"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    summary["Notes"] = summary["Notes"].fillna("—")

    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Project description
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Sources")
        st.markdown(
            """
            - **OpenFEMA**: 41K+ disaster declarations (2004–2024)
            - **NFIP Claims**: 1.5M+ flood insurance claims
            - **Census ACS**: Demographics for 3,200+ counties
            - **FRED**: Federal Funds Rate, CPI, Home Price Index, Mortgage Rates
            """
        )
    with col2:
        st.subheader("Methodology Highlights")
        st.markdown(
            """
            - **Module 1**: SARIMA & Prophet time series forecasting
            - **Module 2**: Gamma GLM, Tweedie GLM, Logistic Regression
            - **Module 3**: Gradient Boosting & Random Forest with SHAP
            - **Module 4**: Sensitivity, geographic CV, calibration analysis
            """
        )

# ============================================================
# PAGE 2: DISASTER EXPLORER
# ============================================================
elif page == "Disaster Explorer":
    st.title("Disaster Trend Explorer")

    disasters = data["disasters"].copy()

    # Annual trends
    st.subheader("Annual Disaster Declarations (2004–2024)")
    annual = (
        disasters.groupby("declaration_year")
        .agg(
            total_declarations=("disasterNumber", "count"),
            unique_events=("disasterNumber", "nunique"),
            counties_affected=("county_fips", "nunique"),
        )
        .reset_index()
    )

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total Declarations", "Unique Events", "Counties Affected"],
    )
    fig.add_trace(
        go.Bar(x=annual["declaration_year"], y=annual["total_declarations"],
               marker_color="#1976D2", name="Declarations"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=annual["declaration_year"], y=annual["unique_events"],
               marker_color="#FF9800", name="Events"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=annual["declaration_year"], y=annual["counties_affected"],
               marker_color="#4CAF50", name="Counties"),
        row=1, col=3,
    )
    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # By disaster type
    st.subheader("Disasters by Type")
    col1, col2 = st.columns([1, 2])
    with col1:
        type_counts = (
            disasters["disaster_category"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        type_counts.columns = ["type", "count"]
        fig_pie = px.pie(
            type_counts, values="count", names="type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Trends by type over time
        type_year = (
            disasters.groupby(["declaration_year", "disaster_category"])
            .size()
            .reset_index(name="count")
        )
        top_types = type_counts["type"].head(6).tolist()
        type_year_top = type_year[type_year["disaster_category"].isin(top_types)]
        fig_line = px.line(
            type_year_top,
            x="declaration_year", y="count", color="disaster_category",
            labels={"declaration_year": "Year", "count": "Declarations",
                    "disaster_category": "Type"},
        )
        fig_line.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

    # State-level explorer
    st.subheader("State-Level Disaster Counts")
    selected_year = st.slider(
        "Select Year",
        min_value=int(disasters["declaration_year"].min()),
        max_value=int(disasters["declaration_year"].max()),
        value=2020,
    )
    state_data = (
        disasters[disasters["declaration_year"] == selected_year]
        .groupby("state")
        .size()
        .reset_index(name="count")
    )
    fig_map = px.choropleth(
        state_data, locations="state", locationmode="USA-states",
        color="count", scope="usa",
        color_continuous_scale="YlOrRd",
        labels={"count": "Disasters", "state": "State"},
        title=f"Disaster Declarations by State ({selected_year})",
    )
    fig_map.update_layout(height=450, margin=dict(t=50, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# PAGE 3: RISK RANKINGS
# ============================================================
elif page == "Risk Rankings":
    st.title("County Uninsurability Risk Rankings")

    risk = data["risk_scores"].copy()

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Counties Scored", f"{len(risk):,}")
    with col2:
        st.metric("High-Risk (>50%)", f"{(risk['mean_risk'] > 0.5).sum():,}")
    with col3:
        st.metric("Median Risk Score", f"{risk['mean_risk'].median():.3f}")

    st.markdown("---")

    # Risk by FEMA region
    st.subheader("Risk Distribution by FEMA Region")
    fig_box = px.box(
        risk.dropna(subset=["fema_region"]),
        x="fema_region", y="mean_risk",
        color="fema_region",
        labels={"fema_region": "FEMA Region", "mean_risk": "Mean Risk Score"},
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Top risk counties table
    st.subheader("Top 30 At-Risk Counties")
    col1, col2 = st.columns([1, 2])

    with col1:
        state_filter = st.multiselect(
            "Filter by State",
            options=sorted(risk["state_abbr"].dropna().unique()),
            default=[],
        )
        min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.01)

    filtered = risk.copy()
    if state_filter:
        filtered = filtered[filtered["state_abbr"].isin(state_filter)]
    filtered = filtered[filtered["mean_risk"] >= min_risk]

    with col2:
        top30 = filtered.nlargest(30, "mean_risk")[
            ["risk_rank", "county_fips", "state_abbr", "mean_risk", "max_risk",
             "n_years", "fema_region"]
        ].rename(columns={
            "risk_rank": "Rank", "county_fips": "FIPS", "state_abbr": "State",
            "mean_risk": "Mean Risk", "max_risk": "Max Risk",
            "n_years": "Years", "fema_region": "FEMA Region",
        })
        st.dataframe(
            top30.style.background_gradient(subset=["Mean Risk"], cmap="YlOrRd")
            .format({"Mean Risk": "{:.3f}", "Max Risk": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # Risk score distribution
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        risk, x="mean_risk", nbins=50,
        labels={"mean_risk": "Mean Risk Score", "count": "Counties"},
        color_discrete_sequence=["#1976D2"],
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                       annotation_text="50% threshold")
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

    # State-level risk map
    st.subheader("Average Risk Score by State")
    state_risk = (
        risk.groupby("state_abbr")["mean_risk"]
        .mean()
        .reset_index()
    )
    fig_map = px.choropleth(
        state_risk, locations="state_abbr", locationmode="USA-states",
        color="mean_risk", scope="usa",
        color_continuous_scale="YlOrRd",
        labels={"mean_risk": "Avg Risk Score", "state_abbr": "State"},
    )
    fig_map.update_layout(height=450, margin=dict(t=30, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance & Explainability")

    # ROC curves
    st.subheader("ROC Curves — Gradient Boosting vs Random Forest")
    roc = data["roc"]
    fig_roc = go.Figure()
    gb_auc = data["comparison"][
        data["comparison"]["model"] == "Gradient Boosting"
    ]["auc_roc"].values[0]
    rf_auc = data["comparison"][
        data["comparison"]["model"] == "Random Forest"
    ]["auc_roc"].values[0]

    fig_roc.add_trace(go.Scatter(
        x=roc["gb_fpr"], y=roc["gb_tpr"], mode="lines",
        name=f"Gradient Boosting (AUC = {gb_auc:.3f})",
        line=dict(color="#1976D2", width=2.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=roc["rf_fpr"], y=roc["rf_tpr"], mode="lines",
        name=f"Random Forest (AUC = {rf_auc:.3f})",
        line=dict(color="#FF9800", width=2.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450, legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Model comparison table
    st.subheader("Model Comparison")
    comp = data["comparison"].copy()
    st.dataframe(
        comp.style.format({
            c: "{:.3f}" for c in comp.columns if comp[c].dtype == "float64"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # SHAP Feature Importance
    st.subheader("SHAP Feature Importance — Top Risk Drivers")
    shap = data["shap"].copy().head(15)
    fig_shap = px.bar(
        shap.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature", orientation="h",
        labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": ""},
        color="mean_abs_shap",
        color_continuous_scale="Viridis",
    )
    fig_shap.update_layout(height=450, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # Gamma GLM Coefficients
    st.subheader("Gamma GLM — Drivers of Claim Severity")
    gamma = data["gamma_coef"].copy()
    gamma = gamma[~gamma["feature"].str.startswith("fema_region")]
    gamma = gamma[gamma["feature"] != "const"]
    gamma["color"] = gamma["significant"].map({True: "#d32f2f", False: "#9e9e9e"})

    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(
        x=gamma["exp_coef"], y=gamma["feature"],
        mode="markers",
        marker=dict(
            size=10,
            color=gamma["color"],
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=gamma["ci_upper"] - gamma["exp_coef"],
            arrayminus=gamma["exp_coef"] - gamma["ci_lower"],
        ),
        hovertemplate="<b>%{y}</b><br>exp(β) = %{x:.3f}<extra></extra>",
    ))
    fig_gamma.add_vline(x=1, line_dash="dash", line_color="gray")
    fig_gamma.update_layout(
        xaxis_title="exp(β) — Multiplicative Effect on Claim Severity",
        height=500,
        margin=dict(l=200),
    )
    st.plotly_chart(fig_gamma, use_container_width=True)
    st.caption("Red = statistically significant (p < 0.05). Dashed line = no effect.")

# ============================================================
# PAGE 5: VALIDATION & ROBUSTNESS
# ============================================================
elif page == "Validation & Robustness":
    st.title("Model Validation & Robustness")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Sensitivity Analysis",
        "Geographic Robustness",
        "Temporal Stability",
        "Feature Ablation",
    ])

    # --- Tab 1: Sensitivity ---
    with tab1:
        st.subheader("Threshold Sensitivity Analysis")
        st.markdown(
            "AUC-ROC across 16 configurations of composite target thresholds "
            "(quantile × minimum signals required)."
        )
        sens = data["sensitivity"].dropna(subset=["auc_roc"])
        pivot = sens.pivot_table(
            index="quantile", columns="min_signals", values="auc_roc"
        )
        fig_heat = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            color_continuous_scale="YlOrRd",
            labels={"x": "Min Signals", "y": "Quantile", "color": "AUC-ROC"},
            text_auto=".3f",
            zmin=0.5, zmax=1.0,
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "The model achieves strong AUC across most configurations, "
            "confirming the composite target is not an artifact of a single threshold choice."
        )

    # --- Tab 2: Geographic ---
    with tab2:
        st.subheader("Leave-One-FEMA-Region-Out Cross-Validation")
        geo = data["geo_cv"].copy()
        mean_auc = geo["auc_roc"].mean()

        fig_geo = go.Figure()
        fig_geo.add_trace(go.Bar(
            x=geo["fema_region"].astype(str),
            y=geo["auc_roc"],
            marker_color=px.colors.qualitative.Set2[:len(geo)],
            text=geo["auc_roc"].round(3),
            textposition="outside",
        ))
        fig_geo.add_hline(
            y=mean_auc, line_dash="dash", line_color="red",
            annotation_text=f"Mean = {mean_auc:.3f}",
        )
        fig_geo.update_layout(
            xaxis_title="FEMA Region",
            yaxis_title="AUC-ROC",
            yaxis_range=[0.5, 1.0],
            height=400,
        )
        st.plotly_chart(fig_geo, use_container_width=True)

        # Region details
        st.dataframe(
            geo.style.format({
                c: "{:.3f}" for c in geo.columns if geo[c].dtype == "float64"
            }),
            use_container_width=True,
            hide_index=True,
        )

    # --- Tab 3: Temporal ---
    with tab3:
        st.subheader("Expanding-Window Temporal Validation")
        temp = data["temporal"].copy()

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temp["test_year"], y=temp["auc_roc"],
            mode="lines+markers", name="AUC-ROC",
            line=dict(color="#1976D2", width=2.5),
            marker=dict(size=10),
        ))
        if "f1_score" in temp.columns:
            fig_temp.add_trace(go.Scatter(
                x=temp["test_year"], y=temp["f1_score"],
                mode="lines+markers", name="F1 Score",
                line=dict(color="#FF9800", width=2, dash="dash"),
                marker=dict(size=8),
            ))
        fig_temp.update_layout(
            xaxis_title="Test Year",
            yaxis_title="Score",
            yaxis_range=[0.3, 1.0],
            height=400,
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        st.caption(
            "AUC-ROC remains stable (0.74–0.85) across all test years, "
            "confirming the model generalizes across different climate regimes."
        )

    # --- Tab 4: Feature Ablation ---
    with tab4:
        st.subheader("Feature Ablation Study")
        st.markdown("Impact on AUC-ROC when removing each feature group.")

        abl = data["ablation"].copy().sort_values("auc_drop")
        colors = ["#d32f2f" if x < 0 else "#388e3c" for x in abl["auc_drop"]]

        fig_abl = go.Figure()
        fig_abl.add_trace(go.Bar(
            x=abl["auc_drop"],
            y=abl["group_removed"],
            orientation="h",
            marker_color=colors,
            text=abl["pct_drop"].apply(lambda x: f"{x:+.1f}%"),
            textposition="outside",
        ))
        fig_abl.add_vline(x=0, line_color="black", line_width=1)
        fig_abl.update_layout(
            xaxis_title="AUC-ROC Change (vs Full Model)",
            height=350,
        )
        st.plotly_chart(fig_abl, use_container_width=True)
        st.caption(
            "Disaster features are the most critical group — removing them "
            "causes the largest AUC drop. This confirms the model captures "
            "genuine climate risk signal."
        )

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit | Data: OpenFEMA, Census ACS, FRED")
