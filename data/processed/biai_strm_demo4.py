import os
import glob
import math
import base64
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import altair as alt
import requests

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TxDOT | Austin Safety Intelligence Elite",
    layout="wide",
    page_icon="🛣️"
)

# --- 2. LOGO ASSET MANAGEMENT ---
# Standardized naming for corporate branding
LOGO_FILENAME = "CGI_logo_color_rgb.jpg"
GITHUB_LOGO_URL = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/CGI_logo_color_rgb.jpg"

def get_logo():
    """Locates logo asset: checks local filesystem first, then remote GitHub repository."""
    if os.path.exists(LOGO_FILENAME):
        return LOGO_FILENAME
    try:
        response = requests.head(GITHUB_LOGO_URL, timeout=5)
        if response.status_code == 200:
            return GITHUB_LOGO_URL
    except Exception:
        pass
    return None

LOGO_PATH = get_logo()

# Display logo at the top of the main dashboard area
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)
else:
    st.sidebar.error(f"⚠️ Logo asset '{LOGO_FILENAME}' missing from local and remote paths.")

alt.data_transformers.disable_max_rows()

# --- 3. RESOURCE PATH CONFIGURATION ---
DATA_DIR = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed"
LOCAL_DATA_DIR = 'data/processed'

CSV_PATH1 = f"{DATA_DIR}/atx_crash_data_2018-2026_clean.csv"
CSV_PATH2 = f"{DATA_DIR}/outputs/df_prescriptive_final_20260204_102224.csv"

MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"

# ----------------------------
# Data Loading & Processing
# ----------------------------

@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    """Fetches CSV data from a URL with basic connectivity validation."""
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    """Primary pipeline for historical data: standardizes types, severity, and mode flags."""
    try:
        df = read_csv_url(url)
    except Exception:
        # Fallback to local data folder if remote fails
        local_path = os.path.join(LOCAL_DATA_DIR, os.path.split(url)[-1])
        df = pd.read_csv(local_path)

    # Temporal feature engineering
    df["Crash timestamp"] = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    df["Year"] = df["Crash timestamp"].dt.year
    df["HOUR"] = df["Crash timestamp"].dt.hour
    df["DAY_NAME"] = df["Crash timestamp"].dt.day_name()

    # Domain-specific label mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    # Sanitize numeric columns for visualization
    numeric_cols = ["tot_injry_cnt", "crash_speed_limit", "Estimated Total Comprehensive Cost", "death_cnt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Standardize Involvement Flags (handles varying column naming conventions)
    mapping = {
        "Passenger Car": ["passenger_car_involved", "is_car"],
        "Bicycle": ["bicycle_involved", "is_bike"],
        "Pedestrian": ["pedestrian_involved", "is_ped"],
        "Motorcycle": ["motorcycle_involved", "is_mc"],
        "Commercial Veh": ["comml_mtr_veh_fl", "is_truck"],
    }
    for label, variations in mapping.items():
        actual_col = next((v for v in variations if v in df.columns), None)
        df[label] = df[actual_col].apply(lambda x: 1 if str(x).strip().upper() in ["Y", "1", "TRUE", "YES"] else 0) if actual_col else 0

    df["marker_size"] = (df["crash_speed_limit"] / 5).clip(lower=2)
    return df.dropna(subset=["latitude", "longitude"])

def fmt_dollars(x):
    """Formats numeric values into localized currency strings."""
    return f"${float(x):,.0f}" if pd.notnull(x) else "—"

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive):
    """Processes AI-generated prescriptive actions for dashboard display."""
    df = df_prescriptive.copy()
    num_cols = ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_reduction_amount", "pct_reduction"]
    for c in num_cols: 
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["address"] = df["Address"].astype(str).fillna("").str.strip()
    # Truncate rationale for table readability
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: (x[:157] + '...') if len(x) > 160 else x)
    return df

# ----------------------------
# Application Execution
# ----------------------------

# Load core datasets
df_raw1 = load_partner_data(CSV_PATH1)
try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception:
    df_prescriptive_raw = None

# --- SIDEBAR FILTERS ---
with st.sidebar:
    if LOGO_PATH: 
        st.image(LOGO_PATH, use_container_width=True)
    st.title("Global Filters")
    
    # Filter by Year
    years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    sel_years = st.multiselect("📅 Fiscal Years:", years, default=years[-4:])
    
    # Filter by Corridor (Top 10 by Impact + Full Alpha List)
    top_10 = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    corridor_list = ["All Corridors"] + top_10 + sorted(df_raw1["rpt_street_name"].unique().tolist())
    sel_street = st.selectbox("📍 Corridor:", corridor_list)

# Apply filter state to global dataframe
df = df_raw1[df_raw1["Year"].isin(sel_years)]
if sel_street != "All Corridors":
    df = df[df["rpt_street_name"] == sel_street]

# --- DASHBOARD HEADER ---
st.title("Safety Intelligence Dashboard")
st.caption(f"Analyzing Data for: **{sel_street}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")
st.markdown("---")

# --- TABBED ANALYSIS SECTIONS ---
tabs = st.tabs(["Top Predictors", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"])

# TAB 0: AI MODEL CONTEXT
with tabs[0]:
    st.subheader("Random Forest Model Analysis")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.write("##### Feature Importance (SHAP)")
        st.image("https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/BI%20to%20AI%20SHAP%20vf.png", use_container_width=True)
    with c2:
        st.write("##### Historical Economic Trends")
        cost_yr = df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        st.plotly_chart(px.bar(cost_yr, x="Year", y="Estimated Total Comprehensive Cost", color_continuous_scale="Purples", height=350), use_container_width=True)

# TAB 1: GEOGRAPHIC SPATIAL ANALYSIS
with tabs[1]:
    st.subheader("Spatial Risk Distribution")
    lat, lon = df["latitude"].median(), df["longitude"].median()
    st.plotly_chart(px.density_mapbox(
        df, lat="latitude", lon="longitude", z="Estimated Total Comprehensive Cost", 
        radius=10, center=dict(lat=lat, lon=lon), zoom=10, 
        mapbox_style="carto-positron", color_continuous_scale="Purples", height=600
    ), use_container_width=True)

# TAB 2: INCIDENT RISK PROFILE
with tabs[2]:
    st.subheader("Severity and Mode Involvement")
    c1, c2 = st.columns(2)
    with c1:
        sev_counts = df["Severity_Label"].value_counts().reset_index()
        st.plotly_chart(px.pie(
            sev_counts, values='count', names='Severity_Label', 
            title="Distribution by Severity", hole=0.4, 
            color_discrete_sequence=px.colors.sequential.Purples_r
        ), use_container_width=True)
    with c2:
        modes = ["Pedestrian", "Bicycle", "Motorcycle", "Commercial Veh"]
        mode_counts = pd.DataFrame([{"Mode": m, "Count": df[m].sum()} for m in modes])
        st.plotly_chart(px.bar(
            mode_counts, x="Mode", y="Count", 
            title="Involvement by Transport Mode", color_discrete_sequence=["#4B0082"]
        ), use_container_width=True)

# TAB 3: TEMPORAL PATTERNS
with tabs[3]:
    st.subheader("Temporal Pattern Detection")
    c1, c2 = st.columns(2)
    with c1:
        hourly = df.groupby("HOUR").size().reset_index(name="Count")
        st.plotly_chart(px.line(
            hourly, x="HOUR", y="Count", title="Hourly Frequency (24hr Clock)", 
            markers=True
        ).update_traces(line_color="#9370DB"), use_container_width=True)
    with c2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df["DAY_NAME"].value_counts().reindex(day_order).reset_index()
        st.plotly_chart(px.bar(
            dow_counts, x="DAY_NAME", y="count", 
            title="Crashes by Day of Week", color_discrete_sequence=["#4B0082"]
        ), use_container_width=True)

# TAB 4: ECONOMIC ANALYSIS
with tabs[4]:
    st.subheader("Economic Impact Deep-Dive")
    st.plotly_chart(px.scatter(
        df, x="crash_speed_limit", y="Estimated Total Comprehensive Cost", 
        color="Severity_Label", size="marker_size", 
        title="Impact Correlation: Speed vs. Cost", 
        color_discrete_sequence=px.colors.qualitative.Prism
    ), use_container_width=True)
    
    st.write("#### Top 5 High-Impact Road Segments (Aggregated Cost)")
    top_impact = df.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(5).reset_index()
    top_impact["Estimated Total Comprehensive Cost"] = top_impact["Estimated Total Comprehensive Cost"].map(fmt_dollars)
    st.table(top_impact)

# TAB 5: AI PRESCRIPTIVE ACTIONS
with tabs[5]:
    st.subheader("AI-Recommended Countermeasures")
    if df_prescriptive_raw is not None:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
        # Filter prescriptions based on the active Corridor selection
        if sel_street != "All Corridors":
            dfp = dfp[dfp["address"].str.contains(sel_street, case=False, na=False)]
        
        if not dfp.empty:
            st.metric("Potential Economic Savings (Targeted)", fmt_dollars(dfp["expected_reduction_amount"].sum()))
            st.dataframe(
                dfp[["address", "best_action", "ai_rationale_short"]].sort_values("address"), 
                use_container_width=True
            )
        else:
            st.warning("No specific prescriptive actions found for this corridor selection.")
    else:
        st.error("Prescriptive analytics dataset failed to load.")
