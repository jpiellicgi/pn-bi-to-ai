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
# Path configuration for corporate branding
LOGO_FILENAME = "CGI_logo_color_rgb.jpg"
GITHUB_LOGO_URL = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/CGI_logo_color_rgb.jpg"

def get_logo():
    """Locates logo: checks local directory first, then falls back to GitHub."""
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

# Render logo at the top of the app
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)
else:
    st.sidebar.error(f"⚠️ Logo asset '{LOGO_FILENAME}' not found.")

alt.data_transformers.disable_max_rows()

# --- 3. RESOURCE PATHS ---
DATA_DIR = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed"
LOCAL_DATA_DIR = 'data/processed'

CSV_PATH1 = f"{DATA_DIR}/atx_crash_data_2018-2026_clean.csv"
CSV_PATH2 = f"{DATA_DIR}/outputs/df_prescriptive_final_20260204_102224.csv"

MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"

# ----------------------------
# Data Loading & Transformation
# ----------------------------

@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    """Fetches CSV with validation for connectivity and content."""
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    """Primary pipeline for historical data: cleans, maps severity, and detects transport modes."""
    try:
        df = read_csv_url(url)
    except Exception:
        local_path = os.path.join(LOCAL_DATA_DIR, os.path.split(url)[-1])
        df = pd.read_csv(local_path)

    # Feature Engineering
    df["Crash timestamp"] = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    df["Year"] = df["Crash timestamp"].dt.year
    df["HOUR"] = df["Crash timestamp"].dt.hour
    df["DAY_NAME"] = df["Crash timestamp"].dt.day_name()

    # Domain Mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    # Numeric formatting for KPIs
    cols = ["tot_injry_cnt", "crash_speed_limit", "Estimated Total Comprehensive Cost", "death_cnt"]
    for col in cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Standardize Transport Mode columns (handles varied naming conventions)
    mapping = {
        "Passenger Car": ["passenger_car_involved", "car_fl", "is_car"],
        "Bicycle": ["bicycle_involved", "bicycle_fl", "is_bike"],
        "Pedestrian": ["pedestrian_involved", "pedestrian_fl", "is_ped"],
        "Motorcycle": ["motorcycle_involved", "motorcycle_fl", "is_mc"],
        "Commercial Veh": ["comml_mtr_veh_fl", "cmv_involved", "is_truck"],
    }
    for label, vars in mapping.items():
        actual = next((v for v in vars if v in df.columns), None)
        df[label] = df[actual].apply(lambda x: 1 if str(x).strip().upper() in ["Y", "1", "TRUE", "YES"] else 0) if actual else 0

    df["marker_size"] = (df["crash_speed_limit"] / 5).clip(lower=2)
    df["Speed_Bin"] = pd.cut(df["crash_speed_limit"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 110], 
                             labels=["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"])

    return df.dropna(subset=["latitude", "longitude"])

# ----------------------------
# Prescriptive Helpers
# ----------------------------

def fmt_dollars(x):
    return f"${float(x):,.0f}" if pd.notnull(x) else "—"

def normalize_pct_reduction(series):
    """Detects if scale is 0-1 or 0-100 and standardizes to 0-1 ratio."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty: return s
    return s / 100.0 if s.dropna().quantile(0.95) > 1.5 else s

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive):
    df = df_prescriptive.copy()
    num_cols = ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_cost_after_action", "expected_reduction_amount", "pct_reduction"]
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["address"] = df["Address"].astype(str).fillna("").str.strip()
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: x[:159] + "…" if len(x) > 160 else x)
    return df

# ----------------------------
# Main Logic
# ----------------------------

# Load core datasets
df_raw1 = load_partner_data(CSV_PATH1)
try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception as e:
    df_prescriptive_raw = None
    prescriptive_load_error = str(e)

# --- SIDEBAR FILTERS ---
with st.sidebar:
    if LOGO_PATH: st.image(LOGO_PATH, use_container_width=True)
    st.title("Global Filters")
    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    
    top_10 = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    corridor_options = ["All Corridors"] + top_10 + sorted(df_raw1["rpt_street_name"].unique().tolist())
    selected_street = st.selectbox("📍 Corridor:", corridor_options)

# Apply global filters
df = df_raw1[df_raw1["Year"].isin(selected_years)]
if selected_street != "All Corridors":
    df = df[df["rpt_street_name"] == selected_street]

# --- DASHBOARD HEADER ---
st.title("Safety Intelligence Dashboard")
st.caption(f"Currently Analyzing: **{selected_street}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")
st.markdown("---")

# --- TABBED ANALYSIS ---
tabs = st.tabs(["Top Predictors", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"])

# Tab 0: Predictors
with tabs[0]:
    st.write("##### Feature importance derived from Random Forest model training.")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.subheader("Top Predictors")
        st.image("https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/BI%20to%20AI%20SHAP%20vf.png", width=800)
    with c2:
        st.subheader("Historical Cost Trends")
        cost_yr = df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        st.plotly_chart(px.bar(cost_yr, x="Year", y="Estimated Total Comprehensive Cost", color_continuous_scale="Purples", height=350), use_container_width=True)

# Tab 1: Geographic
with tabs[1]:
    lat_c, lon_c = (df["latitude"].median(), df["longitude"].median()) if not df.empty else (30.2672, -97.7431)
    st.plotly_chart(px.density_mapbox(df, lat="latitude", lon="longitude", z="Estimated Total Comprehensive Cost", radius=10, center=dict(lat=lat_c, lon=lon_c), zoom=10, mapbox_style="open-street-map", color_continuous_scale="Purples", height=600), use_container_width=True)

# Tab 2: Incident Risk (FIXED)
with tabs[2]:
    st.subheader("Severity & Mode Involvement")
    c1, c2 = st.columns(2)
    with c1:
        sev_data = df["Severity_Label"].value_counts().reset_index()
        st.plotly_chart(px.pie(sev_data, values='count', names='Severity_Label', hole=0.4, title="Severity Distribution", color_discrete_sequence=px.colors.sequential.Purples_r), use_container_width=True)
    with c2:
        modes = ["Pedestrian", "Bicycle", "Motorcycle", "Commercial Veh"]
        mode_data = pd.DataFrame([{"Mode": m, "Count": df[m].sum()} for m in modes])
        st.plotly_chart(px.bar(mode_data, x="Mode", y="Count", title="Involvement by Mode", color_discrete_sequence=["#6A0DAD"]), use_container_width=True)

# Tab 3: Temporal (FIXED)
with tabs[3]:
    st.subheader("Temporal Distribution")
    c1, c2 = st.columns(2)
    with c1:
        hourly = df.groupby("HOUR").size().reset_index(name="Crashes")
        st.plotly_chart(px.line(hourly, x="HOUR", y="Crashes", title="Hourly Volume", markers=True).update_traces(line_color="#4B0082"), use_container_width=True)
    with c2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df["DAY_NAME"].value_counts().reindex(day_order).reset_index()
        st.plotly_chart(px.bar(dow, x="DAY_NAME", y="count", title="Day of Week Volume", color_discrete_sequence=["#9370DB"]), use_container_width=True)

# Tab 4: Economic (FIXED)
with tabs[4]:
    st.subheader("Economic Impact Analysis")
    st.plotly_chart(px.scatter(df, x="crash_speed_limit", y="Estimated Total Comprehensive Cost", color="Severity_Label", title="Cost vs Speed Correlation", size_max=15), use_container_width=True)
    st.write("#### Top 5 High-Impact Streets (by Cost)")
    top_streets = df.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(5).reset_index()
    top_streets["Estimated Total Comprehensive Cost"] = top_streets["Estimated Total Comprehensive Cost"].map(fmt_dollars)
    st.table(top_streets)

# Tab 5: Prescriptive
with tabs[5]:
    st.subheader("AI-Recommended Prescriptive Actions")
    if df_prescriptive_raw is not None:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
        dfp = dfp[dfp["best_action"] != "no_change"]
        if selected_street != "All Corridors":
            dfp = dfp[dfp["address"].str.contains(selected_street, case=False, na=False)]
        
        if not dfp.empty:
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Expected Savings", fmt_dollars(dfp["expected_reduction_amount"].sum()))
            p2.metric("Avg. Reduction %", f"{(dfp['pct_reduction_norm'].mean()*100):.1f}%")
            p3.metric("Primary Recommendation", dfp["best_action"].mode()[0])
            st.dataframe(dfp[["address", "best_action", "expected_reduction_amount", "ai_rationale_short"]].sort_values("expected_reduction_amount", ascending=False), use_container_width=True)
        else:
            st.warning("No prescriptive records match the current filters.")
    else:
        st.error("Prescriptive data unavailable.")
