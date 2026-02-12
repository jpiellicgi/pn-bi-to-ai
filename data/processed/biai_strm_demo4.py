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

# --- 2. ROBUST LOGO LOADER ---
# This looks for your file locally, then on GitHub
LOGO_FILENAME = "CGI_logo_color_rgb.jpg"
GITHUB_LOGO_URL = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/CGI_logo_rgb_white.png"

def get_logo():
    """Returns a valid path or URL for the logo, or None if not found."""
    if os.path.exists(LOGO_FILENAME):
        return LOGO_FILENAME
    # Test if GitHub link is alive
    try:
        response = requests.head(GITHUB_LOGO_URL, timeout=5)
        if response.status_code == 200:
            return GITHUB_LOGO_URL
    except:
        pass
    return None

LOGO_PATH = get_logo()

# Render logo at the top of the main area
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)
else:
    st.sidebar.error(f"⚠️ Logo file '{LOGO_FILENAME}' not found locally or on GitHub.")

# Make Altair nicer
alt.data_transformers.disable_max_rows()

# --- 3. PATH CONFIGURATION ---
DATA_DIR = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed"
LOCAL_DATA_DIR = 'data/processed'

CSV_FILENAME1 = "atx_crash_data_2018-2026_clean.csv"
CSV_PATH1 = f"{DATA_DIR}/{CSV_FILENAME1}"

CSV_FILENAME2 = "df_prescriptive_final_20260204_102224.csv"
CSV_PATH2 = f"{DATA_DIR}/outputs/{CSV_FILENAME2}"

MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"

# ----------------------------
# Data Loaders
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")
    if len(r.content) <= 10:
        raise ValueError(f"Remote file is too small. URL: {url}")
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    try:
        df = read_csv_url(url)
    except Exception:
        local_path = os.path.join(LOCAL_DATA_DIR, os.path.split(url)[-1])
        df = pd.read_csv(local_path)

    df["Crash timestamp"] = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    df["Year"] = df["Crash timestamp"].dt.year
    df["HOUR"] = df["Crash timestamp"].dt.hour
    df["DAY_NAME"] = df["Crash timestamp"].dt.day_name()

    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    cols_to_fix = ["tot_injry_cnt", "crash_speed_limit", "Estimated Total Comprehensive Cost"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    mapping = {
        "Passenger Car": ["passenger_car_involved", "car_fl", "is_car"],
        "Bicycle": ["bicycle_involved", "bicycle_fl", "is_bike"],
        "Pedestrian": ["pedestrian_involved", "pedestrian_fl", "is_ped"],
        "Motorcycle": ["motorcycle_involved", "motorcycle_fl", "is_mc"],
        "Commercial Veh": ["comml_mtr_veh_fl", "cmv_involved", "is_truck"],
    }

    for clean_label, variations in mapping.items():
        actual_col = next((v for v in variations if v in df.columns), None)
        if actual_col:
            df[clean_label] = df[actual_col].apply(lambda x: 1 if str(x).strip().upper() in ["Y", "1", "TRUE", "YES"] else 0)
        else:
            df[clean_label] = 0

    df["marker_size"] = (df["crash_speed_limit"] / 5).clip(lower=2)
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 110]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
    df["Speed_Bin"] = pd.cut(df["crash_speed_limit"], bins=bins, labels=labels)

    return df.dropna(subset=["latitude", "longitude"])

# ----------------------------
# Prescriptive Helpers
# ----------------------------
REQUIRED_COLS = ["latitude", "longitude", "Address", "pred_est_ttl_comp_cost", "best_action", "expected_cost_after_action", "expected_reduction_amount", "pct_reduction", "ai_rationale"]

def _coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def normalize_pct_reduction(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty: return s
    return s / 100.0 if s.dropna().quantile(0.95) > 1.5 else s

def make_location_id(df):
    return df["latitude"].round(5).astype(str) + ", " + df["longitude"].round(5).astype(str)

def fmt_dollars(x):
    try:
        return f"${float(x):,.0f}" if pd.notnull(x) else "—"
    except:
        return "—"

def compact_text(s, n=140):
    s = str(s).strip()
    return s if len(s) <= n else s[:n-1] + "…"

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive):
    df = df_prescriptive.copy()
    df = _coerce_numeric(df, ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_cost_after_action", "expected_reduction_amount", "pct_reduction"])
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["location_id"] = make_location_id(df)
    df["address"] = df["Address"].astype(str).fillna("").str.strip()
    df["address_short"] = df["address"].map(lambda x: compact_text(x, 80))
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x, 160))
    return df

# ----------------------------
# Load Core Data
# ----------------------------
try:
    df_raw1 = load_partner_data(CSV_PATH1)
except Exception as e:
    st.error(f"🛑 Crash dataset load failed: {e}")
    st.stop()

try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception as e:
    df_prescriptive_raw = None
    prescriptive_load_error = str(e)

# --- 4. SIDEBAR ---
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.title("Global Filters")
    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    top_10_names = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    corridor_options = ["All Corridors"] + top_10_names + ["--- Full Street List ---"] + sorted(df_raw1["rpt_street_name"].unique().tolist())
    selected_street = st.selectbox("📍 Corridor:", corridor_options)

# Filter Logic
df = df_raw1[df_raw1["Year"].isin(selected_years)]
if selected_street not in ["All Corridors", "--- Full Street List ---"]:
    df = df[df["rpt_street_name"] == selected_street]
    current_focus = selected_street
else:
    current_focus = "Austin District (Full View)"

# --- 5. DASHBOARD HEADER ---
st.title("Safety Intelligence Dashboard")
st.caption(f"Currently Analyzing: **{current_focus}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")
st.markdown("---")

# --- 6. TABS ---
tabs = st.tabs(["Top Predictors", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"])

with tabs[0]:
    st.write("##### Predictors determined via Random Forest model on Austin crash data (2018-Present).")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.subheader("Top Predictors")
        st.image("https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/BI%20to%20AI%20SHAP%20vf.png", width=800)
    with c2:
        st.subheader("Historical Trends")
        cost_yr = df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        st.plotly_chart(px.bar(cost_yr, x="Year", y="Estimated Total Comprehensive Cost", color_continuous_scale="Purples", height=300), use_container_width=True)

with tabs[1]:
    col_list, col_map = st.columns([1, 2])
    with col_list:
        risk_df = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(risk_df, x="Estimated Total Comprehensive Cost", y="rpt_street_name", orientation='h', color_discrete_sequence=["#4B0082"]), use_container_width=True)
    with col_map:
        lat_c, lon_c = (df["latitude"].median(), df["longitude"].median()) if not df.empty else (30.2672, -97.7431)
        st.plotly_chart(px.density_mapbox(df, lat="latitude", lon="longitude", z="Estimated Total Comprehensive Cost", radius=10, center=dict(lat=lat_c, lon=lon_c), zoom=10, mapbox_style="open-street-map", color_continuous_scale="Purples", height=600), use_container_width=True)

with tabs[5]:
    st.subheader("Prescriptive Actions & Savings")
    if df_prescriptive_raw is not None:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
        dfp = dfp[dfp["best_action"] != "no_change"]
        if selected_street not in ["All Corridors", "--- Full Street List ---"]:
            dfp = dfp[dfp["address"].str.contains(selected_street, case=False, na=False)]
        
        if not dfp.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Expected Savings", fmt_dollars(dfp["expected_reduction_amount"].sum()))
            c2.metric("Avg. Reduction %", f"{(dfp['pct_reduction_norm'].mean()*100):.1f}%")
            c3.metric("Top Action", dfp["best_action"].mode()[0])
            st.dataframe(dfp[["address", "best_action", "expected_reduction_amount", "ai_rationale_short"]].sort_values("expected_reduction_amount", ascending=False), use_container_width=True)
        else:
            st.warning("No recommendations for this selection.")
