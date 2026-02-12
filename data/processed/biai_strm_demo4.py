import os
import glob
import math
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
    page_title="CGI | Austin Safety Intelligence Elite",
    layout="wide",
    page_icon="🛣️"
)

# Make Altair nicer
alt.data_transformers.disable_max_rows()

# --- 2. PATH CONFIGURATION ---
DATA_DIR = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed"
LOCAL_DATA_DIR = 'data/processed'

CSV_FILENAME1 = "atx_crash_data_2018-2026_clean.csv"
CSV_PATH1 = f"{DATA_DIR}/{CSV_FILENAME1}"

CSV_FILENAME2 = "df_prescriptive_final_20260204_102224.csv"
CSV_PATH2 = f"{DATA_DIR}/outputs/{CSV_FILENAME2}"

MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"

# --- 3. SMART ASSET LOADER ---
def get_cgi_logo():
    """
    Locates the CGI corporate logo by checking local files and remote URL.
    """
    logo_filename = "CGI_logo_color_rgb.jpg"
    github_logo_url = f"{DATA_DIR}/{logo_filename}"
    
    # Check local path first
    if os.path.exists(logo_filename):
        return logo_filename
    
    # Check GitHub URL
    try:
        response = requests.head(github_logo_url, timeout=5)
        if response.status_code == 200:
            return github_logo_url
    except:
        pass
        
    return None

LOGO_PATH = get_cgi_logo()

# Render logo at the very top of the main area
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)

# ----------------------------
# Shared: Safe remote CSV loader
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    """
    Robustly fetch CSV from a URL and fail clearly if empty.
    """
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")
    if len(r.content) <= 10:
        raise ValueError(f"Remote file is too small. URL: {url}")
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

# ----------------------------
# Data Pipeline
# ----------------------------
@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    """
    Standardizes crash data, mappings, and involvement flags.
    """
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
# Prescriptive Tab Helpers
# ----------------------------
# (Defining a simple kpi_row since it was called but not defined in the snippet)
def kpi_row(dfp):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reduction", fmt_dollars(dfp["expected_reduction_amount"].sum()))
    c2.metric("Avg % Reduction", f"{(dfp['pct_reduction_norm'].mean()*100):.1f}%")
    c3.metric("Locations", len(dfp))

def _coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def normalize_pct_reduction(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty: return s
    p95 = s.dropna().quantile(0.95)
    return s / 100.0 if p95 > 1.5 else s

def make_location_id(df):
    return df["latitude"].round(5).astype(str) + ", " + df["longitude"].round(5).astype(str)

def compact_text(s, n=140):
    s = str(s).strip()
    return s[:n-1] + "…" if len(s) > n else s

def fmt_dollars(x):
    return f"${float(x):,.0f}" if pd.notnull(x) else "—"

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive):
    df = df_prescriptive.copy()
    df = _coerce_numeric(df, ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_reduction_amount", "pct_reduction"])
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["location_id"] = make_location_id(df)
    df["address_short"] = df["Address"].astype(str).map(lambda x: compact_text(x, 80))
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x, 160))
    return df

def build_map(df, top_n, all_actions):
    df_map = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    ACTION_COLORS_RGB = {"reduce_speed_limit": (227, 25, 55), "increase_enforcement": (82, 54, 171), "improve_crosswalks": (110, 63, 237)}
    df_map["color"] = df_map["best_action"].map(lambda a: list(ACTION_COLORS_RGB.get(a, (120, 120, 120))))
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=df_map, get_position="[longitude, latitude]", get_fill_color="color", get_radius=100, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=df_map["latitude"].mean(), longitude=df_map["longitude"].mean(), zoom=10),
        map_style="mapbox://styles/mapbox/streets-v12"
    ))

def action_bars(df):
    agg = df.groupby("best_action").agg(total_reduction=("expected_reduction_amount", "sum"), locations=("location_id", "count")).reset_index()
    c1, c2 = st.columns(2)
    c1.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x="best_action:N", y="total_reduction:Q"), use_container_width=True)
    c2.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x="best_action:N", y="locations:Q"), use_container_width=True)

def ranked_table_and_details(df, top_n):
    ranked = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    st.dataframe(ranked[["Address", "best_action", "expected_reduction_amount", "ai_rationale_short"]], use_container_width=True)

# ----------------------------
# Execution & UI
# ----------------------------
df_raw1 = load_partner_data(CSV_PATH1)
try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception:
    df_prescriptive_raw = None

# --- SIDEBAR ---
with st.sidebar:
    if LOGO_PATH: 
        st.image(LOGO_PATH, use_container_width=True)
    st.title("Global Filters")
    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    top_10 = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    selected_street = st.selectbox("📍 Corridor:", ["All Corridors"] + top_10)

# Filter Logic
df = df_raw1[df_raw1["Year"].isin(selected_years)]
if selected_street != "All Corridors":
    df = df[df["rpt_street_name"] == selected_street]

st.title("Safety Intelligence Dashboard")
st.caption(f"Analyzing: **{selected_street}**")

# --- TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs(["Top Predictors", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"])

with t1:
    st.image("https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed/BI%20to%20AI%20SHAP%20vf.png", width=800)
    st.plotly_chart(px.bar(df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index(), x="Year", y="Estimated Total Comprehensive Cost", color_continuous_scale="Purples"), use_container_width=True)



with t2:
    st.plotly_chart(px.density_mapbox(df, lat="latitude", lon="longitude", z="Estimated Total Comprehensive Cost", radius=12, zoom=10, mapbox_style="open-street-map", color_continuous_scale="Purples"), use_container_width=True)

with t3:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(df, names="Severity_Label", hole=0.4, color_discrete_sequence=px.colors.sequential.Purples_r), use_container_width=True)
    c2.plotly_chart(px.bar(df.groupby("Speed_Bin", observed=False).size().reset_index(name="Count"), x="Speed_Bin", y="Count", color_discrete_sequence=["#4B0082"]), use_container_width=True)

with t4:
    st.plotly_chart(px.density_heatmap(df.groupby(["DAY_NAME", "HOUR"]).size().reset_index(name="Count"), x="HOUR", y="DAY_NAME", z="Count", color_continuous_scale="Purples"), use_container_width=True)

with t5:
    modes = ["Bicycle", "Pedestrian", "Motorcycle"]
    mode_stats = pd.DataFrame([{"Mode": m, "Avg Cost": df[df[m]==1]["Estimated Total Comprehensive Cost"].mean()} for m in modes])
    st.plotly_chart(px.bar(mode_stats, x="Mode", y="Avg Cost", color="Avg Cost", color_continuous_scale="Purples"), use_container_width=True)

with t6:
    if df_prescriptive_raw is not None:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
        if selected_street != "All Corridors": dfp = dfp[dfp["Address"].str.contains(selected_street, case=False, na=False)]
        kpi_row(dfp)
        build_map(dfp, 50, dfp["best_action"].unique())
        action_bars(dfp)
        ranked_table_and_details(dfp, 50)
    else:
        st.error("Prescriptive data unavailable.")
