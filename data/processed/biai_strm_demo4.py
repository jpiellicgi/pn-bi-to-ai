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
from plotly.subplots import make_subplots
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

# ----------------------------
# Shared: Safe remote CSV loader
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    """
    Robustly fetch CSV from a URL.
    Fails clearly if the remote file is empty or non-CSV.
    """
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")

    # If tiny response, it's effectively empty (common cause of 'No columns to parse')
    if len(r.content) <= 10:
        raise ValueError(f"Remote file is too small ({len(r.content)} bytes). URL: {url}")

    # Let pandas parse from bytes
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

# ----------------------------
# Data Formatting Function Definitions
# ----------------------------
def format_hour(hour):
    if hour == 0: return "12 AM"
    if hour < 12: return f"{hour} AM"
    if hour == 12: return "12 PM"
    return f"{hour - 12} PM"

# ----------------------------
# Data Pipeline
# ----------------------------
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

    #Hour label formatting for visuals
    df['hour_label'] = df['HOUR'].apply(format_hour)
    labels_in_order = [format_hour(h) for h in range(24)]
    df['hour_label'] = pd.Categorical(df['hour_label'], categories=labels_in_order, ordered=True)

    #Hour range formatting for heatmap visual
    #heat_df["Time_Range"] = heat_df["HOUR"].apply(get_range_label)

    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    cols_to_fix = ["tot_injry_cnt", "crash_speed_limit", "Estimated Total Comprehensive Cost"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Transportation Mode Mapping (Updated with more modes from data source)
    mapping = {
        "Passenger Car": ["passenger car_involved", "passenger_car_involved", "car_fl", "is_car"],
        "Bicycle": ["bicycle_involved", "bicycle_fl", "is_bike"],
        "Pedestrian": ["pedestrian_involved", "pedestrian_fl", "is_ped"],
        "Motorcycle": ["motorcycle_involved", "motorcycle_fl", "is_mc"],
        "Commercial Veh": ["comml_mtr_veh_fl", "cmv_involved", "is_truck"],
        "Micromobility": ["micromobility device_involved", "micromobility_fl"],
        "E-Scooter": ["e-scooter_involved", "scooter_fl"],
        "Large Passenger Veh": ["large passenger vehicle_involved", "large_veh_fl"],
        "Train": ["train_involved", "train_fl"],
        "Motor Vehicle": ["motor vehicle_involved", "motor_veh_fl"],
        "Other": ["other_involved", "other_fl"],
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
    p95 = s.dropna().quantile(0.95)
    return s / 100.0 if p95 > 1.5 else s

def make_location_id(df):
    return df["latitude"].round(5).astype(str) + ", " + df["longitude"].round(5).astype(str)

def compact_text(s, n=140):
    if s is None or (isinstance(s, float) and math.isnan(s)): return ""
    s = str(s).strip()
    return s[:n-1] + "…" if len(s) > n else s

def fmt_dollars(x):
    try:
        return f"${float(x):,.0f}" if pd.notnull(x) else "—"
    except: return "—"

def pretty_action(a: str) -> str:
    """Display-friendly action label: 'micromobility_zones' -> 'micromobility zones'"""
    if a is None:
        return ""
    return str(a).replace("_", " ").strip()

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive):
    df = df_prescriptive.copy()
    df = _coerce_numeric(df, ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_cost_after_action", "expected_reduction_amount", "pct_reduction"])
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[df["best_action"] != "no_change"].copy()
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["location_id"] = make_location_id(df)
    df["address_short"] = df["Address"].astype(str).map(lambda x: compact_text(x, 80))
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x, 160))
    df["best_action_label"] = df["best_action"].apply(pretty_action)
    return df
    return df

def build_map(df, top_n=50, all_actions=None):
    # Ensure display label column exists
    if "best_action_label" not in df.columns:
        df = df.copy()
        df["best_action_label"] = df["best_action"].apply(pretty_action)

    # Sort and take top N
    df_map = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

    # Legend/category order
    if all_actions is None:
        all_actions = list(df_map["best_action"].dropna().unique())
    # Build label order in the same order as actions, but de-duplicated and aligned
    pairs = (
        df_map[["best_action", "best_action_label"]]
        .dropna()
        .drop_duplicates()
    )
    # Preserve the original actions order but map to their labels
    label_order = [pairs.loc[pairs["best_action"] == a, "best_action_label"].iloc[0] 
                   for a in all_actions if a in pairs["best_action"].values]

    # Color mapping (fallback to gray if missing)
    ACTION_COLORS_RGB = {
        "reduce_speed_limit": (227, 25, 55),
        "increase_enforcement": (82, 54, 171),
        "improve_crosswalks": (110, 63, 237),
        "add_speed_bumps": (168, 36, 101)
    }
    def _rgb_to_plotly(rgb_tuple):
        r, g, b = rgb_tuple
        return f"rgb({r},{g},{b})"
    ACTION_COLORS = {k: _rgb_to_plotly(v) for k, v in ACTION_COLORS_RGB.items()}
    DEFAULT_COLOR = "rgb(120,120,120)"

    # Map pretty labels to colors using the original action color if we have it
    label_to_color = {}
    for _, row in pairs.iterrows():
        orig = row["best_action"]
        lbl = row["best_action_label"]
        label_to_color[lbl] = ACTION_COLORS.get(orig, DEFAULT_COLOR)

    # Center map on data
    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        color="best_action_label",  # <- pretty labels in legend
        color_discrete_map=label_to_color,
        category_orders={"best_action_label": label_order},
        hover_name="best_action_label",
        hover_data={
            "expected_reduction_amount": ":,.0f",
            "latitude": False,
            "longitude": False,
        },
        zoom=10,
        center=dict(lat=center_lat, lon=center_lon),
        height=550,
    )

    # Marker and layout tweaks
    fig.update_traces(marker=dict(size=10, opacity=0.9))
    fig.update_layout(
        mapbox_style="open-street-map",  # <- no Mapbox token required
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Recommended action",
    )

    st.plotly_chart(fig, use_container_width=True)
# def build_map(df, top_n=50, all_actions=None):
#     # Sort and take top N
#     df_map = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

#     # Legend/category order
#     if all_actions is None:
#         all_actions = list(df_map["best_action"].dropna().unique())

#     # Color mapping (fallback to gray if missing)
#     ACTION_COLORS_RGB = {"reduce_speed_limit": (227, 25, 55), "increase_enforcement": (82, 54, 171), "improve_crosswalks": (110, 63, 237), "add_speed_bumps": (168, 36, 101)}    
#     def _rgb_to_plotly(rgb_tuple):
#         r, g, b = rgb_tuple
#         return f"rgb({r},{g},{b})"
#     ACTION_COLORS = {k: _rgb_to_plotly(v) for k, v in ACTION_COLORS_RGB.items()}
#     DEFAULT_COLOR = "rgb(120,120,120)"
#     color_map = {a: ACTION_COLORS.get(a, DEFAULT_COLOR) for a in all_actions}

#     # Center map on data
#     center_lat = df_map["latitude"].mean()
#     center_lon = df_map["longitude"].mean()

#     fig = px.scatter_mapbox(
#         df_map,
#         lat="latitude",
#         lon="longitude",
#         color="best_action",
#         color_discrete_map=color_map,
#         category_orders={"best_action": all_actions},
#         hover_name="best_action",
#         hover_data={
#             "expected_reduction_amount": ":,.0f",
#             "latitude": False,
#             "longitude": False,
#         },
#         zoom=10,
#         center=dict(lat=center_lat, lon=center_lon),
#         height=550,
#     )

#     # Marker and layout tweaks
#     fig.update_traces(marker=dict(size=10, opacity=0.9))
#     fig.update_layout(
#         mapbox_style="open-street-map",  # <- no Mapbox token required
#         margin=dict(l=0, r=0, t=0, b=0),
#         legend_title_text="Recommended action",
#     )

#     st.plotly_chart(fig, use_container_width=True)

# def action_bars(df,top_n=50):
#     # Sort and take top N
#     df_bar = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
#     agg = df_bar.groupby("best_action").agg(total_reduction=("expected_reduction_amount", "sum"), locations=("location_id", "count")).reset_index()
#     c1, c2 = st.columns(2)
#     c1.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x=alt.X("best_action:N", sort='-y'), y="total_reduction:Q"), use_container_width=True)
#     c2.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x=alt.X("best_action:N", sort='-y'), y="locations:Q"), use_container_width=True)
def action_bars(df, top_n=50):
    # Ensure label column exists
    if "best_action_label" not in df.columns:
        df = df.copy()
        df["best_action_label"] = df["best_action"].apply(pretty_action)

    # Sort and take top N
    df_bar = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    agg = (
        df_bar
        .groupby("best_action_label")
        .agg(total_reduction=("expected_reduction_amount", "sum"),
             locations=("location_id", "count"))
        .reset_index()
    )

    c1, c2 = st.columns(2)
    c1.altair_chart(
        alt.Chart(agg).mark_bar(color="#5236ab").encode(
            x=alt.X("best_action_label:N", sort='-y', title="Recommended action"),
            y=alt.Y("total_reduction:Q", title="Total expected reduction ($)")
        ),
        use_container_width=True
    )
    c2.altair_chart(
        alt.Chart(agg).mark_bar(color="#5236ab").encode(
            x=alt.X("best_action_label:N", sort='-y', title="Recommended action"),
            y=alt.Y("locations:Q", title="Number of locations")
        ),
        use_container_width=True
    )
def ranked_table_and_details(df, top_n):
    left, right = st.columns([1.35, 1])
    ranked = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")
        # show_columns = ["Address", "location_id", "best_action", "expected_reduction_amount", "pct_reduction_norm","pred_est_ttl_comp_cost", "expected_cost_after_action","ai_rationale_short"]
        # ranked_display = ranked[show_columns].rename(columns={"address": "Address","pct_reduction_norm": "pct_reduction","ai_rationale_short": "ai_rationale (short)",})
        show_columns = [
            "Address", "location_id", "best_action_label", "expected_reduction_amount",
            "pct_reduction_norm", "pred_est_ttl_comp_cost", "expected_cost_after_action", "ai_rationale_short"
        ]
        ranked_display = ranked[show_columns].rename(columns={
            "best_action_label": "best_action",
            "pct_reduction_norm": "pct_reduction",
            "ai_rationale_short": "ai_rationale (short)",
        })
        ranked_display["expected_reduction_amount"] = ranked_display["expected_reduction_amount"].map(fmt_dollars)
        ranked_display["expected_cost_after_action"] = ranked_display["expected_cost_after_action"].map(fmt_dollars)
        ranked_display["pred_est_ttl_comp_cost"] = ranked_display["pred_est_ttl_comp_cost"].map(fmt_dollars)
        st.dataframe(ranked_display, use_container_width=True, hide_index=True)
        
    with right:
        st.subheader("")
        options = ranked[["address_short", "location_id"]].fillna("").copy()
        options["label"] = options["address_short"] + "  (" + options["location_id"] + ")"
        selected_label = st.selectbox("Select an address to see full rationale", options=options["label"].tolist(),index=0 if len(options) else None)
        if selected_label:
            selected_loc = options.loc[options["label"] == selected_label, "location_id"].iloc[0]
            row = df.loc[df["location_id"] == selected_loc].iloc[0]
            st.markdown(
                f"""
                **Action:** {row.get('best_action_label', pretty_action(row.get('best_action', '')))}
                **Risk score:** `{fmt_dollars(row['pred_est_ttl_comp_cost'])}`   
                **Expected reduction:** `{fmt_dollars(row['expected_reduction_amount'])}`  
                **% reduction:** {row['pct_reduction_norm'] * 100:.1f}%  
                **Expected cost after action:** {fmt_dollars(row['expected_cost_after_action'])}  
                """
            )
            st.markdown("**Rationale:**")
            st.write(str(row.get("ai_rationale", "")))
# ----------------------------
# Execution & UI
# ----------------------------
try:
    df_raw1 = load_partner_data(CSV_PATH1)
except Exception as e:
    st.error(f"Crash dataset load failed: {e}")
    st.stop()

try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception as e:
    df_prescriptive_raw = None

# --- SIDEBAR (Logo Removed from here) ---
with st.sidebar:
    st.title("Global Filters")
    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    top_10_names = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    selected_street = st.selectbox("📍 Corridor:", ["All Corridors"] + top_10_names)
    corridor_options = ["All Corridors"] + top_10_names + ["--- Full Street List ---"] + sorted(df_raw1["rpt_street_name"].unique().tolist())

# Filter Logic
df = df_raw1[df_raw1["Year"].isin(selected_years)]
if selected_street != "All Corridors":
    df = df[df["rpt_street_name"] == selected_street]
    current_focus = selected_street
else:
    current_focus = "Austin District (Full View)"

# --- MAIN DASHBOARD AREA ---
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)

st.title("Safety Intelligence Dashboard")
st.caption(f"Analyzing: **{current_focus}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚶Top Predictors", "🗺️ Geographic Risk", "📊 Speed and Severity", "⏰ Temporal Patterns", "💰 Transportation Mode Analysis", "🧠 Prescriptive Actions"])

with tab1:
    st.write("##### The top predictors and prescriptive actions were determined through a random forest model trained on crash data from the City of Austin from the 2018 to present.")
    shap_output, historicaloverview = st.columns([1, 2], gap="large")
    with shap_output:
        st.subheader("Top Predictors of Estimated Cost")
        st.image("data/processed/outputs/BI to AI SHAP vf.png", width=800)
        st.write("This shows the feature importances assigned by SHAP for each feature for the prediction of estimated cost in our random forest model. This SHAP summary plot shows how each feature influences the model's predicted crash cost relative to the average. Features are ranked by importance (top = most impactful). Each dot represents an individual crash — red dots indicate a high feature value, blue dots indicate a low feature value. Dots to the right of center (positive SHAP) mean that feature increased the predicted cost; dots to the left (negative SHAP) mean it decreased the predicted cost. The three most influential predictors of estimated crash cost are pedestrian involved, motorcycle involved, and crash speed limit.")
    
    with historicaloverview:
        st.subheader("Historical Trends")

        st.write("**Estimated Total Comprehensive Cost per Year**")
        df_total_cost= df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        fig_cost_bar= px.bar(df_total_cost, x="Year", y="Estimated Total Comprehensive Cost", text_auto=True)
        fig_cost_bar.update_layout(
            height=400, 
            width=800,
            margin=dict(l=100, r=100, t=20, b=20), # Tighten whitespace
            yaxis_tickprefix='$'
            )
        avg_annual_cost_ref= df_total_cost["Estimated Total Comprehensive Cost"].mean()
        fig_cost_bar.add_hline(
            y=avg_annual_cost_ref,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average Annual Cost: ${avg_annual_cost_ref:,.0f}",
            annotation_position="bottom right"
        )
        fig_cost_bar.update_xaxes(type='category')
        fig_cost_bar.update_traces(marker_color='#5236ab')
        st.plotly_chart(fig_cost_bar)

        st.write("**Total Number of Crashes per Year**")
        df_crash_count= df.groupby("Year")["ID"].count().reset_index()
        df_crash_count.columns = ["Year", "Number of Crashes"]
        fig_crash_count= px.bar(df_crash_count, x="Year", y="Number of Crashes", text_auto=".2s")
        fig_crash_count.update_layout(
            height=400, 
            width=800,
            margin=dict(l=100, r=100, t=20, b=20) # Tighten whitespace
            )
        fig_crash_count.update_xaxes(type='category')
        fig_crash_count.update_traces(marker_color='#5236ab')
        st.plotly_chart(fig_crash_count)

with tab2:
    col_list, col_map = st.columns([1, 2])   
    with col_list:
        st.subheader("🔥 Top 10 Risk Corridors")      
        # Data Processing
        risk_df = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).reset_index()
        risk_df.columns = ["Street", "Cost"]        
        # Color Logic
        bar_colors = ["#4B0082" if s == selected_street else "#D8BFD8" for s in risk_df["Street"]]        
        # Create Figure
        fig_bar = px.bar(
            risk_df, 
            x="Cost", 
            y="Street", 
            orientation="h", 
            template="plotly_white"
        )       
        # Update Traces and Axis Formatting
        fig_bar.update_traces(marker_color=bar_colors)
        
        fig_bar.update_layout(
            xaxis_title="Cost",
            yaxis_title="Street",
            xaxis=dict(
                tickprefix="$", 
                tickformat=",d"  # Adds commas for thousands (e.g., $1,000)
            )
        )      
        # Adjust Y-axis to ensure the bars are sorted correctly (highest at top)
        fig_bar.update_yaxes(autorange="reversed")
        
        st.plotly_chart(fig_bar, use_container_width=True)

with col_map:
    severity_color_map = {
        'Fatal': '#991f3d',
        'Serious Injury': '#e31937',
        'Minor Injury': '#ff6a00',
        'Possible Injury': '#f1a425',
        'No Injury': '#128354',
        'Unknown': '#cccccc'
    }
    
    map_type = st.radio("Map Layer:", ["Economic Heatmap", "Incident Clusters"], horizontal=True)
    lat_c, lon_c = (df["latitude"].median(), df["longitude"].median()) if not df.empty else (30.2672, -97.7431)

    if map_type == "Economic Heatmap":
        fig_m = px.density_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            z="Estimated Total Comprehensive Cost",
            radius=12,
            center=dict(lat=lat_c, lon=lon_c),
            zoom=10,
            mapbox_style="open-street-map",
            color_continuous_scale="Purples",           
        )
        # Update the colorbar to show currency
        fig_m.update_layout(
            coloraxis_colorbar=dict(
                title="Total Comprehensive Cost",
                tickprefix="$",
                tickformat=",d" # Adds commas for thousands
            )
        )
    else:
        # We set the order from least serious to most serious. 
        # Plotly draws these in order, so 'Fatal' (the last one) will be layered on top.
        layer_order = ["Unknown", "No Injury", "Possible Injury", "Minor Injury", "Serious Injury", "Fatal"]
        
        fig_m = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="Severity_Label",
            color_discrete_map=severity_color_map,
            category_orders={"Severity_Label": layer_order},
            labels={"Severity_Label": "Severity Label"},
            size="marker_size",
            center=dict(lat=lat_c, lon=lon_c),
            zoom=10,
            mapbox_style="open-street-map",
        )
        
        # This ensures the legend still shows 'Fatal' at the top, even though it's drawn last
        fig_m.update_layout(
            legend=dict(
                traceorder="reversed",
                title_font_family="Arial",
                font=dict(size=12)
            )
        )

    fig_m.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
    st.plotly_chart(fig_m, use_container_width=True)

with tab3:
    st.subheader(f"🛡️ Crash Risk Profile: {current_focus}")
    
    # --- ROW 1: Existing Formatting ---
    r1c1, r1c2, r1c3 = st.columns(3) 

    with r1c1:
        severity_color_map = {
            'Fatal': '#991f3d',
            'Serious Injury': '#e31937',
            'Minor Injury': '#ff6a00',
            'Possible Injury': '#f1a425',
            'No Injury': '#128354',
            'Unknown': '#cccccc'
        }
        fig_pie = px.pie(
            df, 
            names="Severity_Label", 
            hole=0.4, 
            color="Severity_Label", 
            color_discrete_map=severity_color_map, 
            category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
            labels={"Severity_Label": "Severity Label"}, 
            title='Crash Severity Breakdown'
        )
        fig_pie.update_layout(
            height=450, 
            legend_title_text='Severity Label',
            legend=dict(x=0.85, y=0.5)   
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with r1c2:
        df_speed_severity = df.groupby(["Speed_Bin", "Severity_Label"]).size().reset_index(name="Accident_Count")
        fig_bar = px.bar(
            df_speed_severity,
            x="Speed_Bin",
            y="Accident_Count",
            color="Severity_Label",
            title="Crashes by Speed Limit and Severity",
            labels={"Accident_Count": "Number of Crashes", "Speed_Bin": "Speed Limit (mph)", "Severity_Label": "Severity Label"},
            barmode="stack",
            category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
            color_discrete_map=severity_color_map
        )
        fig_bar.update_layout(
            legend=dict(x=0.85, y=0.5),
            xaxis_title="Speed Limit (mph)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with r1c3:
        df_avg_cost_speed = df.groupby("Speed_Bin")["Estimated Total Comprehensive Cost"].mean().reset_index()
        fig_avg_cost_speed = px.bar(
            df_avg_cost_speed, 
            x="Speed_Bin", 
            y="Estimated Total Comprehensive Cost", 
            color="Estimated Total Comprehensive Cost",
            title="Average Estimated Cost by Speed Bin",
            labels={"Estimated Total Comprehensive Cost": "Average Estimated Cost", "Speed_Bin": "Speed Limit (mph)"}, 
            text_auto=".2s"
        )
        # Ensuring $ formatting remains on the Y-Axis
        fig_avg_cost_speed.update_layout(
            yaxis_tickprefix='$',
            yaxis_tickformat=',.0f'
        )
        fig_avg_cost_speed.update_traces(marker_color='#5236ab')
        st.plotly_chart(fig_avg_cost_speed, use_container_width=True)

    # --- ROW 2: New Information on Other Modes ---
    st.markdown("---")
    st.write("### 🚲 Transportation Mode Involvement")
    st.info("The chart below shows the frequency of each mode involved in the current selection. Note: A single crash may involve multiple modes (e.g., a Car vs. Bicycle incident).")

    # 1. Prepare data for the 11 modes
    modes = [
        "Passenger Car", "Bicycle", "Pedestrian", "Motorcycle", 
        "Commercial Veh", "Micromobility", "E-Scooter", 
        "Large Passenger Veh", "Train", "Motor Vehicle", "Other"
    ]
    
    mode_counts = []
    for m in modes:
        if m in df.columns:
            count = df[m].sum()
            if count > 0:
                mode_counts.append({"Mode": m, "Incident Count": count})
    
    if mode_counts:
        mode_count_df = pd.DataFrame(mode_counts).sort_values("Incident Count", ascending=True)

        # 2. Create a Horizontal Bar Chart for Mode Distribution
        fig_modes = px.bar(
            mode_count_df,
            x="Incident Count",
            y="Mode",
            orientation='h',
            title="Crash Frequency by Mode of Transportation",
            text_auto=True,
            template="plotly_white",
            color="Incident Count",
            color_continuous_scale="Purples"
        )
        
        fig_modes.update_layout(
            height=500,
            xaxis_title="Total Number of Incidents",
            yaxis_title=None,
            coloraxis_showscale=False # Clean look
        )
        
        st.plotly_chart(fig_modes, use_container_width=True)
    else:
        st.warning("No specific mode data found for the current selection.")

with tab4:
    st.subheader(f"Temporal Patterns: {current_focus}")
    st.write("""The visuals on this page shows the number of crashes that occurred during different timeframes as well the average estimated cost and the severity of those crashes.""")
    st.info("""
            **💡High-Level Insights:**
            - Most crashes occur during Monday-Friday from 3 PM - 6 PM.
            - The most expensive crashes occur at 6 AM and 8 PM.
            - The most severe crashes (those with serious or fatal injuries) occur between ---
            
            *Recommendation: Deploying additional resources during afternoon rush hour. Targeting solutions for pedestrian related accidents from 6 PM - 9 PM and solutions for speed-related accidents from 5 AM- 7 AM.*
            *See the prescriptive actions tab for specific solutions.*
        """)

    #Density Heatmap for Number of Crashes for Day of Week and Time Frame
    heat_df = df.groupby(["DAY_NAME", "HOUR"]).size().reset_index(name="Count")
    fig_heat = px.density_heatmap(
        heat_df,
        x="HOUR",
        y="DAY_NAME",
        z="Count",
        title= "Number of Crashes by Day and Hour",
        labels={"DAY_NAME": "Day", "HOUR": "Hour", "Count": "Number of Crashes"},
        color_continuous_scale="Purples",
        category_orders={"DAY_NAME": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
    )
    fig_heat.update_traces(
        xbins=dict(start=0, end=24, size=3),
        autobinx=False,
        hovertemplate=(
        "<b>Number of Crashes:</b> %{z}<extra></extra>"
        )
    )
    tick_vals = [0, 3, 6, 9, 12, 15, 18, 21]
    tick_text = ["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"]
    fig_heat.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_text
        )
    )
    fig_heat.update_layout(
    coloraxis_colorbar=dict(
        title="Number of Crashes"
        )
    )   
    st.plotly_chart(fig_heat, use_container_width=True)
    st.text("The visual above shows the number of crashes each day of the week in 3 hour timeframes. The darker the shade of purple, the more crashes that occurred during that timeframe.")

    #Crash Severity vs. Average Estimated Costs
    df_avg_cost = df.groupby(["hour_label"], observed=False)["Estimated Total Comprehensive Cost"].mean().reset_index()
    df_severity = df.groupby(["hour_label", "Severity_Label"], observed=False).size().reset_index(name="Accident_Count")
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #Add the Stacked Bars (Primary Y-Axis)
    severity_order = ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]
    for severity in severity_order:
        mask = df_severity["Severity_Label"] == severity
        fig.add_trace(
            go.Bar(
                x=df_severity[mask]["hour_label"],
                y=df_severity[mask]["Accident_Count"],
                name=severity,
                marker_color=severity_color_map.get(severity, '#cccccc'),
                hovertemplate=f"<b>{severity}</b>: %{{y}} crashes<extra></extra>"
            ),
            secondary_y=False,
        )
    #Add the Average Cost Line (Secondary Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=df_avg_cost["hour_label"],
            y=df_avg_cost["Estimated Total Comprehensive Cost"],
            name="Avg Cost ($)",
            mode='lines+markers',
            line=dict(color='#5236ab', width=4),
            marker=dict(size=8),
            hovertemplate="<b>Avg Cost:</b> $%{y:,.2f}<extra></extra>"
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="Crash Severity vs. Average Estimated Cost",
        barmode='stack',
        hovermode="x unified", # Shows both cost and count in one tooltip
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    fig.update_yaxes(title_text="Number of Crashes", secondary_y=False)
    fig.update_yaxes(title_text="Average Estimated Cost ($)", secondary_y=True, tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)


    #Explanatory Charts for Spikes in Average Cost 
    st.text("The spikes in the average estimated total cost can be explained by pedestrian-involvement in the crash, higher average speed limits, and outliers in the data. The visuals below show these patterns. The first visual shows which hours have the most crashes with pedestrians involved. The second visual shows the average speed limit by hour. The spike in average cost for crashes at 1 AM is due to outliers in the data. Most crashes that occur between 1 AM and 2 AM fall in the average cost range of $20k - $70k, but there were some exceptionally costly crashes that drove up the average cost.")

    st.write("##### Explanatory Visuals for Spikes in Average Cost")
    #Number of Crashes Involving Pedestrians by Hour
    df_pedestrian = df[df['pedestrian_involved'] == True]
    df_ped_hour = df_pedestrian.groupby("hour_label", observed=False).size().reset_index(name="Pedestrian_Crash_Count")
    fig_ped = px.bar(
        df_ped_hour, 
        x="hour_label", 
        y="Pedestrian_Crash_Count",
        title="Pedestrian-Involved Crashes by Hour",
        labels={"hour_label": "Hour", "Pedestrian_Crash_Count": "Number of Crashes"},
        text_auto=True # Shows the count number on top of each bar
    )
    fig_ped.update_traces(marker_color='#5236ab')
    st.plotly_chart(fig_ped, use_container_width=True) 


    #Average Crash Speed Limit by Hour
    df_avg_speed = df.groupby("hour_label", observed=False)["crash_speed_limit"].mean().reset_index()
    fig_speed = px.line(
        df_avg_speed, 
        x="hour_label", 
        y="crash_speed_limit",
        title="Average Speed Limit of Crashes by Hour",
        markers=True, # Adds dots to each hour for better readability
        labels={"hour_label": "Hour of Day", "crash_speed_limit": "Avg Speed Limit (MPH)"},
        template="plotly_white"
    )
    fig_speed.update_traces(
        line=dict(color='#5236ab', width=3),
        marker=dict(size=8)
    )
    fig_speed.update_yaxes(ticksuffix=" MPH")
    st.plotly_chart(fig_speed, use_container_width=True)

with tab5:
    st.subheader(f"📊 Economic Impact by Transportation Type: {current_focus}")
    st.write("""
        This analysis breaks down the economic burden of crashes based on the modes of transportation involved. 
        **Comprehensive Cost** includes medical expenses, lost productivity, property damage, and the monetized value of pain and suffering.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Define all modes including the new ones from the data pipeline
    modes = [
        "Passenger Car", "Bicycle", "Pedestrian", "Motorcycle", 
        "Commercial Veh", "Micromobility", "E-Scooter", 
        "Large Passenger Veh", "Train", "Motor Vehicle", "Other"
    ]
    
    mode_stats = []

    # 2. Calculate statistics for each mode
    for m in modes:
        if m in df.columns:
            subset = df[df[m] == 1]
            if not subset.empty:
                avg_cost = subset["Estimated Total Comprehensive Cost"].mean()
                total_impact = subset["Estimated Total Comprehensive Cost"].sum()
                mode_stats.append({
                    "Transportation Mode": m, 
                    "Average Cost per Accident": avg_cost, 
                    "Total Economic Burden": total_impact, 
                    "Number of Accidents": len(subset)
                })

    if mode_stats:
        mode_df = pd.DataFrame(mode_stats).sort_values("Average Cost per Accident", ascending=False)

        # 3. Summary Metrics for "Additional Helpful Information"
        top_mode = mode_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Highest Avg Cost Mode", top_mode["Transportation Mode"])
        m2.metric("Avg Cost (Highest)", f"${top_mode['Average Cost per Accident']:,.0f}")
        m3.metric("Total Modes of Transportation Analyzed", len(mode_df))

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Average Economic Cost Bar Chart (Spaced out full-width)
        st.write("### 💰 Average Economic Cost per Accident")
        st.info("This chart identifies which types of accidents are the most 'expensive' on average, often highlighting the severity of incidents involving vulnerable road users.")
        
        fig_avg = px.bar(
            mode_df, 
            x="Transportation Mode", 
            y="Average Cost per Accident", 
            text_auto=".2s",
            template="plotly_white",
            labels={"Average Cost per Accident": "Average Cost ($)"}
        )
        fig_avg.update_traces(marker_color='#5236ab', textposition="outside")
        fig_avg.update_layout(
            yaxis_tickprefix='$',
            yaxis_tickformat=',.0f',
            xaxis_title=None,
            height=500
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        st.markdown("---")

        # 5. Enhanced Bubble Chart (Vulnerability Matrix)
        st.write("### 🎯 Mode Vulnerability Matrix")
        st.write("""
            **How to read this chart:**
            - **X-Axis (Horizontal):** Higher numbers mean these accidents happen more frequently.
            - **Y-Axis (Vertical):** Higher positions mean these accidents are more severe/costly per incident.
            - **Bubble Size:** Represents the **Total Economic Burden** (the sum of all costs for that mode).
            
            *Target the top-left for high-severity/low-volume risks and the bottom-right for high-volume systemic issues.*
        """)

        fig_bubble = px.scatter(
            mode_df, 
            x="Number of Accidents", 
            y="Average Cost per Accident", 
            size="Total Economic Burden",
            color="Transportation Mode", 
            hover_name="Transportation Mode",
            size_max=60,
            template="plotly_white",
            labels={
                "Number of Accidents": "Total Number of Accidents",
                "Average Cost per Accident": "Average Cost per Incident ($)",
                "Total Economic Burden": "Total Economic Impact ($)"
            }
        )
        
        fig_bubble.update_layout(
            yaxis_tickprefix='$',
            yaxis_tickformat=',.0f',
            xaxis_tickformat=',d',
            legend_title="Mode",
            height=600,
            hovermode="closest"
        )
        
        # Add a reference line for average across all modes
        avg_all = mode_df["Average Cost per Accident"].mean()
        fig_bubble.add_hline(y=avg_all, line_dash="dot", annotation_text="Mean Avg Cost", annotation_position="bottom right")
        
        st.plotly_chart(fig_bubble, use_container_width=True)
        
    else:
        st.warning("No Mode-specific data found in the current selection.")

with tab6:
    st.subheader("Prescriptive Actions: Recommended Interventions & Savings")
    st.caption("Explore high-impact locations, recommended interventions, and expected reductions.")
    
    if df_prescriptive_raw is not None:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
        if selected_street != "All Corridors":
            dfp = dfp[dfp["address_short"].str.contains(selected_street, case=False, na=False)]
        all_actions = sorted(dfp["best_action"].dropna().unique().tolist())
        all_action_labels = [pretty_action(a) for a in all_actions]  # (used implicitly by build_map)
        
        # top layout: filters + KPIs
        colL, colR = st.columns([3, 2], gap="large")

        with colL:
            st.markdown('<div class="left-panel sticky-col">', unsafe_allow_html=True)
    
            selected_actions = st.multiselect(
                "Recommended action",
                options=all_actions,
                default=all_actions,
                key="presc_actions"
            )
    
            top_n = st.slider(
                "Top N locations",
                min_value=10,
                max_value=300,
                value=50,
                step=10,
                key="presc_topn"
            )
    
            with st.expander("More filters"):
                if "severity" in dfp.columns:
                    sevs = sorted(dfp["severity"].dropna().unique().tolist())
                    st.multiselect("Severity", sevs, key="presc_severity")
    
                if "district" in dfp.columns:
                    dists = sorted(dfp["district"].dropna().unique().tolist())
                    st.multiselect("District", dists, key="presc_district")
    
            st.caption("Note: The **Year** filter in the global sidebar does not apply to this tab.")
    
            st.markdown('</div>', unsafe_allow_html=True)
            
            dfp_f = dfp[dfp["best_action"].isin(st.session_state.get("presc_actions", all_actions))].copy()
        
            if st.session_state.get("presc_severity"):
                if "severity" in dfp_f.columns:
                    dfp_f = dfp_f[dfp_f["severity"].isin(st.session_state["presc_severity"])]
        
            if st.session_state.get("presc_district"):
                if "district" in dfp_f.columns:
                    dfp_f = dfp_f[dfp_f["district"].isin(st.session_state["presc_district"])]
        
            if dfp_f.empty:
                st.warning("No data matches your filters. Select more options.")
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()
        
            df_topn = (
                dfp_f.sort_values("expected_reduction_amount", ascending=False)
                .head(st.session_state["presc_topn"])
            )
        with colR:
            # KPI 1 — total expected reduction
            total_reduction = float(df_topn["expected_reduction_amount"].sum())
    
            # KPI 2 — median pct reduction
            if "pct_reduction" in df_topn.columns:
                pct_series = df_topn["pct_reduction"]
                if pct_series.max() > 1.0:
                    pct_series = pct_series / 100.0
                median_pct_display = f"{float(pct_series.median()):.1%}"
            else:
                median_pct_display = "—"
    
            # KPI 3 — number of top-N locations
            locations_display = f"{len(df_topn):,}"
           
            with st.container():
                st.metric("Total Expected Reduction", f"{total_reduction:,.0f}")
                st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
            
                st.metric("Median % Reduction", median_pct_display)
                st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
            
                st.metric("Locations in Scope", locations_display)

            # end top layout
    
        selected_action_labels = [pretty_action(a) for a in st.session_state.get("presc_actions", all_actions)]
        build_map(dfp_f, top_n=st.session_state["presc_topn"], all_actions=st.session_state.get("presc_actions", all_actions))
        # build_map(dfp_f, top_n=st.session_state["presc_topn"], all_actions=all_actions)
        action_bars(dfp_f, top_n=st.session_state["presc_topn"])
        ranked_table_and_details(dfp_f, top_n=st.session_state["presc_topn"])
    else:
        st.error("Prescriptive data unavailable.")
