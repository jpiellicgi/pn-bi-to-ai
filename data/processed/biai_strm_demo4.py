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
    return df

def build_map(df, top_n=50, all_actions=None):
    # Sort and take top N
    df_map = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

    # Legend/category order
    if all_actions is None:
        all_actions = list(df_map["best_action"].dropna().unique())

    # Color mapping (fallback to gray if missing)
    ACTION_COLORS_RGB = {"reduce_speed_limit": (227, 25, 55), "increase_enforcement": (82, 54, 171), "improve_crosswalks": (110, 63, 237), "add_speed_bumps": (168, 36, 101)}    
    def _rgb_to_plotly(rgb_tuple):
        r, g, b = rgb_tuple
        return f"rgb({r},{g},{b})"
    ACTION_COLORS = {k: _rgb_to_plotly(v) for k, v in ACTION_COLORS_RGB.items()}
    DEFAULT_COLOR = "rgb(120,120,120)"
    color_map = {a: ACTION_COLORS.get(a, DEFAULT_COLOR) for a in all_actions}

    # Center map on data
    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        color="best_action",
        color_discrete_map=color_map,
        category_orders={"best_action": all_actions},
        hover_name="best_action",
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

def action_bars(df,top_n=50):
    # Sort and take top N
    df_bar = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    agg = df_bar.groupby("best_action").agg(total_reduction=("expected_reduction_amount", "sum"), locations=("location_id", "count")).reset_index()
    c1, c2 = st.columns(2)
    c1.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x=alt.X("best_action:N", sort='-y'), y="total_reduction:Q"), use_container_width=True)
    c2.altair_chart(alt.Chart(agg).mark_bar(color="#5236ab").encode(x=alt.X("best_action:N", sort='-y'), y="locations:Q"), use_container_width=True)

def ranked_table_and_details(df, top_n):
    left, right = st.columns([1.35, 1])
    ranked = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")
        show_columns = ["Address", "location_id", "best_action", "expected_reduction_amount", "pct_reduction_norm","pred_est_ttl_comp_cost", "expected_cost_after_action","ai_rationale_short"]
        ranked_display = ranked[show_columns].rename(columns={"address": "Address","pct_reduction_norm": "pct_reduction","ai_rationale_short": "ai_rationale (short)",})
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
                **Action:** {row['best_action']}  
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚶Top Predictors", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"])

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
        fig_cost_bar= px.bar(df_total_cost, x="Year", y="Estimated Total Comprehensive Cost", text_auto=".2s")
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
        else:
            fig_m = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="Severity_Label",
                color_discrete_map= severity_color_map,
                category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
                labels={"Severity_Label": "Severity Label"},
                size="marker_size",
                center=dict(lat=lat_c, lon=lon_c),
                zoom=10,
                mapbox_style="open-street-map",
            )

        fig_m.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_m, use_container_width=True)    

with tab3:
    st.subheader(f"Crash Risk Profile: {current_focus}")
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
        fig_pie = px.pie(df, names="Severity_Label", hole=0.4, color= "Severity_Label", color_discrete_map=severity_color_map, category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
                         labels={"Severity_Label": "Severity Label"}, title='Crash Severity Breakdown')
        fig_pie.update_layout(
            height=450, 
            width=500,
            legend_title_text='Severity Label',
            legend=dict(
                x=0.85,        
                y=0.5
                )   
            )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with r1c2:
        #Crashes by Speed Limit and Severity
        severity_color_map = {
            'Fatal': '#991f3d',
            'Serious Injury': '#e31937',
            'Minor Injury': '#ff6a00',
            'Possible Injury': '#f1a425',
            'No Injury': '#128354',
            'Unknown': '#cccccc'
            }
        df_speed_severity= df.groupby(["Speed_Bin", "Severity_Label"]).size().reset_index(name="Accident_Count")
        fig_bar = px.bar(
            df_speed_severity,
            x="Speed_Bin",
            y="Accident_Count",
            color="Severity_Label",
            title="Crashes by Speed Limit and Severity",
            labels={"Accident_Count": "Number of Crashes", "Speed_Bin": "Speed Limit (mph)", "Severity_Label": "Severity Label"},
            # This ensures the bars are stacked rather than grouped
            barmode="stack",
            # Optional: Define a specific order for the severity levels in the legend
            category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
            color_discrete_map=severity_color_map
            #color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig_bar.update_layout(
            legend=dict(
                x=0.85,        
                y=0.5
                )   
            )
        st.plotly_chart(fig_bar, use_container_width=True)

    with r1c3:
        #Average cost by speed bin
        df_avg_cost_speed= df.groupby("Speed_Bin")["Estimated Total Comprehensive Cost"].mean().reset_index()
        fig_avg_cost_speed= px.bar(df_avg_cost_speed, x="Speed_Bin", y="Estimated Total Comprehensive Cost", color="Estimated Total Comprehensive Cost",
            title= "Average Estimated Cost by Speed Bin",labels={"Estimated Total Comprehensive Cost": "Average Estimated Cost", "Speed_Bin": "Speed Limit (mph)"}, text_auto=".2s")
        fig_avg_cost_speed.update_layout(yaxis_tickprefix='$')
        fig_avg_cost_speed.update_traces(marker_color='#5236ab')
        st.plotly_chart(fig_avg_cost_speed)  

with tab4:
    st.subheader(f"Temporal Patterns: {current_focus}")
    heat_df = df.groupby(["DAY_NAME", "HOUR"]).size().reset_index(name="Count")
   # hour_map = {h: pd.to_datetime(h, format='%H').strftime('%-I %p') for h in range(24)}
    # heat_df['PRETTY_HOUR'] = heat_df['HOUR'].map(hour_map)
    fig_heat = px.density_heatmap(
        heat_df,
        x="HOUR",
        y="DAY_NAME",
        z="Count",
       # custom_data=["PRETTY_HOUR"], 
        title= "Number of Crashes by Day and Hour",
        labels={"DAY_NAME": "Day", "HOUR": "Hour", "Count": "Number of Crashes"},
        color_continuous_scale="Purples",
        category_orders={"DAY_NAME": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
    )
   # fig_heat.update_traces(
   #     hovertemplate="<b>Time:</b> %{customdata[0]}<br><b>Day:</b> %{y}<br><b>Crashes:</b> %{z}<extra></extra>"
   # )    
    tick_vals = [0, 3, 6, 9, 12, 15, 18, 21]
    tick_text = ["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"]
    fig_heat.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_text
        )
    )
    #Defining number of bins
    fig_heat.update_traces(
        xbins=dict(
            start=0,
            end=24,
            size=3  # 24 hours / 8 bins = size of 3
        ),
        autobinx=False
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    #Average cost by speed bin
    df_avg_cost_hour= df.groupby("HOUR")["Estimated Total Comprehensive Cost"].mean().reset_index()
    fig_avg_cost_hour= px.bar(df_avg_cost_hour, x="HOUR", y="Estimated Total Comprehensive Cost", title= "Average Estimated Cost by Hour", text_auto=".2s")
    fig_avg_cost_hour.update_layout(yaxis_tickprefix='$')
    fig_avg_cost_hour.update_xaxes(
        tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
        ticktext=["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"]
    )
    fig_avg_cost_hour.update_traces(marker_color='#5236ab')
    st.plotly_chart(fig_avg_cost_hour)

    #Severity Breakdown by Hour
    severity_color_map = {
        'Fatal': '#991f3d',
        'Serious Injury': '#e31937',
        'Minor Injury': '#ff6a00',
        'Possible Injury': '#f1a425',
        'No Injury': '#128354',
        'Unknown': '#cccccc'
        }
    df_hour_severity= df.groupby(["HOUR", "Severity_Label"]).size().reset_index(name="Accident_Count")
    fig_bar = px.bar(
        df_hour_severity,
        x="HOUR",
        y="Accident_Count",
        color="Severity_Label",
        title="Number of Crashes by Hour and Severity",
        labels={"Accident_Count": "Number of Crashes", "Severity_Label": "Severity Label"},
        barmode="stack",
        category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
        color_discrete_map= severity_color_map
    )
    fig_bar.update_xaxes(
        tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
        ticktext=["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"]
    )
    st.plotly_chart(fig_bar, use_container_width=True)


with tab5:
    st.subheader(f"Economic Impact by Transportation Type: {current_focus}")
    #st.subheader(f"Crash Risk Profile: {current_focus}")
    st.write("##### This page shows the cost of crashes in which a pedestrian, bicycle, or motorcycle were involved.")
    st.markdown("<br>", unsafe_allow_html=True)

    modes = ["Passenger Car", "Bicycle", "Pedestrian", "Motorcycle", "Commercial Veh"]
    mode_stats = []

    for m in modes:
        subset = df[df[m] == 1]
        if not subset.empty:
            avg_cost = subset["Estimated Total Comprehensive Cost"].mean()
            total_impact = subset["Estimated Total Comprehensive Cost"].sum()
            mode_stats.append({"Mode": m, "Average Cost": avg_cost, "Total Impact": total_impact, "Count": len(subset)})

    if mode_stats:
        mode_df = pd.DataFrame(mode_stats)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Average Economic Cost per Crash**")
            fig_avg = px.bar(mode_df, x="Mode", y="Average Cost", text_auto=".2s")
            fig_avg.update_traces(marker_color='#5236ab')
            fig_avg.update_layout(yaxis_tickprefix='$')
            st.plotly_chart(fig_avg, use_container_width=True)

        with c2:
            st.write("**Total Economic Burden (Sum)**")
            fig_total = px.pie(mode_df, names="Mode", values="Total Impact",
                               color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig_total, use_container_width=True)

        st.markdown("---")
        st.write("**Mode Vulnerability Matrix (Volume vs. Average Cost)**")
        fig_bubble = px.scatter(mode_df, x="Count", y="Average Cost", size="Total Impact",
                                color="Mode", hover_name="Mode", size_max=60)
        fig_bubble.update_layout(yaxis_tickprefix='$')
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
        build_map(dfp_f, top_n=st.session_state["presc_topn"], all_actions=all_actions)
        action_bars(dfp_f, top_n=st.session_state["presc_topn"])
        ranked_table_and_details(dfp_f, top_n=st.session_state["presc_topn"])
    else:
        st.error("Prescriptive data unavailable.")
