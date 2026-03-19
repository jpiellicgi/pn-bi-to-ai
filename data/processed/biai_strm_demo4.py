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

CSV_FILENAME3 = "cost_forecast_2026.csv"
CSV_PATH3 = f"{DATA_DIR}/{CSV_FILENAME3}"

CSV_FILENAME4 = "cost_forecast_2026_per_crash.csv"
CSV_PATH4 = f"{DATA_DIR}/{CSV_FILENAME4}"

CSV_FILENAME5 = "crash_forecast_2026.csv"
CSV_PATH5 = f"{DATA_DIR}/{CSV_FILENAME5}"

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

def build_map(df, top_n=50):
    # Ensure display label column exists
    if "best_action_label" not in df.columns:
        df = df.copy()
        df["best_action_label"] = df["best_action"].apply(pretty_action)

    # Sort and take top N
    df_map = (
        df.sort_values("expected_reduction_amount", ascending=False)
          .head(top_n)
          .copy()
          .reset_index(drop=True)
    )

    # ---- FIX: category order based only on df_map actions ----
    pairs = (
        df_map[["best_action", "best_action_label"]]
        .dropna()
        .drop_duplicates()
    )
    label_order = pairs["best_action_label"].tolist()

    # ---- Colors ----
    ACTION_COLORS_RGB = {
        "reduce_speed_limit": (195, 10, 50),
        "increase_enforcement": (40, 90, 180),
        "improve_crosswalks": (142, 84, 255),
        "add_speed_bumps": (215, 45, 125),
        "work_zone_controls": (230, 126, 34),
        "micromobility_zone_controls": (82, 54, 171)
    }

    def _rgb_to_plotly(rgb_tuple):
        r, g, b = rgb_tuple
        return f"rgb({r},{g},{b})"

    # MUST be defined BEFORE full_actions
    ACTION_COLORS = {k: _rgb_to_plotly(v) for k, v in ACTION_COLORS_RGB.items()}
    DEFAULT_COLOR = "rgb(120,120,120)"

    # ---- FULL list of ALL actions (even if not in df_map) ----
    full_actions = list(ACTION_COLORS.keys())
    full_action_labels = [pretty_action(a) for a in full_actions]

    # ---- Map pretty label → color for ALL actions ----
    label_to_color = {
        pretty_action(a): ACTION_COLORS.get(a, DEFAULT_COLOR)
        for a in full_actions
    }

    # ---- FULL LEGEND FIX — build figure manually ----
    fig = go.Figure()
    # 1) Dummy legend traces (ensures legend shows ALL actions)
    
    for action_label in full_action_labels:
        fig.add_trace(
            go.Scattermapbox(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(size=10, color=label_to_color[action_label]),
                name=action_label,
                showlegend=True
            )
        )

    # 2) Real plotted points (one trace per action)
    for action_label in label_order:
        subset = df_map[df_map["best_action_label"] == action_label]

        fig.add_trace(
            go.Scattermapbox(
                lat=subset["latitude"],
                lon=subset["longitude"],
                mode="markers",
                marker=dict(size=10, color=label_to_color[action_label], opacity=0.9),
                name=action_label,
                showlegend=False   # legend handled by dummy traces
            )
        )

    # ---- Tooltip data PER REAL TRACE ----
    # IMPORTANT: skip dummy traces (first len(label_to_color) traces)
    dummy_count = len(label_to_color)

    for i, trace in enumerate(fig.data):
        if i < dummy_count:
            continue  # skip dummy legend traces

        action_label = trace.name
        mask = df_map["best_action_label"] == action_label

        trace_customdata = np.stack([
            df_map.loc[mask, "best_action_label"].astype(str),
            df_map.loc[mask, "pred_est_ttl_comp_cost"].astype(float),
            df_map.loc[mask, "expected_reduction_amount"].astype(float),
            df_map.loc[mask, "pct_reduction_norm"].astype(float),
            df_map.loc[mask, "address_short"].astype(str),
        ], axis=-1)

        trace.customdata = trace_customdata
        trace.hovertemplate = (
            "<b>%{customdata[0]}</b><br>" +
            "Estimated loss: %{customdata[1]:$,.0f}<br>" +
            "Expected reduction: %{customdata[2]:$,.0f}<br>" +
            "Percent reduction: %{customdata[3]:.1%}<br>" +
            "Address: %{customdata[4]}<extra></extra>"
        )

    # Layout
    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10),
        margin=dict(l=0, r=0, t=0, b=0),
    
        legend_title_text="Recommended action",
        legend=dict(
            itemsizing="constant",
            traceorder="normal",  
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
    
        showlegend=True   # <-- IMPORTANT
    )

    st.plotly_chart(fig, use_container_width=True)

def action_bars(df, top_n=50):

    # Sort and take top N
    df_bar = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

    # Full list of all actions (use internal keys)
    full_actions = [
        "reduce_speed_limit",
        "increase_enforcement",
        "improve_crosswalks",
        "add_speed_bumps",
        "work_zone_controls",
        "micromobility_zone_controls"
    ]
    full_action_labels = [pretty_action(a) for a in full_actions]

    # Aggregate *actual* data (may contain fewer than all actions)
    agg = df_bar.groupby("best_action_label").agg(
        total_reduction=("expected_reduction_amount", "sum"),
        locations=("location_id", "count")
    ).reset_index()

    # --- FORCE AGG TO INCLUDE ALL ACTIONS ---
    # Create a full-frame template
    full_frame = pd.DataFrame({
        "Recommended action": full_action_labels,
        "total_reduction": [0.0] * len(full_action_labels),
        "locations": [0] * len(full_action_labels),
    })

    # Normalize agg column names BEFORE merging
    agg = agg.rename(columns={"best_action_label": "Recommended action"})

    # Merge actual values into the full list
    agg_full = full_frame.merge(agg, on="Recommended action", how="left", suffixes=("", "_actual"))
    agg_full = agg_full.sort_values("total_reduction", ascending=False)

    # Fill missing values
    agg_full["total_reduction"] = agg_full["total_reduction_actual"].fillna(agg_full["total_reduction"])
    agg_full["locations"] = agg_full["locations_actual"].fillna(agg_full["locations"]).astype(int)

    # Final cleanup
    agg_full = agg_full[["Recommended action", "total_reduction", "locations"]]

    # LEFT chart — total expected reduction
    c1, c2 = st.columns(2)

    c1.altair_chart(
        alt.Chart(agg_full).mark_bar(color="#5236ab").encode(
            x=alt.X("Recommended action:N", title="Recommended action", sort='-y'),
            # y=alt.Y("total_reduction:Q", title="Total expected reduction ($)"),
            y=alt.Y("total_reduction:Q", title="Total expected reduction ($)"),
            tooltip=[
                alt.Tooltip("Recommended action:N"),
                alt.Tooltip("total_reduction:Q", format="$,.0f"),
            ]
        ),
        use_container_width=True
    )

    # RIGHT chart — number of locations
    c2.altair_chart(
        alt.Chart(agg_full).mark_bar(color="#5236ab").encode(
            x=alt.X("Recommended action:N", title="Recommended action", sort=alt.SortField(field="locations",order="descending")),
            y=alt.Y("locations:Q", title="Number of locations"),
            tooltip=[
                alt.Tooltip("Recommended action:N"),
                alt.Tooltip("locations:Q")
            ]
        ),
        use_container_width=True
    )

def clean_rationale(text: str) -> str:
    import re

    if not isinstance(text, str) or not text.strip():
        return text

    t = text.strip()

    # Split into sentence-like units
    sentences = [s.strip() for s in re.split(r"[.]\s*", t) if s.strip()]

    cleaned = []

    for s in sentences:

        # --- SPECIFIC TRANSFORMATIONS for your actual patterns ---
        if s.lower().startswith("pedestrian involvement suggests reducing conflict points"):
            cleaned.append(
                "Improving pedestrian visibility and reducing conflict points here would help lower crash risk."
            )
            continue

        if s.lower().startswith("pedestrian involvement detected"):
            cleaned.append(
                "This area sees meaningful pedestrian activity, which increases the chance of conflicts."
            )
            continue

        if s.lower().startswith("nighttime conditions detected"):
            cleaned.append(
                "Crashes here often occur at night, when visibility is lower."
            )
            continue

        if s.lower().startswith("higher-speed environment detected"):
            cleaned.append(
                "The roadway environment supports higher speeds, which increases crash severity."
            )
            continue

        if s.lower().startswith("work zone flag indicates"):
            cleaned.append(
                "Work zone indicators suggest temporary controls such as signage, barriers, or speed management would be appropriate."
            )
            continue

        if s.lower().startswith("work zone context detected"):
            cleaned.append(
                "Work zone activity has been identified at this location."
            )
            continue

        if s.lower().startswith("fatality flag increases priority"):
            cleaned.append(
                "A recent fatality increases the priority for stronger interventions at this location."
            )
            continue

        # fallback: lightly cleaned sentence with period added later
        cleaned.append(s)

    # Add periods back cleanly
    cleaned_with_periods = []
    for s in cleaned:
        s = s.strip()
        if not s.endswith("."):
            s = s + "."
        cleaned_with_periods.append(s)

    # Join into natural paragraph
    final_text = " ".join(cleaned_with_periods)
    final_text = re.sub(r"\s+", " ", final_text)

    return final_text.strip()

def ranked_table_and_details(df, top_n):
    left, right = st.columns([1.35, 1])
    ranked = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()
    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")
        show_columns = [
            "Address", "location_id", "best_action_label", "expected_reduction_amount",
            "pct_reduction_norm", "pred_est_ttl_comp_cost", "expected_cost_after_action", "ai_rationale_short"
        ]
        ranked_display = ranked[show_columns].rename(columns={"address": "Address","pct_reduction_norm": "% reduction","ai_rationale_short": "Rationale", "expected_cost_after_action":"Cost after action","pred_est_ttl_comp_cost":"Crash cost est.","expected_reduction_amount":"Expected reduction","best_action_label":"Recommended action","location_id":"Location",})
        
        ranked_display["Expected reduction"] = ranked_display["Expected reduction"].map(fmt_dollars)
        ranked_display["Cost after action"] = ranked_display["Cost after action"].map(fmt_dollars)
        ranked_display["Crash cost est."] = ranked_display["Crash cost est."].map(fmt_dollars)
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
                **Crash cost estimate:** `{fmt_dollars(row['pred_est_ttl_comp_cost'])}`   
                **Expected reduction:** `{fmt_dollars(row['expected_reduction_amount'])}`  
                **% reduction:** {row['pct_reduction_norm'] * 100:.1f}%  
                **Expected cost after action:** {fmt_dollars(row['expected_cost_after_action'])}  
                """
            )
            st.markdown("**Rationale:**")
            cleaned = clean_rationale(str(row.get("ai_rationale", "")))
            st.write(cleaned)

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

try:
    df_cost_forecast_2026 = pd.read_csv(CSV_PATH3)
except Exception as e:
    st.error(f"Forecasting dataset load failed: {e}")
    st.stop()   

try:
    df_cost_forecast_2026_per_crash = pd.read_csv(CSV_PATH4)
except Exception as e:
    st.error(f"Forecasting per crash dataset load failed: {e}")
    st.stop()   

try:
    df_crash_forecast_2026 = pd.read_csv(CSV_PATH5)
except Exception as e:
    st.error(f"Forecasting crash count dataset load failed: {e}")
    st.stop()   

# --- SIDEBAR ---
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🚶Top Predictors", "🗺️ Geographic Risk", "🚨 Speed and Severity", "⏰ Temporal Patterns", "💰 Transportation Mode Analysis", "🧠 Prescriptive Actions", "📈 2026 Monthly Forecast" ])

with tab1:
    st.subheader("Top Predictors and Historical Trends for Crash Cost and Volume")
    st.write(""" The top predictors of estimated cost were determined through a random forest model trained on crash data from the City of Austin from the 2018 to present.""")
    st.info("""
        **💡High-Level Insights:**
        - The top 3 predictors for estimated cost are **pedestrian involved**, **motorcycle involved**, and **crash speed limit**.
        - Within our dataset, **2025** had the lowest estimated total cost and the fewest crashes. **2018** had the highest estimated total cost and the most crashes. 
        - 2020 had low estimated costs and number of crashes compared to previous years and the two years immediately after due to work from home during COVID.
    """)

    shap_output, historicaloverview = st.columns([1, 2], gap="large")
    
    with shap_output:
        st.write("#### Top Predictors of Estimated Cost")
        st.info("The visual below shows the feature importances assigned by SHAP for each feature for the prediction of estimated cost in our random forest model.")
        st.image("data/processed/outputs/BI to AI SHAP vf.png", width=1200)
    
    with historicaloverview:
        st.write("#### Historical Trends")
        st.info("The below graphs update based on the fiscal years and corridors selected in the Global Filters side bar.")

        st.write("**Estimated Total Comprehensive Cost per Year**")
        df_total_cost= df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        fig_cost_bar= px.bar(df_total_cost, x="Year", y="Estimated Total Comprehensive Cost", text_auto=True)
        fig_cost_bar.update_layout(
            height=400, 
            width=800,
            margin=dict(l=100, r=100, t=20, b=20),
            yaxis_tickprefix='$'
            )
        avg_annual_cost_ref= df_total_cost["Estimated Total Comprehensive Cost"].mean()
        fig_cost_bar.add_hline(
            y=avg_annual_cost_ref,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average Annual Cost: ${avg_annual_cost_ref:,.0f}",
            annotation_position="top left"
            )
        st.plotly_chart(fig_cost_bar, use_container_width=True)

        st.write("**Total Number of Crashes per Year**")
        df_crash_volume= df.groupby("Year").size().reset_index(name="Crash Count")
        fig_crash_volume= px.bar(df_crash_volume, x="Year", y="Crash Count", text_auto=True)
        fig_crash_volume.update_layout(
            height=400, 
            width=800,
            margin=dict(l=100, r=100, t=20, b=20)
            )
        st.plotly_chart(fig_crash_volume, use_container_width=True)

with tab2:
    st.subheader("Geographic Risk Distribution")
    st.info("""
        **💡High-Level Insights:**
        - **Concentrated Risk:** High-density crash clusters are primarily localized along major highways (I-35 and MoPac) and high-speed arterial corridors.
        - **Corridor Vulnerability:** While urban centers show high crash volume, peripheral corridors often exhibit higher severity per incident due to increased speed limits.
        - **Geospatial Outliers:** Specific intersections consistently account for a disproportionate percentage of the total economic impact within the selected corridor.
    """)
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=df["latitude"].mean(),
            longitude=df["longitude"].mean(),
            zoom=10,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                'HeatmapLayer',
                data=df,
                get_position='[longitude, latitude]',
                get_weight="tot_injry_cnt",
                radius_pixels=60,
            ),
        ],
    ))

with tab3:
    st.subheader("Speed Limits and Crash Severity")
    st.info("""
        **💡High-Level Insights:**
        - **The Speed Threshold:** Crashes in zones with speed limits of **50+ MPH** result in exponentially higher comprehensive costs compared to urban 30 MPH zones.
        - **Severity Correlation:** Fatalities are significantly more likely to occur on roads categorized with higher speed bins, confirming speed as a primary severity driver.
    """)
    fig_speed = px.box(df, x="Speed_Bin", y="Estimated Total Comprehensive Cost", color="Severity_Label", 
                      title="Cost Distribution by Speed Limit Bin")
    st.plotly_chart(fig_speed, use_container_width=True)

with tab4:
    st.subheader("Temporal Patterns")
    st.info("""
        **💡High-Level Insights:**
        - **Peak Risk Windows:** Crash volume peaks during morning and evening rush hours (7-9 AM, 4-6 PM), but **economic impact** often spikes during late-night hours due to increased severity.
        - **Weekend Trends:** Saturday and Sunday nights show a distinct pattern of high-cost incidents, likely correlated with lower visibility and potential impairment.
    """)
    temp_df = df.groupby(["DAY_NAME", "hour_label"])["Estimated Total Comprehensive Cost"].mean().reset_index()
    fig_temp = px.line(temp_df, x="hour_label", y="Estimated Total Comprehensive Cost", color="DAY_NAME",
                      title="Average Crash Cost by Time of Day")
    st.plotly_chart(fig_temp, use_container_width=True)

with tab5:
    st.subheader("Transportation Mode Analysis")
    st.info("""
        **💡High-Level Insights:**
        - **Vulnerable Road Users:** Pedestrian and motorcycle-involved crashes account for only a small fraction of total volume but over **40% of total economic impact**.
        - **Mode Disparity:** While passenger cars represent the highest total volume, the cost per incident for micromobility and bicycles is rising in specific urban sectors.
        - **Commercial Impact:** Commercial vehicle crashes, though less frequent, result in significant corridor delays and high secondary economic costs.
    """)
    modes = ["Passenger Car", "Bicycle", "Pedestrian", "Motorcycle", "Commercial Veh"]
    mode_sums = [df[m].sum() for m in modes]
    fig_modes = px.pie(values=mode_sums, names=modes, title="Volume by Transportation Mode")
    st.plotly_chart(fig_modes, use_container_width=True)

with tab6:
    st.subheader("Prescriptive Actions")
    if df_prescriptive_raw is not None:
        df_p = prepare_prescriptive_df(df_prescriptive_raw)
        build_map(df_p)
        action_bars(df_p)
        ranked_table_and_details(df_p, 50)
    else:
        st.warning("Prescriptive data not available.")

with tab7:
    st.subheader("2026 Monthly Forecast")
    st.write("Projected Economic Impact and Crash Volume for the next fiscal year.")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig_f1 = px.line(df_cost_forecast_2026, x="Month", y="Projected_Cost", title="2026 Cost Forecast")
        st.plotly_chart(fig_f1, use_container_width=True)
    with col_f2:
        fig_f2 = px.line(df_crash_forecast_2026, x="Month", y="Projected_Crashes", title="2026 Volume Forecast")
        st.plotly_chart(fig_f2, use_container_width=True)
