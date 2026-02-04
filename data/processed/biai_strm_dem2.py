import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import math
import textwrap
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TxDOT | Austin Safety Intelligence Elite", 
    layout="wide",
    page_icon="🛣️"
)

# --- 2. PATH CONFIGURATION ---
DATA_DIR = 'https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed'
CSV_FILENAME1 = 'atx_crash_data_2018-2026_cleansed.csv'
CSV_PATH1 = f"{DATA_DIR}/{CSV_FILENAME1}"

# CSV_FILENAME2 = "df_prescriptive_final_20260204_102224.csv"
# CSV_PATH2 = f"{DATA_DIR}/{CSV_FILENAME2}"
# df_raw2 = pd.read_csv(CSV_PATH2, low_memory=False)

# --- 3. SMART ASSET LOADER ---
def get_txdot_logo():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.webp']
    for ext in extensions:
        pattern = os.path.join(DATA_DIR1, 'txdot' + ext)
        files = glob.glob(pattern)
        if files: return files[0]
    return None

LOGO_PATH = get_txdot_logo()

# Make Altair nicer
alt.data_transformers.disable_max_rows()

# MAPBOX_API_KEY = "pk.your_mapbox_token_here"
# # Set Mapbox token (required for mapbox:// styles)
# if "MAPBOX_API_KEY" in st.secrets:
#     pdk.settings.mapbox_api_key = st.secrets["MAPBOX_API_KEY"]
MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"

# ----------------------------
# Helpers
# ----------------------------
REQUIRED_COLS = [
    "latitude",
    "longitude",
    "Address",
    "pred_est_ttl_comp_cost",
    "best_action",
    "expected_cost_after_action",
    "expected_reduction_amount",
    "pct_reduction",
    "ai_rationale",
]


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_pct_reduction(series: pd.Series) -> pd.Series:
    """
    Normalize pct_reduction into a 0..1 range.
    Handles cases where values are already 0..1 OR 0..100.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s

    # Heuristic: if typical values exceed 1.5, assume it's in 0..100 scale
    p95 = s.dropna().quantile(0.95)
    if p95 > 1.5:
        return s / 100.0
    return s


def make_location_id(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable id you can show in UI.
    Uses rounded lat/lon + index fallback.
    """
    lat = df["latitude"].round(5).astype(str)
    lon = df["longitude"].round(5).astype(str)
    return "loc_" + lat + "_" + lon + "_i" + df.index.astype(str)


def action_color_map(actions) -> Dict[str, Tuple[int, int, int]]:
    """
    Assign consistent colors to actions.
    Uses a fixed palette then cycles if you add more actions.
    """
    palette = [
        (31, 119, 180),   # blue
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189, 34),   # olive
        (23, 190, 207),   # teal
    ]
    actions = list(actions)
    cmap = {}
    for i, a in enumerate(actions):
        cmap[a] = palette[i % len(palette)]
    return cmap


def scale_sizes(values: pd.Series, min_size=40, max_size=400) -> pd.Series:
    """
    Map a numeric column into a visually useful size range.
    Uses robust scaling to reduce outlier domination.
    """
    v = pd.to_numeric(values, errors="coerce").fillna(0.0)

    if v.nunique() <= 1:
        return pd.Series(np.full(len(v), (min_size + max_size) / 2), index=v.index)

    # Robust: clip at 5th/95th percentiles
    lo, hi = v.quantile(0.05), v.quantile(0.95)
    if hi <= lo:
        lo, hi = v.min(), v.max()

    v_clip = v.clip(lo, hi)
    # Normalize to 0..1
    t = (v_clip - lo) / (hi - lo + 1e-9)
    # Slightly nonlinear so mid values remain visible
    t = np.sqrt(t)
    return min_size + t * (max_size - min_size)


def compact_text(s: str, n=140) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"

def fmt_dollars(x) -> str:
    """Format numeric as $#,### (nearest dollar)."""
    try:
        if pd.isna(x):
            return "—"
        return f"${float(x):,.0f}"
    except Exception:
        return "—"
    
@st.cache_data(show_spinner=False)
def prepare_df(df_prescriptive: pd.DataFrame) -> pd.DataFrame:
    df = df_prescriptive.copy()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _coerce_numeric(
        df,
        ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_cost_after_action",
         "expected_reduction_amount", "pct_reduction"]
    )

    # Clean coords
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Normalize pct_reduction to 0..1
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])

    # Location id
    df["location_id"] = make_location_id(df)

    df["address"] = df["Address"].astype(str).fillna("").str.strip()
    df["address_short"] = df["address"].map(lambda x: compact_text(x, 80))

    # Tooltip-friendly truncated rationale
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x, 160))

    # Precompute size for map points
    df["point_size"] = scale_sizes(df["expected_reduction_amount"], min_size=40, max_size=360)

    return df


def kpi_row(df: pd.DataFrame):
    total_reduction = df["expected_reduction_amount"].sum(skipna=True)
    avg_pct = df["pct_reduction_norm"].mean(skipna=True)

    top_action = (
        df["best_action"].mode().iloc[0] if not df["best_action"].dropna().empty else "—"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total expected reduction", fmt_dollars(total_reduction))
    c2.metric("Avg % reduction", f"{(avg_pct*100):.1f}%")
    c3.metric("Most recommended action", top_action)


def build_map(df: pd.DataFrame, top_n: int, all_actions: list):
    # Only plot top N locations
    df = (
        df.sort_values("expected_reduction_amount", ascending=False)
        .head(top_n)
        .copy()
    )

    actions = all_actions
    cmap = action_color_map(actions)

    df["color"] = df["best_action"].map(lambda x: list(cmap.get(x, (120, 120, 120))))
    # Cleaner diplay for tooltip
    df["pct_reduction_display"] = (df["pct_reduction_norm"] * 100).round(1).astype(str) + "%"

    df["pred_est_ttl_comp_cost_display"] = df["pred_est_ttl_comp_cost"].map(fmt_dollars)
    df["expected_reduction_amount_display"] = df["expected_reduction_amount"].map(fmt_dollars)
    df["expected_cost_after_action_display"] = df["expected_cost_after_action"].map(fmt_dollars)

    # Center the map
    center_lat = float(df["latitude"].mean()) if len(df) else 39.0
    center_lon = float(df["longitude"].mean()) if len(df) else -96.0

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10 if len(df) else 4,
        pitch=0,
    )

    # Pydeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="point_size",       # uses meters-ish; relative sizing is what we want
        radius_scale=1,
        radius_min_pixels=2,
        radius_max_pixels=80,
        pickable=True,
        opacity=0.75,
        stroked=True,
        get_line_color=[20, 20, 20],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": """
        <div style="max-width: 360px;">
          <div><b>Address</b>: {address_short}</div>  
          <div><b>Location</b>: {location_id}</div>
          <div><b>Action</b>: {best_action}</div>
          <div><b>Risk score</b>: {pred_est_ttl_comp_cost_display}</div>
          <div><b>Expected reduction</b>: {expected_reduction_amount_display}</div>
          <div><b>% reduction</b>: {pct_reduction_display}</div>
          <div><b>Cost after action</b>: {expected_cost_after_action_display}</div>
          <hr style="margin:6px 0;" />
          <div><b>Rationale</b>: {ai_rationale_short}</div>
        </div>
        """,
        "style": {"backgroundColor": "rgba(25, 25, 25, 0.92)", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        # navigation-day-v1
        # streets-v12
        map_style="mapbox://styles/mapbox/streets-v12",
        api_keys={"mapbox": MAPBOX_TOKEN},
        map_provider="mapbox"
    )

    st.pydeck_chart(deck, use_container_width=True)


def action_bars(df: pd.DataFrame):  
    # Exclude "no_change" from portfolio summaries
    df_plot = df[df["best_action"].ne("no_change")].copy()  

    # Aggregations
    agg = df_plot.groupby("best_action", dropna=False).agg(
        total_reduction=("expected_reduction_amount", "sum"),
        locations=("location_id", "count"),
        avg_pct=("pct_reduction_norm", "mean"),
    ).reset_index()

    # Total expected reduction by action
    bar1 = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("best_action:N", title="Best action", sort="-y"),
            y=alt.Y("total_reduction:Q", title="Total expected reduction ($)"),
            tooltip=["best_action", alt.Tooltip("total_reduction:Q", format=",.0f"),
                     "locations", alt.Tooltip("avg_pct:Q", format=".1%")],
        )
        .properties(height=260, title="Total expected reduction by action")
    )

    # Count of locations by action
    bar2 = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("best_action:N", title="Best action", sort="-y"),
            y=alt.Y("locations:Q", title="# Locations"),
            tooltip=["best_action", "locations"],
        )
        .properties(height=260, title="Locations by action")
    )

    c1, c2 = st.columns(2)
    c1.altair_chart(bar1, use_container_width=True)
    c2.altair_chart(bar2, use_container_width=True)


def ranked_table_and_details(df: pd.DataFrame, top_n: int):
    # Rank by biggest expected reduction (default)
    ranked = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

    show_cols = [
        "address",
        "location_id",
        "best_action",
        "pred_est_ttl_comp_cost",
        "expected_reduction_amount",
        "pct_reduction_norm",
        "expected_cost_after_action",
        "ai_rationale_short",
    ]
    ranked_display = ranked[show_cols].rename(columns={
        "address": "Address",
        "pct_reduction_norm": "pct_reduction",
        "ai_rationale_short": "ai_rationale (short)",
    })

    ranked_display["expected_reduction_amount"] = ranked_display["expected_reduction_amount"].map(fmt_dollars)
    ranked_display["expected_cost_after_action"] = ranked_display["expected_cost_after_action"].map(fmt_dollars)
    ranked_display["pred_est_ttl_comp_cost"] = ranked_display["pred_est_ttl_comp_cost"].map(fmt_dollars)

    ranked_display["pct_reduction"] = ranked["pct_reduction_norm"].map(
        lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "—"
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")
        st.dataframe(
            ranked_display,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Location details")
 
        options = ranked[["address", "location_id"]].fillna("").copy()
        options["label"] = options["address"] + "  (" + options["location_id"] + ")"

        selected_label = st.selectbox(
            "Select an address to see full rationale",
            options=options["label"].tolist(),
            index=0 if len(options) else None,
        )

        if selected_label:
            selected_loc = options.loc[options["label"] == selected_label, "location_id"].iloc[0]
            row = df.loc[df["location_id"] == selected_loc].iloc[0]
            st.markdown(
                f"""
                **Action:** {row['best_action']}  
                **Risk score:** `{fmt_dollars(row['pred_est_ttl_comp_cost'  ])}`  
                **Expected reduction:** `{fmt_dollars(row['expected_reduction_amount'])}`   
                **% reduction:** {row['pct_reduction_norm']*100:.1f}%  
                **Expected cost after action:** {fmt_dollars(row['expected_cost_after_action'])}  
                """
            )
            st.markdown("**Rationale:**")
            st.write(str(row.get("ai_rationale", "")))

# --- 4. DATA PIPELINE ---
@st.cache_data
def load_data(url:str):
    
    # Check if the file exists remotely
    # r = requests.get(url, timeout=30)
    # if r.status_code != 200:
    #     st.error(f"Dataset not found or not accessible (HTTP {r.status_code}) at: {url}")
    #     return None

    df = pd.read_csv(url, low_memory=False)
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Year'] = df['Crash timestamp'].dt.year
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    # Severity Mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    # Robust Numeric Handling
    cols_to_fix = ['tot_injry_cnt', 'crash_speed_limit', 'death_cnt', 'Estimated Total Comprehensive Cost']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # --- NEW: STANDARDIZE MODE COLUMNS ---
    # We map whatever names are in your CSV to these clean categories
    mapping = {
        'Passenger Car': ['passenger_car_involved', 'car_fl', 'is_car'],
        'Bicycle': ['bicycle_involved', 'bicycle_fl', 'is_bike'],
        'Pedestrian': ['pedestrian_involved', 'pedestrian_fl', 'is_ped'],
        'Motorcycle': ['motorcycle_involved', 'motorcycle_fl', 'is_mc'],
        'Commercial Veh': ['comml_mtr_veh_fl', 'cmv_involved', 'is_truck']
    }
    
    for clean_label, variations in mapping.items():
        # Find the first variation that actually exists in the CSV
        actual_col = next((v for v in variations if v in df.columns), None)
        if actual_col:
            df[clean_label] = df[actual_col].apply(lambda x: 1 if str(x).strip().upper() in ['Y', '1', 'TRUE', 'YES'] else 0)
        else:
            df[clean_label] = 0 # Default to 0 if not found
            
    # Marker size logic
    df['marker_size'] = (df['crash_speed_limit'] / 5).clip(lower=2)
    
    # Speed Binning
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 110]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    df['Speed_Bin'] = pd.cut(df['crash_speed_limit'], bins=bins, labels=labels)
    
    return df.dropna(subset=['latitude', 'longitude'])

df_raw1 = load_data(CSV_PATH1)

if df_raw1 is None:
    st.error(f"🛑 Dataset not found at {CSV_PATH1}")
    st.stop()

# --- 5. SIDEBAR FILTERS ---
with st.sidebar:
    if LOGO_PATH: st.image(LOGO_PATH, use_container_width=True)
    st.title("Strategic Filters")
    all_years = sorted(df_raw1['Year'].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    
    top_10_names = df_raw1.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).index.tolist()
    
    st.subheader("📍 Target Selection")
    corridor_options = ["All Corridors"] + top_10_names + ["--- Full Street List ---"] + sorted(df_raw1['rpt_street_name'].unique().tolist())
    selected_street = st.selectbox("Select a Corridor to Focus Analysis:", corridor_options)

# --- 6. APPLY FILTERS ---
df = df_raw1[df_raw1['Year'].isin(selected_years)]

if selected_street not in ["All Corridors", "--- Full Street List ---"]:
    df = df[df['rpt_street_name'] == selected_street]
    current_focus = selected_street
else:
    current_focus = "Austin District (Full View)"

# --- 7. HEADER & KPIs ---
st.title("Vision Zero: Safety Intelligence Dashboard")
st.caption(f"Currently Analyzing: **{current_focus}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df['death_cnt'].sum()))
k3.metric("Avg Speed Limit", f"{df['crash_speed_limit'].mean():.1f} MPH")
k4.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum()/1e9:.2f}B")

st.markdown("---")

# --- 8. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis"])

# --- TAB 1, 2, 3: (Logic remains same as previous version) ---
with tab1:
    col_list, col_map = st.columns([1, 2])
    with col_list:
        st.subheader("🔥 Top 10 Risk Corridors")
        risk_df = df_raw1.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
        risk_df.columns = ['Street', 'Cost']
        bar_colors = ["#4B0082" if s == selected_street else "#D8BFD8" for s in risk_df['Street']]
        fig_bar = px.bar(risk_df, x='Cost', y='Street', orientation='h', template="plotly_white")
        fig_bar.update_traces(marker_color=bar_colors)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_map:
        map_type = st.radio("Map Layer:", ["Economic Heatmap", "Incident Clusters"], horizontal=True)
        lat_c, lon_c = (df['latitude'].median(), df['longitude'].median()) if not df.empty else (30.2672, -97.7431)
        if map_type == "Economic Heatmap":
            fig_m = px.density_mapbox(df, lat='latitude', lon='longitude', z='Estimated Total Comprehensive Cost', 
                                      radius=12, center=dict(lat=lat_c, lon=lon_c), zoom=10, 
                                      mapbox_style="open-street-map", color_continuous_scale="Purples")
        else:
            fig_m = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='Severity_Label', 
                                      size='marker_size', center=dict(lat=lat_c, lon=lon_c), 
                                      zoom=10, mapbox_style="open-street-map")
        fig_m.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=600)
        st.plotly_chart(fig_m, use_container_width=True)

with tab2:
    st.subheader(f"Crash Risk Profile: {current_focus}")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        hr_vol = df.groupby('HOUR').size().reset_index(name='Volume')
        st.plotly_chart(px.line(hr_vol, x='HOUR', y='Volume', markers=True, color_discrete_sequence=['#6A0DAD']), use_container_width=True)
    with r1c2:
        fig_pie = px.pie(df, names='Severity_Label', hole=0.4, color_discrete_sequence=px.colors.sequential.Purples_r)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    heat_df = df.groupby(['DAY_NAME', 'HOUR']).size().reset_index(name='Count')
    fig_heat = px.density_heatmap(heat_df, x='HOUR', y='DAY_NAME', z='Count', color_continuous_scale='Purples',
                                  category_orders={'DAY_NAME': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    st.plotly_chart(fig_heat, use_container_width=True)

# --- NEW TAB 4: ECONOMIC ANALYSIS BY MODE ---
with tab4:
    st.subheader("Economic Impact by Transportation Type")
    
    # Calculate Avg Cost per mode
    modes = ['Passenger Car', 'Bicycle', 'Pedestrian', 'Motorcycle', 'Commercial Veh']
    mode_stats = []
    
    for m in modes:
        subset = df[df[m] == 1]
        if not subset.empty:
            avg_cost = subset['Estimated Total Comprehensive Cost'].mean()
            total_impact = subset['Estimated Total Comprehensive Cost'].sum()
            mode_stats.append({'Mode': m, 'Average Cost': avg_cost, 'Total Impact': total_impact, 'Count': len(subset)})
    
    if mode_stats:
        mode_df = pd.DataFrame(mode_stats)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Average Economic Cost per Incident**")
            # Using Purple theme
            fig_avg = px.bar(mode_df, x='Mode', y='Average Cost', color='Average Cost', 
                             color_continuous_scale='Purples', text_auto='.2s')
            st.plotly_chart(fig_avg, use_container_width=True)
            
        with c2:
            st.write("**Total Economic Burden (Sum)**")
            fig_total = px.pie(mode_df, names='Mode', values='Total Impact', 
                               color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig_total, use_container_width=True)
            
        st.markdown("---")
        st.write("**Mode Vulnerability Matrix (Volume vs. Average Cost)**")
        # Bubbles show how frequent vs how deadly/expensive each mode is
        fig_bubble = px.scatter(mode_df, x='Count', y='Average Cost', size='Total Impact', 
                                color='Mode', hover_name='Mode', size_max=60)
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("No Mode-specific data found in the current selection. Ensure columns like 'passenger_car_involved' are present.")
        
