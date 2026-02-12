import os
import glob
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
import altair as alt
import requests

# --- 1. PAGE CONFIGURATION (KEEP PARTNER DEFAULT) ---
st.set_page_config(
    page_title="TxDOT | Austin Safety Intelligence Elite",
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
CSV_PATH2 = f"{DATA_DIR}/{CSV_FILENAME2}"

MAPBOX_TOKEN = "pk.eyJ1IjoianBpZWxsaWNnaSIsImEiOiJjbWw2c21tdGgwaThvM2RvY25iaTc5aWR1In0.1zrdRIL8deHfHNMikwdKMw"


# --- 3. SMART ASSET LOADER (NOTE: glob works for local files, not URLs) ---
def get_txdot_logo():
    # If you later add a local logo in the repo (e.g., ./data/processed/txdot.png),
    # you can switch DATA_DIR to a local path or directly use a URL with st.image(URL).
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.webp"]
    for ext in extensions:
        pattern = os.path.join(DATA_DIR, "txdot" + ext)
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


LOGO_PATH = get_txdot_logo()


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
# Partner Dashboard Data Pipeline (unchanged logic, just uses safe loader)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    try:
        df = read_csv_url(url)
    except Exception as e:
        # local load for dev testing
        local_path = os.path.join(LOCAL_DATA_DIR,os.path.split(url)[-1])
        df = pd.read_csv(local_path)

    # Preprocessing
    df["Crash timestamp"] = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    df["Year"] = df["Crash timestamp"].dt.year
    df["HOUR"] = df["Crash timestamp"].dt.hour
    df["DAY_NAME"] = df["Crash timestamp"].dt.day_name()

    # Severity Mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    # Robust Numeric Handling
    cols_to_fix = ["tot_injry_cnt", "crash_speed_limit", "Estimated Total Comprehensive Cost"] #"death_cnt"
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # STANDARDIZE MODE COLUMNS
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

    # Marker size logic
    df["marker_size"] = (df["crash_speed_limit"] / 5).clip(lower=2)

    # Speed Binning
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 110]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
    df["Speed_Bin"] = pd.cut(df["crash_speed_limit"], bins=bins, labels=labels)

    return df.dropna(subset=["latitude", "longitude"])


# ----------------------------
# Your Prescriptive Tab Helpers
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

    p95 = s.dropna().quantile(0.95)
    if p95 > 1.5:
        return s / 100.0
    return s


def make_location_id(df: pd.DataFrame) -> pd.Series:
    lat = df["latitude"].round(5).astype(str)
    lon = df["longitude"].round(5).astype(str)
    return "loc_" + lat + "_" + lon + "_i" + df.index.astype(str)


def action_color_map(actions) -> Dict[str, Tuple[int, int, int]]:
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
    return {a: palette[i % len(palette)] for i, a in enumerate(actions)}


def scale_sizes(values: pd.Series, min_size=40, max_size=400) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0)

    if v.nunique() <= 1:
        return pd.Series(np.full(len(v), (min_size + max_size) / 2), index=v.index)

    lo, hi = v.quantile(0.05), v.quantile(0.95)
    if hi <= lo:
        lo, hi = v.min(), v.max()

    v_clip = v.clip(lo, hi)
    t = (v_clip - lo) / (hi - lo + 1e-9)
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
    try:
        if pd.isna(x):
            return "—"
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(df_prescriptive: pd.DataFrame) -> pd.DataFrame:
    df = df_prescriptive.copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _coerce_numeric(
        df,
        ["latitude", "longitude", "pred_est_ttl_comp_cost", "expected_cost_after_action",
         "expected_reduction_amount", "pct_reduction"]
    )

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["location_id"] = make_location_id(df)

    df["address"] = df["Address"].astype(str).fillna("").str.strip()
    df["address_short"] = df["address"].map(lambda x: compact_text(x, 80))
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x, 160))

    df["point_size"] = scale_sizes(df["expected_reduction_amount"], min_size=40, max_size=360)
    return df


def kpi_row(df: pd.DataFrame):
    total_reduction = df["expected_reduction_amount"].sum(skipna=True)
    avg_pct = df["pct_reduction_norm"].mean(skipna=True)
    top_action = df["best_action"].mode().iloc[0] if not df["best_action"].dropna().empty else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total expected reduction", fmt_dollars(total_reduction))
    c2.metric("Avg % reduction", f"{(avg_pct * 100):.1f}%")
    c3.metric("Most recommended action", top_action)


def build_map(df: pd.DataFrame, top_n: int, all_actions: list):
    df = df.sort_values("expected_reduction_amount", ascending=False).head(top_n).copy()

    cmap = action_color_map(all_actions)
    df["color"] = df["best_action"].map(lambda x: list(cmap.get(x, (120, 120, 120))))

    df["pct_reduction_display"] = (df["pct_reduction_norm"] * 100).round(1).astype(str) + "%"
    df["pred_est_ttl_comp_cost_display"] = df["pred_est_ttl_comp_cost"].map(fmt_dollars)
    df["expected_reduction_amount_display"] = df["expected_reduction_amount"].map(fmt_dollars)
    df["expected_cost_after_action_display"] = df["expected_cost_after_action"].map(fmt_dollars)

    center_lat = float(df["latitude"].mean()) if len(df) else 30.2672
    center_lon = float(df["longitude"].mean()) if len(df) else -97.7431

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10 if len(df) else 4, pitch=0)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="point_size",
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
        map_style="mapbox://styles/mapbox/streets-v12",
        api_keys={"mapbox": MAPBOX_TOKEN},
        map_provider="mapbox",
    )

    st.pydeck_chart(deck, use_container_width=True)
    
def action_bars(df: pd.DataFrame):
    df_plot = df[df["best_action"].ne("no_change")].copy()

    agg = df_plot.groupby("best_action", dropna=False).agg(
        total_reduction=("expected_reduction_amount", "sum"),
        locations=("location_id", "count"),
        avg_pct=("pct_reduction_norm", "mean"),
    ).reset_index()

    # --- Chart 1: Total reduction ---
    bar1 = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "best_action:N",
                title="Best action",
                sort="-y",
                axis=alt.Axis(labelAngle=-35, labelLimit=180),
            ),
            y=alt.Y(
                "total_reduction:Q",
                title="Total expected reduction ($)",
                axis=alt.Axis(format="~s"),  # ✅ 1.2M instead of 1200000
            ),
            tooltip=[
                "best_action",
                alt.Tooltip("total_reduction:Q", format=",.0f"),
                "locations",
                alt.Tooltip("avg_pct:Q", format=".1%"),
            ],
        )
        .properties(height=340, title="Total expected reduction by action", padding={"top":30})
        .configure_title(fontSize=14, offset=24)  # ✅ prevents title clipping
        .configure_view(strokeWidth=0)
    )

    # --- Chart 2: Locations ---
    bar2 = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "best_action:N",
                title="Best action",
                sort="-y",
                axis=alt.Axis(labelAngle=-35, labelLimit=180),
            ),
            y=alt.Y("locations:Q", title="# Locations"),
            tooltip=["best_action", "locations"],
        )
        .properties(height=340, title="Locations by action")
        .configure_title(fontSize=14, offset=16)
        .configure_view(strokeWidth=0)
    )

    c1, c2 = st.columns(2)
    c1.altair_chart(bar1, use_container_width=True)
    c2.altair_chart(bar2, use_container_width=True)

def ranked_table_and_details(df: pd.DataFrame, top_n: int):
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
        lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "—"
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")
        st.dataframe(ranked_display, use_container_width=True, hide_index=True)

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
                **Risk score:** `{fmt_dollars(row['pred_est_ttl_comp_cost'])}`   
                **Expected reduction:** `{fmt_dollars(row['expected_reduction_amount'])}`  
                **% reduction:** {row['pct_reduction_norm'] * 100:.1f}%  
                **Expected cost after action:** {fmt_dollars(row['expected_cost_after_action'])}  
                """
            )
            st.markdown("**Rationale:**")
            st.write(str(row.get("ai_rationale", "")))


# ----------------------------
# Load datasets
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


# --- 5. SIDEBAR FILTERS (PARTNER DEFAULT) ---
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.title("Strategic Filters")

    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])

    top_10_names = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()

    st.subheader("📍 Target Selection")
    corridor_options = ["All Corridors"] + top_10_names + ["--- Full Street List ---"] + sorted(df_raw1["rpt_street_name"].unique().tolist())
    selected_street = st.selectbox("Select a Corridor to Focus Analysis:", corridor_options)


# --- 6. APPLY FILTERS ---
df = df_raw1[df_raw1["Year"].isin(selected_years)]

if selected_street not in ["All Corridors", "--- Full Street List ---"]:
    df = df[df["rpt_street_name"] == selected_street]
    current_focus = selected_street
else:
    current_focus = "Austin District (Full View)"


# --- 7. HEADER & KPIs ---
st.title("Safety Intelligence Dashboard")
st.caption(f"Currently Analyzing: **{current_focus}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")

st.markdown("---")


# --- 8. TABS (ADD YOUR 5TH TAB) ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Top Predictors and Historical Overview", "🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns", "💰 Economic Analysis", "🧠 Prescriptive Actions"]
)

# --- TAB 1 ---
with tab1:
    st.write("##### The top predictors and prescriptive actions were determined through a random forest model trained on crash data from the City of Austin from the 2018 to present.")
    shap_output, historicaloverview = st.columns([1, 2], gap="large")
    with shap_output:
        st.subheader("Top Predictors of Estimated Cost")
        st.image("data/processed/BI to AI SHAP vf.png", width=800)
        st.write("This shows the feature importances assigned by SHAP for each feature for the prediction of estimated cost in our random forest model. The most important features for predicting estimated cost were pedestrian involved, motorcycle involved, and On TxDOT highway system. ")
    
    with historicaloverview:
        st.subheader("Historical Trends")

        st.write("**Estimated Total Comprehensive Cost per Year**")
        df_total_cost= df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        fig_cost_bar= px.bar(df_total_cost, x="Year", y="Estimated Total Comprehensive Cost", color="Estimated Total Comprehensive Cost", 
                             color_continuous_scale="Purples", text_auto=".2s")
        fig_cost_bar.update_layout(
            height=300, 
            width=400,
            margin=dict(l=100, r=100, t=20, b=20) # Tighten whitespace
            )
        st.plotly_chart(fig_cost_bar)

        st.write("**Total Number of Crashes per Year**")
        df_crash_count= df.groupby("Year")["ID"].count().reset_index()
        df_crash_count.columns = ["Year", "Number of Crashes"]
        fig_crash_count= px.bar(df_crash_count, x="Year", y="Number of Crashes", color="Number of Crashes", 
        color_continuous_scale="Purples", text_auto=".2s")
        fig_crash_count.update_layout(
            height=300, 
            width=400,
            margin=dict(l=100, r=100, t=20, b=20) # Tighten whitespace
            )
        st.plotly_chart(fig_crash_count)
  

# --- TAB 2 ---
with tab2:
    col_list, col_map = st.columns([1, 2])
    with col_list:
        st.subheader("🔥 Top 10 Risk Corridors")
        risk_df = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).reset_index()
        risk_df.columns = ["Street", "Cost"]
        bar_colors = ["#4B0082" if s == selected_street else "#D8BFD8" for s in risk_df["Street"]]
        fig_bar = px.bar(risk_df, x="Cost", y="Street", orientation="h", template="plotly_white")
        fig_bar.update_traces(marker_color=bar_colors)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_map:
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
                size="marker_size",
                center=dict(lat=lat_c, lon=lon_c),
                zoom=10,
                mapbox_style="open-street-map",
            )

        fig_m.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_m, use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.subheader(f"Crash Risk Profile: {current_focus}")
    r1c1, r1c2, r1c3 = st.columns(3) #removed r1c1 for testing
    # with r1c1:
    #     hr_vol = df.groupby("HOUR").size().reset_index(name="Volume")
    #     st.plotly_chart(
    #         px.line(hr_vol, x="HOUR", y="Volume", markers=True, color_discrete_sequence=["#6A0DAD"]),
    #         use_container_width=True,
    #     )
    with r1c1:
        fig_pie = px.pie(df, names="Severity_Label", hole=0.4, color_discrete_sequence=px.colors.sequential.Purples_r, title='Accident Severity Breakdown')
        fig_pie.update_layout(
            height=450, 
            width=500,
            legend=dict(
                x=0.85,          # Pulls legend closer to the center
                #xanchor="left",
                #yanchor="middle",
                y=0.5
                )   
            )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with r1c2:
        #Accidents by Speed Limit and Severity
        df_speed_severity= df.groupby(["Speed_Bin", "Severity_Label"]).size().reset_index(name="Accident_Count")
        #df_speed_severity= df.groupby(["Speed_Bin", "Severity_Label"])["ID"].count().reset_index(name="Accident_Count")
        fig_bar = px.bar(
            df_speed_severity,
            x="Speed_Bin",
            y="Accident_Count",
            color="Severity_Label",
            title="Accidents by Speed Limit and Severity",
            labels={"Accident_Count": "Number of Accidents", "Speed_Bin": "Speed Limit (mph)", "Severity_Label": "Severity Label"},
            # This ensures the bars are stacked rather than grouped
            barmode="stack",
            # Optional: Define a specific order for the severity levels in the legend
            category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with r1c3:
        #Average cost by speed bin
        df_avg_cost_speed= df.groupby("Speed_Bin")["Estimated Total Comprehensive Cost"].mean().reset_index()
        fig_avg_cost_speed= px.bar(df_avg_cost_speed, x="Speed_Bin", y="Estimated Total Comprehensive Cost", color="Estimated Total Comprehensive Cost",
            title= "Average Estimated Cost by Speed Bin",labels={"Estimated Total Comprehensive Cost": "Average Estimated Cost", "Speed_Bin": "Speed Limit (mph)"},                            
            color_continuous_scale="Purples", text_auto=".2s")
        # fig_avg_cost_speed.update_layout(
        #     height=300, 
        #     width=400,
        #     margin=dict(l=100, r=100, t=20, b=20) # Tighten whitespace
        #     )
        st.plotly_chart(fig_avg_cost_speed)


# --- TAB 4 ---
with tab4:    
    heat_df = df.groupby(["DAY_NAME", "HOUR"]).size().reset_index(name="Count")
    fig_heat = px.density_heatmap(
        heat_df,
        x="HOUR",
        y="DAY_NAME",
        z="Count",
        title= "Number of Accidents by Day and Hour",
        color_continuous_scale="Purples",
        category_orders={"DAY_NAME": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    #Average cost by speed bin
    df_avg_cost_hour= df.groupby("HOUR")["Estimated Total Comprehensive Cost"].mean().reset_index()
    fig_avg_cost_hour= px.bar(df_avg_cost_hour, x="HOUR", y="Estimated Total Comprehensive Cost", color="Estimated Total Comprehensive Cost",
        title= "Average Estimated Cost by Hour",                            
        color_continuous_scale="Purples", text_auto=".2s")
    # fig_avg_cost_speed.update_layout(
    #     height=300, 
    #     width=400,
    #     margin=dict(l=100, r=100, t=20, b=20) # Tighten whitespace
    #     )
    st.plotly_chart(fig_avg_cost_hour)

    #Severity Breakdown by Hour
    df_hour_severity= df.groupby(["HOUR", "Severity_Label"]).size().reset_index(name="Accident_Count")
    fig_bar = px.bar(
        df_hour_severity,
        x="HOUR",
        y="Accident_Count",
        color="Severity_Label",
        title="Number of Accidents by Hour and Severity",
        labels={"Accident_Count": "Number of Accidents", "Severity_Label": "Severity Label"},
        # This ensures the bars are stacked rather than grouped
        barmode="stack",
        # Optional: Define a specific order for the severity levels in the legend
        category_orders={"Severity_Label": ["Fatal", "Serious Injury", "Minor Injury", "Possible Injury", "No Injury", "Unknown"]},
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# --- TAB 5 ---
with tab5:
    st.subheader("Economic Impact by Transportation Type")
    st.write("##### This page shows the cost of accidents in which a pedestrian, bicycle, or motorcycle were involved.")
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
            st.write("**Average Economic Cost per Incident**")
            fig_avg = px.bar(mode_df, x="Mode", y="Average Cost", color="Average Cost",
                             color_continuous_scale="Purples", text_auto=".2s")
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
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("No Mode-specific data found in the current selection.")


# --- ✅ TAB 6 (YOUR PRESCRIPTIVE ACTIONS) ---
with tab6:
    st.subheader("Prescriptive Actions: Recommended Interventions & Savings")
    st.caption("Explore high-impact locations, recommended interventions, and expected reductions.")

    if df_prescriptive_raw is None:
        st.error(f"Prescriptive dataset failed to load: {prescriptive_load_error}")
        st.stop()

    # Prepare + validate
    try:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)
    except Exception as e:
        st.error(f"Prescriptive data validation error: {e}")
        st.write("Columns found:", list(df_prescriptive_raw.columns))
        st.stop()

    # Drop no_change by default
    dfp = dfp[dfp["best_action"] != "no_change"].copy()

    # Optional alignment with partner corridor selection:
    # If user picked a specific corridor, filter addresses containing that corridor name.
    if selected_street not in ["All Corridors", "--- Full Street List ---"]:
        dfp = dfp[dfp["address"].str.contains(selected_street, case=False, na=False)].copy()

    if dfp.empty:
        st.warning("No prescriptive records match the current corridor selection. Try 'All Corridors'.")
        st.stop()

    all_actions = sorted(dfp["best_action"].dropna().unique().tolist())

    # Controls inside the tab (so we don't fight the shared sidebar)
    cA, cB = st.columns([2, 1])
    with cA:
        selected_actions = st.multiselect(
            "Filter by recommended action",
            options=all_actions,
            default=all_actions,
        )
    with cB:
        top_n = st.slider("Top N locations", min_value=10, max_value=300, value=50, step=10)

    dfp_f = dfp[dfp["best_action"].isin(selected_actions)].copy()

    if dfp_f.empty:
        st.warning("No data matches your action filters. Select more actions.")
        st.stop()

    df_topn = dfp_f.sort_values("expected_reduction_amount", ascending=False).head(top_n)

    # KPI row
    kpi_row(df_topn)
    st.divider()

    # Map
    st.subheader(f"Impact map (Top {top_n} | color = action)")
    build_map(dfp_f, top_n=top_n, all_actions=all_actions)
    st.divider()

    # Portfolio summary
    st.subheader("Action portfolio summary")
    action_bars(dfp_f)
    st.divider()

    # Ranked table + details
    ranked_table_and_details(dfp_f, top_n=top_n)

    with st.expander("Notes & tips"):
        st.markdown(
            """
            **Map encoding**
            - **Color**: `best_action`

            **Percent normalization**
            - If `pct_reduction` is 0–100, it is converted to 0–1 automatically.
            - If it's already 0–1, it stays as-is.
            """
        )
