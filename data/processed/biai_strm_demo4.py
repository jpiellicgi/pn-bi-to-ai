import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import requests

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="CGI | Austin Safety Intelligence Elite",
    layout="wide",
    page_icon="🛣️"
)

alt.data_transformers.disable_max_rows()

DATA_DIR = "https://raw.githubusercontent.com/jpiellicgi/pn-bi-to-ai/main/data/processed"
LOCAL_DATA_DIR = "data/processed"

CSV_FILENAME1 = "atx_crash_data_2018-2026_clean.csv"
CSV_FILENAME2 = "df_prescriptive_final_20260204_102224.csv"

CSV_PATH1 = f"{DATA_DIR}/{CSV_FILENAME1}"
CSV_PATH2 = f"{DATA_DIR}/outputs/{CSV_FILENAME2}"

# ----------------------------
# Helpers
# ----------------------------
def get_cgi_logo():
    """Return local or remote logo filepath."""
    logo_filename = "CGI_logo_color_rgb.jpg"
    if os.path.exists(logo_filename):
        return logo_filename
    try:
        url = f"{DATA_DIR}/{logo_filename}"
        if requests.head(url, timeout=5).status_code == 200:
            return url
    except:
        pass
    return None

LOGO_PATH = get_cgi_logo()

@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code} fetching {url}")
    if len(r.content) <= 10:
        raise ValueError(f"Remote file too small: {url}")
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

def format_hour(hour):
    if hour == 0: return "12 AM"
    if hour < 12: return f"{hour} AM"
    if hour == 12: return "12 PM"
    return f"{hour - 12} PM"

# ----------------------------
# Load Crash Data
# ----------------------------
@st.cache_data(show_spinner=False)
def load_partner_data(url: str) -> pd.DataFrame:
    try:
        df = read_csv_url(url)
    except Exception:
        df = pd.read_csv(os.path.join(LOCAL_DATA_DIR, os.path.split(url)[-1]))

    df["Crash timestamp"] = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    df["Year"] = df["Crash timestamp"].dt.year
    df["HOUR"] = df["Crash timestamp"].dt.hour
    df["DAY_NAME"] = df["Crash timestamp"].dt.day_name()

    df["hour_label"] = pd.Categorical(
        df["HOUR"].apply(format_hour),
        categories=[format_hour(h) for h in range(24)],
        ordered=True
    )

    sev_map = {1:"Fatal",2:"Serious Injury",3:"Minor Injury",4:"Possible Injury",0:"No Injury",5:"Unknown"}
    df["Severity_Label"] = df["crash_sev_id"].map(sev_map)

    # FIX APPLIED HERE
    numeric_cols = ["tot_injry_cnt","crash_speed_limit","Estimated Total Comprehensive Cost"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0

    mode_map = {
        "Passenger Car": ["passenger car_involved","passenger_car_involved","car_fl","is_car"],
        "Bicycle": ["bicycle_involved","bicycle_fl","is_bike"],
        "Pedestrian": ["pedestrian_involved","pedestrian_fl","is_ped"],
        "Motorcycle": ["motorcycle_involved","motorcycle_fl","is_mc"],
        "Commercial Veh": ["comml_mtr_veh_fl","cmv_involved","is_truck"],
        "Micromobility": ["micromobility device_involved","micromobility_fl"],
        "E-Scooter": ["e-scooter_involved","scooter_fl"],
        "Large Passenger Veh": ["large passenger vehicle_involved","large_veh_fl"],
        "Train": ["train_involved","train_fl"],
        "Motor Vehicle": ["motor vehicle_involved","motor_veh_fl"],
        "Other": ["other_involved","other_fl"]
    }

    for clean, cols in mode_map.items():
        col = next((c for c in cols if c in df.columns), None)
        df[clean] = df[col].apply(lambda x: 1 if str(x).upper() in ["Y","1","TRUE","YES"] else 0) if col else 0

    df["marker_size"] = (df["crash_speed_limit"] / 5).clip(lower=2)
    df["Speed_Bin"] = pd.cut(
        df["crash_speed_limit"],
        bins=[0,20,30,40,50,60,70,80,110],
        labels=["<20","20-30","30-40","40-50","50-60","60-70","70-80","80+"]
    )

    return df.dropna(subset=["latitude","longitude"])

# ----------------------------
# Prescriptive Helpers
# ----------------------------
def _coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def normalize_pct_reduction(s):
    s = pd.to_numeric(s, errors="coerce")
    return s/100 if s.dropna().quantile(0.95) > 1.5 else s

def make_location_id(df):
    return df["latitude"].round(5).astype(str) + ", " + df["longitude"].round(5).astype(str)

def compact_text(s, n=140):
    if not s or (isinstance(s,float) and math.isnan(s)): return ""
    return s[:n-1] + "…" if len(s) > n else s

def fmt_dollars(x):
    try:
        return f"${float(x):,.0f}" if pd.notnull(x) else "—"
    except:
        return "—"

def pretty_action(a):
    return a.replace("_"," ").strip() if a else ""

@st.cache_data(show_spinner=False)
def prepare_prescriptive_df(dfp):
    df = dfp.copy()
    df = _coerce_numeric(df, [
        "latitude","longitude",
        "pred_est_ttl_comp_cost","expected_cost_after_action",
        "expected_reduction_amount","pct_reduction"
    ])
    df = df.dropna(subset=["latitude","longitude"])
    df = df[df["best_action"] != "no_change"]
    df["pct_reduction_norm"] = normalize_pct_reduction(df["pct_reduction"])
    df["location_id"] = make_location_id(df)
    df["address_short"] = df["Address"].astype(str).map(lambda x: compact_text(x,80))
    df["ai_rationale_short"] = df["ai_rationale"].astype(str).map(lambda x: compact_text(x,160))
    df["best_action_label"] = df["best_action"].apply(pretty_action)
    return df

# ----------------------------
# Build Map (Fully Working)
# ----------------------------
def build_map(df, top_n=50):
    if "best_action_label" not in df.columns:
        df = df.copy()
        df["best_action_label"] = df["best_action"].apply(pretty_action)

    df_map = (
        df.sort_values("expected_reduction_amount", ascending=False)
          .head(top_n)
          .copy()
          .reset_index(drop=True)
    )

    ACTION_COLORS_RGB = {
        "reduce_speed_limit": (195,10,50),
        "increase_enforcement": (40,90,180),
        "improve_crosswalks": (142,84,255),
        "add_speed_bumps": (215,45,125),
        "work_zone_controls": (230,126,34),
        "micromobility_zone_controls": (82,54,171)
    }
    ACTION_COLORS = {
        k: f"rgb({r},{g},{b})"
        for k,(r,g,b) in ACTION_COLORS_RGB.items()
    }

    full_actions = list(ACTION_COLORS.keys())
    full_action_labels = [pretty_action(a) for a in full_actions]

    label_to_color = {
        pretty_action(a): ACTION_COLORS[a]
        for a in full_actions
    }

    pairs = df_map[["best_action_label"]].drop_duplicates()
    label_order = pairs["best_action_label"].tolist()

    fig = go.Figure()

    # Dummy legend traces (show all actions)
    for action_label in full_action_labels:
        fig.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="markers",
            marker=dict(size=10, color=label_to_color[action_label]),
            name=action_label,
            showlegend=True
        ))

    # Real plotted points
    for action_label in label_order:
        subset = df_map[df_map["best_action_label"] == action_label]
        fig.add_trace(go.Scattermapbox(
            lat=subset["latitude"],
            lon=subset["longitude"],
            mode="markers",
            marker=dict(size=10,color=label_to_color[action_label],opacity=0.9),
            name=action_label,
            showlegend=False
        ))

    dummy_count = len(full_action_labels)

    for i, trace in enumerate(fig.data):
        if i < dummy_count:
            continue
        action_label = trace.name
        mask = df_map["best_action_label"] == action_label

        trace.customdata = np.stack([
            df_map.loc[mask,"best_action_label"].astype(str),
            df_map.loc[mask,"pred_est_ttl_comp_cost"].astype(float),
            df_map.loc[mask,"expected_reduction_amount"].astype(float),
            df_map.loc[mask,"pct_reduction_norm"].astype(float),
            df_map.loc[mask,"address_short"].astype(str)
        ], axis=-1)

        trace.hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Estimated loss: %{customdata[1]:$,.0f}<br>"
            "Expected reduction: %{customdata[2]:$,.0f}<br>"
            "Percent reduction: %{customdata[3]:.1%}<br>"
            "Address: %{customdata[4]}<extra></extra>"
        )

    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat,lon=center_lon), zoom=10),
        margin=dict(l=0,r=0,t=0,b=0),
        legend_title_text="Recommended action",
        showlegend=True,
        legend=dict(
            x=0.02,y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Action Bars
# ----------------------------
def action_bars(df, top_n=50):
    df_bar = (
        df.sort_values("expected_reduction_amount", ascending=False)
          .head(top_n)
          .copy()
    )

    full_actions = [
        "reduce_speed_limit", "increase_enforcement", "improve_crosswalks",
        "add_speed_bumps","work_zone_controls","micromobility_zone_controls"
    ]
    full_labels = [pretty_action(a) for a in full_actions]

    agg = df_bar.groupby("best_action_label").agg(
        total_reduction=("expected_reduction_amount","sum"),
        locations=("location_id","count")
    ).reset_index()

    full_frame = pd.DataFrame({
        "Recommended action": full_labels,
        "total_reduction": [0.0]*len(full_labels),
        "locations": [0]*len(full_labels)
    })

    agg = agg.rename(columns={"best_action_label":"Recommended action"})
    agg_full = full_frame.merge(agg, on="Recommended action", how="left", suffixes=("","_actual"))

    agg_full["total_reduction"] = agg_full["total_reduction_actual"].fillna(agg_full["total_reduction"])
    agg_full["locations"] = agg_full["locations_actual"].fillna(agg_full["locations"]).astype(int)
    agg_full = agg_full[["Recommended action","total_reduction","locations"]]
    agg_full = agg_full.sort_values("total_reduction", ascending=False)

    c1, c2 = st.columns(2)

    c1.altair_chart(
        alt.Chart(agg_full).mark_bar(color="#5236ab").encode(
            x=alt.X("Recommended action:N", sort='-y'),
            y=alt.Y("total_reduction:Q", title="Total expected reduction ($)"),
            tooltip=[
                alt.Tooltip("Recommended action:N"),
                alt.Tooltip("total_reduction:Q", format="$,.0f")
            ]
        ),
        use_container_width=True
    )

    c2.altair_chart(
        alt.Chart(agg_full).mark_bar(color="#5236ab").encode(
            x=alt.X("Recommended action:N", sort=alt.SortField(field="locations", order="descending")),
            y=alt.Y("locations:Q", title="Number of locations"),
            tooltip=[
                alt.Tooltip("Recommended action:N"),
                alt.Tooltip("locations:Q")
            ]
        ),
        use_container_width=True
    )

# ----------------------------
# Rationale Cleaner
# ----------------------------
def clean_rationale(text: str) -> str:
    import re
    if not isinstance(text,str) or not text.strip():
        return text

    t = text.strip()
    sents = [s.strip() for s in re.split(r"[.]\s*", t) if s.strip()]
    cleaned = []

    for s in sents:
        low = s.lower()

        if low.startswith("pedestrian involvement suggests reducing conflict points"):
            cleaned.append("Improving pedestrian visibility and reducing conflict points here would help lower crash risk.")
        elif low.startswith("pedestrian involvement detected"):
            cleaned.append("This area sees meaningful pedestrian activity, increasing the chance of conflicts.")
        elif low.startswith("nighttime conditions detected"):
            cleaned.append("Crashes here often occur at night when visibility is lower.")
        elif low.startswith("higher-speed environment detected"):
            cleaned.append("This roadway supports higher speeds, increasing crash severity.")
        elif low.startswith("work zone flag indicates"):
            cleaned.append("Work zone indicators suggest temporary controls such as signage, barriers, or speed management.")
        elif low.startswith("work zone context detected"):
            cleaned.append("Work zone activity has been identified at this location.")
        elif low.startswith("fatality flag increases priority"):
            cleaned.append("A recent fatality increases the priority for stronger interventions.")
        else:
            cleaned.append(s)

    return " ".join(s + "." if not s.endswith(".") else s for s in cleaned).strip()

# ----------------------------
# Ranked Table + Details
# ----------------------------
def ranked_table_and_details(df, top_n):
    left, right = st.columns([1.35,1])
    ranked = (
        df.sort_values("expected_reduction_amount", ascending=False)
          .head(top_n).copy()
    )

    with left:
        st.subheader(f"Top {top_n} locations by expected reduction")

        cols = [
            "Address","location_id","best_action_label",
            "expected_reduction_amount","pct_reduction_norm",
            "pred_est_ttl_comp_cost","expected_cost_after_action","ai_rationale_short"
        ]

        display = ranked[cols].rename(columns={
            "best_action_label":"Recommended action",
            "expected_reduction_amount":"Expected reduction",
            "pct_reduction_norm":"% reduction",
            "pred_est_ttl_comp_cost":"Crash cost est.",
            "expected_cost_after_action":"Cost after action",
            "ai_rationale_short":"Rationale",
            "location_id":"Location"
        })

        display["Expected reduction"] = display["Expected reduction"].map(fmt_dollars)
        display["Cost after action"] = display["Cost after action"].map(fmt_dollars)
        display["Crash cost est."] = display["Crash cost est."].map(fmt_dollars)

        st.dataframe(display, use_container_width=True, hide_index=True)

    with right:
        options = ranked[["address_short","location_id"]].fillna("").copy()
        options["label"] = options["address_short"] + " (" + options["location_id"] + ")"

        sel = st.selectbox(
            "Select an address to see full rationale",
            options["label"].tolist(),
            index=0 if len(options) else None
        )

        if sel:
            loc = options.loc[options["label"] == sel, "location_id"].iloc[0]
            row = df.loc[df["location_id"] == loc].iloc[0]

            st.markdown(f"""
                **Action:** {row.get('best_action_label', pretty_action(row.get('best_action','')))}  
                **Crash cost estimate:** `{fmt_dollars(row['pred_est_ttl_comp_cost'])}`  
                **Expected reduction:** `{fmt_dollars(row['expected_reduction_amount'])}`  
                **% reduction:** {row['pct_reduction_norm']*100:.1f}%  
                **Expected cost after action:** {fmt_dollars(row['expected_cost_after_action'])}  
            """)

            st.markdown("**Rationale:**")
            st.write(clean_rationale(str(row.get("ai_rationale",""))))

# ====================================================
# ====================== UI ==========================
# ====================================================
try:
    df_raw1 = load_partner_data(CSV_PATH1)
except Exception as e:
    st.error(f"Crash dataset load failed: {e}")
    st.stop()

try:
    df_prescriptive_raw = read_csv_url(CSV_PATH2)
except Exception:
    df_prescriptive_raw = None

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("Global Filters")
    all_years = sorted(df_raw1["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    top_corr = df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"].sum().nlargest(10).index.tolist()
    selected_street = st.selectbox("📍 Corridor:", ["All Corridors"] + top_corr)

df = df_raw1[df_raw1["Year"].isin(selected_years)]
current_focus = selected_street if selected_street != "All Corridors" else "Austin District (Full View)"
if selected_street != "All Corridors":
    df = df[df["rpt_street_name"] == selected_street]

# ----------------------------
# Header
# ----------------------------
if LOGO_PATH:
    st.image(LOGO_PATH, width=180)

st.title("Safety Intelligence Dashboard")
st.caption(f"Analyzing: **{current_focus}**")

k1, k2, k3 = st.columns(3)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df["death_cnt"].sum()))
k3.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum() / 1e9:.2f}B")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚶Top Predictors", "🗺️ Geographic Risk", "📊 Speed and Severity",
    "⏰ Temporal Patterns", "💰 Transportation Mode Analysis",
    "🧠 Prescriptive Actions"
])

# ----------------------------
# Tab 1
# ----------------------------
with tab1:
    st.write(
        "##### The top predictors and prescriptive actions come from a random forest model trained on Austin crash data (2018–present)."
    )

    colL, colR = st.columns([1,2], gap="large")

    with colL:
        st.subheader("Top Predictors of Estimated Cost")
        st.image("data/processed/outputs/BI to AI SHAP vf.png", width=800)
        st.write(
            "Pedestrian involvement, motorcycle involvement, and posted speed limit "
            "are the strongest predictors of estimated crash cost."
        )

    with colR:
        st.subheader("Historical Trends")

        df_total = df.groupby("Year")["Estimated Total Comprehensive Cost"].sum().reset_index()
        fig_total = px.bar(df_total, x="Year", y="Estimated Total Comprehensive Cost", text_auto=True)
        fig_total.update_layout(
            height=400, width=800, margin=dict(l=100,r=100,t=20,b=20),
            yaxis_tickprefix="$"
        )
        avg_cost = df_total["Estimated Total Comprehensive Cost"].mean()
        fig_total.add_hline(
            y=avg_cost, line_width=3, line_dash="dash", line_color="red",
            annotation_text=f"Avg Annual Cost: ${avg_cost:,.0f}",
            annotation_position="bottom right"
        )
        fig_total.update_traces(marker_color="#5236ab")
        st.plotly_chart(fig_total)

        df_cnt = df.groupby("Year")["ID"].count().reset_index(name="Number of Crashes")
        fig_cnt = px.bar(df_cnt, x="Year", y="Number of Crashes", text_auto=True)
        fig_cnt.update_layout(height=400, width=800, margin=dict(l=100,r=100,t=20,b=20))
        fig_cnt.update_traces(marker_color="#5236ab")
        st.plotly_chart(fig_cnt)

# ----------------------------
# Tab 2
# ----------------------------
with tab2:
    colL, colR = st.columns([1,2])

    with colL:
        st.subheader("🔥 Top 10 Risk Corridors")
        risk_df = (
            df_raw1.groupby("rpt_street_name")["Estimated Total Comprehensive Cost"]
            .sum().nlargest(10).reset_index()
        )
        risk_df.columns = ["Street","Cost"]
        bar_colors = ["#4B0082" if s == selected_street else "#D8BFD8" for s in risk_df["Street"]]

        fig_bar = px.bar(risk_df, x="Cost", y="Street", orientation="h", template="plotly_white")
        fig_bar.update_traces(marker_color=bar_colors)
        fig_bar.update_layout(
            xaxis_title="Cost", yaxis_title="Street",
            xaxis=dict(tickprefix="$", tickformat=",d")
        )
        fig_bar.update_yaxes(autorange="reversed")

        st.plotly_chart(fig_bar, use_container_width=True)

    with colR:
        severity_colors = {
            "Fatal":"#991f3d","Serious Injury":"#e31937","Minor Injury":"#ff6a00",
            "Possible Injury":"#f1a425","No Injury":"#128354","Unknown":"#cccccc"
        }

        map_type = st.radio("Map Layer:", ["Economic Heatmap","Incident Clusters"], horizontal=True)
        center_lat = df["latitude"].median() if not df.empty else 30.2672
        center_lon = df["longitude"].median() if not df.empty else -97.7431

        if map_type == "Economic Heatmap":
            fig_map = px.density_mapbox(
                df,
                lat="latitude", lon="longitude",
                z="Estimated Total Comprehensive Cost",
                radius=12,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10, mapbox_style="open-street-map",
                color_continuous_scale="Purples"
            )
            fig_map.update_layout(coloraxis_colorbar=dict(
                title="Total Cost",
                tickprefix="$", tickformat=",d"
            ))

        else:
            layer_order = ["Unknown","No Injury","Possible Injury","Minor Injury","Serious Injury","Fatal"]
            fig_map = px.scatter_mapbox(
                df,
                lat="latitude", lon="longitude",
                color="Severity_Label",
                size="marker_size",
                category_orders={"Severity_Label": layer_order},
                color_discrete_map=severity_colors,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10, mapbox_style="open-street-map"
            )
            fig_map.update_layout(legend=dict(traceorder="reversed"))

        fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=600)
        st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Tab 3
# ----------------------------
with tab3:
    st.subheader(f"🛡️ Crash Risk Profile: {current_focus}")

    c1, c2, c3 = st.columns(3)

    severity_colors = {
        "Fatal":"#991f3d","Serious Injury":"#e31937","Minor Injury":"#ff6a00",
        "Possible Injury":"#f1a425","No Injury":"#128354","Unknown":"#cccccc"
    }

    with c1:
        fig_pie = px.pie(
            df, names="Severity_Label", hole=0.4,
            color="Severity_Label",
            color_discrete_map=severity_colors,
            category_orders={"Severity_Label": list(severity_colors.keys())},
            title="Crash Severity Breakdown"
        )
        fig_pie.update_layout(height=450, legend=dict(x=0.85,y=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        df_speed = df.groupby(["Speed_Bin","Severity_Label"]).size().reset_index(name="Count")
        fig_speed = px.bar(
            df_speed, x="Speed_Bin", y="Count",
            color="Severity_Label",
            barmode="stack",
            category_orders={"Severity_Label": list(severity_colors.keys())},
            color_discrete_map=severity_colors,
            title="Crashes by Speed Limit and Severity"
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    with c3:
        df_cost_speed = df.groupby("Speed_Bin")["Estimated Total Comprehensive Cost"].mean().reset_index()
        fig_costs = px.bar(
            df_cost_speed, x="Speed_Bin", y="Estimated Total Comprehensive Cost",
            title="Average Estimated Cost by Speed Bin",
            text_auto=".2s"
        )
        fig_costs.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
        fig_costs.update_traces(marker_color="#5236ab")
        st.plotly_chart(fig_costs, use_container_width=True)

    st.markdown("---")
    st.write("### 🚲 Transportation Mode Involvement")

    mode_list = [
        "Passenger Car","Bicycle","Pedestrian","Motorcycle",
        "Commercial Veh","Micromobility","E-Scooter",
        "Large Passenger Veh","Train","Motor Vehicle","Other"
    ]

    mode_counts = []
    for m in mode_list:
        if m in df.columns and df[m].sum() > 0:
            mode_counts.append({"Mode": m, "Count": df[m].sum()})

    if mode_counts:
        df_modes = pd.DataFrame(mode_counts).sort_values("Count", ascending=True)
        fig_modes = px.bar(
            df_modes, x="Count", y="Mode", orientation="h",
            title="Crash Frequency by Mode",
            color="Count", text_auto=True,
            color_continuous_scale="Purples"
        )
        fig_modes.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_modes, use_container_width=True)
    else:
        st.warning("No mode data available for this selection.")

# ----------------------------
# Tab 4
# ----------------------------
with tab4:
    st.subheader(f"Temporal Patterns: {current_focus}")

    heat_df = df.groupby(["DAY_NAME","HOUR"]).size().reset_index(name="Count")
    fig_heat = px.density_heatmap(
        heat_df,
        x="HOUR", y="DAY_NAME", z="Count",
        color_continuous_scale="Purples",
        category_orders={"DAY_NAME":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]},
        title="Crashes by Day and Hour"
    )

    fig_heat.update_traces(
        xbins=dict(start=0,end=24,size=3),
        hovertemplate="Crashes: %{z}<extra></extra>"
    )
    fig_heat.update_layout(
        xaxis=dict(
            tickvals=[0,3,6,9,12,15,18,21],
            ticktext=["12 AM","3 AM","6 AM","9 AM","12 PM","3 PM","6 PM","9 PM"]
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    df_avg_cost = df.groupby("hour_label")["Estimated Total Comprehensive Cost"].mean().reset_index()
    df_sev = df.groupby(["hour_label","Severity_Label"], observed=False).size().reset_index(name="Count")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sev in ["Fatal","Serious Injury","Minor Injury","Possible Injury","No Injury","Unknown"]:
        m = df_sev["Severity_Label"] == sev
        fig.add_trace(
            go.Bar(
                x=df_sev[m]["hour_label"],
                y=df_sev[m]["Count"],
                name=sev,
                marker_color=severity_colors.get(sev,"#ccc")
            ),
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=df_avg_cost["hour_label"],
            y=df_avg_cost["Estimated Total Comprehensive Cost"],
            name="Avg Cost ($)",
            mode="lines+markers",
            line=dict(color="#5236ab", width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Crash Severity vs. Average Cost",
        barmode="stack",
        hovermode="x unified",
        height=600
    )
    fig.update_yaxes(title_text="Crashes", secondary_y=False)
    fig.update_yaxes(title_text="Avg Cost ($)", secondary_y=True, tickprefix="$")

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tab 5
# ----------------------------
with tab5:
    st.subheader(f"📊 Economic Impact by Transportation Type: {current_focus}")
    st.write(
        "Comprehensive cost includes medical, productivity, property damage, "
        "and monetized pain & suffering."
    )

    modes = [
        "Passenger Car","Bicycle","Pedestrian","Motorcycle",
        "Commercial Veh","Micromobility","E-Scooter",
        "Large Passenger Veh","Train","Motor Vehicle","Other"
    ]

    stats = []
    for m in modes:
        if m in df.columns:
            subset = df[df[m] == 1]
            if not subset.empty:
                stats.append({
                    "Transportation Mode": m,
                    "Average Cost per Accident": subset["Estimated Total Comprehensive Cost"].mean(),
                    "Total Economic Burden": subset["Estimated Total Comprehensive Cost"].sum(),
                    "Number of Accidents": len(subset)
                })

    if stats:
        mode_df = pd.DataFrame(stats).sort_values("Average Cost per Accident", ascending=False)
        top_mode = mode_df.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Highest Avg Cost Mode", top_mode["Transportation Mode"])
        c2.metric("Avg Cost (Highest)", f"${top_mode['Average Cost per Accident']:,.0f}")
        c3.metric("Modes Analyzed", len(mode_df))

        st.write("### 💰 Average Economic Cost per Accident")
        fig_avg = px.bar(
            mode_df, x="Transportation Mode", y="Average Cost per Accident",
            text_auto=".2s"
        )
        fig_avg.update_traces(marker_color="#5236ab")
        fig_avg.update_layout(
            yaxis_tickprefix="$",
            height=500
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        st.markdown("---")
        st.write("### 🎯 Mode Vulnerability Matrix")

        fig_bubble = px.scatter(
            mode_df,
            x="Number of Accidents",
            y="Average Cost per Accident",
            size="Total Economic Burden",
            color="Transportation Mode",
            hover_name="Transportation Mode",
            size_max=60
        )
        fig_bubble.update_layout(
            yaxis_tickprefix="$",
            xaxis_tickformat=",",
            height=600
        )
        avg_all = mode_df["Average Cost per Accident"].mean()
        fig_bubble.add_hline(
            y=avg_all, line_dash="dot",
            annotation_text="Mean Avg Cost",
            annotation_position="bottom right"
        )

        st.plotly_chart(fig_bubble, use_container_width=True)

    else:
        st.warning("No mode data found for this selection.")

# ----------------------------
# Tab 6 — Prescriptive Actions
# ----------------------------
with tab6:
    st.subheader("Prescriptive Actions: Recommended Interventions & Savings")
    st.caption("Explore high-impact locations and recommended strategies.")

    if df_prescriptive_raw is None:
        st.error("Prescriptive data unavailable.")
    else:
        dfp = prepare_prescriptive_df(df_prescriptive_raw)

        # Corridor filter
        if selected_street != "All Corridors":
            dfp = dfp[dfp["address_short"].str.contains(selected_street, case=False, na=False)]

        all_actions = sorted(dfp["best_action"].dropna().unique())

        selected_actions = st.multiselect(
            "Recommended action",
            options=all_actions,
            default=all_actions,
            format_func=pretty_action,
            key="presc_actions"
        )

        top_n = st.slider("Top N locations", 10, 300, 50, 10, key="presc_topn")

        with st.expander("More filters"):
            if "severity" in dfp.columns:
                st.multiselect("Severity", sorted(dfp["severity"].dropna().unique()), key="presc_severity")
            if "district" in dfp.columns:
                st.multiselect("District", sorted(dfp["district"].dropna().unique()), key="presc_district")

        dfp_f = dfp[dfp["best_action"].isin(selected_actions)].copy()

        if st.session_state.get("presc_severity"):
            dfp_f = dfp_f[dfp_f["severity"].isin(st.session_state["presc_severity"])]
        if st.session_state.get("presc_district"):
            dfp_f = dfp_f[dfp_f["district"].isin(st.session_state["presc_district"])]

        if dfp_f.empty:
            st.warning("No data matches your filters.")
            st.stop()

        df_topn = (
            dfp_f.sort_values("expected_reduction_amount", ascending=False)
                 .head(top_n)
        )

        cLeft, cRight = st.columns([3,2], gap="large")

        with cRight:
            total_reduction = float(df_topn["expected_reduction_amount"].sum())
            pct_series = df_topn["pct_reduction"] / (100 if df_topn["pct_reduction"].max() > 1 else 1)
            median_pct = f"{pct_series.median():.1%}"

            st.metric("Total Expected Reduction", f"${total_reduction:,.0f}")
            st.metric("Median % Reduction", median_pct)
            st.metric("Locations in Scope", f"{len(df_topn):,}")

        build_map(dfp_f, top_n=top_n)
        action_bars(dfp_f, top_n=top_n)
        ranked_table_and_details(dfp_f, top_n=top_n)
