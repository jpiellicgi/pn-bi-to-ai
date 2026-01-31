import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TxDOT | Austin Safety Intelligence", 
    layout="wide",
    page_icon="🛣️"
)

# --- PATH CONFIGURATION ---
DATA_DIR = r'C:\Users\itai.makubise\code_nova\pn-bi-to-ai\data'
CSV_FILENAME = 'atx_crash_data_2018-2026_cleansed.csv'
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

# --- SMART LOGO LOADER ---
def get_txdot_logo():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.webp']
    for ext in extensions:
        pattern = os.path.join(DATA_DIR, 'txdot*' + ext)
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None

LOGO_PATH = get_txdot_logo()

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Preprocessing Timestamps
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Year'] = df['Crash timestamp'].dt.year
    df['Month'] = df['Crash timestamp'].dt.month_name()
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    # Severity Mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    # Coordinate Cleaning
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # --- DATA SANITIZATION (The Fix for your Error) ---
    # Convert 'N/A' strings to NaN, then fill NaN with 0
    if 'tot_injry_cnt' in df.columns:
        df['tot_injry_cnt'] = pd.to_numeric(df['tot_injry_cnt'], errors='coerce').fillna(0)
    
    df['crash_speed_limit'] = pd.to_numeric(df['crash_speed_limit'], errors='coerce').fillna(0)
    df['Estimated Total Comprehensive Cost'] = pd.to_numeric(df['Estimated Total Comprehensive Cost'], errors='coerce').fillna(0)
    
    return df.dropna(subset=['latitude', 'longitude'])

df_raw = load_data()

if df_raw is None:
    st.error(f"🛑 Dataset not found: {CSV_PATH}")
    st.stop()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### **District Operations Control**")
    
    years = sorted(df_raw['Year'].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", years, default=years[-3:])
    
    streets = sorted(df_raw['rpt_street_name'].dropna().unique().tolist())
    selected_street = st.selectbox("🎯 Corridor Focus:", ["All Corridors"] + streets)
    
    selected_sev = st.multiselect("⚠️ Severity Filter:", 
                                   df_raw['Severity_Label'].unique().tolist(), 
                                   default=["Fatal", "Serious Injury", "Minor Injury"])

# --- FILTER LOGIC ---
df = df_raw.copy()
df = df[df['Year'].isin(selected_years)]
if selected_street != "All Corridors":
    df = df[df['rpt_street_name'] == selected_street]
df = df[df['Severity_Label'].isin(selected_sev)]

# --- HEADER ---
h1, h2 = st.columns([1, 5])
with h1:
    if LOGO_PATH: st.image(LOGO_PATH, width=120)
with h2:
    st.title("Vision Zero: Austin District Command")
    st.subheader(f"Safety Analysis Period: {min(selected_years)} - {max(selected_years)}")

# --- ROW 1: KPI METRICS ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Total Incidents", f"{len(df):,}")
with m2: 
    fatal_pct = (len(df[df['Severity_Label'] == 'Fatal']) / len(df) * 100) if len(df) > 0 else 0
    st.metric("Fatality Rate", f"{fatal_pct:.1f}%")
with m3: 
    # Safe calculation after sanitization
    total_injuries = int(df['tot_injry_cnt'].sum()) if 'tot_injry_cnt' in df.columns else 0
    st.metric("Total Injuries", f"{total_injuries:,}")
with m4:
    avg_speed = df['crash_speed_limit'].mean()
    st.metric("Avg Speed Limit", f"{avg_speed:.0f} MPH")
with m5:
    total_cost = df['Estimated Total Comprehensive Cost'].sum()
    st.metric("Economic Impact", f"${total_cost/1e6:.1f}M")

st.markdown("---")

# --- TABS: DRILL-THROUGH ---
tab1, tab2, tab3, tab4 = st.tabs(["📉 Trend Analysis", "🗺️ GIS Hotspots", "🏎️ Speed & Risk", "⏰ Temporal Patterns"])

# (Tab logic remains the same as previous version)
with tab1:
    st.subheader("Monthly Incident Volume (Year-over-Year)")
    trend_df = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    fig_trend = px.line(trend_df, x='Month', y='Count', color='Year', 
                        category_orders={'Month': month_order}, markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Geospatial High-Injury Network")
    fig_map = px.density_mapbox(df, lat='latitude', lon='longitude', z='Estimated Total Comprehensive Cost',
                                radius=10, center=dict(lat=30.2672, lon=-97.7431), zoom=10,
                                mapbox_style="carto-darkmatter")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=550)
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Top 10 High-Risk Corridors")
    street_risk = df.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
    fig_risk = px.bar(street_risk, x='Estimated Total Comprehensive Cost', y='rpt_street_name', orientation='h', color_continuous_scale='Reds')
    st.plotly_chart(fig_risk, use_container_width=True)

with tab4:
    st.subheader("Peak Risk Windows (Hour vs Day)")
    heat_df = df.groupby(['DAY_NAME', 'HOUR']).size().reset_index(name='Count')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig_heat = px.density_heatmap(heat_df, x='HOUR', y='DAY_NAME', z='Count', category_orders={'DAY_NAME': day_order})
    st.plotly_chart(fig_heat, use_container_width=True)