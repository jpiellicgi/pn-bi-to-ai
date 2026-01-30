import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Austin Vision Zero Command Center", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # ADJUST THIS PATH to your actual file location
    file_path = r'C:\Users\itai.makubise\code_nova\poc_land\data\atx_crash_2025.csv'
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Date'] = df['Crash timestamp'].dt.date
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    # Map Severity
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    # System Labels
    df['Road_Type'] = df['onsys_fl'].map({True: "Highway/On-System", False: "City Street/Off-System"})
    
    return df

df_raw = load_data()

if df_raw is None:
    st.error("🛑 File not found. Please check your file path.")
    st.stop()

# --- SIDEBAR: CONTROL PANEL ---
st.sidebar.header("🕹️ Vision Zero Filters")

# 1. Street Search (New Feature)
all_streets = sorted(df_raw['rpt_street_name'].dropna().unique().tolist())
selected_street = st.sidebar.selectbox("🎯 Street-Level Deep Dive:", ["All Streets"] + all_streets)

# 2. Time/Date Filters
hour_range = st.sidebar.slider("Hour of Day:", 0, 23, (0, 23))
selected_days = st.sidebar.multiselect("Days of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# 3. Severity & Road System
selected_sev = st.sidebar.multiselect("Severity:", df_raw['Severity_Label'].unique(), default=df_raw['Severity_Label'].unique())

# --- FILTER LOGIC ---
df = df_raw.copy()
if selected_street != "All Streets":
    df = df[df['rpt_street_name'] == selected_street]

df = df[
    (df['Severity_Label'].isin(selected_sev)) &
    (df['DAY_NAME'].isin(selected_days)) &
    (df['HOUR'].between(hour_range[0], hour_range[1]))
]

# --- MAIN DASHBOARD HEADER ---
st.title("🏙️ Austin Vision Zero Command Center")
if selected_street != "All Streets":
    st.subheader(f"Street Profile: {selected_street}")

# --- ROW 1: KPI METRICS WITH LOGOS ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Crashes", f"{len(df):,}")
with m2:
    st.metric("🚶 Pedestrian Deaths", int(df['pedestrian_death_count'].sum()))
with m3:
    st.metric("🚲 Bicycle Deaths", int(df['bicycle_death_count'].sum()))
with m4:
    st.metric("🏍️ Motorcycle Deaths", int(df['motorcycle_death_count'].sum()))
with m5:
    total_cost = df['Estimated Total Comprehensive Cost'].sum()
    st.metric("Total Economic Drain", f"${total_cost/1e6:.1f}M")

st.markdown("---")

# --- ROW 2: TIME SERIES & STREET DRAIN ---
tab1, tab2, tab3 = st.tabs(["📈 Time Series Analysis", "🏘️ Street & Road Analysis", "💸 Financial Impact"])

with tab1:
    st.subheader("Cumulative Economic Impact Over Time")
    # Prepare time series data
    ts_data = df.groupby('Date')['Estimated Total Comprehensive Cost'].sum().cumsum().reset_index()
    fig_ts = px.line(ts_data, x='Date', y='Estimated Total Comprehensive Cost', 
                     title="Cumulative Cost of Crashes (Year-to-Date)",
                     labels={'Estimated Total Comprehensive Cost': 'Cumulative Cost ($)'},
                     color_discrete_sequence=['#ff4b4b'])
    fig_ts.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    st.info("💡 **Discussion Point:** This chart shows the 'Burn Rate' of city resources. Each jump represents a high-cost major incident.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Crashes by Speed Limit")
        # Vision Zero core analysis: Speed vs Frequency
        speed_data = df.groupby('crash_speed_limit').size().reset_index(name='count')
        fig_speed = px.bar(speed_data, x='crash_speed_limit', y='count', 
                           title="Incident Volume by Speed Limit",
                           color='count', color_continuous_scale='YlOrRd')
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with c2:
        st.subheader("Top High-Strain Corridors")
        # Street Drain
        drain_data = df.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
        fig_drain = px.bar(drain_data, x='Estimated Total Comprehensive Cost', y='rpt_street_name', 
                           orientation='h', title="Top 10 Streets by Economic Drain",
                           color='Estimated Total Comprehensive Cost', color_continuous_scale='Purples')
        fig_drain.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_drain, use_container_width=True)

with tab3:
    st.subheader("Vulnerable User Vulnerability Split")
    vru_cols = ['pedestrian_death_count', 'bicycle_death_count', 'motorcycle_death_count', 'motor_vehicle_death_count']
    vru_sums = df[vru_cols].sum()
    
    fig_vru_pie = px.pie(values=vru_sums.values, names=['Pedestrian', 'Bicycle', 'Motorcycle', 'Motor Vehicle'],
                         title="Fatality Distribution by User Type",
                         hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_vru_pie, use_container_width=True)

    # Comparison Scatter
    st.subheader("Crash Severity vs. Economic Cost")
    fig_scatter = px.scatter(df, x="crash_speed_limit", y="Estimated Total Comprehensive Cost",
                             color="Severity_Label", size="tot_injry_cnt", hover_data=['rpt_street_name'],
                             title="High Speed vs. High Cost Correlation")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- MAP VIEW ---
st.subheader("🗺️ Geographic Incident Distribution")
st.map(df[['latitude', 'longitude']].dropna())

# --- RAW DATA VIEW ---
with st.expander("🔍 Detailed Records"):
    st.write(df[['Crash timestamp', 'rpt_street_name', 'crash_speed_limit', 'Severity_Label', 'Estimated Total Comprehensive Cost']].sort_values(by='Estimated Total Comprehensive Cost', ascending=False))