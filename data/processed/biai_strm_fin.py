import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Austin Crash Command Center 2025", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # ADJUST THIS PATH to your actual file location
    # Use the 'r' before the quotes for Windows paths
    file_path = r'C:\Users\itai.makubise\code_nova\poc_land\data\atx_crash_2025.csv'

    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    # Map Severity IDs to Names
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    # Road Type Label
    df['Road_Type'] = df['onsys_fl'].map({True: "Highway/On-System", False: "City Street/Off-System"})
    
    # Flag for Vulnerable Road Users (VRU)
    df['is_vru_fatal'] = (df['pedestrian_death_count'] > 0) | (df['bicycle_death_count'] > 0)
    
    return df

df_raw = load_data()

# --- SAFETY GATE ---
if df_raw is None:
    st.error("🛑 File not found. Please update the 'file_path' in the code to your local CSV path.")
    st.stop()

# --- SIDEBAR: ADVANCED FILTERS ---
st.sidebar.header("🕹️ Control Panel")

# 1. Financial Filter
min_cost = float(df_raw['Estimated Total Comprehensive Cost'].min())
max_cost = float(df_raw['Estimated Total Comprehensive Cost'].max())
selected_cost = st.sidebar.slider("Financial Impact Range ($):", min_cost, max_cost, (min_cost, max_cost))

# 2. Severity Filter
all_severities = df_raw['Severity_Label'].unique().tolist()
selected_sev = st.sidebar.multiselect("Crash Severity:", all_severities, default=all_severities)

# 3. Road System Filter
road_types = df_raw['Road_Type'].unique().tolist()
selected_roads = st.sidebar.multiselect("Road System:", road_types, default=road_types)

# 4. Time Filter
day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_days = st.sidebar.multiselect("Days of Week:", day_list, default=day_list)
hour_range = st.sidebar.slider("Hour of Day:", 0, 23, (0, 23))

# APPLY ALL FILTERS
df = df_raw[
    (df_raw['Estimated Total Comprehensive Cost'].between(selected_cost[0], selected_cost[1])) &
    (df_raw['Severity_Label'].isin(selected_sev)) &
    (df_raw['Road_Type'].isin(selected_roads)) &
    (df_raw['DAY_NAME'].isin(selected_days)) &
    (df_raw['HOUR'].between(hour_range[0], hour_range[1]))
]

# --- MAIN DASHBOARD ---
st.title("🚔 Austin Traffic Safety & Economic Command Center")

# --- ROW 1: KPI METRICS (Including Financials) ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
with col_kpi1:
    st.metric("Total Incidents", f"{len(df):,}")
with col_kpi2:
    st.metric("Fatalities", int(df['death_cnt'].sum()))
with col_kpi3:
    vru_deaths = int(df['pedestrian_death_count'].sum() + df['bicycle_death_count'].sum())
    st.metric("Ped/Bike Deaths", vru_deaths, delta="Vulnerable Users", delta_color="inverse")
with col_kpi4:
    total_cost = df['Estimated Total Comprehensive Cost'].sum()
    st.metric("Economic Impact", f"${total_cost/1e6:.1f}M")
with col_kpi5:
    avg_cost = df['Estimated Total Comprehensive Cost'].mean() if len(df) > 0 else 0
    st.metric("Avg Cost / Crash", f"${avg_cost/1e3:.1f}K")

st.markdown("---")

# --- ROW 2: ANALYSIS TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Incident Trends", "📍 Geographic Risk", "💰 Financial & VRU Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly Peak Analysis")
        fig_hour = px.area(df.groupby('HOUR').size().reset_index(name='count'), 
                           x='HOUR', y='count', title="Crash Volume by Hour",
                           color_discrete_sequence=['#ef233c'])
        st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        st.subheader("Volume by Injury Severity")
        fig_sev = px.bar(df['Severity_Label'].value_counts().reset_index(), 
                         x='count', y='Severity_Label', orientation='h', 
                         color='Severity_Label', color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_sev, use_container_width=True)

with tab2:
    col_map, col_street = st.columns([2, 1])
    with col_map:
        st.subheader("Collision Heatmap")
        st.map(df[['latitude', 'longitude']].dropna())
    with col_street:
        st.subheader("Top High-Risk Streets")
        top_streets = df['rpt_street_name'].value_counts().head(10).reset_index()
        fig_top = px.bar(top_streets, x='count', y='rpt_street_name', orientation='h',
                         title="Highest Frequency", color='count', color_continuous_scale='Reds')
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

with tab3:
    st.subheader("Economic Impact & Vulnerable Road Users")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Cost by Street Treemap
        street_cost = df.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
        fig_tree = px.treemap(street_cost, path=['rpt_street_name'], values='Estimated Total Comprehensive Cost',
                              title="Economic Drain by Street (Top 10)",
                              color='Estimated Total Comprehensive Cost', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with col_f2:
        # Pedestrian/Bike Fatality counts
        vru_stats = pd.DataFrame({
            'Type': ['Pedestrian Deaths', 'Bicycle Deaths', 'Motorcycle Deaths'],
            'Count': [df['pedestrian_death_count'].sum(), 
                      df['bicycle_death_count'].sum(), 
                      df['motorcycle_death_count'].sum()]
        })
        fig_vru = px.pie(vru_stats, values='Count', names='Type', title="Vulnerable User Fatality Split",
                         hole=0.4, color_discrete_sequence=px.colors.sequential.OrRd_r)
        st.plotly_chart(fig_vru, use_container_width=True)

    # Max vs Total Cost comparison
    st.subheader("Max vs. Total Comprehensive Cost Comparison")
    fig_comp = px.scatter(df, x='Estimated Total Comprehensive Cost', y='Estimated Maximum Comprehensive Cost',
                          color='Severity_Label', size='death_cnt', hover_data=['rpt_street_name'],
                          title="Cost Variance Analysis (Bubble size = Death Count)")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- RAW DATA ---
with st.expander("📝 View Detailed Records"):
    st.dataframe(df[['Crash timestamp', 'rpt_street_name', 'Severity_Label', 
                     'Estimated Total Comprehensive Cost', 'pedestrian_death_count', 'bicycle_death_count']])