import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TxDOT | Austin Safety Intelligence Elite", 
    layout="wide",
    page_icon="🛣️"
)

# --- 2. PATH CONFIGURATION ---
DATA_DIR = r'C:\Users\itai.makubise\code_nova\pn-bi-to-ai\data'
CSV_FILENAME = 'atx_crash_data_2018-2026_cleansed.csv'
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

# --- 3. SMART ASSET LOADER ---
def get_txdot_logo():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.webp']
    for ext in extensions:
        pattern = os.path.join(DATA_DIR, 'txdot*' + ext)
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None

LOGO_PATH = get_txdot_logo()

# --- 4. DATA PIPELINE ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Year'] = df['Crash timestamp'].dt.year
    df['Month'] = df['Crash timestamp'].dt.month_name()
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    def safe_numeric(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        return pd.Series([0] * len(df))

    df['tot_injry_cnt'] = safe_numeric('tot_injry_cnt')
    df['crash_speed_limit'] = safe_numeric('crash_speed_limit')
    df['death_cnt'] = safe_numeric('death_cnt')
    df['Estimated Total Comprehensive Cost'] = safe_numeric('Estimated Total Comprehensive Cost')
    
    # Safe clipping for markers
    df['marker_size'] = (df['crash_speed_limit'] / 5).clip(lower=0) + 2
    
    return df.dropna(subset=['latitude', 'longitude'])

df_raw = load_data()

if df_raw is None:
    st.error(f"🛑 Dataset not found at: {CSV_PATH}")
    st.stop()

# --- 5. SIDEBAR FILTERS (Global Controls) ---
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.title("Strategic Filters")
    st.markdown("---")
    years = sorted(df_raw['Year'].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", years, default=years[-4:])
    
    # Pre-calculate Top 10 for the Sidebar Shortcut
    top_10_names = df_raw.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).index.tolist()
    
    streets = sorted(df_raw['rpt_street_name'].dropna().unique().tolist())
    selected_street = st.selectbox("🎯 Corridor Focus:", ["All Corridors"] + streets)
    
    st.info("💡 **Tip:** Selecting a corridor will auto-zoom the map and filter all analytics tabs.")

# Apply Filters
df = df_raw.copy()
df = df[df['Year'].isin(selected_years)]
if selected_street != "All Corridors":
    df = df[df['rpt_street_name'] == selected_street]

# --- 6. HEADER & KPIs ---
h1, h2 = st.columns([1, 5])
with h1: 
    if LOGO_PATH: st.image(LOGO_PATH, width=120)
with h2:
    st.title("Vision Zero: Enterprise Safety Command")
    st.caption("Strategic Decision Support Platform | TxDOT Austin District")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Crash Volume", f"{len(df):,}")
m2.metric("Lives Lost", int(df['death_cnt'].sum()))
m3.metric("Avg Speed Limit", f"{df['crash_speed_limit'].mean():.1f} MPH")
m4.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum()/1e9:.2f}B")

st.markdown("---")

# --- 7. TOP-LEVEL NAVIGATION TABS ---
tab_exec, tab_geo, tab_temp, tab_speed = st.tabs([
    "🚀 Executive Overview", 
    "🗺️ Geographic Intelligence", 
    "⏰ Temporal Patterns", 
    "🏎️ Speed & Risk Analysis"
])

# TAB 1: EXECUTIVE OVERVIEW
with tab_exec:
    st.subheader("Historical Volume Trends")
    yearly_data = df.groupby('Year').size().reset_index(name='Crashes')
    fig_area = px.area(yearly_data, x='Year', y='Crashes', line_shape='spline', color_discrete_sequence=['#FF4B4B'])
    st.plotly_chart(fig_area, use_container_width=True)

# TAB 2: GEOGRAPHIC INTELLIGENCE (INTERACTIVE)
with tab_geo:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("🔥 Top 10 High-Risk Corridors")
        # Calculate Top 10 for the visual
        top_10_df = df_raw.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
        top_10_df.columns = ['Street', 'Total_Cost']
        
        fig_top10 = px.bar(top_10_df, x='Total_Cost', y='Street', orientation='h',
                           color='Total_Cost', color_continuous_scale='Reds',
                           title="Economic Burden Leaderboard")
        fig_top10.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top10, use_container_width=True)
        st.caption("Selection above in 'Corridor Focus' will isolate these on the map.")

    with c2:
        map_choice = st.radio("Select Layer:", ["Economic Heatmap", "Incident Clusters"], horizontal=True)
        
        # Smart Map Centering: Zoom in if a street is selected, otherwise show Austin overview
        if selected_street != "All Corridors" and not df.empty:
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            zoom_level = 13
        else:
            center_lat = 30.2672
            center_lon = -97.7431
            zoom_level = 10

        if map_choice == "Economic Heatmap":
            fig_map = px.density_mapbox(df, lat='latitude', lon='longitude', z='Estimated Total Comprehensive Cost',
                                        radius=12, center=dict(lat=center_lat, lon=center_lon), zoom=zoom_level,
                                        mapbox_style="carto-darkmatter", color_continuous_scale="Viridis")
        else:
            fig_map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='Severity_Label', 
                                        size='marker_size', size_max=12, 
                                        center=dict(lat=center_lat, lon=center_lon), zoom=zoom_level,
                                        mapbox_style="carto-positron")
        
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=550)
        st.plotly_chart(fig_map, use_container_width=True)

# TAB 3: TEMPORAL PATTERNS
with tab_temp:
    st.subheader("Crash Density Heatmap (Hour vs Day)")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heat_df = df.groupby(['DAY_NAME', 'HOUR']).size().reset_index(name='Count')
    fig_heat = px.density_heatmap(heat_df, x='HOUR', y='DAY_NAME', z='Count', 
                                  category_orders={'DAY_NAME': day_order}, color_continuous_scale='YlOrRd')
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 4: SPEED & RISK ANALYSIS
with tab_speed:
    st.subheader("The Lethality Matrix")
    risk_mat = df.groupby('rpt_street_name').agg({'death_cnt': 'sum', 'Estimated Total Comprehensive Cost': 'count', 'crash_speed_limit': 'mean'}).reset_index()
    risk_mat.columns = ['Street', 'Deaths', 'Crashes', 'Avg_Speed']
    risk_mat = risk_mat[risk_mat['Crashes'] > 5]
    fig_scat = px.scatter(risk_mat, x='Crashes', y='Deaths', size='Avg_Speed', color='Avg_Speed', 
                          hover_name='Street', color_continuous_scale='Reds')
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("---")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Severity vs. Speed Limit")
        st.plotly_chart(px.box(df, x='Severity_Label', y='crash_speed_limit', color='Severity_Label'), use_container_width=True)
    with col_v2:
        st.subheader("High-Speed Fatality Audit")
        high_speed_fatal = df[df['death_cnt'] > 0].groupby('rpt_street_name')['crash_speed_limit'].mean().nlargest(10).reset_index()
        st.table(high_speed_fatal)