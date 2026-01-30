import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TxDOT | Austin Vision Zero Command Center", 
    layout="wide",
    page_icon="🛣️"
)

# --- PATH CONFIGURATION ---
DATA_DIR = r'C:\Users\itai.makubise\code_nova\poc_land\data'
CSV_PATH = os.path.join(DATA_DIR, 'atx_crash_2025.csv')

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
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Date'] = df['Crash timestamp'].dt.date
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['crash_speed_limit'] = pd.to_numeric(df['crash_speed_limit'], errors='coerce').fillna(0)
    df['map_size'] = df['crash_speed_limit'].apply(lambda x: x if x > 0 else 5)
    df['Estimated Total Comprehensive Cost'] = pd.to_numeric(df['Estimated Total Comprehensive Cost'], errors='coerce').fillna(0)
    
    return df.dropna(subset=['latitude', 'longitude'])

df_raw = load_data()

if df_raw is None:
    st.error(f"🛑 CSV file not found at: {CSV_PATH}")
    st.stop()

# --- SIDEBAR: BRANDING & FILTERS ---
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### **Austin District Operations**")
    st.divider()
    
    clean_street_list = df_raw['rpt_street_name'].dropna()
    clean_street_list = clean_street_list[~clean_street_list.str.contains("NOT REPORTED|UNKNOWN", case=False)]
    all_streets = sorted(clean_street_list.unique().tolist())

    selected_street = st.selectbox("🎯 Target Corridor:", ["All Streets"] + all_streets)
    hour_range = st.slider("Hour of Day:", 0, 23, (0, 23))
    selected_sev = st.multiselect("Severity Level:", df_raw['Severity_Label'].unique().tolist(), default=df_raw['Severity_Label'].unique().tolist())

# --- FILTER LOGIC ---
df = df_raw.copy()
if selected_street != "All Streets":
    df = df[df['rpt_street_name'] == selected_street]
df = df[(df['Severity_Label'].isin(selected_sev)) & (df['HOUR'].between(hour_range[0], hour_range[1]))]

# --- MAIN DASHBOARD HEADER ---
head_col1, head_col2 = st.columns([1, 5])
with head_col1:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=150)
with head_col2:
    st.title("Vision Zero Intelligence Portal")
    st.subheader("Texas Department of Transportation | Austin District")

# --- ROW 1: KPI METRICS ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Total Crashes", f"{len(df):,}")
with m2: st.metric("🚶 Pedestrian Deaths", int(df['pedestrian_death_count'].sum()))
with m3: st.metric("🚲 Bicycle Deaths", int(df['bicycle_death_count'].sum()))
with m4: st.metric("🏍️ Motorcycle Deaths", int(df['motorcycle_death_count'].sum()))
with m5:
    total_cost = df['Estimated Total Comprehensive Cost'].sum()
    st.metric("Economic Impact", f"${total_cost/1e6:.1f}M")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["💸 Financial Burden", "🗺️ GIS Mapping", "🛣️ Street Intelligence", "🚲 VRU Analysis"])

# TAB 1: FINANCIAL BURDEN
with tab1:
    st.subheader("Economic Burden Analysis")
    cost_sev = df.groupby('Severity_Label')['Estimated Total Comprehensive Cost'].sum().reset_index()
    fig_donut = px.pie(cost_sev, values='Estimated Total Comprehensive Cost', names='Severity_Label', 
                       hole=0.4, title="Comprehensive Cost by Severity",
                       color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig_donut, use_container_width=True)

# TAB 2: GIS MAPPING
with tab2:
    st.subheader("Geospatial Incident Intelligence")
    view_mode = st.radio("Overlay Type:", ["Heatmap", "Incident Markers"], horizontal=True)
    
    if view_mode == "Heatmap":
        fig_map = px.density_mapbox(df, lat='latitude', lon='longitude', z='Estimated Total Comprehensive Cost',
                                    radius=10, center=dict(lat=30.2672, lon=-97.7431), zoom=10,
                                    mapbox_style="carto-darkmatter")
    else:
        fig_map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='Severity_Label', 
                                    size='map_size', size_max=12, center=dict(lat=30.2672, lon=-97.7431), zoom=10,
                                    mapbox_style="carto-positron")
    
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    st.plotly_chart(fig_map, use_container_width=True)

# TAB 3: STREET INTELLIGENCE
with tab3:
    st.subheader("📍 High-Risk Street Intelligence Index")
    street_df = df[~df['rpt_street_name'].str.contains("NOT REPORTED|UNKNOWN", case=False, na=True)]
    risk_index = street_df.groupby('rpt_street_name').agg({
        'ID': 'count', 'death_cnt': 'sum', 'sus_serious_injry_cnt': 'sum', 'Estimated Total Comprehensive Cost': 'sum'
    }).reset_index()
    
    risk_index.columns = ['Street Name', 'Total Incidents', 'Death Count', 'Serious Injuries', 'Total Comprehensive Cost']
    risk_index = risk_index.nlargest(10, 'Total Incidents')

    st.dataframe(risk_index.style.format({
        'Total Comprehensive Cost': '${:,.0f}', 'Total Incidents': '{:,}', 
        'Death Count': '{:,}', 'Serious Injuries': '{:,}'
    }).background_gradient(subset=['Total Comprehensive Cost'], cmap='YlOrRd'), use_container_width=True, hide_index=True)

# TAB 4: VRU ANALYSIS
with tab4:
    st.subheader("Vulnerable Road User (VRU) Safety")
    vru_counts = pd.DataFrame({
        'User Type': ['Pedestrian', 'Bicycle', 'Motorcycle'],
        'Fatalities': [df['pedestrian_death_count'].sum(), df['bicycle_death_count'].sum(), df['motorcycle_death_count'].sum()]
    })
    fig_vru = px.bar(vru_counts, x='User Type', y='Fatalities', color='User Type', 
                     title="Fatality Breakdown by Mode",
                     color_discrete_map={'Pedestrian':'#E63946', 'Bicycle':'#F1FAEE', 'Motorcycle':'#A8DADC'})
    st.plotly_chart(fig_vru, use_container_width=True)