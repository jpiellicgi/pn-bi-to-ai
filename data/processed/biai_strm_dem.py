import streamlit as st
import pandas as pd
import plotly.express as px
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
        pattern = os.path.join(DATA_DIR, 'txdot' + ext)
        files = glob.glob(pattern)
        if files: return files[0]
    return None

LOGO_PATH = get_txdot_logo()

# --- 4. DATA PIPELINE ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH): return None
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Preprocessing Timestamps
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['Year'] = df['Crash timestamp'].dt.year
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    
    # Severity Mapping
    sev_map = {1: "Fatal", 2: "Serious Injury", 3: "Minor Injury", 4: "Possible Injury", 0: "No Injury", 5: "Unknown"}
    df['Severity_Label'] = df['crash_sev_id'].map(sev_map)
    
    # --- ROBUST NUMERIC HANDLING (Fixes fillna error) ---
    cols_to_fix = ['tot_injry_cnt', 'crash_speed_limit', 'death_cnt', 'Estimated Total Comprehensive Cost']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = pd.Series([0] * len(df))
    
    # --- TRANSPORT MODE MAPPING ---
    mode_map = {
        'pedestrian_fl': ['pedestrian_fl', 'ped_fl'],
        'bicycle_fl': ['bicycle_fl', 'bike_fl'],
        'motorcycle_fl': ['motorcycle_fl', 'mc_fl'],
        'comml_mtr_veh_fl': ['comml_mtr_veh_fl', 'cmv_fl']
    }
    
    for standard_col, names in mode_map.items():
        existing = next((n for n in names if n in df.columns), None)
        if existing:
            df[standard_col] = df[existing].apply(lambda x: 1 if str(x).strip().upper() in ['Y', '1', '1.0'] else 0)
        else:
            df[standard_col] = 0

    # Speed Binning
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    df['Speed_Bin'] = pd.cut(df['crash_speed_limit'], bins=bins, labels=labels)
    
    # Marker size logic
    df['marker_size'] = (df['crash_speed_limit'] / 5).clip(lower=2)
    
    return df.dropna(subset=['latitude', 'longitude'])

df_raw = load_data()

if df_raw is None:
    st.error(f"🛑 Dataset not found at {CSV_PATH}")
    st.stop()

# --- 5. SIDEBAR FILTERS ---
with st.sidebar:
    if LOGO_PATH: st.image(LOGO_PATH, use_container_width=True)
    st.title("Strategic Filters")
    all_years = sorted(df_raw['Year'].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Fiscal Years:", all_years, default=all_years[-4:])
    
    top_10_names = df_raw.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).index.tolist()
    selected_street = st.selectbox("🎯 Corridor Focus:", ["All Corridors"] + top_10_names + ["--- Full List ---"] + sorted(df_raw['rpt_street_name'].unique().tolist()))

# Apply Filters
df = df_raw[df_raw['Year'].isin(selected_years)]
if selected_street not in ["All Corridors", "--- Full List ---"]:
    df = df[df['rpt_street_name'] == selected_street]
    current_focus = selected_street
else:
    current_focus = "Austin District (Full)"

# --- 6. HEADER & KPIs ---
st.title("Vision Zero: Safety Intelligence Dashboard")
st.caption(f"Currently Analyzing: **{current_focus}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Crash Volume", f"{len(df):,}")
k2.metric("Lives Lost", int(df['death_cnt'].sum()))
k3.metric("Avg Speed Limit", f"{df['crash_speed_limit'].mean():.1f} MPH")
k4.metric("Economic Impact", f"${df['Estimated Total Comprehensive Cost'].sum()/1e9:.2f}B")

# --- 7. TABS ---
tab1, tab2, tab3 = st.tabs(["🗺️ Geographic Risk", "📊 Incident Risk Profile", "⏰ Temporal Patterns"])

# --- TAB 1: GEOGRAPHIC RISK ---
with tab1:
    col_list, col_map = st.columns([1, 2])
    with col_list:
        st.subheader("🔥 Top 10 High-Risk Streets")
        risk_df = df_raw.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
        risk_df.columns = ['Street', 'Cost']
        bar_colors = ['#FF4B4B' if s == selected_street else '#31333F' for s in risk_df['Street']]
        fig_bar = px.bar(risk_df, x='Cost', y='Street', orientation='h', template="plotly_white")
        fig_bar.update_traces(marker_color=bar_colors)
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_map:
        map_type = st.radio("Map Layer:", ["Incident Clusters", "Economic Heatmap"], horizontal=True)
        lat_c, lon_c, zoom_c = (df['latitude'].median(), df['longitude'].median(), 13) if selected_street != "All Corridors" else (30.2672, -97.7431, 10)
        
        if map_type == "Economic Heatmap":
            fig_m = px.density_mapbox(df, lat='latitude', lon='longitude', z='Estimated Total Comprehensive Cost', radius=10, 
                                     center=dict(lat=lat_c, lon=lon_c), zoom=zoom_c, mapbox_style="carto-darkmatter")
        else:
            fig_m = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='Severity_Label', size='marker_size',
                                     center=dict(lat=lat_c, lon=lon_c), zoom=zoom_c, mapbox_style="carto-positron")
        fig_m.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=550)
        st.plotly_chart(fig_m, use_container_width=True)

# --- TAB 2: INCIDENT RISK PROFILE ---
with tab2:
    st.subheader("Risk & Physics Profile")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.write("**Crash Volume per Hour**")
        hr_vol = df.groupby('HOUR').size().reset_index(name='Volume')
        st.plotly_chart(px.line(hr_vol, x='HOUR', y='Volume', markers=True, color_discrete_sequence=['#FF4B4B']), use_container_width=True)
    with r1c2:
        st.write("**Volume by Severity**")
        # FIXED: Using sequential palette to avoid Reds_r error
        fig_pie = px.pie(df, names='Severity_Label', hole=0.4, color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig_pie, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.write("**Crash Distribution by Speed Bracket**")
        speed_df = df['Speed_Bin'].value_counts().sort_index().reset_index()
        st.plotly_chart(px.bar(speed_df, x='Speed_Bin', y='count', color='count', color_continuous_scale='OrRd'), use_container_width=True)
        
    with r2c2:
        st.write("**Avg Economic Cost by Transport Mode**")
        mode_data = []
        target_modes = {'Pedestrian': 'pedestrian_fl', 'Bicycle': 'bicycle_fl', 'Motorcycle': 'motorcycle_fl', 'CMV': 'comml_mtr_veh_fl'}
        for label, col in target_modes.items():
            if col in df.columns:
                avg_cost = df[df[col] > 0]['Estimated Total Comprehensive Cost'].mean()
                if not pd.isna(avg_cost): mode_data.append({'Mode': label, 'Avg Cost': avg_cost})
        
        if mode_data:
            st.plotly_chart(px.bar(pd.DataFrame(mode_data), x='Mode', y='Avg Cost', color='Avg Cost', color_continuous_scale='Reds'), use_container_width=True)
            

# --- TAB 3: TEMPORAL PATTERNS ---
with tab3:
    st.subheader("Temporal Risk Density")
    heat_df = df.groupby(['DAY_NAME', 'HOUR']).size().reset_index(name='Count')
    fig_heat = px.density_heatmap(heat_df, x='HOUR', y='DAY_NAME', z='Count', 
                                  category_orders={'DAY_NAME': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
                                  color_continuous_scale='YlOrRd')
    st.plotly_chart(fig_heat, use_container_width=True)

    