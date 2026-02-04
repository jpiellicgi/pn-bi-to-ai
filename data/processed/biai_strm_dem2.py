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
    
    st.subheader("📍 Target Selection")
    corridor_options = ["All Corridors"] + top_10_names + ["--- Full Street List ---"] + sorted(df_raw['rpt_street_name'].unique().tolist())
    selected_street = st.selectbox("Select a Corridor to Focus Analysis:", corridor_options)

# --- 6. APPLY FILTERS ---
df = df_raw[df_raw['Year'].isin(selected_years)]

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
        risk_df = df_raw.groupby('rpt_street_name')['Estimated Total Comprehensive Cost'].sum().nlargest(10).reset_index()
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