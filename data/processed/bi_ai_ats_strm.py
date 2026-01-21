import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import os


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Austin Crash Intelligence", layout="wide")

# --- DATA LOADING ---
@st.cache_data # This keeps the app fast by "remembering" the data
def load_data():
    # Use absolute path or relative path
    file_name = 'atx_crash_2025.csv'
    if not os.path.exists(file_name):
        return None
    
    df = pd.read_csv(file_name, low_memory=False)
    
    # Preprocessing
    df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
    df['HOUR'] = df['Crash timestamp'].dt.hour
    df['DAY_NAME'] = df['Crash timestamp'].dt.day_name()
    df['high_severity'] = ((df['death_cnt'] > 0) | (df['sus_serious_injry_cnt'] > 0)).astype(int)
    return df

df_raw = load_data()

if df_raw is None:
    st.error("Could not find 'atx_crash_2025.csv'. Please ensure it's in the same folder.")
    st.stop()


# --- SIDEBAR NAVIGATION & FILTERS ---
st.sidebar.title("📊 Navigation & Filters")
page = st.sidebar.radio("Go to:", ["Business Intelligence", "AI Deep Dive"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

# Filter 1: Hour of Day
hour_range = st.sidebar.slider("Select Hour Range:", 0, 23, (0, 23))

# Filter 2: Day of Week
days = df_raw['DAY_NAME'].dropna().unique().tolist()
selected_days = st.sidebar.multiselect("Select Days:", days, default=days)

# Filter 3: Speed Limit
min_speed = int(df_raw['crash_speed_limit'].min())
max_speed = int(df_raw['crash_speed_limit'].max())
speed_limit = st.sidebar.slider("Minimum Speed Limit:", min_speed, max_speed, min_speed)

# Apply Filters
df = df_raw[
    (df_raw['HOUR'].between(hour_range[0], hour_range[1])) &
    (df_raw['DAY_NAME'].isin(selected_days)) &
    (df_raw['crash_speed_limit'] >= speed_limit)
]

# --- MAIN PAGE ---
st.title(f"🚀 {page}")
st.write(f"Showing **{len(df):,}** crashes based on current filters.")

if page == "Business Intelligence":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hourly Distribution")
        fig_hour = px.histogram(df, x="HOUR", nbins=24, color_discrete_sequence=['#636EFA'])
        fig_hour.update_layout(xaxis_title="Hour (0-23)", yaxis_title="Incident Count")
        st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        st.subheader("Severity Breakdown")
        severity_sums = df[['death_cnt', 'sus_serious_injry_cnt', 'tot_injry_cnt']].sum()
        fig_pie = px.pie(
            values=severity_sums, 
            names=['Deaths', 'Serious Injuries', 'Other Injuries'],
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Interactive Map (General Density)")
    st.map(df[['latitude', 'longitude']].dropna())

elif page == "AI Deep Dive":
    st.subheader("🤖 Machine Learning Insights")
    
    col_ai1, col_ai2 = st.columns([2, 1])

    with col_ai1:
        st.markdown("**AI Geospatial Risk Clustering**")
        # Run K-Means on filtered data
        geo_df = df[['latitude', 'longitude']].dropna()
        if not geo_df.empty:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            geo_df['Risk_Cluster'] = kmeans.fit_predict(geo_df)
            fig_map = px.scatter_mapbox(
                geo_df, lat="latitude", lon="longitude", color="Risk_Cluster",
                zoom=10, mapbox_style="carto-positron"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Not enough data points for clustering.")

    with col_ai2:
        st.markdown("**Severity Predictors (Random Forest)**")
        features = ['crash_speed_limit', 'HOUR']
        model_df = df[features + ['high_severity']].dropna()
        
        if len(model_df) > 10:
            rf = RandomForestClassifier(n_estimators=50)
            rf.fit(model_df[features], model_df['high_severity'])
            
            importance = pd.DataFrame({'Factor': features, 'Weight': rf.feature_importances_})
            fig_imp = px.bar(importance, x='Weight', y='Factor', orientation='h', color='Weight')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Filter less data to see AI feature importance.")
