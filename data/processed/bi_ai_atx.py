#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# 1. LOAD DATA (Handling potential FileNotFoundError)
try:
    df_atx = pd.read_csv('atx_crash_2025.csv', low_memory=False)
    print("Successfully loaded ATX Crash Data.")
except FileNotFoundError:
    print("Error: 'atx_crash_2025.csv' not found. Please check the file path.")
    exit()


# 2. DATA PREPROCESSING (Fixing KeyErrors and extracting Time features)
# Convert timestamp to datetime to extract HOUR and DAY_WEEK
df_atx['Crash timestamp'] = pd.to_datetime(df_atx['Crash timestamp (US/Central)'])
df_atx['HOUR'] = df_atx['Crash timestamp'].dt.hour
df_atx['DAY_WEEK'] = df_atx['Crash timestamp'].dt.dayofweek # 0=Monday, 6=Sunday


# Define Severity Target for AI (High Severity = Death or Serious Injury)
df_atx['high_severity'] = ((df_atx['death_cnt'] > 0) | (df_atx['sus_serious_injry_cnt'] > 0)).astype(int)

# ==========================================
# PHASE 1: BUSINESS INTELLIGENCE (BI)
# ==========================================

# Insight 1: Hourly Distribution of Crashes
plt.figure(figsize=(10, 5))
sns.countplot(data=df_atx, x='HOUR', palette='viridis')
plt.title('BI: Austin Crashes by Hour of Day (2025)')
plt.savefig('bi_atx_hourly_distribution.png')

# Insight 2: Fatality vs Injury Contribution
totals = df_atx[['death_cnt', 'sus_serious_injry_cnt', 'tot_injry_cnt']].sum()
plt.figure(figsize=(8, 8))
plt.pie(totals, labels=['Deaths', 'Serious Injuries', 'Other Injuries'], autopct='%1.1f%%', colors=['#ff4d4d', '#ff9933', '#66b3ff'])
plt.title('BI: Proportion of Crash Outcomes in Austin')
plt.savefig('bi_atx_severity_pie.png')

# ==========================================
# PHASE 2: ARTIFICIAL INTELLIGENCE (AI)
# ==========================================

# Insight 3: AI Geospatial Risk Clustering
# Focusing on where accidents happen regardless of street names
geo_data = df_atx[['latitude', 'longitude']].dropna()
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
geo_data['Risk_Cluster'] = kmeans.fit_predict(geo_data)

plt.figure(figsize=(10, 8))
plt.scatter(geo_data['longitude'], geo_data['latitude'], c=geo_data['Risk_Cluster'], cmap='tab10', s=2, alpha=0.4)
plt.title('AI: Machine Learning Hotspot Identification (Austin Clusters)')
plt.savefig('ai_atx_hotspots.png')

# Insight 4: Feature Importance for Crash Severity
# Using AI to see what predicts high-severity outcomes
features = ['crash_speed_limit', 'HOUR', 'DAY_WEEK', 'onsys_fl', 'private_dr_fl']
model_df = df_atx[features + ['high_severity']].dropna()

X = model_df[features]
y = model_df['high_severity']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.DataFrame({'Factor': features, 'Weight': rf.feature_importances_}).sort_values(by='Weight', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=importance, x='Weight', y='Factor', palette='magma')
plt.title('AI: Top Predictors of Serious/Fatal Crashes in Austin')
plt.savefig('ai_atx_severity_drivers.png')

print("BI and AI Insights generated successfully.")