import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import os

print(os.getcwd())
# Load the Austin dataset
df = pd.read_csv('atx_crash_2025.csv', low_memory=False)

# Preprocessing
df['Crash timestamp'] = pd.to_datetime(df['Crash timestamp (US/Central)'], errors='coerce')
df['HOUR'] = df['Crash timestamp'].dt.hour
df['DAY_WEEK'] = df['Crash timestamp'].dt.dayofweek

# Define high severity: deaths or serious injuries
df['high_severity'] = ((df['death_cnt'] > 0) | (df['sus_serious_injry_cnt'] > 0)).astype(int)



# --- BI PLOT 1: Hourly Distribution ---
plt.figure(figsize=(12, 6))
sns.countplot(data=df.dropna(subset=['HOUR']), x='HOUR', palette='viridis', hue='HOUR', legend=False)
plt.title('BI Insight: Hourly Distribution of Austin Crashes (2025)', fontsize=14)
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Number of Incidents')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('bi_atx_hourly_distribution.png')
#plt.close()


# --- BI PLOT 2: Severity Pie Chart ---
## stacked bar chart 
severity_sums = df[['death_cnt', 'sus_serious_injry_cnt', 'tot_injry_cnt']].sum()
labels = ['Deaths', 'Serious Injuries', 'Total Injuries (All)']
plt.figure(figsize=(8, 8))
plt.pie(severity_sums, labels=labels, autopct='%1.1f%%', colors=['#e63946', '#f4a261', '#2a9d8f'], startangle=140, explode=[0.1, 0.05, 0])
plt.title('BI Insight: Austin Crash Severity Breakdown', fontsize=14)
plt.savefig('bi_atx_severity_pie.png')
# plt.close()


# --- AI PLOT 1: Geospatial Risk Clustering ---
geo_df = df[['latitude', 'longitude']].dropna()
# Using 6 clusters to find major hotspots in the Austin area
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
geo_df['Cluster'] = kmeans.fit_predict(geo_df)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(geo_df['longitude'], geo_df['latitude'], c=geo_df['Cluster'], cmap='Set1', s=5, alpha=0.5)
plt.title('AI Insight: K-Means Hotspot Clustering (Austin)', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(*scatter.legend_elements(), title="Risk Clusters", loc="upper right")
plt.grid(True, alpha=0.3)
plt.savefig('ai_atx_hotspots.png')
# plt.close()


# --- AI PLOT 2: Feature Importance ---
# Convert booleans to ints
# maps based on cost and see if we can filters on streamlit 
df['onsys_fl_int'] = df['onsys_fl'].astype(int)
df['private_dr_int'] = df['private_dr_fl'].astype(int)

features = ['crash_speed_limit', 'HOUR', 'DAY_WEEK', 'onsys_fl_int', 'private_dr_int']
model_df = df[features + ['high_severity']].dropna()

X = model_df[features]
y = model_df['high_severity']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': ['Speed Limit', 'Hour', 'Day of Week', 'On-System Road', 'Private Drive'],
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma', hue='Feature', legend=False)
plt.title('AI Insight: Key Predictors of High Severity Crashes (Austin)', fontsize=14)
plt.tight_layout()
plt.savefig('ai_atx_severity_drivers.png')
# plt.close()

print("All plots generated and saved.")
