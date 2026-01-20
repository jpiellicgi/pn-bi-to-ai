#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#1. Load the dataset
# Ensure 'accident.csv' is in your working directory
df = pd.read_csv('accident_2017to2023.csv',sep='|', low_memory=False)

# Prints the first row with headers
print(df.head(1))

#Accessing the Raw Header from the File
with open('accident_2017to2023.csv', 'r') as f:
    first_line = f.readline()
    print(first_line)

# 2. Data Cleaning & Numeric Conversion
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUD'] = pd.to_numeric(df['LONGITUD'], errors='coerce')
df['FATALS'] = pd.to_numeric(df['FATALS'], errors='coerce')

# Filter for valid coordinates within the contiguous United States
valid_coords = df[
    (df['LATITUDE'] < 50) & (df['LATITUDE'] > 24) &
    (df['LONGITUD'] > -125) & (df['LONGITUD'] < -66)
].dropna(subset=['LATITUDE', 'LONGITUD', 'FATALS']).copy()

# --- HEATMAP 1: General Accident Density ---
plt.figure(figsize=(14, 8))
hb1 = plt.hexbin(valid_coords['LONGITUD'], valid_coords['LATITUDE'], 
                 gridsize=80, cmap='YlOrRd', bins='log', mincnt=1)
plt.colorbar(hb1, label='Log10(Number of Accidents)')
plt.title('Heatmap 1: General Accident Density (2017-2023)', fontsize=15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(alpha=0.2)
plt.savefig('heatmap_density_2017_2023.png', dpi=300)

# --- HEATMAP 2: Fatality Risk Hotspots ---
plt.figure(figsize=(14, 8))
hb2 = plt.hexbin(valid_coords['LONGITUD'], valid_coords['LATITUDE'], 
                 C=valid_coords['FATALS'], reduce_C_function=np.sum, 
                 gridsize=80, cmap='Reds', mincnt=1)
plt.colorbar(hb2, label='Total Fatalities (Sum)')
plt.title('Heatmap 2: Fatality Risk Hotspots (Severity Weighted)', fontsize=15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(alpha=0.2)
plt.savefig('heatmap_fatality_2017_2023.png', dpi=300)

# --- HEATMAP 3: Nighttime Accident Risk ---
night_df = valid_coords[valid_coords['LGT_CONDNAME'].str.contains('Dark', case=False, na=False)]
plt.figure(figsize=(14, 8))
hb3 = plt.hexbin(night_df['LONGITUD'], night_df['LATITUDE'], 
                 gridsize=80, cmap='magma', bins='log', mincnt=1)
plt.colorbar(hb3, label='Log10(Nighttime Accidents)')
plt.title('Heatmap 3: Nighttime Accident Risk Analysis', fontsize=15)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(alpha=0.2)
plt.savefig('heatmap_night_2017_2023.png', dpi=300)