import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



# Load the dataset
df = pd.read_csv('accident_2017to2023.csv',sep="|", low_memory=False)


print(df.columns.tolist()) 

# --- 1. BI VISUAL: Day of Week Fatality Distribution ---
# Ensure days are plotted in chronological order
day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
day_data = df.groupby('DAY_WEEKNAME')['FATALS'].sum().reindex(day_order)



plt.figure(figsize=(10, 6))
sns.barplot(x=day_data.index, y=day_data.values, palette='viridis')
plt.title('BI Insight: Total Fatalities by Day of Week', fontsize=14)
plt.ylabel('Total Fatalities')
plt.xlabel('Day of the Week')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('bi_day_fatalities.png')


  
# --- 2. BI VISUAL: Route Type Distribution ---
route_data = df['ROUTENAME'].value_counts().head(7) # Top 7 road types
plt.figure(figsize=(8, 8))
plt.pie(route_data, labels=route_data.index, autopct='%1.1f%%', startangle=140, 
        colors=sns.color_palette('pastel'))
plt.title('BI Insight: Accident Distribution by Route Type', fontsize=14)
plt.savefig('bi_route_distribution.png')


# --- 3. AI/DS VISUAL: Predictive Feature Importance ---
# Goal: Predict "High Severity" (accidents with more than 1 fatality)
df['high_severity'] = (df['FATALS'] > 1).astype(int)

# Filter for relevant numerical features (cleaning unknown hours/codes)
model_df = df[['HOUR', 'PERSONS', 'VE_TOTAL', 'DAY_WEEKNAME', 'high_severity']].dropna()
model_df = model_df[model_df['HOUR'] <= 23] 

X = model_df.drop('high_severity', axis=1)
y = model_df['high_severity']



# Train a simple Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

"""
# Extract and plot feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='magma')
plt.title('AI Insight: Key Predictors of High-Severity Accidents', fontsize=14)
plt.xlabel('Importance Score (Machine Learning Weight)')
plt.ylabel('Accident Features')
plt.savefig('ai_feature_importance.png')

"""