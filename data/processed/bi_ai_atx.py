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