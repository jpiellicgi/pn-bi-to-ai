import streamlit as st
import pandas as pd
from pathlib import Path

# EXAMPLE CODE BELOW TO GET STARTED WITH STREAMLIT

# Set page config
st.set_page_config(page_title="Vehicle Analysis", layout="wide")

# Define path to your processed CSV
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_PATH = ROOT / "data" / "processed" / "vehicles_2017to2023.csv"

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

st.title("Top 10 Vehicle MAKE & MODEL Counts")

# Group by MAKE and MODEL
top_models = (
    df.groupby(["MAKE", "MODEL"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)

# Display table
st.write("### Top 10 MAKE & MODEL combinations")
st.dataframe(top_models)

# Add a bar chart
st.bar_chart(top_models.set_index("MODEL")["count"])
