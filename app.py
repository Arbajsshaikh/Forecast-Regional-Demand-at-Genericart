import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import numpy as np
import random
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import utils

from utils import predict_medicine_demand



# Assuming you've already imported your function
# from your_script import predict_medicine_demand

# Dummy function placeholder (replace with your real one)
# from prediction_module import predict_medicine_demand
# For this example, we assume it's already in memory

# Load unique locations from your CSV
@st.cache_data
def load_locations():
    df = pd.read_csv("medicinal_sales_data_large.csv")
    return sorted(df['location'].unique())

# App title
st.title("🧪 Medicinal Demand Forecast")
st.subheader("Predict monthly demand for medicines by location")

# Sidebar Inputs
locations = load_locations()
selected_location = st.selectbox("Select Location", locations)

years = list(range(2015, datetime.now().year + 2))
months = list(range(1, 13))

selected_year = st.selectbox("Select Year", years, index=years.index(datetime.now().year))
selected_month = st.selectbox("Select Month", months, index=datetime.now().month - 1)

# Button to trigger forecast
if st.button("📈 Predict Demand"):
    st.info("Running prediction, please wait...")
    try:
        result_df = predict_medicine_demand(selected_location, selected_year, selected_month)
        if result_df.empty:
            st.warning("No data available for this location.")
        else:
            st.success(f"Predicted demand for {selected_location} in {selected_month}/{selected_year}")
            st.dataframe(result_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
