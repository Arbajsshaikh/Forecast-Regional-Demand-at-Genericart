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


def predict_medicine_demand(location: str, year: int, month: int):
    results = []
    products=('Medicine_4', 'Medicine_10', 'Medicine_5', 'Medicine_7','Medicine_3', 'Medicine_2', 'Medicine_8', 'Medicine_1','Medicine_6', 'Medicine_9')
    for product in products:
        try:
            monthly_data=pd.read_csv('monthly_data.csv')
            # Get last 3 values from the data
            ts_df = monthly_data[(monthly_data['product'] == product) & (monthly_data['location'] == location)].copy()
            ts_df.set_index('date', inplace=True)
            ts_df = ts_df.sort_index()
            ts = ts_df['quantity']
            if len(ts) < 6:
                continue

            # Load ARIMA model
            arima_model = joblib.load(f"models/arima_{product}_{location}.pkl")
            arima_forecast = arima_model.forecast(steps=1)[0]

            # Load ANN
            ann_model = tf.keras.models.load_model(f"models/ann_{product}_{location}.h5")
            scaler_X = joblib.load(f"models/ann_scalerX_{product}_{location}.pkl")
            scaler_y = joblib.load(f"models/ann_scalerY_{product}_{location}.pkl")

            last_3 = ts[-3:].values.reshape(1, -1)
            last_scaled = scaler_X.transform(last_3)
            ann_pred_scaled = ann_model.predict(last_scaled, verbose=0)
            ann_forecast = scaler_y.inverse_transform(ann_pred_scaled)[0][0]

            # Load LSTM
            lstm_model = tf.keras.models.load_model(f"models/lstm_{product}_{location}.h5")
            lstm_scaler = joblib.load(f"models/lstm_scaler_{product}_{location}.pkl")

            series_scaled = lstm_scaler.transform(ts.values.reshape(-1, 1)).flatten()
            last_seq = series_scaled[-3:].reshape(1, 3, 1)
            lstm_pred_scaled = lstm_model.predict(last_seq, verbose=0)[0][0]
            lstm_forecast = lstm_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]

            results.append({
                "Product": product,
                "ARIMA Pred": round(arima_forecast, 2),
                "ANN Pred": round(ann_forecast, 2),
                "LSTM Pred": round(lstm_forecast, 2),
                "Average Requirement":round((arima_forecast+ann_forecast+lstm_forecast)/3,2)
            })

        except Exception as e:
            print(f"Failed to predict for {product} - {location}: {e}")
    return pd.DataFrame(results)
