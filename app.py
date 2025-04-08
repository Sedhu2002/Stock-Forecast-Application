
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📈 Stock Price Forecasting with NeuralProphet")

# Input Section
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., INFY.NS)", value="INFY.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-10-01"))

if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching and preparing data..."):
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for this ticker and date range.")
        else:
            data = data[['Close']].reset_index()
            data.columns = ['ds', 'y']

            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            model = NeuralProphet(
                n_forecasts=30,
                n_lags=60,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )

            metrics = model.fit(train_data, freq='D')

            future = model.make_future_dataframe(test_data, periods=len(test_data))
            forecast = model.predict(future)

            merged = test_data.merge(forecast[['ds', 'yhat1']], on='ds', how='inner').dropna()
            y_true = merged['y'].values
            y_pred = merged['yhat1'].values

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            st.success(f"R²: {r2:.4f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

            st.subheader("📊 Forecast Components (Trend, Seasonality)")
future_full = model.make_future_dataframe(data, periods=30)
forecast_full = model.predict(future_full)

# Plot each component separately
fig_components = model.plot_components(forecast_full)
for fig in fig_components:
    st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔍 Forecast Data Preview")
            st.dataframe(forecast_full[['ds', 'yhat1']].tail(30))
else:
    st.info("Enter ticker and click 'Run Forecast' to begin.")
