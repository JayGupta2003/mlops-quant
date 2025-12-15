import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
import datetime
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.train import add_technical_indicators, train_model
from src.data_loader import load_stock_data

st.set_page_config(page_title="Quant Pipeline", layout="wide")

@st.cache_resource
def load_model():
    model_path = "models/model.joblib"

    if not os.path.exists(model_path):
        st.warning("Model not found! Training a new one... (This may take a minute)")
        os.makedirs("models", exist_ok=True)
        train_model()
        st.success("Training Complete!")
    
    return joblib.load(model_path)

model = load_model()

st.sidebar.title("Quant Pipeline")
ticker = st.sidebar.text_input("Enter Ticker:", value="NVDA")
st.sidebar.markdown("---")

if st.sidebar.button("Analyse and Predict"):
    try:
        with st.spinner(f"Fetching data for {ticker}"):
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365)

            df = load_stock_data(ticker, start_date, end_date)
            df = add_technical_indicators(df)

            latest = df.iloc[-1]
            features = pd.DataFrame([latest[["RSI", "EMA_Diff", "BB_Width"]]])

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            signal = "BUY" if prediction == 1 else "SELL"
            st.sidebar.subheader("AI Signal")
            st.sidebar.header(f"{signal}")
            st.sidebar.write(f"Confidence: {probability * 100:.2f}%")

            st.title(f"{ticker} Stock Analysis")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'
            ))

            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='gray', width=1), name='BB Lower'))

            fig.update_layout(height=600, title=f"{ticker} Price Action & Bollinger Bands")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("RSI (14)", f"{latest['RSI']:.2f}")
            col2.metric("EMA Diff", f"{latest['EMA_Diff']:.2f}")
            col3.metric("Bollinger Width", f"{latest['BB_Width']:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")