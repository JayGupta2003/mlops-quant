from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from src.data_loader import load_stock_data
from src.train import add_technical_indicators
import datetime
import os

app = FastAPI(title="Quant API")

MODEL_PATH = os.path.join("models", "model.joblib")
model = joblib.load(MODEL_PATH)

class TickerRequest(BaseModel):
    ticker: str

def get_latest_indicators(ticker_symbol):
    """
    Fetches the last year of data to calculate today's indicators.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    df = load_stock_data(ticker_symbol, start_date, end_date)
   
    if len(df) < 30:
        raise ValueError("Not enough data to calculate indicators")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = add_technical_indicators(df)
    
    # Return only the most recent row (Today's data)
    latest = df.iloc[-1][["RSI", "EMA_Diff", "BB_Width"]]
    return pd.DataFrame([latest])

@app.post("/predict")
def predict(request: TickerRequest):
    """
    Accepts stock features and returns a buy/sell signal.
    """
    try:
        input_data = get_latest_indicators(request.ticker)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        signal = "BUY" if prediction == 1 else "SELL"
        confidence = f"{probability * 100:.2f}%"

        return {
            "ticker": request.ticker,
            "signal": signal,
            "confidence": confidence,
            "features": input_data.to_dict(orient="records")[0]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))