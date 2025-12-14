from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="AAPL Trading Signal API")

MODEL_PATH = os.path.join("models", "model.joblib")
model = joblib.load(MODEL_PATH)

class StockFeatures(BaseModel):
    Return: float
    Volatility: float
    Dist_MA10: float

@app.get("/")
def home():
    return {"message": "Quant API is running. Go to /docs to test it."}

@app.post("/predict")
def predict(features: StockFeatures):
    """
    Accepts stock features and returns a buy/sell signal.
    """
    try:
        input_data = pd.DataFrame([features.dict()])
        prediction = model.predict(input_data)[0]
        signal = "BUY" if prediction == 1 else "SELL"
        return {"signal": signal, "prediction_int": int(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))