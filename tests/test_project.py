import os
import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.data_loader import load_stock_data

def test_data_loader():
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-01-07"
    save_path = "data/test_data.csv"

    df = load_stock_data(ticker, start, end, save_path)

    assert os.path.exists(save_path)
    assert len(df) > 0               
    assert "Close" in df.columns

    # Clean up the test file
    if os.path.exists(save_path):
        os.remove(save_path)

client = TestClient(app)

def test_api_prediction():
    payload = {
        "Return": 0.01,
        "Volatility": 0.02,
        "Dist_MA10": 0.05
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "signal" in json_response
    assert "prediction_int" in json_response
    assert json_response["signal"] in ["BUY", "SELL"]

if __name__ == "__main__":
    pytest.main()