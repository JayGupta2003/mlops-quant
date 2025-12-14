import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_api_prediction():
    payload = {"ticker": "AAPL"}
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    json_response = response.json()
    
    assert "signal" in json_response
    assert "confidence" in json_response
    assert "RSI" in json_response["features"]