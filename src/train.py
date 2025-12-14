import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import mlflow
import mlflow.sklearn
import joblib
import os

from src.data_loader import load_stock_data
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

EXPERIMENT_NAME = "Quant API Experiment"

def add_technical_indicators(df):
    df = df.copy()
    
    # 1. RSI (Momentum) - Classic Overbought/Oversold
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    # 2. EMA Crossover (Trend)
    ema_short = EMAIndicator(close=df["Close"], window=12)
    ema_long = EMAIndicator(close=df["Close"], window=26)
    df["EMA_Diff"] = ema_short.ema_indicator() - ema_long.ema_indicator()

    # 3. Bollinger Bands (Volatility)
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["Close"]
    
    df.dropna(inplace=True)
    return df

def train_model():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Fetch Data (We train on SPY as a general market proxy)
    print("Fetching SPY data for training...")
    df = load_stock_data("SPY", "2015-01-01", None)

    df = add_technical_indicators(df)

    # Target: 1 if Next Day Return > 0.0 (Positive day)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    features = ["RSI", "EMA_Diff", "BB_Width"]
    X = df[features]
    y = df["Target"]

    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print(f"Training on {len(X_train)} days, Testing on {len(X_test)} days.")

    with mlflow.start_run():
        
        # Hyperparameters
        n_estimators = 200
        min_samples = 10
        max_depth = 10

        # Log them
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_split", min_samples)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", features)

        print(f"Training with n_estimators={n_estimators}...")
        clf = RandomForestClassifier(n_estimators=n_estimators, 
                                     min_samples_split=min_samples,
                                     max_depth=max_depth, 
                                     random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"Model Precision: {precision:.2f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.sklearn.log_model(clf, "random_forest_model")

        print("Model and metrics logged to MLflow.")

        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, os.path.join("models", "model.joblib"))
        print("Model saved locally to models/model.joblib")

if __name__ == "__main__":
    train_model()