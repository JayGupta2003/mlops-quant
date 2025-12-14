import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

EXPERIMENT_NAME = "AAPL_Trading_Signal"

def feature_engineering(df):
    """
    Creates Technical indicators.
    """
    df = df.copy()

    # 1. Daily Return (Today's Close / Yesterday's Close - 1)
    df["Return"] = df["Close"].pct_change()

    # 2. Volatility (5-day rolling standard deviation of returns)
    df["Volatility"] = df["Return"].rolling(window=5).std()

    # 3. Moving Average (10-day) relative to price
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["Dist_MA10"] = df["Close"] / df["MA_10"] - 1

    # Target: 1 if Next Day's Close > Today's Close, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

def train_model():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load Data
    data_path = os.path.join("data", "raw", "aapl_data.csv")
    df = pd.read_csv(data_path)
    df_processed = feature_engineering(df)

    # Define features (X) and target (y)
    features = ['Return', 'Volatility', 'Dist_MA10']
    X = df_processed[features]
    y = df_processed['Target']

    split_point = int(len(df_processed) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print(f"Training on {len(X_train)} days, Testing on {len(X_test)} days.")

    with mlflow.start_run():
        
        # Hyperparameters
        n_estimators = 200
        min_samples = 10

        # Log them
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_split", min_samples)
        mlflow.log_param("features", features)

        print(f"Training with n_estimators={n_estimators}...")
        clf = RandomForestClassifier(n_estimators=n_estimators, 
                                     min_samples_split=min_samples, 
                                     random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(clf, "random_forest_model")
        print("Model and metrics logged to MLflow.")

        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, os.path.join("models", "model.joblib"))
        print("Model saved locally to models/model.joblib")

if __name__ == "__main__":
    train_model()