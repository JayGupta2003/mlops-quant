import pandas as pd
import yfinance as yf
import os

def load_stock_data(ticker, start_date, end_date):
    """
    Centralized logic to download and clean stock data.
    Returns a DataFrame.
    """
    print(f"Downloading data for {ticker} ({start_date} to {end_date})...")
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)

    if len(df) == 0:
        raise ValueError(f"No Data found for {ticker}")
    
    return df

def save_data_to_csv(df, save_path):
    """
    Helper to save the dataframe to disk.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    return df

if __name__ == "__main__":
    TICKER = "SPY"
    START = "2015-01-01"
    END = None
    SAVE_PATH = os.path.join("data", "raw", "spy_data.csv")
    
    df = load_stock_data(TICKER, START, END)
    save_data_to_csv(df, SAVE_PATH)