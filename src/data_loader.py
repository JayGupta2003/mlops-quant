import pandas as pd
import yfinance as yf
import os

def load_stock_data(ticker, start_date, end_date, save_path):
    """
    Downloads historical stock data from Yahoo Finance and saves it.
    """
    print(f"Downloading data for {ticker} ({start_date} to {end_date})...")
    
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    print(f"Rows: {len(df)}")
    return df

if __name__ == "__main__":
    TICKER = "AAPL"
    START = "2018-01-01"
    END = None
    SAVE_PATH = os.path.join("data", "raw", "aapl_data.csv")
    
    load_stock_data(TICKER, START, END, SAVE_PATH)