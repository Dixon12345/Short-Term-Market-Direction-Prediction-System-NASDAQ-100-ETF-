import pandas as pd

def normalize_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]

    # Remove any ticker/header rows
    df = df[~df.index.astype(str).str.contains("Ticker|Price", case=False)]

    # Fix datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna()

    # Fix Adj Close if exists
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
        df.drop(columns=["Adj Close"], inplace=True)

    # Keep OHLCV only
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in wanted if c in df.columns]]

    return df.sort_index()
