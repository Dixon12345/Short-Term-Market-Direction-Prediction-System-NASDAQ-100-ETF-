import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from .cleaner import normalize_yf

def fetch_latest(symbol):
    today = datetime.now().date()
    start = today - timedelta(days=5)

    df = yf.download(
        symbol,
        start=start,
        end=today + timedelta(days=1),
        interval="1d"
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return normalize_yf(df)
