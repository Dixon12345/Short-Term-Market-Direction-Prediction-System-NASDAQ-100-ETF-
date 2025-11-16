import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================================================
# 1. Fetch LIVE last row from yfinance
# =========================================================
def fetch_live_yf(ticker="QQQ"):
    df = yf.download(ticker, period="7d", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df.index.name = "Date"
    df.reset_index(inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("Date", inplace=True)

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.iloc[[-1]]


# =========================================================
# 2. Load CLEAN pipeline dataset
# =========================================================
def load_clean_pipeline_dataset(path):
    try:
        return pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    except:
        return None


# =========================================================
# 3. Feature Engineering (12 features EXACTLY matching training)
# =========================================================
def compute_atr(df, window=5):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_rsi(series, window=7):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta).clip(lower=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def make_features_v3(df):
    df = df.copy()

    # flatten in case MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["returns"] = df["Close"].pct_change()
    df["ATR5"] = compute_atr(df, 5)
    df["std5"] = df["returns"].rolling(5).std()
    df["norm_tr"] = (df["High"] - df["Low"]) / df["Close"]
    df["vol_ratio_5"] = df["ATR5"] / df["std5"]
    df["roc2"] = df["Close"].pct_change(2)
    df["roc5"] = df["Close"].pct_change(5)
    df["rsi7"] = compute_rsi(df["Close"], 7)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["candle_body_pct"] = ((df["Close"] - df["Open"]).abs() /
                             (df["High"] - df["Low"]).replace(0, np.nan)).fillna(0)
    df["vol_ma5"] = df["Volume"].rolling(5).mean()
    df["vol_spike_5"] = df["Volume"] / df["vol_ma5"]
    df["return_1d"] = df["Close"].pct_change(1)

    final_cols = [
        "ATR5", "std5", "norm_tr", "vol_ratio_5",
        "roc2", "roc5", "rsi7",
        "MA5", "MA10",
        "candle_body_pct",
        "vol_spike_5",
        "return_1d"
    ]

    return df[final_cols].dropna()


# =========================================================
# 4. Build LIVE + CLEAN merged dataset
# =========================================================
def build_live_dataset(clean_df, ticker="QQQ"):
    live_row = fetch_live_yf(ticker)
    if live_row.empty:
        return clean_df

    df = pd.concat([clean_df, live_row])
    df = df[~df.index.duplicated(keep="last")]
    return df


# =========================================================
# 5. Model + Prediction
# =========================================================
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba_model(model, X):
    if isinstance(model, xgb.Booster):
        d = xgb.DMatrix(X)
        return model.predict(d)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    preds = model.predict(X)
    return np.array(preds, float)
