import pandas as pd
import os
from .fetcher import fetch_latest
from .merger import merge_datasets

# Ensure required folders exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw_data", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

# Correct file paths for your structure
HIST_PATH = "data/processed/new_QQQ_data.csv"       # cleaned master dataset
RAW_PATH  = "data/raw_data/yfinance_data.csv"       # raw downloaded data
LOG_PATH  = "data/logs/pipeline.log"                # pipeline log file


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg)


def run_pipeline(symbol="QQQ"):
    log("\n=== Running QQQ Pipeline ===")

    # Load historical clean data
    old_df = pd.read_csv(HIST_PATH, parse_dates=["Date"], index_col="Date")

    # Fetch last few days
    new_df = fetch_latest(symbol)

    if new_df.empty:
        log("No new data from yfinance. Skipping update.")
        return old_df

    # Keep rows newer than last existing date
    last_date = old_df.index.max()
    new_df = new_df[new_df.index > last_date]

    if new_df.empty:
        log("No newer rows. Data already up to date.")
        return old_df

    # Save raw fetch
    new_df.to_csv(RAW_PATH)
    log(f"Raw new data saved to {RAW_PATH}")

    # Merge datasets
    full_df = merge_datasets(old_df, new_df)

    # Save updated historical dataset
    full_df.to_csv(HIST_PATH)
    log(f"Historical updated at {HIST_PATH}")

    log("Done.")
    return full_df


if __name__ == "__main__":
    run_pipeline()
