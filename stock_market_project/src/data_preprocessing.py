import os
from functools import lru_cache

import pandas as pd


DATA_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "..")
# Use absolute path to the dataset folder in the workspace
NIFTY50_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "..", "NSE-Data-main", "Nifty50 Stocks 20 Year Data")
)
# Additional NSE stocks dataset
NSE_STOCKS_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "..", "NSE-stock-market-historical-data-main", "v1")
)


@lru_cache(maxsize=32)
def list_nifty50_companies():
    """List available Nifty 50 company tickers from the data directory."""
    if not os.path.isdir(NIFTY50_FOLDER):
        raise FileNotFoundError(f"Data folder not found: {NIFTY50_FOLDER}")

    files = [f for f in os.listdir(NIFTY50_FOLDER) if f.endswith(".csv")]
    tickers = sorted([os.path.splitext(f)[0] for f in files])
    return tickers


@lru_cache(maxsize=32)
def list_available_companies():
    """List all available company tickers from both Nifty50 and NSE datasets."""
    tickers = set()

    # Add Nifty50 stocks
    if os.path.isdir(NIFTY50_FOLDER):
        files = [f for f in os.listdir(NIFTY50_FOLDER) if f.endswith(".csv")]
        nifty_tickers = [os.path.splitext(f)[0] for f in files]
        tickers.update(nifty_tickers)

    # Add NSE stocks (remove .NS suffix)
    if os.path.isdir(NSE_STOCKS_FOLDER):
        files = [f for f in os.listdir(NSE_STOCKS_FOLDER) if f.endswith(".NS.csv")]
        nse_tickers = [os.path.splitext(os.path.splitext(f)[0])[0] for f in files]  # Remove both .NS and .csv
        tickers.update(nse_tickers)

    return sorted(list(tickers))


@lru_cache(maxsize=32)
def load_stock_data(ticker: str, data_dir: str = None) -> pd.DataFrame:
    """Load stock historical data for a given ticker from the local CSV dataset."""
    if data_dir is None:
        # First try Nifty50 folder
        path = os.path.join(NIFTY50_FOLDER, f"{ticker}.csv")
        if not os.path.isfile(path):
            # Try NSE stocks folder with .NS.csv extension
            path = os.path.join(NSE_STOCKS_FOLDER, f"{ticker}.NS.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Stock CSV not found for ticker '{ticker}' in either data directory")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stock CSV not found for ticker '{ticker}' at {path}")

    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # convert date and ensure sorting
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        raise ValueError(f"CSV for {ticker} does not contain a 'Date' column")

    # ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_by_date(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Filter the data between start_date and end_date inclusive."""
    out = df.copy()
    if start_date is not None:
        out = out[out["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out["Date"] <= pd.to_datetime(end_date)]
    return out.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_nifty_index() -> pd.DataFrame:
    """Build a simple NIFTY index proxy using the mean of all constituent closing prices."""
    tickers = list_nifty50_companies()
    frames = []
    for ticker in tickers:
        try:
            df = load_stock_data(ticker)[["Date", "Close"]].rename(columns={"Close": ticker})
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("No stock data found to calculate NIFTY index")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.ffill().bfill()

    merged["NIFTY_INDEX"] = merged[[t for t in tickers if t in merged.columns]].mean(axis=1)
    return merged[["Date", "NIFTY_INDEX"]]
